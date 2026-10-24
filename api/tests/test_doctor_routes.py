"""Doctor report routes (handlers called directly, DB + queue faked)."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api.routes import doctor as routes
from api.services.workflow.doctor import analysis
from api.services.workflow.doctor.contracts import CreateReportRequest

USER = SimpleNamespace(id=3, selected_organization_id=1)


class FakeDB:
    def __init__(self):
        self.report = None
        self.opted_in = True

    async def get_workflow(self, workflow_id, organization_id=None):
        return SimpleNamespace(id=workflow_id) if workflow_id == 9 else None

    async def create_analysis_report(
        self, *, organization_id, workflow_id, created_by, params
    ):
        self.report = SimpleNamespace(
            report_uuid="r-1",
            organization_id=organization_id,
            workflow_id=workflow_id,
            created_by=created_by,
            status="queued",
            params=params,
            result={},
            error=None,
            created_at=datetime.now(UTC),
        )
        return self.report

    async def get_analysis_report(self, report_uuid, organization_id):
        if (
            self.report
            and report_uuid == self.report.report_uuid
            and organization_id == 1
        ):
            return self.report
        return None

    async def update_analysis_report(
        self, report_uuid, organization_id, *, status=None, result=None, error=None
    ):
        if status is not None:
            self.report.status = getattr(status, "value", status)
        if result is not None:
            self.report.result = result
        if error is not None:
            self.report.error = error
        return self.report

    async def count_active_reports(self, organization_id=None):
        return 2

    async def get_configuration_value(self, organization_id, key, default=None):
        return self.opted_in


@pytest.fixture
def env(monkeypatch):
    db = FakeDB()
    jobs = []

    async def fake_enqueue(name, *args, **kwargs):
        jobs.append((name, args, kwargs))

    monkeypatch.setattr(routes, "db_client", db)
    monkeypatch.setattr(analysis, "db_client", db)
    monkeypatch.setattr(routes, "enqueue_job", fake_enqueue)
    return db, jobs


async def test_create_enqueues_and_reports_ai_state(env, monkeypatch):
    db, jobs = env
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    resp = await routes.create_doctor_report(9, CreateReportRequest(limit=20), USER)
    assert resp.status.value == "queued" and resp.queue_position == 2
    assert resp.result.ai_enabled is False
    assert resp.result.ai_disabled_reason == "no_key"
    ((name, args, kwargs),) = jobs
    assert name == "analyze_workflow_calls" and args == ("r-1", 1)
    assert kwargs["_job_id"].startswith("doctor:r-1:")


async def test_create_for_foreign_workflow_is_404(env):
    with pytest.raises(HTTPException) as exc:
        await routes.create_doctor_report(77, CreateReportRequest(), USER)
    assert exc.value.status_code == 404


async def test_resume_only_partial_or_failed(env, monkeypatch):
    db, jobs = env
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    await routes.create_doctor_report(9, CreateReportRequest(), USER)
    with pytest.raises(HTTPException) as exc:
        await routes.resume_doctor_report("r-1", USER)
    assert exc.value.status_code == 409

    db.report.status = "partial"
    db.report.error = "Analysis stopped at problems"
    resp = await routes.resume_doctor_report("r-1", USER)
    assert resp.status.value == "queued" and resp.error is None
    assert resp.result.ai_enabled is True
    assert len(jobs) == 2 and jobs[0][2]["_job_id"] != jobs[1][2]["_job_id"]


async def test_get_report_is_org_scoped(env):
    await routes.create_doctor_report(9, CreateReportRequest(), USER)
    other = SimpleNamespace(id=4, selected_organization_id=2)
    with pytest.raises(HTTPException) as exc:
        await routes.get_doctor_report("r-1", other)
    assert exc.value.status_code == 404
    resp = await routes.get_doctor_report("r-1", USER)
    assert resp.report_uuid == "r-1"


async def test_doctor_settings_round_trip(monkeypatch):
    """PUT then GET /doctor/settings stores the opt-in for the user's org."""
    from types import SimpleNamespace

    from api.routes import doctor as doctor_routes

    store: dict = {}

    async def fake_upsert(org_id, key, value, last_validated_at=None):
        store[(org_id, key)] = value

    async def fake_get(org_id, key, default=None):
        return store.get((org_id, key), default)

    monkeypatch.setattr(doctor_routes.db_client, "upsert_configuration", fake_upsert)
    monkeypatch.setattr(doctor_routes.db_client, "get_configuration_value", fake_get)
    user = SimpleNamespace(selected_organization_id=7)

    before = await doctor_routes.get_doctor_settings(user=user)
    assert before.ai_opt_in is False

    await doctor_routes.update_doctor_settings(
        doctor_routes.DoctorSettings(ai_opt_in=True), user=user
    )
    assert store[(7, doctor_routes.DOCTOR_AI_OPT_IN_KEY)] == {"enabled": True}
    after = await doctor_routes.get_doctor_settings(user=user)
    assert after.ai_opt_in is True
    # Another org is unaffected.
    other = await doctor_routes.get_doctor_settings(
        user=SimpleNamespace(selected_organization_id=8)
    )
    assert other.ai_opt_in is False
