"""AnalysisReportClient (Agent Doctor) against Postgres test_db.

Covers org scoping, report lifecycle, candidate definitions staying out of
version history and draft lookups (D17), and scenario CRUD.
"""

from datetime import UTC, datetime, timedelta

import pytest

from api.db.models import WorkflowRunModel
from api.services.workflow.doctor.contracts import (
    CANDIDATE_DEFINITION_STATUS,
    DOCTOR_REPLAY_RUN_MODE,
    REPORT_RETENTION_DAYS,
    ReportStatus,
)


async def _org_with_workflow(db_session, slug: str):
    user, _ = await db_session.get_or_create_user_by_provider_id(f"{slug}_user")
    org, _ = await db_session.get_or_create_organization_by_provider_id(
        f"{slug}_org", user.id
    )
    await db_session.update_user_selected_organization(user.id, org.id)
    workflow = await db_session.create_workflow(
        name=f"{slug} workflow",
        workflow_definition={"nodes": [], "edges": []},
        user_id=user.id,
        organization_id=org.id,
    )
    return user, org, workflow


async def test_report_lifecycle_is_org_scoped(db_session):
    user, org, workflow = await _org_with_workflow(db_session, "doctor_report")
    _, other_org, _ = await _org_with_workflow(db_session, "doctor_report_other")

    report = await db_session.create_analysis_report(
        organization_id=org.id,
        workflow_id=workflow.id,
        created_by=user.id,
        params={"limit": 50},
    )
    assert report.status == ReportStatus.queued.value
    assert report.params == {"limit": 50}
    assert len(report.report_uuid) == 36
    retention = report.expires_at - report.created_at
    assert abs(retention - timedelta(days=REPORT_RETENTION_DAYS)) < timedelta(seconds=5)

    assert (
        await db_session.get_analysis_report(report.report_uuid, other_org.id) is None
    )
    assert (
        await db_session.update_analysis_report(
            report.report_uuid, other_org.id, status=ReportStatus.failed
        )
        is None
    )

    assert await db_session.count_active_reports(org.id) == 1
    updated = await db_session.update_analysis_report(
        report.report_uuid,
        org.id,
        status=ReportStatus.completed,
        result={"run_ids": [1]},
        error="partial note",
    )
    assert updated.status == "completed"
    assert updated.result == {"run_ids": [1]}
    assert updated.error == "partial note"
    assert await db_session.count_active_reports(org.id) == 0

    listed = await db_session.list_analysis_reports(workflow.id, org.id)
    assert [r.report_uuid for r in listed] == [report.report_uuid]
    assert await db_session.list_analysis_reports(workflow.id, other_org.id) == []


async def test_create_report_rejects_foreign_workflow(db_session):
    _, org, _ = await _org_with_workflow(db_session, "doctor_foreign_a")
    _, _, other_workflow = await _org_with_workflow(db_session, "doctor_foreign_b")
    with pytest.raises(ValueError):
        await db_session.create_analysis_report(
            organization_id=org.id,
            workflow_id=other_workflow.id,
            created_by=None,
            params={},
        )


async def test_candidate_hidden_from_history_and_drafts(db_session):
    _, org, workflow = await _org_with_workflow(db_session, "doctor_candidate")
    _, other_org, _ = await _org_with_workflow(db_session, "doctor_candidate_x")

    versions_before = await db_session.get_workflow_versions(workflow.id)
    candidate = await db_session.create_candidate_definition(
        workflow_id=workflow.id,
        organization_id=org.id,
        workflow_json={"nodes": [{"id": "n1"}], "edges": []},
        workflow_configurations={"max_call_duration": 300},
    )
    assert candidate.status == CANDIDATE_DEFINITION_STATUS
    assert candidate.is_current is False
    assert candidate.version_number is None

    versions_after = await db_session.get_workflow_versions(workflow.id)
    assert [v.id for v in versions_after] == [v.id for v in versions_before]
    assert await db_session.get_draft_version(workflow.id) is None
    reloaded = await db_session.get_workflow(workflow.id, organization_id=org.id)
    assert reloaded.current_definition.id != candidate.id
    assert reloaded.released_definition.id != candidate.id

    assert (await db_session.get_candidate_definition(candidate.id, org.id)).id == (
        candidate.id
    )
    assert await db_session.get_candidate_definition(candidate.id, other_org.id) is None
    # Published definitions are never returned or deleted as candidates.
    published_id = reloaded.released_definition.id
    assert await db_session.get_candidate_definition(published_id, org.id) is None
    assert await db_session.delete_candidate_definition(published_id, org.id) is False

    assert (
        await db_session.delete_candidate_definition(candidate.id, other_org.id)
        is False
    )
    assert await db_session.delete_candidate_definition(candidate.id, org.id) is True
    assert await db_session.get_candidate_definition(candidate.id, org.id) is None


async def test_runs_with_logs_for_workflow(db_session, async_session):
    _, org, workflow = await _org_with_workflow(db_session, "doctor_input")
    _, other_org, _ = await _org_with_workflow(db_session, "doctor_input_x")
    now = datetime.now(UTC)

    def run(mode, minutes_ago, completed=True):
        return WorkflowRunModel(
            name=f"run-{mode}",
            workflow_id=workflow.id,
            mode=mode,
            is_completed=completed,
            created_at=now - timedelta(minutes=minutes_ago),
            logs={"realtime_feedback_events": [{"type": "x"}]},
            usage_info={"call_duration_seconds": 12},
            annotations={"qa": 1},
        )

    newest, older = run("twilio", 1), run("webrtc", 5)
    async_session.add_all(
        [newest, older, run(DOCTOR_REPLAY_RUN_MODE, 2), run("twilio", 3, False)]
    )
    await async_session.flush()

    rows = await db_session.get_runs_with_logs_for_workflow(
        workflow.id, org.id, limit=10
    )
    assert [r["id"] for r in rows] == [newest.id, older.id]
    assert set(rows[0]) == {
        "id",
        "created_at",
        "definition_id",
        "mode",
        "logs",
        "usage_info",
        "gathered_context",
        "initial_context",
        "annotations",
    }
    assert rows[0]["logs"] == {"realtime_feedback_events": [{"type": "x"}]}

    limited = await db_session.get_runs_with_logs_for_workflow(
        workflow.id, org.id, limit=1
    )
    assert [r["id"] for r in limited] == [newest.id]

    filtered = await db_session.get_runs_with_logs_for_workflow(
        workflow.id,
        org.id,
        limit=10,
        filters=[
            {"attribute": "callChannel", "type": "radio", "value": {"status": "web"}}
        ],
    )
    assert [r["id"] for r in filtered] == [older.id]

    assert (
        await db_session.get_runs_with_logs_for_workflow(
            workflow.id, other_org.id, limit=10
        )
        == []
    )


async def test_regression_scenario_crud(db_session, async_session):
    user, org, workflow = await _org_with_workflow(db_session, "doctor_scenario")
    _, other_org, _ = await _org_with_workflow(db_session, "doctor_scenario_x")
    report = await db_session.create_analysis_report(
        organization_id=org.id, workflow_id=workflow.id, created_by=user.id, params={}
    )
    source = WorkflowRunModel(
        name="source", workflow_id=workflow.id, mode="twilio", is_completed=True
    )
    async_session.add(source)
    await async_session.flush()

    scenario = await db_session.create_regression_scenario(
        organization_id=org.id,
        workflow_id=workflow.id,
        report_id=report.id,
        cluster_id="C1",
        source_run_id=source.id,
        turn=3,
        expected_behavior="Offer a callback instead of inventing a price.",
    )
    assert scenario.expected_confirmed is False
    assert scenario.verification == {}

    assert await db_session.get_regression_scenario(scenario.id, other_org.id) is None
    assert [
        s.id
        for s in await db_session.list_regression_scenarios(
            workflow.id, org.id, report_id=report.id
        )
    ] == [scenario.id]
    assert await db_session.list_regression_scenarios(workflow.id, other_org.id) == []

    candidate = await db_session.create_candidate_definition(
        workflow_id=workflow.id, organization_id=org.id, workflow_json={}
    )
    updated = await db_session.update_regression_scenario(
        scenario.id,
        org.id,
        expected_confirmed=True,
        candidate_definition_id=candidate.id,
        verification={"status": "verifying", "cluster_id": "C1"},
    )
    assert updated.expected_confirmed is True
    assert updated.candidate_definition_id == candidate.id
    assert updated.verification["status"] == "verifying"

    with pytest.raises(ValueError):
        await db_session.update_regression_scenario(scenario.id, org.id, workflow_id=9)
    assert (
        await db_session.update_regression_scenario(scenario.id, other_org.id, turn=1)
        is None
    )


async def test_archive_candidate_keeps_row_hidden(db_session):
    _, org, workflow = await _org_with_workflow(db_session, "doctor_archive")
    _, other_org, _ = await _org_with_workflow(db_session, "doctor_archive_x")
    candidate = await db_session.create_candidate_definition(
        workflow_id=workflow.id,
        organization_id=org.id,
        workflow_json={"nodes": [], "edges": []},
    )
    assert (
        await db_session.archive_candidate_definition(candidate.id, other_org.id)
        is False
    )
    assert await db_session.archive_candidate_definition(candidate.id, org.id) is True
    # Archived: no longer a live candidate, still absent from version history.
    assert await db_session.get_candidate_definition(candidate.id, org.id) is None
    versions = await db_session.get_workflow_versions(workflow.id)
    assert candidate.id not in [v.id for v in versions]
    # Idempotent: archiving again is a no-op.
    assert await db_session.archive_candidate_definition(candidate.id, org.id) is False


async def test_runs_with_logs_for_exact_run_ids(db_session, async_session):
    _, org, workflow = await _org_with_workflow(db_session, "doctor_run_ids")
    _, other_org, foreign = await _org_with_workflow(db_session, "doctor_run_ids_x")

    def run(wf, mode="twilio"):
        return WorkflowRunModel(
            name=f"run-{mode}", workflow_id=wf.id, mode=mode, is_completed=True
        )

    picked = [run(workflow) for _ in range(3)]
    unpicked = run(workflow, "webrtc")
    replay = run(workflow, DOCTOR_REPLAY_RUN_MODE)
    foreign_run = run(foreign)
    async_session.add_all([*picked, unpicked, replay, foreign_run])
    await async_session.flush()
    wanted = [r.id for r in picked]

    # limit and filters are ignored when run_ids is given.
    rows = await db_session.get_runs_with_logs_for_workflow(
        workflow.id,
        org.id,
        limit=1,
        filters=[
            {"attribute": "callChannel", "type": "radio", "value": {"status": "web"}}
        ],
        run_ids=wanted,
    )
    assert sorted(r["id"] for r in rows) == sorted(wanted)

    # Still workflow / org scoped and without internal runs.
    rows = await db_session.get_runs_with_logs_for_workflow(
        workflow.id,
        org.id,
        limit=10,
        run_ids=[wanted[0], replay.id, foreign_run.id],
    )
    assert [r["id"] for r in rows] == [wanted[0]]
    assert (
        await db_session.get_runs_with_logs_for_workflow(
            workflow.id, other_org.id, limit=10, run_ids=wanted
        )
        == []
    )
    assert (
        await db_session.get_runs_with_logs_for_workflow(
            workflow.id, org.id, limit=10, run_ids=[]
        )
        == []
    )
