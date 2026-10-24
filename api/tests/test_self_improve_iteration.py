"""Self-improve round job: stages, resume, outcomes, sweep (RI-6, O4, O5, D6)."""

from datetime import timedelta

import pytest

from api.services.self_improve import carryover, iteration, status
from api.services.workflow.doctor.contracts import (
    Cluster,
    Evidence,
    ReportResult,
    StageName,
    StageState,
)
from api.tasks import self_improve as task
from api.tests.self_improve_fakes import ORG, USER_ID, WF, FakeSelfImproveDB, ai


def clustered_result(clusters_done=True, clusters=None):
    result = ReportResult(run_ids=[1, 2, 3])
    for stage in result.stages:
        result.stages[stage] = StageState(status="done")
    if not clusters_done:
        result.stages[StageName.clusters] = StageState(
            status="failed", error="model overloaded"
        )
    result.clusters = clusters or [
        Cluster(
            cluster_id="C1",
            title="Answers pricing",
            category="knowledge_gap",
            severity="major",
            fix_target="Welcome",
            root_cause="prices in prompt",
            runs_affected=2,
            runs_analysed=3,
            evidence=[Evidence(run_id=1, quote="$50"), Evidence(run_id=2, quote="$5")],
        )
    ]
    return result


class Doctor:
    """run_analysis stand-in: writes a result into the fake report."""

    def __init__(self, db, clusters_done=True, error=None):
        self.db = db
        self.calls = []
        self.clusters_done = clusters_done
        self.error = error

    async def __call__(self, report_uuid, organization_id):
        self.calls.append(report_uuid)
        report = self.db.reports[report_uuid]
        report.result = clustered_result(self.clusters_done).model_dump(mode="json")
        report.status = "completed" if self.clusters_done else "partial"
        report.error = self.error
        return report.status


class Copilot:
    """run_self_improve_turn stand-in."""

    def __init__(self, db, propose=True, summary="Moved pricing out", error=None):
        self.db = db
        self.propose = propose
        self.summary = summary
        self.error = error
        self.calls = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        if not self.propose:
            return None, self.summary
        proposal = self.db.add_proposal(iteration_id=kwargs["iteration_id"])
        return proposal.proposal_uuid, self.summary


@pytest.fixture
def env(monkeypatch):
    db = FakeSelfImproveDB()
    db.configure(every_n_calls=5)
    for module in (iteration, carryover, status, task):
        monkeypatch.setattr(module, "db_client", db)
    monkeypatch.setattr(iteration, "resolve_ai_access", ai(True))
    doctor = Doctor(db)
    copilot = Copilot(db)
    monkeypatch.setattr(iteration, "run_analysis", doctor)
    monkeypatch.setattr(iteration, "_run_self_improve_turn", copilot)
    round_ = db.add_iteration(number=1, run_ids=[1, 2, 3])
    return db, doctor, copilot, round_


def outcomes(db):
    return [o[1] for o in db.outcomes]


async def test_round_with_a_proposal_is_ready(env):
    db, doctor, copilot, round_ = env
    assert await iteration.run_iteration(round_.id, ORG) == "ready"
    stored = db.iterations[round_.id]
    assert stored.status == "ready" and stored.proposal_id is not None
    assert stored.summary == "Moved pricing out"
    assert outcomes(db) == ["changes"]

    # The Doctor ran on exactly the round's calls.
    (report,) = db.reports.values()
    assert report.params["run_ids"] == [1, 2, 3] and report.params["limit"] == 3
    assert report.created_by == USER_ID and stored.report_id == report.id
    # Status walked queued -> diagnosing -> proposing -> ready.
    walked = [u["status"] for u in db.iteration_updates if "status" in u]
    assert walked == ["diagnosing", "proposing", "ready"]

    (call,) = copilot.calls
    assert call["iteration_id"] == round_.id
    assert call["report_uuid"] == report.report_uuid
    assert call["user"].id == USER_ID
    assert call["session_uuid"] == db.sessions[WF].session_uuid
    assert "Self-improve round 1." in call["text"]
    assert report.report_uuid in call["text"]


async def test_round_without_a_proposal_is_no_changes_and_backs_off(env):
    db, _, copilot, round_ = env
    copilot.propose, copilot.summary = False, ""
    assert await iteration.run_iteration(round_.id, ORG) == "no_changes"
    stored = db.iterations[round_.id]
    assert stored.summary == "No changes needed in 3 calls."
    assert outcomes(db) == ["no_changes"]
    assert db.settings[WF].window_multiplier == 2


async def test_unfinished_problem_grouping_fails_the_round(env, monkeypatch):
    db, _, copilot, round_ = env
    monkeypatch.setattr(
        iteration,
        "run_analysis",
        Doctor(db, clusters_done=False, error="Analysis stopped at problem grouping"),
    )
    assert await iteration.run_iteration(round_.id, ORG) == "failed"
    stored = db.iterations[round_.id]
    assert "did not finish grouping problems" in stored.error
    assert "Analysis stopped at problem grouping" in stored.error
    assert copilot.calls == [] and outcomes(db) == ["failed"]


async def test_copilot_error_fails_the_round_once(env):
    db, _, copilot, round_ = env
    copilot.error = RuntimeError("provider down")
    assert await iteration.run_iteration(round_.id, ORG) == "failed"
    assert db.iterations[round_.id].error == "Unexpected error: provider down"
    # A retry of the ended round is a no-op: the outcome is not applied twice.
    assert await iteration.run_iteration(round_.id, ORG) == "failed"
    assert await task.self_improve_iteration({}, round_.id, ORG) == "failed"
    assert outcomes(db) == ["failed"]


async def test_two_failed_rounds_pause(env):
    db, _, copilot, round_ = env
    copilot.error = RuntimeError("provider down")
    await iteration.run_iteration(round_.id, ORG)
    second = db.add_iteration(number=2, run_ids=[4, 5])
    await iteration.run_iteration(second.id, ORG)
    assert db.settings[WF].paused is True
    assert db.settings[WF].paused_reason == "Unexpected error: provider down"


async def test_ready_outcome_is_applied_once_across_retries(env):
    db, _, _, round_ = env
    await iteration.run_iteration(round_.id, ORG)
    await iteration.run_iteration(round_.id, ORG)
    assert outcomes(db) == ["changes"]


async def test_retry_reuses_the_existing_report(env):
    db, doctor, _, round_ = env
    # First attempt died mid-diagnosis after creating the report.
    report = await db.create_analysis_report(
        organization_id=ORG, workflow_id=WF, created_by=USER_ID, params={}
    )
    db.iterations[round_.id].report_id = report.id
    db.iterations[round_.id].status = "diagnosing"
    assert await iteration.run_iteration(round_.id, ORG) == "ready"
    assert db.reports_created == 1 and doctor.calls == [report.report_uuid]


async def test_retry_in_proposing_reuses_the_existing_proposal(env):
    db, doctor, copilot, round_ = env
    report = await db.create_analysis_report(
        organization_id=ORG, workflow_id=WF, created_by=USER_ID, params={}
    )
    report.result = clustered_result().model_dump(mode="json")
    stored = db.iterations[round_.id]
    stored.report_id, stored.status = report.id, "proposing"
    # The copilot proposed, then the worker died before the round was ready.
    proposal = db.add_proposal(iteration_id=round_.id, summary="Earlier summary")
    assert await iteration.run_iteration(round_.id, ORG) == "ready"
    assert copilot.calls == [] and doctor.calls == []
    assert stored.proposal_id == proposal.id and stored.summary == "Earlier summary"
    assert outcomes(db) == ["changes"]


async def test_enabled_by_user_who_left_the_org_fails_the_round(env):
    db, doctor, copilot, round_ = env
    db.members.clear()
    assert await iteration.run_iteration(round_.id, ORG) == "failed"
    assert "no longer in this organization" in db.iterations[round_.id].error
    assert doctor.calls == [] and copilot.calls == []
    assert outcomes(db) == ["failed"]


async def test_ai_turned_off_after_the_claim_fails_plainly(env, monkeypatch):
    db, doctor, _, round_ = env
    monkeypatch.setattr(iteration, "resolve_ai_access", ai(False, "not_opted_in"))
    assert await iteration.run_iteration(round_.id, ORG) == "failed"
    assert db.iterations[round_.id].error == (
        "AI analysis is not enabled for this organization."
    )
    assert doctor.calls == []


async def test_round_message_has_previous_rounds_and_carryover(env, monkeypatch):
    db, _, copilot, round_ = env
    # Renumber: an earlier no_changes round and an earlier applied round.
    db.iterations[round_.id].number = 3
    db.add_iteration(number=1, status="no_changes")
    applied = db.add_iteration(number=2, status="ready", summary="Moved pricing")
    db.add_proposal(iteration_id=applied.id, status="applied")

    async def fake_carryover(**kwargs):
        from api.services.self_improve.contracts import FixCarryover, FixOutcome

        return [
            FixCarryover(
                proposal_uuid="p",
                change_id="n:1",
                title="Welcome step answers pricing",
                outcome=FixOutcome.insufficient,
                eligible_calls=2,
            )
        ]

    monkeypatch.setattr(iteration, "build_carryover", fake_carryover)
    await iteration.run_iteration(round_.id, ORG)
    text = copilot.calls[0]["text"]
    assert "Round 1: no changes needed." in text
    assert "Round 2: the user applied its proposal. Proposal: Moved pricing" in text
    assert "– Welcome step answers pricing: not enough calls yet" in text
    assert db.iterations[round_.id].carryover[0]["outcome"] == "insufficient"


async def test_admission_exhausted_fails_the_round(env, monkeypatch):
    db, _, _, round_ = env

    async def no_permit(ctx, organization_id, job_label, work, on_exhausted):
        assert job_label == f"self-improve:{round_.id}"
        return await on_exhausted()

    monkeypatch.setattr(task, "run_with_admission", no_permit)
    assert await task.self_improve_iteration({}, round_.id, ORG) == "failed"
    assert outcomes(db) == ["failed"]


async def test_job_runs_inside_admission(env, monkeypatch):
    db, _, _, round_ = env
    seen = []

    async def admit(ctx, organization_id, job_label, work, on_exhausted):
        seen.append(organization_id)
        return await work()

    monkeypatch.setattr(task, "run_with_admission", admit)
    assert await task.self_improve_iteration({}, round_.id, ORG) == "ready"
    assert seen == [ORG]


async def test_sweep_re_enqueues_stuck_rounds_with_deterministic_ids(env, monkeypatch):
    db, _, _, round_ = env
    db.iterations[round_.id].stage_started_at = db.now - timedelta(days=1)
    db.add_iteration(number=2, status="ready")  # ended: never swept
    jobs = []

    async def fake_enqueue(name, *args, **kwargs):
        jobs.append((name, args, kwargs))

    import api.tasks.arq as arq_module

    monkeypatch.setattr(arq_module, "enqueue_job", fake_enqueue)
    assert await task.sweep_stuck_self_improve_rounds({}) == 1
    assert jobs == [
        (
            "self_improve_iteration",
            (round_.id, ORG),
            {"_job_id": f"self-improve-{WF}-1"},
        )
    ]


# --------------------------------------------------- Doctor run_ids plumbing


async def test_doctor_loads_exactly_the_round_runs_when_run_ids_given(monkeypatch):
    from api.services.workflow.doctor import analysis

    seen = []

    class DB:
        async def get_runs_with_logs_for_workflow(
            self, workflow_id, organization_id, **kwargs
        ):
            seen.append(kwargs)
            return [{"id": 5, "logs": {}}, {"id": 7, "logs": {}}]

    monkeypatch.setattr(analysis, "db_client", DB())
    run = analysis._Analysis("r", ORG, ReportResult())
    await analysis._load_runs(run, WF, {"limit": 2, "run_ids": [7, 5]})
    await analysis._load_runs(
        analysis._Analysis("r", ORG, ReportResult()), WF, {"limit": 50}
    )
    assert seen[0] == {"limit": 2, "filters": None, "run_ids": [7, 5]}
    # Without run_ids the Doctor's query is unchanged (no run_ids kwarg).
    assert seen[1] == {"limit": 50, "filters": None}
    assert sorted(run.result.run_ids) == [5, 7]


def test_create_report_request_accepts_run_ids():
    from api.services.workflow.doctor.contracts import CreateReportRequest

    params = CreateReportRequest(limit=3, run_ids=[1, 2, 3]).model_dump(mode="json")
    assert params == {"limit": 3, "filters": None, "run_ids": [1, 2, 3]}
    assert CreateReportRequest().run_ids is None


async def test_copilot_turn_error_is_shown_plainly(env):
    from api.services.copilot.agent import SelfImproveTurnError

    db, _, copilot, round_ = env
    copilot.error = SelfImproveTurnError("The model is overloaded.")
    assert await iteration.run_iteration(round_.id, ORG) == "failed"
    assert db.iterations[round_.id].error == (
        "The Copilot could not write fixes: The model is overloaded."
    )
    assert outcomes(db) == ["failed"]


async def test_sweep_fails_rounds_that_never_finish(env, monkeypatch):
    """No endless retries: a round older than MAX_ROUND_AGE_SECONDS is failed
    (outcome counted once) instead of re-enqueued again."""
    from api.services.self_improve.contracts import MAX_ROUND_AGE_SECONDS

    db, _, _, round_ = env
    it = db.iterations[round_.id]
    it.stage_started_at = db.now - timedelta(days=1)
    it.created_at = db.now - timedelta(seconds=MAX_ROUND_AGE_SECONDS + 60)
    jobs = []

    async def fake_enqueue(name, *args, **kwargs):
        jobs.append(name)

    import api.tasks.arq as arq_module

    monkeypatch.setattr(arq_module, "enqueue_job", fake_enqueue)
    assert await task.sweep_stuck_self_improve_rounds({}) == 0
    assert jobs == []
    assert db.iterations[round_.id].status == "failed"
    assert db.iterations[round_.id].error == task.ROUND_TIMED_OUT_MESSAGE
