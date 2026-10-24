"""Self-improve status (DS-3 precedence) and routes (handlers called directly)."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api.routes import self_improve as routes
from api.services.self_improve import counting, status
from api.services.self_improve.contracts import (
    SelfImproveState,
    UpdateSelfImproveRequest,
)
from api.services.workflow.doctor.contracts import (
    Cluster,
    Evidence,
    ReportResult,
    StageName,
    StageState,
)
from api.tests.self_improve_fakes import ORG, OTHER_ORG, WF, FakeSelfImproveDB, ai

USER = SimpleNamespace(id=3, selected_organization_id=ORG)
STRANGER = SimpleNamespace(id=4, selected_organization_id=OTHER_ORG)


@pytest.fixture
def env(monkeypatch):
    db = FakeSelfImproveDB()
    jobs = []

    async def fake_enqueue(name, *args, **kwargs):
        jobs.append((name, args, kwargs))

    for module in (routes, counting, status):
        monkeypatch.setattr(module, "db_client", db)
    for module in (counting, status):
        monkeypatch.setattr(module, "resolve_ai_access", ai(True))
    import api.tasks.arq as arq_module

    monkeypatch.setattr(arq_module, "enqueue_job", fake_enqueue)
    return db, jobs, monkeypatch


def set_ai(monkeypatch, enabled, reason=None):
    for module in (counting, status):
        monkeypatch.setattr(module, "resolve_ai_access", ai(enabled, reason))


async def state(db):
    return (await status.build_status(WF, ORG)).state


# ------------------------------------------------------------------ states


async def test_state_precedence(env):
    db, _, monkeypatch = env
    # Nothing configured: off.
    assert await state(db) == SelfImproveState.off
    db.configure(every_n_calls=10)
    assert await state(db) == SelfImproveState.collecting

    running = db.add_iteration(number=1, status="diagnosing")
    assert await state(db) == SelfImproveState.running

    db.iterations[running.id].status = "ready"
    db.add_proposal(iteration_id=running.id, status="pending")
    assert await state(db) == SelfImproveState.waiting
    # applying is not reviewed (O2)
    db.proposals["prop-1"].status = "applying"
    assert await state(db) == SelfImproveState.waiting

    db.settings[WF].paused = True
    assert await state(db) == SelfImproveState.paused

    db.settings[WF].enabled = False
    assert await state(db) == SelfImproveState.off

    set_ai(monkeypatch, False, "not_opted_in")
    assert await state(db) == SelfImproveState.ai_disabled
    set_ai(monkeypatch, False, "no_key")
    assert await state(db) == SelfImproveState.no_key


async def test_reviewed_round_goes_back_to_collecting(env):
    db, _, _ = env
    db.configure()
    done = db.add_iteration(number=1, status="ready")
    db.add_proposal(iteration_id=done.id, status="discarded")
    st = await status.build_status(WF, ORG)
    assert st.state == SelfImproveState.collecting and st.to_review == 0
    assert st.current_iteration.number == 1 and st.current_iteration.reviewed


async def test_status_numbers(env):
    db, _, _ = env
    db.configure(every_n_calls=10, window_multiplier=2)
    for run_id in range(7):
        db.add_counted(WF, run_id)
    waiting = db.add_iteration(number=4, status="ready")
    db.add_proposal(iteration_id=waiting.id)
    db.add_iteration(number=3, status="no_changes")
    db.brief = SimpleNamespace(brief={"goal": "Book demos", "must_do": []})

    st = await status.build_status(WF, ORG)
    assert (st.collected, st.window_size, st.window_multiplier) == (7, 20, 2)
    assert st.to_review == 1 and st.current_iteration.number == 4
    assert st.has_brief is True and st.settings.every_n_calls == 10

    db.brief = SimpleNamespace(brief={"goal": "", "must_do": []})
    assert (await status.build_status(WF, ORG)).has_brief is False


async def test_problem_count_comes_from_the_report_once_grouped(env):
    db, _, _ = env
    db.configure()
    it = db.add_iteration(number=1, status="proposing")
    report = await db.create_analysis_report(
        organization_id=ORG, workflow_id=WF, created_by=3, params={}
    )
    db.iterations[it.id].report_id = report.id

    def cluster(cid, category="knowledge_gap", watch=False):
        return Cluster(
            cluster_id=cid,
            title=cid,
            category=category,
            severity="major",
            fix_target="Welcome",
            root_cause="x",
            runs_affected=1 if watch else 2,
            runs_analysed=5,
            evidence=[Evidence(run_id=1, quote="q")],
            watch_only=watch,
        )

    result = ReportResult(
        clusters=[cluster("C1"), cluster("C2", "platform"), cluster("C3", watch=True)]
    )
    report.result = result.model_dump(mode="json")
    assert (await status.build_status(WF, ORG)).current_iteration.problem_count is None
    result.stages[StageName.clusters] = StageState(status="done")
    report.result = result.model_dump(mode="json")
    st = await status.build_status(WF, ORG)
    assert st.current_iteration.problem_count == 1
    listed = await routes.list_self_improve_iterations(WF, USER)
    assert listed.iterations[0].problem_count == 1


# ------------------------------------------------------------------ routes


async def test_foreign_workflow_is_404_everywhere(env):
    db, _, _ = env
    db.configure()
    calls = [
        routes.get_self_improve_status(WF, STRANGER),
        routes.update_self_improve_settings(
            WF, UpdateSelfImproveRequest(enabled=True), STRANGER
        ),
        routes.run_self_improve_now(WF, STRANGER),
        routes.resume_self_improve(WF, STRANGER),
        routes.list_self_improve_iterations(WF, STRANGER),
    ]
    for call in calls:
        with pytest.raises(HTTPException) as exc:
            await call
        assert exc.value.status_code == 404


async def test_put_rejects_bad_n(env):
    with pytest.raises(HTTPException) as exc:
        await routes.update_self_improve_settings(
            WF, UpdateSelfImproveRequest(enabled=True, every_n_calls=7), USER
        )
    assert exc.value.status_code == 422


async def test_put_turns_on_and_records_the_user(env):
    db, jobs, _ = env
    st = await routes.update_self_improve_settings(
        WF, UpdateSelfImproveRequest(enabled=True, every_n_calls=5), USER
    )
    assert st.state == SelfImproveState.collecting and st.settings.every_n_calls == 5
    assert db.settings[WF].enabled_by_user_id == USER.id and jobs == []


async def test_smaller_n_starts_a_round_now(env):
    # DS-10: 7/10 collected, N -> 5 starts a round at once.
    db, jobs, _ = env
    db.configure(every_n_calls=10)
    for run_id in range(1, 8):
        db.add_counted(WF, run_id)
    st = await routes.update_self_improve_settings(
        WF, UpdateSelfImproveRequest(enabled=True, every_n_calls=5), USER
    )
    assert st.state == SelfImproveState.running
    assert st.current_iteration.run_ids == [1, 2, 3, 4, 5] and st.collected == 2
    assert jobs[0][2] == {"_job_id": f"self-improve-{WF}-1"}


async def test_larger_n_keeps_collecting(env):
    db, jobs, _ = env
    db.configure(every_n_calls=10)
    for run_id in range(1, 8):
        db.add_counted(WF, run_id)
    st = await routes.update_self_improve_settings(
        WF, UpdateSelfImproveRequest(enabled=True, every_n_calls=20), USER
    )
    assert st.state == SelfImproveState.collecting and st.collected == 7
    assert jobs == []


@pytest.mark.parametrize(
    "setup, reason",
    [
        ("no_key", "no_key"),
        ("ai_off", "ai_disabled"),
        ("off", "off"),
        ("waiting", "waiting_for_review"),
        ("running", "running"),
        ("paused", "paused"),
        ("empty", "no_calls"),
    ],
)
async def test_run_now_reasons(env, setup, reason):
    db, jobs, monkeypatch = env
    db.configure()
    if setup == "no_key":
        set_ai(monkeypatch, False, "no_key")
    elif setup == "ai_off":
        set_ai(monkeypatch, False, "not_opted_in")
    elif setup == "off":
        db.settings[WF].enabled = False
    elif setup == "waiting":
        it = db.add_iteration(number=1, status="ready")
        db.add_proposal(iteration_id=it.id)
    elif setup == "running":
        db.add_iteration(number=1, status="queued")
    elif setup == "paused":
        db.settings[WF].paused = True
    if setup != "empty":
        db.add_counted(WF, 1)
    resp = await routes.run_self_improve_now(WF, USER)
    assert resp.started is False and resp.reason == reason and jobs == []


async def test_run_now_starts_a_round_on_a_partial_window(env):
    db, jobs, _ = env
    db.configure(every_n_calls=10)
    for run_id in (1, 2, 3):
        db.add_counted(WF, run_id)
    resp = await routes.run_self_improve_now(WF, USER)
    assert resp.started is True and resp.reason is None
    assert resp.iteration.run_ids == [1, 2, 3] and resp.iteration.status == "queued"
    assert jobs[0][1] == (resp.iteration.iteration_id, ORG)


async def test_resume_clears_the_pause_and_starts_a_full_window(env):
    db, jobs, _ = env
    db.configure(every_n_calls=5, paused=True, paused_reason="boom")
    db.settings[WF].consecutive_failures = 2
    for run_id in range(1, 6):
        db.add_counted(WF, run_id)
    st = await routes.resume_self_improve(WF, USER)
    assert db.settings[WF].paused is False and db.settings[WF].consecutive_failures == 0
    assert st.state == SelfImproveState.running and len(jobs) == 1


async def test_resume_without_settings_is_404(env):
    with pytest.raises(HTTPException) as exc:
        await routes.resume_self_improve(WF, USER)
    assert exc.value.status_code == 404


async def test_iterations_newest_first(env):
    db, _, _ = env
    db.configure()
    db.add_iteration(number=1, status="no_changes", summary="nothing to fix")
    db.add_iteration(
        number=2,
        status="failed",
        error="model down",
        carryover=[
            {
                "proposal_uuid": "p",
                "change_id": "n:1",
                "title": "Welcome",
                "outcome": "fixed",
                "eligible_calls": 4,
            }
        ],
    )
    resp = await routes.list_self_improve_iterations(WF, USER)
    assert [i.number for i in resp.iterations] == [2, 1]
    assert resp.iterations[0].error == "model down" and resp.iterations[0].reviewed
    assert resp.iterations[0].carryover[0].outcome.value == "fixed"
    assert resp.iterations[1].summary == "nothing to fix"
