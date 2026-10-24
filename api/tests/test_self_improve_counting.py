"""Self-improve counting in the completion job (D3, RI-2, RI-3, RI-5, O7)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from api.services.self_improve import counting
from api.tasks import workflow_completion
from api.tests.self_improve_fakes import ORG, WF, FakeSelfImproveDB, ai


def user_event(text="I want a quote", final=True):
    return {
        "type": "rtf-user-transcription",
        "turn": 1,
        "node_name": "Welcome",
        "payload": {"text": text, "final": final},
    }


def bot_event(text="Hi, this is Sarah from Acme."):
    return {
        "type": "rtf-bot-text",
        "turn": 0,
        "node_name": "Welcome",
        "payload": {"text": text},
    }


def make_run(run_id, mode="twilio", events=None, completed=True):
    events = [bot_event(), user_event()] if events is None else events
    return SimpleNamespace(
        id=run_id,
        workflow_id=WF,
        workflow=SimpleNamespace(organization_id=ORG),
        mode=mode,
        is_completed=completed,
        logs={"realtime_feedback_events": events},
    )


@pytest.fixture
def env(monkeypatch):
    db = FakeSelfImproveDB()
    db.configure(every_n_calls=5)
    jobs = []

    async def fake_enqueue(name, *args, **kwargs):
        jobs.append((name, args, kwargs))

    monkeypatch.setattr(counting, "db_client", db)
    monkeypatch.setattr(counting, "resolve_ai_access", ai(True))
    import api.tasks.arq as arq_module

    monkeypatch.setattr(arq_module, "enqueue_job", fake_enqueue)
    return db, jobs


async def test_connected_web_call_is_counted(env):
    db, _ = env
    db.run_models[1] = make_run(1, mode="smallwebrtc")
    await counting.count_completed_run(1)
    assert [c.run_id for c in db.counted] == [1]


async def test_telephony_call_with_speech_is_counted(env):
    db, _ = env
    db.run_models[1] = make_run(1, mode="twilio")
    await counting.count_completed_run(1)
    assert [c.run_id for c in db.counted] == [1]


@pytest.mark.parametrize("mode", ["textchat", "doctor_replay", "CHAT"])
async def test_text_chat_and_replays_never_count(env, mode):
    db, _ = env
    db.run_models[1] = make_run(1, mode=mode)
    assert await counting.count_completed_run(1) is None
    assert db.counted == []


async def test_no_answer_telephony_call_with_zero_user_turns_is_not_counted(env):
    db, _ = env
    # The bot spoke into voicemail / nobody answered; an interim transcript
    # (final=False) and an empty one are not caller turns either.
    db.run_models[1] = make_run(
        1,
        events=[bot_event(), user_event("uh", final=False), user_event("  ")],
    )
    await counting.count_completed_run(1)
    assert db.counted == []


async def test_unfinished_run_is_not_counted(env):
    db, _ = env
    db.run_models[1] = make_run(1, completed=False)
    await counting.count_completed_run(1)
    assert db.counted == []


async def test_ai_off_means_not_counted(env, monkeypatch):
    db, _ = env
    monkeypatch.setattr(counting, "resolve_ai_access", ai(False, "not_opted_in"))
    db.run_models[1] = make_run(1)
    await counting.count_completed_run(1)
    assert db.counted == []


async def test_self_improve_off_or_paused_skips_before_reading_logs(env, monkeypatch):
    db, _ = env
    spoke = AsyncMock()
    monkeypatch.setattr(counting, "caller_spoke", spoke)
    db.run_models[1] = make_run(1)
    db.settings[WF].enabled = False
    await counting.count_completed_run(1)
    db.settings[WF].enabled, db.settings[WF].paused = True, True
    await counting.count_completed_run(1)
    assert db.counted == [] and spoke.call_count == 0


async def test_full_window_claims_a_round_and_enqueues_with_deterministic_id(env):
    db, jobs = env
    for run_id in range(1, 6):
        db.run_models[run_id] = make_run(run_id)
    started = [await counting.count_completed_run(r) for r in range(1, 6)]
    assert started[:4] == [None] * 4
    iteration = started[4]
    assert iteration is not None and iteration.number == 1
    assert iteration.run_ids == [1, 2, 3, 4, 5]
    ((name, args, kwargs),) = jobs
    assert name == "self_improve_iteration"
    assert args == (iteration.id, ORG)
    assert kwargs == {"_job_id": f"self-improve-{WF}-1"}


async def test_a_retried_completion_still_claims_the_round(env):
    db, jobs = env
    for run_id in range(1, 6):
        db.add_counted(WF, run_id)  # counted earlier; claim never happened
    db.run_models[5] = make_run(5)
    iteration = await counting.count_completed_run(5)
    assert iteration is not None and len(jobs) == 1


async def test_enqueue_failure_keeps_the_round_for_the_sweep(env, monkeypatch):
    db, _ = env
    import api.tasks.arq as arq_module

    monkeypatch.setattr(
        arq_module, "enqueue_job", AsyncMock(side_effect=RuntimeError("redis down"))
    )
    for run_id in range(1, 6):
        db.add_counted(WF, run_id)
    iteration = await counting.claim_and_enqueue(WF, ORG)
    assert iteration is not None and db.iterations[iteration.id].status == "queued"


# --------------------------------------------------------------- the hook


@pytest.fixture
def stubbed_steps(monkeypatch):
    steps = {
        "integrations": AsyncMock(),
        "usage": AsyncMock(),
        "pbx": AsyncMock(),
        "self_improve": AsyncMock(),
    }
    monkeypatch.setattr(
        workflow_completion, "run_integrations_post_workflow_run", steps["integrations"]
    )
    monkeypatch.setattr(
        workflow_completion,
        "report_completed_workflow_run_platform_usage",
        steps["usage"],
    )
    monkeypatch.setattr(
        workflow_completion, "sync_external_pbx_call_record", steps["pbx"]
    )
    monkeypatch.setattr(
        workflow_completion, "count_completed_run", steps["self_improve"]
    )
    return steps


async def test_completion_job_counts_the_run(stubbed_steps):
    await workflow_completion.process_workflow_completion(None, 11)
    stubbed_steps["self_improve"].assert_awaited_once_with(11)


async def test_self_improve_failure_never_breaks_the_completion_job(stubbed_steps):
    stubbed_steps["self_improve"].side_effect = RuntimeError("db down")
    await workflow_completion.process_workflow_completion(None, 11)
    for name in ("integrations", "usage", "pbx"):
        assert stubbed_steps[name].await_count == 1


async def test_other_step_failures_still_count_the_run(stubbed_steps):
    stubbed_steps["integrations"].side_effect = RuntimeError("QA down")
    stubbed_steps["pbx"].side_effect = RuntimeError("PBX down")
    await workflow_completion.process_workflow_completion(None, 11)
    stubbed_steps["self_improve"].assert_awaited_once_with(11)
