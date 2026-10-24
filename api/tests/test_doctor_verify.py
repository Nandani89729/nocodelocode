"""Unit tests for Agent Doctor verification (doctor/verify.py).

Covers D29 checkpoint fidelity + edit classification, D25 concurrency and 429
backoff, the D28 keep rule, D30 hash binding on save, D13 reply grouping and
the O1 dry-run tool results. No DB, no network, no LLM.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from api.services.workflow.doctor import verify
from api.services.workflow.doctor.contracts import (
    BASELINE_MIN_FAILURES,
    REGENERATION_CONCURRENCY,
    SampleResult,
    VerificationResult,
    VerificationStatus,
)
from api.services.workflow.doctor.verify import (
    apply_keep_rule,
    build_checkpoint_from_recorded_run,
    classify_edit,
    compute_definition_hash,
    group_replies,
    is_rate_limit_error,
    run_samples,
)
from api.services.workflow.text_chat_runner import (
    _deserialize_text_chat_checkpoint_messages,
)
from api.services.workflow.tools.dry_run import (
    SYNTHETIC_HTTP_RESULT,
    ToolDryRun,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WORKFLOW = {
    "nodes": [
        {
            "id": "start",
            "type": "startCall",
            "position": {"x": 0, "y": 0},
            "data": {
                "name": "Start",
                "prompt": "Greet the caller.",
                "is_start": True,
                "add_global_prompt": True,
            },
        },
        {
            "id": "billing",
            "type": "agentNode",
            "position": {"x": 0, "y": 200},
            "data": {
                "name": "Billing",
                "prompt": "Answer billing questions. Never quote prices.",
                "add_global_prompt": True,
            },
        },
        {
            "id": "end",
            "type": "endCall",
            "position": {"x": 0, "y": 400},
            "data": {"name": "End", "prompt": "Say goodbye."},
        },
        {
            "id": "global",
            "type": "globalNode",
            "position": {"x": 300, "y": 0},
            "data": {"name": "Global", "prompt": "Be polite."},
        },
    ],
    "edges": [
        {
            "id": "start-billing",
            "source": "start",
            "target": "billing",
            "data": {"label": "Billing", "condition": "Caller asks about a bill."},
        },
        {
            "id": "billing-end",
            "source": "billing",
            "target": "end",
            "data": {"label": "End", "condition": "Caller is done."},
        },
    ],
}


def _edited(**changes):
    out = json.loads(json.dumps(WORKFLOW))
    for node in out["nodes"]:
        if node["id"] in changes:
            node["data"].update(changes[node["id"]])
    return out


def _ev(kind, turn, payload, node_id=None, node_name=None):
    event = {"type": kind, "payload": payload, "turn": turn}
    if node_id:
        event["node_id"] = node_id
    if node_name:
        event["node_name"] = node_name
    return event


RECORDED_EVENTS = [
    _ev("rtf-node-transition", 0, {"node_id": "start", "node_name": "Start"}),
    _ev("rtf-bot-text", 0, {"text": "Hi, this is Acme."}, "start", "Start"),
    _ev("rtf-user-transcription", 1, {"text": "My bill", "final": False}),
    _ev("rtf-user-transcription", 1, {"text": "My bill is wrong.", "final": True}),
    _ev(
        "rtf-function-call-start",
        1,
        {
            "function_name": "lookup_bill",
            "tool_call_id": "call_1",
            "arguments": {"account": "42"},
        },
    ),
    _ev(
        "rtf-function-call-end",
        1,
        {
            "function_name": "lookup_bill",
            "tool_call_id": "call_1",
            "result": '{"amount": 120}',
        },
    ),
    _ev("rtf-bot-text", 1, {"text": "I see a bill of 120."}, "billing", "Billing"),
    _ev("rtf-node-transition", 1, {"node_id": "billing", "node_name": "Billing"}),
    _ev("rtf-user-transcription", 2, {"text": "How much is"}, "billing", "Billing"),
    _ev("rtf-user-transcription", 2, {"text": "the premium plan?"}, "billing"),
    _ev(
        "rtf-function-call-end",
        2,
        {
            "function_name": "lookup_bill",
            "tool_call_id": "call_2",
            "result": '{"amount": 99}',
        },
    ),
    _ev("rtf-bot-text", 2, {"text": "It costs 49 dollars."}, "billing", "Billing"),
    _ev("rtf-user-transcription", 3, {"text": "Thanks"}, "billing", "Billing"),
]


def _recorded_run(**overrides):
    run = SimpleNamespace(
        id=100,
        mode="twilio",
        logs={"realtime_feedback_events": RECORDED_EVENTS},
        initial_context={
            "customer_name": "Ana",
            "mps_correlation_id": "source-call-correlation",
            "workflow_run_id": 100,
        },
        gathered_context={
            "extracted_variables": {"account": "42"},
            "call_disposition": "user_hangup",
            "call_tags": ["x"],
        },
        definition=SimpleNamespace(workflow_json=WORKFLOW),
        workflow_id=9,
    )
    for key, value in overrides.items():
        setattr(run, key, value)
    return run


# ---------------------------------------------------------------------------
# D29 checkpoint fidelity
# ---------------------------------------------------------------------------


def test_checkpoint_history_equals_recorded_transcript_prefix():
    recorded = build_checkpoint_from_recorded_run(_recorded_run(), 2)

    expected_history = [
        {"role": "assistant", "content": "Hi, this is Acme."},
        {"role": "user", "content": "My bill is wrong."},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup_bill",
                        "arguments": json.dumps({"account": "42"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"amount": 120}'},
        {"role": "assistant", "content": "I see a bill of 120."},
    ]
    assert recorded.history == expected_history
    # The injected checkpoint round-trips to exactly the recorded history:
    # nothing before the failing turn is regenerated.
    restored = _deserialize_text_chat_checkpoint_messages(
        recorded.checkpoint["messages"]
    )
    assert restored == expected_history
    # The failing turn's user words become the pending message (all fragments).
    assert recorded.pending_user_message == "How much is the premium plan?"
    # The node active at the failing turn, not the start node.
    assert recorded.checkpoint["current_node_id"] == "billing"
    assert recorded.node_name == "Billing"
    # The failing turn's own recorded tool result is replayed first.
    assert recorded.tool_results["lookup_bill"] == ['{"amount": 99}', '{"amount": 120}']
    # Gathered context without completion-time keys; initial context without
    # the source call's billing correlation id or runner-owned keys.
    assert recorded.checkpoint["gathered_context"] == {
        "extracted_variables": {"account": "42"}
    }
    assert recorded.initial_context == {"customer_name": "Ana"}


def test_checkpoint_resolves_node_by_name_when_ids_missing():
    events = [
        _ev("rtf-bot-text", 0, {"text": "Hello"}, node_name="Start"),
        _ev("rtf-user-transcription", 1, {"text": "Bill?"}, node_name="Billing"),
    ]
    recorded = build_checkpoint_from_recorded_run(
        _recorded_run(logs={"realtime_feedback_events": events}), 1
    )
    assert recorded.node_id == "billing"
    assert recorded.history == [{"role": "assistant", "content": "Hello"}]


def test_checkpoint_drops_tool_call_without_recorded_result():
    events = [
        _ev("rtf-user-transcription", 0, {"text": "Hi"}),
        _ev(
            "rtf-function-call-start",
            0,
            {"function_name": "f", "tool_call_id": "orphan", "arguments": {}},
        ),
        _ev("rtf-bot-text", 0, {"text": "Hello"}),
        _ev("rtf-user-transcription", 1, {"text": "Next"}),
    ]
    recorded = build_checkpoint_from_recorded_run(
        _recorded_run(logs={"realtime_feedback_events": events}), 1
    )
    assert recorded.history == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]


# ---------------------------------------------------------------------------
# D29 which edits are turn-verifiable
# ---------------------------------------------------------------------------


def test_routing_edit_is_not_text_verifiable():
    candidate = json.loads(json.dumps(WORKFLOW))
    candidate["edges"][0]["data"]["condition"] = "Caller mentions money."
    assert classify_edit(WORKFLOW, candidate, "billing") == "not_text_verifiable"


def test_failing_node_prompt_edit_is_turn_verifiable():
    candidate = _edited(billing={"prompt": "Answer billing questions. Quote 49."})
    assert classify_edit(WORKFLOW, candidate, "billing") == "turn_verifiable"
    # failing node given by name works too
    assert classify_edit(WORKFLOW, candidate, "Billing") == "turn_verifiable"


def test_global_prompt_and_kb_edits_are_turn_verifiable():
    candidate = _edited(
        **{"global": {"prompt": "Be polite and exact."}},
        billing={"document_uuids": ["doc-1"]},
    )
    assert classify_edit(WORKFLOW, candidate, "billing") == "turn_verifiable"


def test_earlier_node_or_other_field_edit_is_not_text_verifiable():
    earlier = _edited(start={"prompt": "Greet and ask for the account."})
    assert classify_edit(WORKFLOW, earlier, "billing") == "not_text_verifiable"
    other_field = _edited(billing={"tool_uuids": ["t-1"]})
    assert classify_edit(WORKFLOW, other_field, "billing") == "not_text_verifiable"
    removed = json.loads(json.dumps(WORKFLOW))
    removed["nodes"] = removed["nodes"][:-1]
    assert classify_edit(WORKFLOW, removed, "billing") == "not_text_verifiable"


def test_position_only_change_is_ignored():
    moved = json.loads(json.dumps(WORKFLOW))
    moved["nodes"][1]["position"] = {"x": 999, "y": 999}
    assert classify_edit(WORKFLOW, moved, "billing") == "turn_verifiable"


# ---------------------------------------------------------------------------
# D25 concurrency + 429 backoff
# ---------------------------------------------------------------------------


class _RateLimited(Exception):
    status_code = 429


@pytest.mark.asyncio
async def test_regenerations_never_exceed_concurrency_and_429_is_retried():
    active = 0
    peak = 0
    attempts: dict[int, int] = {}
    sleeps: list[float] = []

    async def sample(index: int) -> SampleResult:
        nonlocal active, peak
        attempts[index] = attempts.get(index, 0) + 1
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.01)
            if index in (2, 7) and attempts[index] < 3:
                raise _RateLimited("429 Too Many Requests")
            return SampleResult(text=f"reply {index}", passed=True)
        finally:
            active -= 1

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    results = await run_samples(sample, 10, sleep=fake_sleep)

    assert len(results) == 10
    assert [r.text for r in results] == [f"reply {i}" for i in range(10)]
    assert peak <= REGENERATION_CONCURRENCY
    assert peak == REGENERATION_CONCURRENCY  # it does run in parallel
    assert attempts[2] == 3 and attempts[7] == 3
    assert len(sleeps) == 4
    # exponential: the second wait is longer than the first base delay
    assert max(sleeps) > min(sleeps)


@pytest.mark.asyncio
async def test_non_rate_limit_failure_propagates_unwrapped():
    async def sample(index: int) -> SampleResult:
        if index == 3:
            raise ValueError("replay broke")
        await asyncio.sleep(0.01)
        return SampleResult(text="ok", passed=True)

    with pytest.raises(ValueError, match="replay broke"):
        await run_samples(sample, 10, sleep=AsyncMock())


def test_rate_limit_detection_follows_cause_chain():
    try:
        try:
            raise _RateLimited("provider")
        except _RateLimited as inner:
            raise RuntimeError("Failed to execute text chat assistant turn") from inner
    except RuntimeError as outer:
        assert is_rate_limit_error(outer)
    assert not is_rate_limit_error(RuntimeError("boom"))


# ---------------------------------------------------------------------------
# D28 keep rule (pure + orchestrated)
# ---------------------------------------------------------------------------


def test_keep_rule_pure():
    assert apply_keep_rule(7, 0) == VerificationStatus.kept
    assert apply_keep_rule(BASELINE_MIN_FAILURES, 0) == VerificationStatus.kept
    assert apply_keep_rule(2, None) == VerificationStatus.not_reproducible
    assert apply_keep_rule(7, 3) == VerificationStatus.not_kept


def _scenario(verification=None, **overrides):
    scenario = SimpleNamespace(
        id=1,
        workflow_id=9,
        cluster_id="C1",
        source_run_id=100,
        turn=2,
        expected_behavior="Do not quote prices; offer to connect sales.",
        candidate_definition_id=55,
        verification=verification
        or VerificationResult(
            status=VerificationStatus.awaiting_start,
            cluster_id="C1",
            scenario_ids=[1],
            candidate_definition_id=55,
            diff_summary="Billing node no longer quotes prices.",
        ).model_dump(mode="json"),
    )
    for key, value in overrides.items():
        setattr(scenario, key, value)
    return scenario


CANDIDATE_JSON = _edited(
    billing={"prompt": "Answer billing questions. Never quote prices; offer sales."}
)


def _patch_verify_db(monkeypatch, scenario, *, candidate_json=CANDIDATE_JSON):
    stored: list[dict] = []

    async def update_regression_scenario(scenario_id, organization_id, **fields):
        stored.append(fields)
        return scenario

    db = verify.db_client
    monkeypatch.setattr(
        db, "get_regression_scenario", AsyncMock(return_value=scenario), raising=False
    )
    monkeypatch.setattr(
        db, "update_regression_scenario", update_regression_scenario, raising=False
    )
    monkeypatch.setattr(
        db, "list_regression_scenarios", AsyncMock(return_value=[]), raising=False
    )
    monkeypatch.setattr(
        db,
        "get_candidate_definition",
        AsyncMock(
            return_value=SimpleNamespace(
                id=55, workflow_id=9, workflow_json=candidate_json
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(
        db,
        "get_workflow",
        AsyncMock(return_value=SimpleNamespace(id=9, user_id=3)),
    )
    monkeypatch.setattr(
        db,
        "get_draft_version",
        AsyncMock(
            return_value=SimpleNamespace(
                id=50, workflow_id=9, workflow_json=WORKFLOW, version_number=4
            )
        ),
    )
    monkeypatch.setattr(db, "get_workflow_run", AsyncMock(return_value=_recorded_run()))
    return stored


def _fake_regenerate(fail_counts: dict[str, int], calls: list[str]):
    async def regenerate_fn(
        run_id, turn, definition, n, *, recorded, label, on_sample, **_kwargs
    ):
        side = label
        calls.append(side)
        # Same recorded checkpoint for both sides (D29), own definition each.
        assert recorded.pending_user_message == "How much is the premium plan?"
        assert definition.id == (50 if side == "baseline" else 55)
        samples = []
        for i in range(n):
            text = "bad: 49 dollars" if i < fail_counts[side] else "good"
            sample = SampleResult(text=text, passed=True)
            samples.append(sample)
            await on_sample(i, sample)
        return samples

    return regenerate_fn


async def _fake_grade(
    samples, expected_behavior, *, context=None, agent_knowledge=None
):
    assert context == "How much is the premium plan?"
    assert agent_knowledge is not None
    return [
        s.model_copy(
            update={
                "passed": not s.text.startswith("bad"),
                "reason": "quoted a price" if s.text.startswith("bad") else "ok",
            }
        )
        for s in samples
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("baseline_fail", "candidate_fail", "status", "candidate_ran"),
    [
        (7, 0, VerificationStatus.kept, True),
        (2, 0, VerificationStatus.not_reproducible, False),
        (7, 3, VerificationStatus.not_kept, True),
    ],
)
async def test_verify_fix_applies_keep_rule(
    monkeypatch, baseline_fail, candidate_fail, status, candidate_ran
):
    scenario = _scenario()
    stored = _patch_verify_db(monkeypatch, scenario)
    calls: list[str] = []

    result = await verify.verify_fix(
        scenario_id=1,
        organization_id=7,
        regenerate_fn=_fake_regenerate(
            {"baseline": baseline_fail, "candidate": candidate_fail}, calls
        ),
        grade_fn=_fake_grade,
    )

    assert result.status == status
    assert result.baseline_failures == baseline_fail
    assert calls == (["baseline", "candidate"] if candidate_ran else ["baseline"])
    if candidate_ran:
        assert result.candidate_failures == candidate_fail
        assert len(result.candidate_samples) == 10
    else:
        assert result.candidate_failures is None
    # D30: the result is bound to the exact candidate JSON that was replayed.
    assert result.candidate_hash == compute_definition_hash(CANDIDATE_JSON)
    final = VerificationResult.model_validate(stored[-1]["verification"])
    assert final.status == status
    # progress was persisted while sampling (UI progress bars)
    assert len(stored) > 3


@pytest.mark.asyncio
async def test_verify_fix_routing_edit_short_circuits(monkeypatch):
    routing = json.loads(json.dumps(WORKFLOW))
    routing["edges"][1]["data"]["condition"] = "Caller asks for a price."
    scenario = _scenario()
    _patch_verify_db(monkeypatch, scenario, candidate_json=routing)
    regenerate_fn = AsyncMock()

    result = await verify.verify_fix(
        scenario_id=1,
        organization_id=7,
        regenerate_fn=regenerate_fn,
        grade_fn=AsyncMock(),
    )

    assert result.status == VerificationStatus.not_text_verifiable
    regenerate_fn.assert_not_awaited()


@pytest.mark.asyncio
async def test_verify_fix_replay_error_marks_failed(monkeypatch):
    scenario = _scenario()
    stored = _patch_verify_db(monkeypatch, scenario)

    async def broken(*_args, **_kwargs):
        raise RuntimeError("pipeline crashed")

    result = await verify.verify_fix(
        scenario_id=1, organization_id=7, regenerate_fn=broken, grade_fn=_fake_grade
    )
    assert result.status == VerificationStatus.failed
    assert "pipeline crashed" in result.error
    assert stored[-1]["verification"]["status"] == "failed"


# ---------------------------------------------------------------------------
# D30 save: hash binding, optimistic lock, archiving
# ---------------------------------------------------------------------------


def _patch_save_db(monkeypatch, scenario, candidate_json, *, draft_version=4):
    db = verify.db_client
    _patch_verify_db(monkeypatch, scenario, candidate_json=candidate_json)
    save_draft = AsyncMock(
        return_value=SimpleNamespace(version_number=draft_version, status="draft")
    )
    archive = AsyncMock(return_value=True)
    monkeypatch.setattr(db, "save_workflow_draft", save_draft)
    monkeypatch.setattr(db, "archive_candidate_definition", archive, raising=False)
    monkeypatch.setattr(
        db,
        "get_workflow",
        AsyncMock(
            return_value=SimpleNamespace(
                id=9, user_id=3, released_definition=None, workflow_definition=None
            )
        ),
    )
    return save_draft, archive


def _kept_verification(candidate_hash):
    return VerificationResult(
        status=VerificationStatus.kept,
        cluster_id="C1",
        scenario_ids=[1],
        candidate_definition_id=55,
        candidate_hash=candidate_hash,
        baseline_failures=7,
        candidate_failures=0,
    ).model_dump(mode="json")


@pytest.mark.asyncio
async def test_save_with_changed_candidate_requires_reverify(monkeypatch):
    tested_hash = compute_definition_hash(
        verify.canonicalize_candidate(CANDIDATE_JSON, WORKFLOW)
    )
    changed = _edited(billing={"prompt": "Edited after the test."})
    scenario = _scenario(verification=_kept_verification(tested_hash))
    save_draft, archive = _patch_save_db(monkeypatch, scenario, changed)

    result = await verify.save_verified(scenario_id=1, organization_id=7)

    assert result["saved"] is False
    assert result["error_code"] == "changed_since_test"
    assert result["requires_reverify"] is True
    save_draft.assert_not_awaited()
    archive.assert_not_awaited()


@pytest.mark.asyncio
async def test_save_with_matching_hash_saves_and_archives(monkeypatch):
    canonical = verify.canonicalize_candidate(CANDIDATE_JSON, WORKFLOW)
    scenario = _scenario(
        verification=_kept_verification(compute_definition_hash(canonical))
    )
    save_draft, archive = _patch_save_db(monkeypatch, scenario, CANDIDATE_JSON)

    result = await verify.save_verified(
        scenario_id=1, organization_id=7, expected_draft_version=4
    )

    assert result["saved"] is True and result["verified"] is True
    assert save_draft.await_args.kwargs["workflow_definition"] == canonical
    # D30 (amended): archived into the record, row retired not deleted.
    archive.assert_awaited_once_with(55, 7)
    archived = scenario.verification["archived_candidate"]
    assert archived["outcome"] == "saved"
    assert archived["workflow_json"] == canonical


@pytest.mark.asyncio
async def test_save_draft_conflict_on_stale_version(monkeypatch):
    canonical = verify.canonicalize_candidate(CANDIDATE_JSON, WORKFLOW)
    scenario = _scenario(
        verification=_kept_verification(compute_definition_hash(canonical))
    )
    save_draft, _archive = _patch_save_db(monkeypatch, scenario, CANDIDATE_JSON)

    result = await verify.save_verified(
        scenario_id=1, organization_id=7, expected_draft_version=3
    )

    assert result["error_code"] == "draft_conflict"
    save_draft.assert_not_awaited()


@pytest.mark.asyncio
async def test_not_kept_fix_cannot_be_saved(monkeypatch):
    verification = VerificationResult(
        status=VerificationStatus.not_kept, cluster_id="C1", scenario_ids=[1]
    ).model_dump(mode="json")
    scenario = _scenario(verification=verification)
    save_draft, _ = _patch_save_db(monkeypatch, scenario, CANDIDATE_JSON)
    result = await verify.save_verified(scenario_id=1, organization_id=7)
    assert result["error_code"] == "not_savable"
    save_draft.assert_not_awaited()


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grade_sets_verdicts_and_passes_context_as_data(monkeypatch):
    seen: list[str] = []

    async def fake_claude(*, system, user_content, output_model, **_kwargs):
        seen.append(user_content)
        passed = "49" not in user_content.split("<reply>")[1]
        return output_model(passed=passed, reason="checked"), {}

    monkeypatch.setattr(verify, "claude_structured", fake_claude)
    samples = [
        SampleResult(text="It costs 49.", passed=True),
        SampleResult(
            text="Let me connect you to sales.", tool_calls=["transfer"], passed=False
        ),
    ]
    graded = await verify.grade(samples, "No prices.", context="How much?", votes=1)

    assert [g.passed for g in graded] == [False, True]
    assert all(g.reason == "checked" for g in graded)
    assert samples[0].passed is True  # inputs are not mutated
    assert "<context>\nHow much?\n</context>" in seen[0]
    assert "transfer" in seen[1]


@pytest.mark.asyncio
async def test_grade_majority_vote_and_agent_knowledge(monkeypatch):
    seen: list[str] = []
    votes = iter([True, False, True])  # one flip; the majority passes

    async def fake_claude(*, system, user_content, output_model, **_kwargs):
        seen.append(user_content)
        passed = next(votes)
        return output_model(passed=passed, reason="yes" if passed else "no"), {}

    monkeypatch.setattr(verify, "claude_structured", fake_claude)
    graded = await verify.grade(
        [SampleResult(text="Self-hosting is free.", passed=False)],
        "Only states facts it was given.",
        agent_knowledge="[Global instructions]\nSelf-hosting is free.",
    )

    assert len(seen) == verify.JUDGE_VOTES == 3
    assert graded[0].passed is True and graded[0].reason == "yes"
    assert "<agent_knowledge>\n[Global instructions]" in seen[0]
    with pytest.raises(ValueError):
        await verify.grade([], "x", votes=2)


def test_agent_knowledge_for_takes_global_and_failing_step_only():
    workflow = {
        "nodes": [
            {
                "id": "g",
                "type": "globalNode",
                "data": {"name": "G", "prompt": "Be kind."},
            },
            {
                "id": "a",
                "type": "agentNode",
                "data": {"name": "Answer", "prompt": "Price is 1c."},
            },
            {
                "id": "b",
                "type": "agentNode",
                "data": {"name": "Other", "prompt": "Secret."},
            },
        ]
    }
    text = verify.agent_knowledge_for(workflow, "Answer")
    assert "Be kind." in text and "Price is 1c." in text
    assert "Secret." not in text
    assert verify.agent_knowledge_for(workflow, "a") == text  # by id too


# ---------------------------------------------------------------------------
# D13 grouping
# ---------------------------------------------------------------------------


def test_group_replies_counts_and_order():
    samples = [
        SampleResult(text="Sure, it is 49 dollars.", passed=True),
        SampleResult(text="sure it is 49 dollars", passed=True),
        SampleResult(text="I can't share prices.", passed=True),
        SampleResult(text="Sure,  it is 49 DOLLARS!", passed=False, reason="price"),
        SampleResult(text="I can't share prices", tool_calls=["transfer"], passed=True),
        SampleResult(text="I can't share prices.", passed=True),
    ]
    groups = group_replies(samples)

    assert [g["count"] for g in groups] == [3, 2, 1]
    assert groups[0]["text"] == "Sure, it is 49 dollars."
    assert groups[0]["flagged"] is True and groups[0]["flag_reason"] == "price"
    assert groups[1]["flagged"] is False
    assert groups[2]["tool_calls"] == ["transfer"]
    assert sum(g["count"] for g in groups) == len(samples)
    assert len(groups[0]["members"]) == 3


# ---------------------------------------------------------------------------
# O1 dry-run results
# ---------------------------------------------------------------------------


def test_dry_run_replays_recorded_then_synthetic_and_failure_mocks():
    dry = ToolDryRun.from_annotations(
        {
            "doctor_replay": {
                "tool_results": {"lookup": ['{"amount": 99}', '{"amount": 120}']},
                "failure_mocks": {"book": {"status": "error", "error": "busy"}},
            }
        }
    )
    synthetic = SYNTHETIC_HTTP_RESULT
    assert dry.result_for("lookup", {}, synthetic=synthetic) == {"amount": 99}
    assert dry.result_for("lookup", {}, synthetic=synthetic) == {"amount": 120}
    # exhausted: repeat the last recorded result
    assert dry.result_for("lookup", {}, synthetic=synthetic) == {"amount": 120}
    assert dry.result_for("other", {"a": 1}, synthetic=synthetic) == synthetic
    assert dry.result_for("book", {}, synthetic=synthetic) == {
        "status": "error",
        "error": "busy",
    }
    assert [c[0] for c in dry.calls] == ["lookup", "lookup", "lookup", "other", "book"]


def test_no_annotations_means_empty_dry_run():
    dry = ToolDryRun.from_annotations(None)
    assert dry.tool_results == {} and dry.failure_mocks == {}


def test_local_hash_matches_save_path_hash():
    import api.mcp_server  # noqa: F401  (import order: see verify.compute_definition_hash)
    from api.services.workflow.sdk_authoring import (
        compute_definition_hash as save_path_hash,
    )

    assert compute_definition_hash(CANDIDATE_JSON) == save_path_hash(CANDIDATE_JSON)
    assert compute_definition_hash(None) == save_path_hash(None)


# ---------------------------------------------------------------------------
# Tester "Regenerate ×10": turn = zero-based index into the session's turns
# ---------------------------------------------------------------------------


def test_text_session_checkpoint_uses_state_stored_before_the_turn():
    from api.services.workflow.text_chat_runner import (
        _serialize_text_chat_checkpoint_messages,
    )

    after_turn_0 = {
        "version": 1,
        "anchor_turn_id": "t0",
        "current_node_id": "billing",
        "messages": _serialize_text_chat_checkpoint_messages(
            [{"role": "assistant", "content": "Hi"}]
        ),
        "gathered_context": {"x": 1},
        "tool_state": {},
    }
    session = SimpleNamespace(
        session_data={
            "turns": [
                {
                    "id": "t0",
                    "status": "completed",
                    "user_message": None,
                    "checkpoint_after_turn": after_turn_0,
                    "events": [],
                },
                {
                    "id": "t1",
                    "status": "completed",
                    "user_message": {"text": "Price?"},
                    "checkpoint_after_turn": {},
                    "events": [
                        {
                            "type": "tool_call_result",
                            "payload": {"function_name": "lookup", "result": {"a": 1}},
                        }
                    ],
                },
            ]
        }
    )
    run = _recorded_run(
        mode="textchat", initial_context={"k": "v", "mps_correlation_id": "c"}
    )

    recorded = verify.checkpoint_from_text_session(run, session, 1)
    assert recorded.history == [{"role": "assistant", "content": "Hi"}]
    assert recorded.pending_user_message == "Price?"
    assert recorded.checkpoint["current_node_id"] == "billing"
    assert recorded.checkpoint["anchor_turn_id"] is None
    assert recorded.tool_results == {"lookup": [{"a": 1}]}
    assert recorded.initial_context == {"k": "v"}

    opening = verify.checkpoint_from_text_session(run, session, 0)
    assert opening.checkpoint["current_node_id"] is None
    assert opening.history == [] and opening.pending_user_message is None
    assert verify.checkpoint_from_text_session(run, session, 5) is None


# ---------------------------------------------------------------------------
# Propose: minimal edit applied, candidate + scenario created, label routed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_propose_fix_creates_candidate_and_scenario(monkeypatch):
    from api.services.workflow.doctor.contracts import (
        Cluster,
        Evidence,
        ReportResult,
    )

    cluster = Cluster(
        cluster_id="C1",
        title="Agent quotes prices",
        category="hallucination",
        severity="major",
        fix_target="Billing",
        root_cause="Prompt does not forbid quoting plan prices strongly enough",
        runs_affected=3,
        runs_analysed=10,
        evidence=[
            Evidence(run_id=100, turn=2, node_name="Billing", quote="49 dollars")
        ],
    )
    report = SimpleNamespace(
        id=11,
        workflow_id=9,
        result=ReportResult(clusters=[cluster], ai_enabled=True).model_dump(
            mode="json"
        ),
    )

    async def fake_claude(*, output_model, **_kwargs):
        return (
            output_model(
                edits=[
                    {
                        "node_id": "billing",
                        "find": "Never quote prices.",
                        "replace": "Never quote prices; offer to connect sales.",
                    }
                ],
                diff_summary="Billing offers sales instead of a price.",
                expected_behavior="Declines to quote a price and offers sales.",
            ),
            {},
        )

    monkeypatch.setattr(verify, "claude_structured", fake_claude)
    created: list[dict] = []
    updates: list[dict] = []

    async def create_regression_scenario(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(id=500 + len(created), **kwargs)

    async def update_regression_scenario(scenario_id, organization_id, **fields):
        updates.append({"id": scenario_id, **fields})

    db = verify.db_client
    monkeypatch.setattr(
        db, "get_workflow", AsyncMock(return_value=SimpleNamespace(id=9, user_id=3))
    )
    monkeypatch.setattr(
        db,
        "get_draft_version",
        AsyncMock(
            return_value=SimpleNamespace(
                id=50, workflow_json=WORKFLOW, workflow_configurations={"a": 1}
            )
        ),
    )
    candidate_create = AsyncMock(return_value=SimpleNamespace(id=55))
    monkeypatch.setattr(
        db, "create_candidate_definition", candidate_create, raising=False
    )
    monkeypatch.setattr(
        db, "create_regression_scenario", create_regression_scenario, raising=False
    )
    monkeypatch.setattr(
        db, "update_regression_scenario", update_regression_scenario, raising=False
    )
    monkeypatch.setattr(db, "get_workflow_run", AsyncMock(return_value=_recorded_run()))
    monkeypatch.setattr(verify, "validate_candidate", lambda _payload: [])

    result, scenario_id = await verify.propose_fix(
        report=report, cluster_id="C1", organization_id=7
    )

    assert scenario_id == 501
    assert result.status == VerificationStatus.awaiting_start
    assert result.scenario_ids == [501]
    assert result.cost_estimate_usd and result.cost_estimate_usd > 0
    candidate_json = candidate_create.await_args.kwargs["workflow_json"]
    billing = next(n for n in candidate_json["nodes"] if n["id"] == "billing")
    assert billing["data"]["prompt"] == (
        "Answer billing questions. Never quote prices; offer to connect sales."
    )
    # only that one line changed
    assert classify_edit(WORKFLOW, candidate_json, "billing") == "turn_verifiable"
    assert result.candidate_hash == compute_definition_hash(candidate_json)
    assert created[0]["turn"] == 2 and created[0]["source_run_id"] == 100
    assert created[0]["expected_behavior"].startswith("Declines")
    assert updates[-1]["verification"]["scenario_ids"] == [501]


def test_prompt_edit_with_unknown_text_is_rejected():
    with pytest.raises(verify.ProposalError):
        verify.apply_prompt_edits(
            WORKFLOW,
            [verify.PromptEdit(node_id="billing", find="not in prompt", replace="x")],
        )


@pytest.mark.asyncio
async def test_previous_rejected_attempts_feed_the_next_proposal(monkeypatch):
    """A retry must see why the last fix failed, or it repeats the same edit."""
    rejected = VerificationResult(
        status=VerificationStatus.not_kept,
        cluster_id="C3",
        candidate_failures=4,
        diff_summary="Told Welcome not to answer product questions.",
        candidate_samples=[
            SampleResult(text="It's one cent a minute.", passed=False, reason="priced"),
            SampleResult(text="", tool_calls=["move_to_questions"], passed=True),
        ],
    )
    scenarios = [
        SimpleNamespace(
            id=1, cluster_id="C3", verification=rejected.model_dump(mode="json")
        ),
        SimpleNamespace(  # secondary call of the same multi-call check: skipped
            id=2,
            cluster_id="C3",
            verification=rejected.model_copy(
                update={"scenario_ids": [1, 2]}
            ).model_dump(mode="json"),
        ),
        SimpleNamespace(  # other cluster: ignored
            id=3, cluster_id="C1", verification=rejected.model_dump(mode="json")
        ),
        SimpleNamespace(  # kept attempt: not a failure to learn from
            id=4,
            cluster_id="C3",
            verification=rejected.model_copy(
                update={"status": VerificationStatus.kept}
            ).model_dump(mode="json"),
        ),
    ]
    lister = AsyncMock(return_value=scenarios)
    monkeypatch.setattr(
        verify.db_client, "list_regression_scenarios", lister, raising=False
    )

    attempts = await verify._previous_attempts(
        SimpleNamespace(id=11, workflow_id=9), "C3", organization_id=7
    )

    lister.assert_awaited_once_with(9, 7, report_id=11)
    assert attempts == [
        {
            "change": "Told Welcome not to answer product questions.",
            "candidate_failures": 4,
            "samples": 2,
            "failed_replies": [
                {"reply": "It's one cent a minute.", "tool_calls": [], "why": "priced"}
            ],
        }
    ]


@pytest.mark.asyncio
async def test_multi_call_verdict_keeps_primary_samples_and_lists_each_call(
    monkeypatch,
):
    """Regression (gate run, 2026-10-01): the worst call's samples overwrote
    the primary's, so the screen showed another call's numbers."""
    primary = _scenario(
        verification=VerificationResult(
            status=VerificationStatus.awaiting_start,
            cluster_id="C1",
            scenario_ids=[1, 2],
            candidate_definition_id=55,
        ).model_dump(mode="json")
    )
    second = _scenario(id=2, source_run_id=101, turn=4)
    stored = _patch_verify_db(monkeypatch, primary)
    monkeypatch.setattr(
        verify.db_client,
        "get_regression_scenario",
        AsyncMock(side_effect=lambda sid, _org: {1: primary, 2: second}[sid]),
        raising=False,
    )
    candidate_fails = {100: 2, 101: 6}

    async def regenerate_fn(run_id, turn, definition, n, *, label, on_sample, **_kw):
        bad = 8 if label == "baseline" else candidate_fails[run_id]
        samples = []
        for i in range(n):
            sample = SampleResult(
                text=f"bad run {run_id}" if i < bad else "good", passed=True
            )
            samples.append(sample)
            await on_sample(i, sample)
        return samples

    async def grade_fn(samples, expected, **_kw):
        return [
            s.model_copy(update={"passed": not s.text.startswith("bad")})
            for s in samples
        ]

    monkeypatch.setattr(
        verify.db_client,
        "get_workflow_run",
        AsyncMock(side_effect=lambda rid, **_kw: _recorded_run()),
    )
    result = await verify.verify_fix(
        scenario_id=1,
        organization_id=7,
        regenerate_fn=regenerate_fn,
        grade_fn=grade_fn,
    )

    assert result.status == VerificationStatus.not_kept
    # The primary's own numbers and replies, not the worst call's.
    assert result.candidate_failures == 2
    assert all("run 100" in s.text for s in result.candidate_samples if not s.passed)
    assert [
        (o.scenario_id, o.source_run_id, o.candidate_failures)
        for o in result.scenario_outcomes
    ] == [(1, 100, 2), (2, 101, 6)]
    assert stored[-1]["verification"]["candidate_failures"] == 2


def test_replay_start_timeout_is_retryable_but_real_errors_are_not():
    """Gate run: one replay missed the 5s pipeline start on a loaded machine
    and failed the whole verification. Start timeouts are transient."""

    class TextChatSessionExecutionError(Exception):
        pass

    try:
        try:
            raise TimeoutError()
        except TimeoutError as timeout:
            raise TextChatSessionExecutionError("turn failed") from timeout
    except TextChatSessionExecutionError as wrapped:
        assert verify.is_retryable_replay_error(wrapped)

    assert not verify.is_retryable_replay_error(RuntimeError("pipeline crashed"))
    assert not verify.is_retryable_replay_error(None)


@pytest.mark.asyncio
async def test_backoff_retries_a_start_timeout_once_then_succeeds():
    attempts = []

    async def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise TimeoutError()
        return "ok"

    async def no_sleep(_seconds):
        return None

    assert await verify.with_rate_limit_backoff(flaky, sleep=no_sleep) == "ok"
    assert len(attempts) == 2


def test_credit_check_outage_is_retryable_but_no_credits_is_not():
    assert verify.is_retryable_replay_error(
        verify.ReplayTransientError("Could not verify Dograh credits.")
    )
    assert not verify.is_retryable_replay_error(
        verify.VerifyError("Insufficient Dograh credits")
    )


@pytest.mark.asyncio
async def test_failure_on_a_later_call_keeps_finished_results(monkeypatch):
    """Gate run: a network blip on call #12 erased call #16's 10+10 replays."""
    primary = _scenario(
        verification=VerificationResult(
            status=VerificationStatus.awaiting_start,
            cluster_id="C1",
            scenario_ids=[1, 2],
            candidate_definition_id=55,
        ).model_dump(mode="json")
    )
    second = _scenario(id=2, source_run_id=101, turn=4)
    stored = _patch_verify_db(monkeypatch, primary)
    monkeypatch.setattr(
        verify.db_client,
        "get_regression_scenario",
        AsyncMock(side_effect=lambda sid, _org: {1: primary, 2: second}[sid]),
        raising=False,
    )

    async def regenerate_fn(run_id, turn, definition, n, *, label, on_sample, **_kw):
        if run_id == 101:
            raise RuntimeError("billing service unreachable")
        samples = []
        for i in range(n):
            sample = SampleResult(text="bad" if i < 7 else "good", passed=True)
            samples.append(sample)
            await on_sample(i, sample)
        return samples

    async def grade_fn(samples, expected, **_kw):
        return [s.model_copy(update={"passed": s.text != "bad"}) for s in samples]

    result = await verify.verify_fix(
        scenario_id=1,
        organization_id=7,
        regenerate_fn=regenerate_fn,
        grade_fn=grade_fn,
    )

    assert result.status == VerificationStatus.failed
    assert "billing service unreachable" in result.error
    assert result.baseline_failures == 7
    assert len(result.baseline_samples) == 10 and len(result.candidate_samples) == 10
    assert stored[-1]["verification"]["baseline_failures"] == 7
