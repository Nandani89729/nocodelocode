"""Copilot agent loop with a fake provider, fake tools and an in-memory DB."""

import sys
import types
from types import SimpleNamespace

import pytest

import api.services.copilot as copilot_pkg
from api.services.copilot import agent, llm
from api.services.copilot.contracts import (
    MAX_PROPOSAL_FIX_ROUNDS,
    MAX_TOOL_ITERATIONS_PER_TURN,
    MAX_TOOL_RESULT_CHARS,
    Plan,
    PlanStep,
    Proposal,
    ProposalKind,
    ToolCall,
    ToolOutcome,
    ToolSpec,
    TranscriptMessage,
)

USER = SimpleNamespace(id=7, selected_organization_id=1)


class FakeDB:
    def __init__(self, workflow_id=5):
        self.session = SimpleNamespace(
            session_uuid="s-1",
            organization_id=1,
            workflow_id=workflow_id,
            title=None,
            transcript=[],
            plan=None,
        )
        self.saves = []  # snapshot of transcript length per save

    async def get_copilot_session(self, session_uuid, organization_id):
        if session_uuid == "s-1" and organization_id == 1:
            return self.session
        return None

    async def update_copilot_session(self, session_uuid, organization_id, **fields):
        assert organization_id == 1
        for key, value in fields.items():
            setattr(self.session, key, value)
        self.saves.append(len(self.session.transcript))
        return self.session

    async def get_workflow(self, workflow_id, organization_id=None):
        return SimpleNamespace(id=workflow_id, name="Lead Qualifier")


class FakeProvider:
    """Plays scripted ModelTurns; records what the loop sent."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []

    async def stream(self, *, system, messages, tools, replay):
        self.calls.append(
            {"system": system, "messages": list(messages), "replay": dict(replay)}
        )
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        for word in turn.text.split(" ") if turn.text else []:
            yield word
        yield turn


class FakeTools:
    def __init__(self, outcomes=None):
        self.outcomes = outcomes or {}
        self.executed = []

    def get_tool_specs(self, ctx):
        return [ToolSpec(name="get_workflow_code", description="read", input_schema={})]

    async def execute_tool(self, name, tool_input, ctx):
        self.executed.append((name, tool_input, ctx.fix_rounds_used))
        outcome = self.outcomes.get(name, ToolOutcome(content={"ok": 1}, summary="Ok"))
        if isinstance(outcome, list):
            outcome = outcome.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def env(monkeypatch):
    db = FakeDB()
    fake_tools = FakeTools()
    module = types.ModuleType("api.services.copilot.tools")
    module.get_tool_specs = fake_tools.get_tool_specs
    module.execute_tool = fake_tools.execute_tool
    monkeypatch.setitem(sys.modules, "api.services.copilot.tools", module)
    monkeypatch.setattr(copilot_pkg, "tools", module, raising=False)
    monkeypatch.setattr(agent, "db_client", db)
    yield db, fake_tools
    llm.set_provider_for_tests(None)


def call(id_, name="get_workflow_code", **input_):
    return ToolCall(id=id_, name=name, input=input_)


async def run(text="Add a qualify node"):
    return [
        e
        async for e in agent.run_turn(
            session_uuid="s-1", organization_id=1, user=USER, text=text
        )
    ]


def types_of(events):
    return [e.type.value for e in events]


async def test_text_only_turn(env):
    db, _ = env
    provider = FakeProvider([llm.ModelTurn(text="Hello there", usage={"x": 1})])
    llm.set_provider_for_tests(provider)

    events = await run("  Help me   build\n a lead qualifier  ")

    assert types_of(events) == ["text_delta", "text_delta", "message_done"]
    assert "".join(e.data["text"] for e in events[:2]) == "Hellothere"
    assert db.session.title == "Help me build a lead qualifier"
    roles = [m["role"] for m in db.session.transcript]
    assert roles == ["user", "assistant"]
    assert db.session.transcript[1]["content"] == "Hello there"
    # Bound workflow reaches the system prompt (session part).
    assert "Lead Qualifier" in provider.calls[0]["system"][1]


async def test_tool_turn_then_final_text(env):
    db, tools = env
    replay_blocks = [{"type": "thinking", "thinking": "t", "signature": "sig"}]
    provider = FakeProvider(
        [
            llm.ModelTurn(
                text="Reading.",
                tool_calls=[call("c1", workflow_id=5)],
                stop_reason=llm.STOP_TOOL_USE,
                replay=replay_blocks,
            ),
            llm.ModelTurn(text="Done."),
        ]
    )
    llm.set_provider_for_tests(provider)

    events = await run()

    assert types_of(events) == [
        "text_delta",
        "tool_call",
        "tool_result",
        "text_delta",
        "message_done",
    ]
    assert events[1].data == {
        "id": "c1",
        "name": "get_workflow_code",
        "summary": "Get workflow code",
    }
    assert events[2].data == {
        "id": "c1",
        "name": "get_workflow_code",
        "ok": True,
        "summary": "Ok",
    }
    assert tools.executed == [("get_workflow_code", {"workflow_id": 5}, 0)]
    transcript = [TranscriptMessage.model_validate(m) for m in db.session.transcript]
    assert [m.role for m in transcript] == ["user", "assistant", "tool", "assistant"]
    assert transcript[2].tool_call_id == "c1" and transcript[2].content == '{"ok": 1}'
    # The second call resends the thinking blocks of the tool-use message.
    assert provider.calls[1]["replay"] == {"c1": replay_blocks}
    assert [m.role for m in provider.calls[1]["messages"]] == [
        "user",
        "assistant",
        "tool",
    ]


async def test_plan_and_proposal_events(env):
    db, tools = env
    plan = Plan(summary="Add qualify", steps=[PlanStep(title="Add node")])
    proposal = Proposal(
        proposal_uuid="p-1",
        session_uuid="s-1",
        kind=ProposalKind.edit,
        workflow_id=5,
        summary="Add qualify node",
    )
    tools.outcomes = {
        "present_plan": ToolOutcome(content="shown", summary="Plan", plan=plan),
        "propose_changes": ToolOutcome(
            content={"proposal_uuid": "p-1"}, summary="1 change", proposal=proposal
        ),
    }
    llm.set_provider_for_tests(
        FakeProvider(
            [
                llm.ModelTurn(
                    tool_calls=[call("c1", "present_plan")],
                    stop_reason=llm.STOP_TOOL_USE,
                ),
                llm.ModelTurn(
                    tool_calls=[call("c2", "propose_changes", code="x")],
                    stop_reason=llm.STOP_TOOL_USE,
                ),
                llm.ModelTurn(text="Ready for review."),
            ]
        )
    )

    events = await run()

    kinds = types_of(events)
    assert kinds[:7] == [
        "tool_call",
        "tool_result",
        "plan",
        "tool_call",
        "tool_result",
        "proposal",
        "text_delta",
    ]
    assert kinds[-1] == "message_done"
    assert events[2].data["plan"]["summary"] == "Add qualify"
    assert events[5].data["proposal"]["proposal_uuid"] == "p-1"
    assert db.session.plan["steps"][0]["title"] == "Add node"


async def test_failed_proposals_count_fix_rounds_and_cap(env):
    db, tools = env
    failure = ToolOutcome(ok=False, content={"errors": ["bad"]}, summary="Invalid")
    tools.outcomes = {"propose_changes": [failure] * 10}
    attempts = MAX_PROPOSAL_FIX_ROUNDS + 2
    turns = [
        llm.ModelTurn(
            tool_calls=[call(f"c{i}", "propose_changes", code="x")],
            stop_reason=llm.STOP_TOOL_USE,
        )
        for i in range(attempts)
    ] + [llm.ModelTurn(text="Could not fix.")]
    llm.set_provider_for_tests(FakeProvider(turns))

    events = await run()

    # The tool saw the running count; the attempt past the cap is refused
    # by the loop without reaching the tool.
    assert [r for _, _, r in tools.executed] == list(range(MAX_PROPOSAL_FIX_ROUNDS + 1))
    results = [e for e in events if e.type.value == "tool_result"]
    assert len(results) == attempts and not any(r.data["ok"] for r in results)
    last_tool = [m for m in db.session.transcript if m["role"] == "tool"][-1]
    assert last_tool["is_error"] and "Fix round limit" in last_tool["content"]
    assert types_of(events)[-1] == "message_done"


async def test_iteration_cap(env):
    db, tools = env
    turns = [
        llm.ModelTurn(tool_calls=[call(f"c{i}")], stop_reason=llm.STOP_TOOL_USE)
        for i in range(MAX_TOOL_ITERATIONS_PER_TURN)
    ]
    provider = FakeProvider(turns)
    llm.set_provider_for_tests(provider)

    events = await run()

    assert len(provider.calls) == MAX_TOOL_ITERATIONS_PER_TURN
    assert len(tools.executed) == MAX_TOOL_ITERATIONS_PER_TURN
    assert events[-2].type.value == "text_delta"
    assert "stopped after" in events[-2].data["text"]
    assert events[-1].type.value == "message_done"
    assert db.session.transcript[-1]["role"] == "assistant"


async def test_exception_becomes_plain_error_and_history_survives(env):
    db, _ = env
    llm.set_provider_for_tests(
        FakeProvider(
            [
                llm.ModelTurn(tool_calls=[call("c1")], stop_reason=llm.STOP_TOOL_USE),
                RuntimeError("secret stack detail"),
            ]
        )
    )

    events = await run()

    assert events[-1].type.value == "error"
    assert events[-1].data["message"] == agent.GENERIC_ERROR
    assert "secret" not in events[-1].data["message"]
    # Persisted incrementally: user, assistant(call), tool result.
    assert [m["role"] for m in db.session.transcript] == ["user", "assistant", "tool"]
    assert db.saves == [1, 2, 3]


async def test_llm_error_message_is_shown(env):
    llm.set_provider_for_tests(
        FakeProvider([llm.CopilotLLMError("The AI service is busy.")])
    )
    events = await run()
    assert types_of(events) == ["error"]
    assert events[0].data["message"] == "The AI service is busy."


async def test_tool_exception_is_contained(env):
    db, tools = env
    tools.outcomes = {"get_workflow_code": RuntimeError("boom")}
    llm.set_provider_for_tests(
        FakeProvider(
            [
                llm.ModelTurn(tool_calls=[call("c1")], stop_reason=llm.STOP_TOOL_USE),
                llm.ModelTurn(text="Sorry."),
            ]
        )
    )
    events = await run()
    result = next(e for e in events if e.type.value == "tool_result")
    assert result.data["ok"] is False
    assert types_of(events)[-1] == "message_done"


@pytest.mark.parametrize(
    "stop,message", [(llm.STOP_REFUSAL, "declined"), (llm.STOP_MAX_TOKENS, "cut off")]
)
async def test_refusal_and_truncation_drop_tool_calls(env, stop, message):
    db, tools = env
    llm.set_provider_for_tests(
        FakeProvider(
            [llm.ModelTurn(text="Partial", tool_calls=[call("c1")], stop_reason=stop)]
        )
    )
    events = await run()
    assert events[-1].type.value == "error" and message in events[-1].data["message"]
    assert tools.executed == []
    assert db.session.transcript[-1]["tool_calls"] == []


async def test_missing_session_and_existing_title_kept(env):
    db, _ = env
    events = [
        e
        async for e in agent.run_turn(
            session_uuid="nope", organization_id=1, user=USER, text="hi"
        )
    ]
    assert types_of(events) == ["error"]

    db.session.title = "Original"
    llm.set_provider_for_tests(FakeProvider([llm.ModelTurn(text="ok")]))
    await run("second message")
    assert db.session.title == "Original"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def msg(role, content="", **kw):
    return TranscriptMessage(role=role, content=content, **kw)


def test_repair_adds_missing_results_and_drops_orphans():
    messages = [
        msg("user", "hi"),
        msg("tool", "orphan", tool_call_id="zz"),
        msg("assistant", tool_calls=[call("a"), call("b")]),
        msg("tool", "ra", tool_call_id="a"),
        msg("user", "again"),
    ]
    repaired = agent.repair_tool_pairs(messages)
    assert [(m.role, m.tool_call_id) for m in repaired] == [
        ("user", None),
        ("assistant", None),
        ("tool", "a"),
        ("tool", "b"),
        ("user", None),
    ]
    assert repaired[3].is_error and repaired[3].tool_name == "get_workflow_code"


def test_trim_keeps_first_user_and_never_starts_with_tool_result():
    messages = [msg("user", "first")]
    for i in range(50):
        messages.append(msg("assistant", tool_calls=[call(f"c{i}")]))
        messages.append(msg("tool", "r", tool_call_id=f"c{i}"))
    trimmed = agent.trim_transcript(messages, limit=10)
    assert trimmed[0].content == "first"
    assert trimmed[1].role == "assistant"
    assert len(trimmed) <= 10
    assert agent.trim_transcript(messages[:5], limit=10) == messages[:5]


def test_tool_result_text_truncates_and_serialises():
    long = ToolOutcome(content="x" * (MAX_TOOL_RESULT_CHARS + 50))
    text = agent.tool_result_text(long)
    assert text.startswith("x" * 100) and "truncated 50 characters" in text
    assert agent.tool_result_text(ToolOutcome(content={"a": "é"})) == '{"a": "é"}'
    assert agent.tool_result_text(ToolOutcome(content=None, summary="Read")) == "Read"


def test_session_title_truncates():
    assert agent.session_title("a" * 100).endswith("…")
    assert len(agent.session_title("a" * 100)) == agent.TITLE_MAX_CHARS


def test_system_prompt_reuses_mcp_guide_and_binds_session():
    from api.mcp_server.instructions import DOGRAH_MCP_INSTRUCTIONS
    from api.services.copilot import prompt

    stable, bound = prompt.build_system_prompt(workflow_id=5, workflow_name="Lead")
    assert DOGRAH_MCP_INSTRUCTIONS in stable
    assert "`present_plan`" in stable and "`propose_changes`" in stable
    assert f"at most {MAX_PROPOSAL_FIX_ROUNDS} fix rounds" in stable
    assert '"Lead" (id 5)' in bound
    _, new = prompt.build_system_prompt()
    assert "new workflow" in new
    # The stable part is identical across sessions (provider cache prefix).
    assert prompt.build_system_prompt(workflow_id=6)[0] == stable
