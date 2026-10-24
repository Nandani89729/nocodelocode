"""Copilot changes for Self-improve (M2b RI-4, O2, O4, O5, O8).

Tools per mode, the brief tools, the report binding, propose-once,
hypotheses mapping, the `applying` apply lifecycle with mark_reviewed, and
`run_self_improve_turn` with a fake provider.
"""

from __future__ import annotations

import shutil
import sys
import types
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import api.services.copilot as copilot_pkg
from api.db import db_client
from api.services.copilot import agent, llm, proposals, tools
from api.services.copilot.contracts import (
    ApplyProposalRequest,
    BriefChange,
    ChangeItem,
    ChangeOp,
    ChangeTarget,
    Proposal,
    ProposalKind,
    ProposalStatus,
    ToolCall,
    ToolContext,
    ToolOutcome,
    ToolSpec,
    WorkflowBrief,
)
from api.services.copilot.proposals import ProposalTarget, map_hypotheses
from api.services.copilot.tools import execute_tool, get_tool_specs
from api.tests.test_copilot_proposals import (
    BASE_CODE,
    ORG,
    PROPOSED_CODE,
    WORKFLOW_ID,
    _Backend,
    _node_id,
    _stored,
)

needs_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="node binary not available"
)


def _ctx(
    workflow_id: int | None = WORKFLOW_ID,
    *,
    mode: str = "chat",
    iteration_id: int | None = None,
    report_uuid: str | None = None,
    organization_id: int = ORG,
) -> ToolContext:
    return ToolContext(
        session_uuid="sess-1",
        organization_id=organization_id,
        user_id=1,
        workflow_id=workflow_id,
        mode=mode,
        iteration_id=iteration_id,
        report_uuid=report_uuid,
    )


def _si_ctx(**kw) -> ToolContext:
    kw.setdefault("iteration_id", 5)
    kw.setdefault("report_uuid", "r-bound")
    return _ctx(mode="self_improve", **kw)


@pytest.fixture
def user():
    return SimpleNamespace(id=1, selected_organization_id=ORG, provider_id="p-1")


# ─── tool specs per mode (RI-4) ──────────────────────────────────────────


def _names(ctx: ToolContext) -> set[str]:
    return {s.name for s in get_tool_specs(ctx)}


def test_tool_specs_per_mode():
    chat = _names(_ctx())
    rounds = _names(_si_ctx())
    build = _names(_ctx(None))

    assert {"present_plan", "update_workflow_brief", "get_workflow_brief"} <= chat
    assert {"get_doctor_report", "propose_changes"} <= chat
    assert "present_plan" not in rounds and "update_workflow_brief" not in rounds
    assert {"get_workflow_brief", "get_doctor_report", "propose_changes"} <= rounds
    assert chat - rounds == {"present_plan", "update_workflow_brief"}
    # Building a new workflow: no brief tools (no workflow yet).
    assert "get_workflow_brief" not in build and "update_workflow_brief" not in build
    assert "present_plan" in build

    bound = next(s for s in get_tool_specs(_si_ctx()) if s.name == "get_doctor_report")
    assert bound.input_schema["properties"] == {}
    assert "bound" in bound.description
    free = next(s for s in get_tool_specs(_ctx()) if s.name == "get_doctor_report")
    assert "report_uuid" in free.input_schema["properties"]
    propose = next(s for s in get_tool_specs(_ctx()) if s.name == "propose_changes")
    assert "targets" in propose.input_schema["properties"]
    assert "targets" in propose.description


@pytest.mark.parametrize("name", ["present_plan", "update_workflow_brief"])
async def test_hidden_tools_are_refused_in_self_improve(name):
    outcome = await execute_tool(name, {"summary": "x", "steps": []}, _si_ctx())
    assert outcome.ok is False
    assert outcome.content["error_code"] == "not_allowed"


async def test_brief_tools_need_a_workflow():
    outcome = await execute_tool("get_workflow_brief", {}, _ctx(None))
    assert outcome.ok is False and outcome.content["error_code"] == "not_allowed"


# ─── brief tools (O8), real DB ───────────────────────────────────────────


async def _org_with_workflow(db, slug: str):
    user, _ = await db.get_or_create_user_by_provider_id(f"{slug}_user")
    org, _ = await db.get_or_create_organization_by_provider_id(f"{slug}_org", user.id)
    workflow = await db.create_workflow(
        name=f"{slug} workflow",
        workflow_definition={"nodes": [], "edges": []},
        user_id=user.id,
        organization_id=org.id,
    )
    return org, workflow


async def test_update_workflow_brief_only_creates_a_pending_change(db_session):
    org, workflow = await _org_with_workflow(db_session, "si_brief_tool")
    ctx = _ctx(workflow.id, organization_id=org.id)
    args = {
        "brief": {"goal": "Book demos", "must_never": ["Quote prices"]},
        "reason": "User: never quote prices",
    }

    outcome = await execute_tool("update_workflow_brief", args, ctx)
    assert outcome.ok is True, outcome.content
    assert isinstance(outcome.brief_change, BriefChange)
    assert outcome.brief_change.status.value == "pending"
    assert outcome.brief_change.proposed.must_never == ["Quote prices"]
    assert "NOT saved" in outcome.content["next"]
    # The brief itself is untouched.
    assert await db_session.get_workflow_brief(workflow.id, org.id) is None

    read = await execute_tool("get_workflow_brief", {}, ctx)
    assert read.ok is True and read.content["version"] == 0
    assert read.content["pending_change"]["change_uuid"] == (
        outcome.brief_change.change_uuid
    )
    read_si = await execute_tool(
        "get_workflow_brief",
        {},
        _si_ctx(workflow_id=workflow.id, organization_id=org.id),
    )
    assert read_si.content["read_only"] is True
    assert "pending_change" not in read_si.content

    same = await execute_tool(
        "update_workflow_brief",
        {"brief": {}, "reason": "nothing"},
        ctx,
    )
    assert same.ok is False and same.content["error_code"] == "no_changes"

    other_org = await execute_tool(
        "update_workflow_brief", args, _ctx(workflow.id, organization_id=org.id + 999)
    )
    assert other_org.ok is False and other_org.content["error_code"] == "not_found"


async def test_update_workflow_brief_rejects_bad_input():
    bad = await execute_tool(
        "update_workflow_brief", {"brief": "goal", "reason": "x"}, _ctx()
    )
    assert bad.ok is False
    extra = await execute_tool(
        "update_workflow_brief", {"brief": {"goal": 3}, "reason": "x"}, _ctx()
    )
    assert extra.ok is False


# ─── get_doctor_report binding (O4) ──────────────────────────────────────


def _report(uuid: str, workflow_id: int = WORKFLOW_ID):
    return SimpleNamespace(
        report_uuid=uuid,
        workflow_id=workflow_id,
        organization_id=ORG,
        status="completed",
        created_at=datetime(2026, 10, 1, tzinfo=UTC),
        result={"run_ids": [1, 2], "clusters": []},
        error=None,
    )


@pytest.fixture
def reports():
    stored = {"r-bound": _report("r-bound"), "r-other": _report("r-other")}
    stored["r-foreign-wf"] = _report("r-foreign-wf", workflow_id=99)

    async def get(report_uuid, organization_id):
        return stored.get(report_uuid) if organization_id == ORG else None

    with (
        patch.object(
            db_client, "get_analysis_report", AsyncMock(side_effect=get)
        ) as get_report,
        patch.object(
            db_client,
            "get_workflow",
            AsyncMock(return_value=SimpleNamespace(id=WORKFLOW_ID)),
        ),
        patch.object(db_client, "list_analysis_reports", AsyncMock()) as listing,
    ):
        yield get_report, listing


async def test_self_improve_report_is_bound_to_the_round(reports):
    get_report, listing = reports
    outcome = await execute_tool("get_doctor_report", {}, _si_ctx())
    assert outcome.ok is True and outcome.content["report_uuid"] == "r-bound"
    get_report.assert_awaited_once_with("r-bound", ORG)
    listing.assert_not_awaited()

    same = await execute_tool(
        "get_doctor_report", {"report_uuid": "r-bound"}, _si_ctx()
    )
    assert same.ok is True
    for args in ({"report_uuid": "r-other"}, {"workflow_id": 99}):
        refused = await execute_tool("get_doctor_report", args, _si_ctx())
        assert refused.ok is False
        assert refused.content["error_code"] == "not_allowed"
        assert "r-bound" in refused.content["error"]

    unbound = await execute_tool("get_doctor_report", {}, _si_ctx(report_uuid=None))
    assert unbound.ok is False
    foreign = await execute_tool(
        "get_doctor_report", {}, _si_ctx(report_uuid="r-foreign-wf")
    )
    assert foreign.ok is False and foreign.content["error_code"] == "not_found"


async def test_chat_can_read_an_explicit_report(reports):
    get_report, listing = reports
    outcome = await execute_tool(
        "get_doctor_report", {"report_uuid": "r-other"}, _ctx()
    )
    assert outcome.ok is True and outcome.content["report_uuid"] == "r-other"
    listing.assert_not_awaited()
    foreign = await execute_tool(
        "get_doctor_report", {"report_uuid": "r-foreign-wf"}, _ctx()
    )
    assert foreign.ok is False and foreign.content["error_code"] == "not_found"


# ─── propose once per round (O5) ─────────────────────────────────────────


async def test_propose_changes_once_per_round():
    sentinel = ToolOutcome(ok=True, summary="Proposed 1 change")
    targets = [{"fix_target": "Qualify", "title": "Skips timeline", "cluster_id": "C1"}]
    with (
        patch.object(
            db_client, "get_proposal_for_iteration", AsyncMock(return_value=None)
        ) as lookup,
        patch.object(
            tools.proposals, "create_proposal", AsyncMock(return_value=sentinel)
        ) as create,
    ):
        first = await execute_tool(
            "propose_changes",
            {"code": "src", "summary": "s", "targets": targets},
            _si_ctx(),
        )
        lookup.assert_awaited_once_with(5, ORG)
        lookup.return_value = SimpleNamespace(proposal_uuid="p-1")
        second = await execute_tool(
            "propose_changes", {"code": "src", "summary": "s"}, _si_ctx()
        )
    assert first is sentinel
    assert create.await_count == 1
    passed = create.await_args.kwargs["targets"]
    assert passed == [ProposalTarget(**targets[0])]
    assert second.ok is False
    assert second.content["error_code"] == "already_proposed"
    assert "Already proposed" in second.content["error"]


async def test_propose_changes_rejects_bad_targets():
    with patch.object(tools.proposals, "create_proposal", AsyncMock()) as create:
        not_list = await execute_tool(
            "propose_changes", {"code": "x", "summary": "s", "targets": {}}, _ctx()
        )
        no_title = await execute_tool(
            "propose_changes",
            {"code": "x", "summary": "s", "targets": [{"cluster_id": "C1"}]},
            _ctx(),
        )
    assert not_list.ok is False and no_title.ok is False
    create.assert_not_awaited()


# ─── hypotheses mapping (O3) ─────────────────────────────────────────────


def _node_change(cid: str, name: str, op=ChangeOp.update) -> ChangeItem:
    return ChangeItem(
        change_id=cid,
        target=ChangeTarget.node,
        op=op,
        ref_id=cid,
        label=name,
        after={"type": "agentNode", "data": {"name": name}},
    )


def test_map_hypotheses():
    changes = [
        _node_change("n:1", "Welcome"),
        _node_change("n:2", "Global"),
        _node_change("n:3", "Goodbye"),
        ChangeItem(
            change_id="e:1",
            target=ChangeTarget.edge,
            op=ChangeOp.add,
            ref_id="e1",
            label="edge",
            after={"source": "1", "target": "3", "data": {"label": "done"}},
        ),
    ]
    names = {"1": "Welcome", "3": "Goodbye"}
    targets = [
        ProposalTarget(
            cluster_id="C1",
            title="Pricing",
            category="knowledge_gap",
            fix_target="welcome",
        ),
        # Knowledge placement: the Global change belongs to the same cluster.
        ProposalTarget(
            node_name="Global", cluster_id="C1", title="Pricing", fix_target="Welcome"
        ),
        ProposalTarget(
            node_name="edge:Welcome -> Goodbye", cluster_id="C2", title="Never ends"
        ),
        ProposalTarget(cluster_id="C9", title="Latency", fix_target="platform"),
    ]
    hypotheses, unmatched = map_hypotheses(changes, targets, names)
    assert set(hypotheses) == {"n:1", "n:2", "e:1"}
    assert hypotheses["n:1"].cluster_id == "C1"
    assert hypotheses["n:1"].category == "knowledge_gap"
    assert hypotheses["n:2"].fix_target == "Welcome"
    assert hypotheses["e:1"].cluster_id == "C2"
    assert hypotheses["e:1"].fix_target == "edge:Welcome -> Goodbye"
    assert [t.cluster_id for t in unmatched] == ["C9"]


@needs_node
async def test_self_improve_proposal_stores_iteration_and_hypotheses():
    stored = await _stored()
    targets = [
        ProposalTarget(cluster_id="C1", title="Skips timeline", fix_target="Qualify"),
        ProposalTarget(node_name="Book demo", cluster_id="C2", title="No demo offer"),
        ProposalTarget(cluster_id="C9", title="Slow", fix_target="platform"),
    ]
    with _Backend(stored) as b:
        outcome = await proposals.create_proposal(
            code=PROPOSED_CODE, summary="Fix", ctx=_si_ctx(), targets=targets
        )
        candidate = b.create_proposal.call_args.kwargs["candidate_json"]
        kwargs = b.create_proposal.call_args.kwargs
    assert outcome.ok is True, outcome.content
    assert kwargs["iteration_id"] == 5
    qualify = f"n:{_node_id(stored, 'Qualify')}"
    book = f"n:{_node_id(candidate, 'Book demo')}"
    assert {k: v.cluster_id for k, v in kwargs["hypotheses"].items()} == {
        qualify: "C1",
        book: "C2",
    }
    assert outcome.proposal.iteration_id == 5
    assert outcome.proposal.hypotheses[book].title == "No demo offer"
    view = {c["change_id"]: c for c in outcome.content["changes"]}
    assert view[qualify]["fixes"] == "Skips timeline"
    assert outcome.content["unmatched_targets"] == [
        {"cluster_id": "C9", "title": "Slow", "fix_target": "platform"}
    ]


@needs_node
async def test_chat_proposal_has_no_iteration_and_race_is_already_proposed():
    stored = await _stored()
    with _Backend(stored) as b:
        chat = await proposals.create_proposal(
            code=PROPOSED_CODE, summary="Fix", ctx=_ctx()
        )
        assert "iteration_id" not in b.create_proposal.call_args.kwargs
        assert chat.ok is True and chat.proposal.iteration_id is None
        b.create_proposal.side_effect = ValueError(
            "Self-improve iteration 5 already has a proposal"
        )
        raced = await proposals.create_proposal(
            code=PROPOSED_CODE, summary="Fix", ctx=_si_ctx()
        )
    assert raced.ok is False and raced.content["error_code"] == "already_proposed"


# ─── apply lifecycle: applying + accepted ids + mark_reviewed (O2) ───────


class _Lifecycle:
    """_Backend + mark_reviewed spy + a draft id from the save."""

    def __init__(self, backend: _Backend):
        self.b = backend
        self.statuses_during_save: list[str] = []
        original = backend._save

        async def save_with_id(**kwargs):
            for model in backend.models.values():
                self.statuses_during_save.append(model.status)
            draft = await original(**kwargs)
            draft.id = 321
            return draft

        backend.save.side_effect = save_with_id
        self.mark_reviewed = AsyncMock(return_value=None)
        self._patch = patch.object(db_client, "mark_reviewed", self.mark_reviewed)

    def __enter__(self):
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()


async def _apply(user, proposal, accepted):
    return await proposals.apply_proposal(
        proposal_uuid=proposal.proposal_uuid,
        organization_id=ORG,
        user=user,
        request=ApplyProposalRequest(accepted_change_ids=accepted),
    )


@needs_node
async def test_round_apply_goes_through_applying_and_marks_reviewed(user):
    stored = await _stored()
    qualify = f"n:{_node_id(stored, 'Qualify')}"
    with _Backend(stored) as b, _Lifecycle(b) as life:
        proposal = (
            await proposals.create_proposal(
                code=PROPOSED_CODE, summary="s", ctx=_si_ctx()
            )
        ).proposal
        response = await _apply(user, proposal, [qualify])
        model = b.models[proposal.proposal_uuid]
    assert response.saved is True, response
    assert life.statuses_during_save == ["applying"]
    assert model.status == "applied"
    assert model.accepted_change_ids == [qualify]
    assert model.applied_definition_id == 321
    assert model.applied_at is not None
    assert response.proposal.accepted_change_ids == [qualify]
    assert response.proposal.applied_definition_id == 321
    life.mark_reviewed.assert_awaited_once_with(
        workflow_id=WORKFLOW_ID, organization_id=ORG
    )


@needs_node
async def test_failed_round_apply_releases_to_pending_without_review(user):
    stored = await _stored()
    with _Backend(stored) as b, _Lifecycle(b) as life:
        proposal = (
            await proposals.create_proposal(
                code=PROPOSED_CODE, summary="s", ctx=_si_ctx()
            )
        ).proposal
        bad = await _apply(user, proposal, ["does-not-exist"])
        status = b.models[proposal.proposal_uuid].status
        b.save.side_effect = RuntimeError("db down")
        with pytest.raises(RuntimeError):
            await _apply(user, proposal, [f"n:{_node_id(stored, 'Qualify')}"])
        status_after_crash = b.models[proposal.proposal_uuid].status
    assert bad.saved is False and status == "pending"
    assert status_after_crash == "pending"
    life.mark_reviewed.assert_not_awaited()


@needs_node
async def test_stale_and_discard_mark_the_round_reviewed(user):
    stored = await _stored()
    with _Backend(stored) as b, _Lifecycle(b) as life:
        stale = (
            await proposals.create_proposal(
                code=PROPOSED_CODE, summary="s", ctx=_si_ctx()
            )
        ).proposal
        b.draft.workflow_json["nodes"][0]["data"]["prompt"] = "Changed in the canvas"
        response = await _apply(user, stale, [stale.changes[0].change_id])
        assert response.error_code == "stale"
        assert b.models[stale.proposal_uuid].status == "stale"
        assert life.mark_reviewed.await_count == 1

        b.draft.workflow_json["nodes"][0]["data"]["prompt"] = "Hi!"
        discard = (
            await proposals.create_proposal(
                code=PROPOSED_CODE, summary="s", ctx=_si_ctx(iteration_id=6)
            )
        ).proposal
        discarded = await proposals.discard_proposal(
            proposal_uuid=discard.proposal_uuid, organization_id=ORG
        )
        assert discarded.status == ProposalStatus.discarded
        assert life.mark_reviewed.await_count == 2
        # A second discard is a no-op (no second review).
        await proposals.discard_proposal(
            proposal_uuid=discard.proposal_uuid, organization_id=ORG
        )
        assert life.mark_reviewed.await_count == 2


@needs_node
async def test_chat_proposal_apply_does_not_mark_reviewed(user):
    stored = await _stored()
    qualify = f"n:{_node_id(stored, 'Qualify')}"
    with _Backend(stored) as b, _Lifecycle(b) as life:
        proposal = (
            await proposals.create_proposal(code=PROPOSED_CODE, summary="s", ctx=_ctx())
        ).proposal
        response = await _apply(user, proposal, [qualify])
        assert b.models[proposal.proposal_uuid].accepted_change_ids == [qualify]
    assert response.saved is True
    life.mark_reviewed.assert_not_awaited()


@needs_node
async def test_applying_proposal_cannot_be_applied_or_discarded(user):
    stored = await _stored()
    with _Backend(stored) as b, _Lifecycle(b) as life:
        proposal = (
            await proposals.create_proposal(
                code=BASE_CODE.replace("Hi!", "Hello!"), summary="s", ctx=_si_ctx()
            )
        ).proposal
        b.models[proposal.proposal_uuid].status = "applying"
        again = await _apply(user, proposal, [proposal.changes[0].change_id])
        discarded = await proposals.discard_proposal(
            proposal_uuid=proposal.proposal_uuid, organization_id=ORG
        )
    assert again.saved is False and again.error_code == "already_applied"
    assert discarded.status == ProposalStatus.applying
    b.save.assert_not_awaited()
    life.mark_reviewed.assert_not_awaited()


def test_proposal_from_model_maps_m2b_fields():
    model = SimpleNamespace(
        proposal_uuid="p",
        session=SimpleNamespace(session_uuid="s"),
        kind="edit",
        workflow_id=3,
        summary="x",
        base_hash="h",
        changes=[],
        status="applying",
        workflow_name=None,
        created_at=None,
        iteration_id=4,
        accepted_change_ids=["n:1"],
        applied_definition_id=9,
        hypotheses={"n:1": {"title": "T", "cluster_id": "C1"}, "bad": {"x": 1}},
    )
    proposal = proposals.proposal_from_model(model)
    assert proposal.status == ProposalStatus.applying
    assert proposal.iteration_id == 4 and proposal.applied_definition_id == 9
    assert proposal.accepted_change_ids == ["n:1"]
    assert list(proposal.hypotheses) == ["n:1"]


# ─── agent: modes, brief_change event, run_self_improve_turn ─────────────


class FakeDB:
    def __init__(self, kind="self_improve"):
        self.session = SimpleNamespace(
            session_uuid="s-1",
            organization_id=1,
            workflow_id=5,
            kind=kind,
            title=None,
            transcript=[
                {"role": "user", "content": "OLD ROUND"},
                {"role": "assistant", "content": "old summary"},
            ],
            plan=None,
        )
        self.proposal = None
        self.brief = SimpleNamespace(
            brief={"goal": "Book demos", "must_never": ["Quote prices"]},
            version=2,
            updated_at=None,
            updated_by="user",
        )

    async def get_copilot_session(self, session_uuid, organization_id):
        return self.session if (session_uuid, organization_id) == ("s-1", 1) else None

    async def update_copilot_session(self, session_uuid, organization_id, **fields):
        for key, value in fields.items():
            setattr(self.session, key, value)
        return self.session

    async def get_workflow(self, workflow_id, organization_id=None):
        return SimpleNamespace(id=workflow_id, name="Lead Qualifier")

    async def get_workflow_brief(self, workflow_id, organization_id):
        return self.brief

    async def get_proposal_for_iteration(self, iteration_id, organization_id):
        return self.proposal


class FakeProvider:
    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []

    async def stream(self, *, system, messages, tools, replay):
        self.calls.append(
            {"system": system, "messages": list(messages), "tools": list(tools)}
        )
        turn = self.turns.pop(0)
        if turn.text:
            yield turn.text
        yield turn


class FakeTools:
    def __init__(self, outcomes=None):
        self.outcomes = outcomes or {}
        self.contexts = []

    def get_tool_specs(self, ctx):
        return [ToolSpec(name=n, description="d", input_schema={}) for n in ("a",)]

    async def execute_tool(self, name, tool_input, ctx):
        self.contexts.append((name, ctx.model_copy()))
        return self.outcomes.get(name, ToolOutcome(content={"ok": 1}, summary="Ok"))


@pytest.fixture
def agent_env(monkeypatch):
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


def _call(id_, name, **input_):
    return ToolCall(id=id_, name=name, input=input_)


def _proposal(uuid="p-9"):
    return Proposal(
        proposal_uuid=uuid,
        session_uuid="s-1",
        kind=ProposalKind.edit,
        workflow_id=5,
        summary="Move pricing",
        iteration_id=5,
    )


async def _round(text="Round 3: report r-1"):
    return await agent.run_self_improve_turn(
        session_uuid="s-1",
        organization_id=1,
        user=SimpleNamespace(id=7),
        iteration_id=5,
        report_uuid="r-1",
        text=text,
    )


async def test_run_self_improve_turn_returns_proposal_and_summary(agent_env):
    db, fake_tools = agent_env
    fake_tools.outcomes["propose_changes"] = ToolOutcome(
        content={"proposal_uuid": "p-9"}, summary="Proposed", proposal=_proposal()
    )
    provider = FakeProvider(
        [
            llm.ModelTurn(
                text="Reading the report.",
                tool_calls=[_call("c1", "get_doctor_report")],
                stop_reason=llm.STOP_TOOL_USE,
            ),
            llm.ModelTurn(
                tool_calls=[_call("c2", "propose_changes", code="x", summary="s")],
                stop_reason=llm.STOP_TOOL_USE,
            ),
            llm.ModelTurn(text="Moved pricing into Welcome (C1)."),
        ]
    )
    llm.set_provider_for_tests(provider)

    proposal_uuid, summary = await _round()

    assert proposal_uuid == "p-9"
    assert summary == "Moved pricing into Welcome (C1)."
    for _, ctx in fake_tools.contexts:
        assert ctx.mode == "self_improve"
        assert ctx.iteration_id == 5 and ctx.report_uuid == "r-1"
    session_context = provider.calls[0]["system"][1]
    assert "Self-improve round" in session_context
    assert "No human is present" in session_context
    assert "Quote prices" in session_context  # the brief
    # Only this round's messages reach the model; history stays stored.
    first = provider.calls[0]["messages"]
    assert [m.content for m in first] == ["Round 3: report r-1"]
    assert db.session.transcript[0]["content"] == "OLD ROUND"
    assert db.session.title == "Self-improve"


async def test_run_self_improve_turn_without_proposal(agent_env):
    llm.set_provider_for_tests(
        FakeProvider([llm.ModelTurn(text="Nothing actionable in 10 calls.")])
    )
    assert await _round() == (None, "Nothing actionable in 10 calls.")


async def test_run_self_improve_turn_reuses_an_existing_proposal(agent_env):
    db, _ = agent_env
    db.proposal = SimpleNamespace(proposal_uuid="p-old", summary="Earlier fix")
    provider = FakeProvider([])
    llm.set_provider_for_tests(provider)
    assert await _round() == ("p-old", "Earlier fix")
    assert provider.calls == []


async def test_run_self_improve_turn_error_without_proposal_raises(agent_env):
    llm.set_provider_for_tests(
        FakeProvider([llm.ModelTurn(text="cut", stop_reason=llm.STOP_MAX_TOKENS)])
    )
    with pytest.raises(agent.SelfImproveTurnError):
        await _round()


async def test_mode_and_session_kind_must_match(agent_env):
    db, _ = agent_env
    llm.set_provider_for_tests(FakeProvider([]))
    chat_on_round = [
        e
        async for e in agent.run_turn(
            session_uuid="s-1", organization_id=1, user=SimpleNamespace(id=7), text="hi"
        )
    ]
    assert [e.type.value for e in chat_on_round] == ["error"]
    assert "Self-improve" in chat_on_round[0].data["message"]

    db.session.kind = "chat"
    with pytest.raises(agent.SelfImproveTurnError):
        await _round()
    db.session.kind = "self_improve"
    missing_report = [
        e
        async for e in agent.run_turn(
            session_uuid="s-1",
            organization_id=1,
            user=SimpleNamespace(id=7),
            text="hi",
            mode="self_improve",
            iteration_id=5,
        )
    ]
    assert [e.type.value for e in missing_report] == ["error"]


async def test_chat_turn_emits_brief_change_and_sees_the_brief(agent_env):
    db, fake_tools = agent_env
    db.session.kind = "chat"
    change = BriefChange(
        change_uuid="bc-1",
        workflow_id=5,
        base_version=2,
        proposed=WorkflowBrief(goal="Book demos", must_never=["Quote prices", "Lie"]),
        reason="User: never lie",
    )
    fake_tools.outcomes["update_workflow_brief"] = ToolOutcome(
        content={"change_uuid": "bc-1"}, summary="Proposed", brief_change=change
    )
    provider = FakeProvider(
        [
            llm.ModelTurn(
                tool_calls=[_call("c1", "update_workflow_brief", reason="r")],
                stop_reason=llm.STOP_TOOL_USE,
            ),
            llm.ModelTurn(text="Proposed a brief update."),
        ]
    )
    llm.set_provider_for_tests(provider)
    events = [
        e
        async for e in agent.run_turn(
            session_uuid="s-1",
            organization_id=1,
            user=SimpleNamespace(id=7),
            text="Never lie to callers",
        )
    ]
    kinds = [e.type.value for e in events]
    assert kinds == [
        "tool_call",
        "tool_result",
        "brief_change",
        "text_delta",
        "message_done",
    ]
    assert events[2].data["brief_change"]["change_uuid"] == "bc-1"
    context = provider.calls[0]["system"][1]
    assert "update_workflow_brief" in context and "Quote prices" in context
    assert "Self-improve round" not in context
    # Chat sees the whole history.
    assert provider.calls[0]["messages"][0].content == "OLD ROUND"
    assert fake_tools.contexts[0][1].mode == "chat"
