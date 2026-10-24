"""`api.services.copilot.proposals`: propose (validate + diff) and apply.

DB access is mocked like `test_sdk_authoring.py`; the real Node TS
validator runs end-to-end so proposals go through the exact save pipeline.
"""

from __future__ import annotations

import copy
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.db import db_client
from api.db.agent_trigger_client import TriggerPathConflictError
from api.mcp_server.tools import create_workflow as create_workflow_module
from api.mcp_server.ts_bridge import generate_code, parse_code
from api.services.configuration.masking import mask_workflow_definition
from api.services.copilot import proposals
from api.services.copilot.contracts import (
    ApplyProposalRequest,
    ChangeOp,
    ProposalKind,
    ProposalStatus,
    ToolContext,
)
from api.services.copilot.diff import compute_changes
from api.services.workflow import sdk_authoring
from api.services.workflow.layout import reconcile_positions
from api.services.workflow.sdk_authoring import (
    compute_definition_hash,
    save_workflow_draft_from_code,
)

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node binary not available"
)

ORG = 1
WORKFLOW_ID = 7
FIXTURES = Path(__file__).parent / "fixtures" / "sdk_authoring"
REAL_TUNER_KEY = "tuner-live-key-000000000000000000000000t3st"
REAL_NOVEUM_KEY = "noveum-live-key-000000000000000n0vm"

BASE_CODE = """import { Workflow } from "@dograh/sdk";
import { agentNode, endCall, startCall } from "@dograh/sdk/typed";

const wf = new Workflow({ name: "Lead flow" });

const greeting = wf.addTyped(startCall({ name: "Greeting", prompt: "Hi!" }));
const qualify = wf.addTyped(agentNode({ name: "Qualify", prompt: "Ask budget." }));
const goodbye = wf.addTyped(endCall({ name: "Goodbye", prompt: "Bye." }));

wf.edge(greeting, qualify, { label: "qualify", condition: "caller is interested" });
wf.edge(qualify, goodbye, { label: "done", condition: "lead is qualified" });
"""

# Qualify prompt edited; "Book demo" inserted between Qualify and Goodbye.
PROPOSED_CODE = """import { Workflow } from "@dograh/sdk";
import { agentNode, endCall, startCall } from "@dograh/sdk/typed";

const wf = new Workflow({ name: "Lead flow" });

const greeting = wf.addTyped(startCall({ name: "Greeting", prompt: "Hi!" }));
const qualify = wf.addTyped(agentNode({ name: "Qualify", prompt: "Ask budget and timeline." }));
const book = wf.addTyped(agentNode({ name: "Book demo", prompt: "Offer a demo slot." }));
const goodbye = wf.addTyped(endCall({ name: "Goodbye", prompt: "Bye." }));

wf.edge(greeting, qualify, { label: "qualify", condition: "caller is interested" });
wf.edge(qualify, book, { label: "book", condition: "lead is qualified" });
wf.edge(book, goodbye, { label: "done", condition: "demo booked or declined" });
"""


# ─── fakes ───────────────────────────────────────────────────────────────


@dataclass
class _FakeDraft:
    version_number: int = 3
    status: str = "draft"
    workflow_json: dict | None = None


@dataclass
class _FakeWorkflow:
    id: int = WORKFLOW_ID
    organization_id: int = ORG
    name: str = "Lead flow"
    status: str = "active"
    released_definition: Any = None
    workflow_definition: dict | None = None
    extra: dict = field(default_factory=dict)


def _ctx(workflow_id: int | None = WORKFLOW_ID) -> ToolContext:
    return ToolContext(
        session_uuid="sess-1", organization_id=ORG, user_id=1, workflow_id=workflow_id
    )


@pytest.fixture
def user() -> MagicMock:
    u = MagicMock()
    u.id = 1
    u.selected_organization_id = ORG
    u.provider_id = "prov-1"
    return u


class _Backend:
    """db_client patched for one workflow, its draft and copilot rows."""

    def __init__(self, stored: dict | None, *, name: str = "Lead flow"):
        self.workflow = _FakeWorkflow(name=name)
        self.draft = _FakeDraft(workflow_json=copy.deepcopy(stored)) if stored else None
        self.models: dict[str, SimpleNamespace] = {}
        self.save = AsyncMock(side_effect=self._save)
        self.update_workflow = AsyncMock(return_value=self.workflow)
        self.assert_paths = AsyncMock(return_value=None)
        self.create_proposal = AsyncMock(side_effect=self._create_proposal)
        self.update_proposal = AsyncMock(side_effect=self._update_proposal)
        self.get_workflow = AsyncMock(side_effect=self._get_workflow)
        self._patches = [
            patch.object(db_client, "get_workflow", self.get_workflow),
            patch.object(
                db_client, "get_draft_version", AsyncMock(side_effect=self._get_draft)
            ),
            patch.object(db_client, "save_workflow_draft", self.save),
            patch.object(db_client, "update_workflow", self.update_workflow),
            patch.object(
                db_client, "assert_trigger_paths_available", self.assert_paths
            ),
            patch.object(db_client, "get_tools_by_uuids", AsyncMock(return_value=[])),
            patch.object(
                db_client,
                "get_copilot_session",
                AsyncMock(return_value=SimpleNamespace(id=11, workflow_id=WORKFLOW_ID)),
            ),
            patch.object(db_client, "create_copilot_proposal", self.create_proposal),
            patch.object(
                db_client,
                "get_copilot_proposal",
                AsyncMock(side_effect=self._get_proposal),
            ),
            patch.object(db_client, "update_copilot_proposal", self.update_proposal),
            # M2c: a create apply rebinds the build session (None = not found,
            # so no background brief draft starts in these tests).
            patch.object(
                db_client, "update_copilot_session", AsyncMock(return_value=None)
            ),
            patch.object(
                db_client,
                "claim_copilot_proposal",
                AsyncMock(side_effect=self._claim_proposal),
            ),
        ]

    async def _get_workflow(self, workflow_id, organization_id=None):
        if workflow_id == self.workflow.id and organization_id == ORG:
            return self.workflow
        return None

    async def _get_draft(self, workflow_id):
        return self.draft if workflow_id == self.workflow.id else None

    async def _save(self, *, workflow_id, workflow_definition):
        self.draft = _FakeDraft(workflow_json=copy.deepcopy(workflow_definition))
        return self.draft

    async def _create_proposal(self, **kwargs):
        uuid = f"prop-{len(self.models) + 1}"
        model = SimpleNamespace(
            id=len(self.models) + 1,
            proposal_uuid=uuid,
            status="pending",
            created_at=None,
            applied_at=None,
            session=SimpleNamespace(session_uuid="sess-1"),
            **copy.deepcopy(kwargs),
        )
        self.models[uuid] = model
        return model

    async def _get_proposal(self, proposal_uuid, organization_id):
        model = self.models.get(proposal_uuid)
        if model is None or model.organization_id != organization_id:
            return None
        return model

    async def _update_proposal(self, proposal_uuid, organization_id, **fields):
        model = await self._get_proposal(proposal_uuid, organization_id)
        if model is None:
            return None
        for key, value in fields.items():
            setattr(model, key, value)
        return model

    async def _claim_proposal(
        self, proposal_uuid, organization_id, *, from_status, to_status
    ):
        model = await self._get_proposal(proposal_uuid, organization_id)
        if model is None or model.status != from_status:
            return False
        model.status = to_status
        return True

    @property
    def saved_payload(self) -> dict:
        return self.save.call_args.kwargs["workflow_definition"]

    def __enter__(self):
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()


async def _parsed(code: str) -> dict:
    result = await parse_code(code)
    assert result.get("ok"), result
    return result["workflow"]


async def _stored(code: str = BASE_CODE) -> dict:
    """A working copy as the canvas would store it: UI ids and positions."""
    workflow = reconcile_positions(await _parsed(code), None)
    mapping = {n["id"]: f"node-{n['id']}" for n in workflow["nodes"]}
    for i, node in enumerate(workflow["nodes"]):
        node["id"] = mapping[node["id"]]
        node["position"] = {"x": 100.0 * i, "y": 50.0 * i}
        node["measured"] = {"width": 200, "height": 80}
    for edge in workflow["edges"]:
        edge["source"] = mapping[edge["source"]]
        edge["target"] = mapping[edge["target"]]
        edge["id"] = f"xy-edge__{edge['source']}-{edge['target']}"
    return workflow


def _node_id(workflow: dict, name: str) -> str:
    return next(n["id"] for n in workflow["nodes"] if n["data"].get("name") == name)


def _change_ids(outcome) -> dict[str, Any]:
    return {c.change_id: c for c in outcome.proposal.changes}


# ─── create_proposal: edit ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_edit_proposal_diffs_against_working_copy():
    stored = await _stored()
    with _Backend(stored) as b:
        outcome = await proposals.create_proposal(
            code=PROPOSED_CODE, summary="Add demo booking", ctx=_ctx()
        )

    assert outcome.ok is True, outcome.content
    proposal = outcome.proposal
    assert proposal.kind == ProposalKind.edit
    assert proposal.session_uuid == "sess-1"
    assert proposal.workflow_id == WORKFLOW_ID
    assert proposal.base_hash == compute_definition_hash(stored)
    assert proposal.status == ProposalStatus.pending

    qualify = _node_id(stored, "Qualify")
    goodbye = _node_id(stored, "Goodbye")
    changes = _change_ids(outcome)
    assert changes[f"n:{qualify}"].op == ChangeOp.update
    assert [d.field for d in changes[f"n:{qualify}"].field_diffs] == ["prompt"]
    new_nodes = [
        c for c in changes.values() if c.op == ChangeOp.add and c.target == "node"
    ]
    (new_node,) = new_nodes
    assert new_node.label == 'Add agentNode node "Book demo"'
    added_edges = [
        c for c in changes.values() if c.op == ChangeOp.add and c.target == "edge"
    ]
    assert len(added_edges) == 2
    assert all(c.depends_on == [new_node.change_id] for c in added_edges)
    removed = [c for c in changes.values() if c.op == ChangeOp.remove]
    assert [c.ref_id for c in removed] == [f"xy-edge__{qualify}-{goodbye}"]
    assert len(changes) == 5

    kwargs = b.create_proposal.call_args.kwargs
    assert kwargs["session_id"] == 11
    assert kwargs["organization_id"] == ORG
    assert kwargs["kind"] == "edit"
    assert kwargs["code"] == PROPOSED_CODE
    assert kwargs["summary"] == "Add demo booking"
    candidate = kwargs["candidate_json"]
    # Candidate ids follow the working copy; the new node gets a fresh id.
    assert _node_id(candidate, "Qualify") == qualify
    assert _node_id(candidate, "Book demo") not in {n["id"] for n in stored["nodes"]}
    b.save.assert_not_awaited()
    b.update_workflow.assert_not_awaited()

    # The model sees ids and labels, no node data.
    assert "Ask budget and timeline" not in str(outcome.content)
    assert {c["change_id"] for c in outcome.content["changes"]} == set(changes)
    assert outcome.summary == "Proposed 5 changes"


@pytest.mark.asyncio
async def test_edit_proposal_with_no_effective_change_is_rejected():
    stored = await _stored()
    projected = await generate_code(
        mask_workflow_definition(stored), workflow_name="Lead flow"
    )
    with _Backend(stored) as b:
        outcome = await proposals.create_proposal(
            code=projected, summary="noop", ctx=_ctx()
        )
    assert outcome.ok is False
    assert outcome.content["error_code"] == "no_changes"
    b.create_proposal.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_proposal_keeps_real_secrets_and_masks_changes():
    code = (FIXTURES / "wf5.ts").read_text(encoding="utf-8")
    stored = await _parsed(code)
    for node in stored["nodes"]:
        if node["type"] == "tuner":
            node["data"]["tuner_api_key"] = REAL_TUNER_KEY
        if node["type"] == "noveum":
            node["data"]["noveum_api_key"] = REAL_NOVEUM_KEY
    projected = await generate_code(
        mask_workflow_definition(stored), workflow_name="Customer Support - inbound"
    )
    agent = next(n for n in stored["nodes"] if n["type"] == "agentNode")
    unique_line = next(
        line
        for line in agent["data"]["prompt"].splitlines()
        if len(line) > 20 and projected.count(line) == 1
    )
    edited = projected.replace(unique_line, unique_line + " (edited)", 1)

    with _Backend(stored, name="Customer Support - inbound") as b:
        outcome = await proposals.create_proposal(
            code=edited, summary="tweak", ctx=_ctx()
        )

    assert outcome.ok is True, outcome.content
    (change,) = outcome.proposal.changes
    assert change.ref_id == agent["id"]
    candidate = b.create_proposal.call_args.kwargs["candidate_json"]
    keys = {
        k: v
        for n in candidate["nodes"]
        for k, v in n["data"].items()
        if k.endswith("api_key")
    }
    assert keys == {"tuner_api_key": REAL_TUNER_KEY, "noveum_api_key": REAL_NOVEUM_KEY}
    dumped = outcome.proposal.model_dump_json() + str(outcome.content)
    assert REAL_TUNER_KEY not in dumped and REAL_NOVEUM_KEY not in dumped


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "label, mutate",
    [
        ("parse_error", lambda c: c + "\nfunction evil() { return 1; }\n"),
        ("validation_error", lambda c: c.replace("prompt:", "promt:", 1)),
        (
            "graph_validation",
            lambda c: re.sub(r"wf\.edge\([^;]*\);\s*", "", c, flags=re.S),
        ),
    ],
)
async def test_validation_failures_match_the_mcp_save_errors(user, label, mutate):
    code = mutate(PROPOSED_CODE)
    stored = await _stored()
    with _Backend(stored) as b:
        outcome = await proposals.create_proposal(code=code, summary="x", ctx=_ctx())
        saved = await save_workflow_draft_from_code(
            user=user, organization_id=ORG, workflow_id=WORKFLOW_ID, code=code
        )
    assert outcome.ok is False
    assert saved["saved"] is False
    assert outcome.content["error_code"] == saved["error_code"] == label
    assert outcome.content["error"] == saved["error"]
    assert "propose_changes again" in outcome.content["hint"]
    b.create_proposal.assert_not_awaited()


@pytest.mark.asyncio
async def test_parse_failure_with_mocked_bridge():
    parse = AsyncMock(
        return_value={
            "ok": False,
            "stage": "validate",
            "errors": [{"message": "Unknown field 'promt'", "line": 4, "column": 9}],
        }
    )
    with _Backend(await _stored()), patch("api.mcp_server.ts_bridge.parse_code", parse):
        outcome = await proposals.create_proposal(code="x", summary="x", ctx=_ctx())
    assert outcome.ok is False
    assert outcome.content["error_code"] == "validation_error"
    assert outcome.content["error"] == "Unknown field 'promt' (line 4, col 9)"
    assert outcome.content["errors"][0]["line"] == 4


@pytest.mark.asyncio
async def test_proposal_for_unknown_workflow_or_session_is_rejected():
    with _Backend(await _stored()):
        outcome = await proposals.create_proposal(
            code=PROPOSED_CODE, summary="x", ctx=_ctx(workflow_id=404)
        )
        assert outcome.content["error_code"] == "not_found"
        with patch.object(
            db_client, "get_copilot_session", AsyncMock(return_value=None)
        ):
            outcome = await proposals.create_proposal(
                code=PROPOSED_CODE, summary="x", ctx=_ctx()
            )
        assert outcome.content["error_code"] == "not_found"


# ─── create_proposal: create ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_proposal_is_all_adds():
    with _Backend(None) as b:
        outcome = await proposals.create_proposal(
            code=BASE_CODE, summary="New lead flow", ctx=_ctx(None)
        )
    assert outcome.ok is True, outcome.content
    proposal = outcome.proposal
    assert proposal.kind == ProposalKind.create
    assert proposal.workflow_id is None
    assert proposal.base_hash is None
    assert proposal.workflow_name == "Lead flow"
    assert len(proposal.changes) == 5
    assert all(c.op == ChangeOp.add for c in proposal.changes)
    kwargs = b.create_proposal.call_args.kwargs
    assert kwargs["workflow_id"] is None and kwargs["kind"] == "create"
    b.get_workflow.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_proposal_requires_a_name():
    parsed = await _parsed(BASE_CODE)
    parse = AsyncMock(
        return_value={"ok": True, "workflowName": "  ", "workflow": parsed}
    )
    with _Backend(None) as b, patch("api.mcp_server.ts_bridge.parse_code", parse):
        outcome = await proposals.create_proposal(code="x", summary="x", ctx=_ctx(None))
    assert outcome.ok is False
    assert outcome.content["error_code"] == "missing_name"
    b.create_proposal.assert_not_awaited()


# ─── apply: edit ─────────────────────────────────────────────────────────


async def _propose(
    b: _Backend, code: str = PROPOSED_CODE, ctx: ToolContext | None = None
):
    outcome = await proposals.create_proposal(code=code, summary="s", ctx=ctx or _ctx())
    assert outcome.ok is True, outcome.content
    return outcome.proposal


async def _apply(user, proposal, accepted, **kwargs):
    return await proposals.apply_proposal(
        proposal_uuid=proposal.proposal_uuid,
        organization_id=ORG,
        user=user,
        request=ApplyProposalRequest(accepted_change_ids=accepted, **kwargs),
    )


@pytest.mark.asyncio
async def test_apply_subset_merges_only_accepted_changes(user):
    stored = await _stored()
    qualify = _node_id(stored, "Qualify")
    with _Backend(stored) as b:
        proposal = await _propose(b)
        response = await _apply(user, proposal, [f"n:{qualify}"])

    assert response.saved is True, response
    saved = b.saved_payload
    assert [n["id"] for n in saved["nodes"]] == [n["id"] for n in stored["nodes"]]
    assert [e["id"] for e in saved["edges"]] == [e["id"] for e in stored["edges"]]
    by_id = {n["id"]: n for n in saved["nodes"]}
    assert by_id[qualify]["data"]["prompt"] == "Ask budget and timeline."
    for node in stored["nodes"]:
        assert by_id[node["id"]]["position"] == node["position"]
        if node["id"] != qualify:
            assert by_id[node["id"]]["data"] == node["data"]
    assert response.workflow_id == WORKFLOW_ID
    assert response.draft_hash == compute_definition_hash(saved)
    assert response.proposal.status == ProposalStatus.applied
    fields = b.update_proposal.call_args.kwargs
    assert fields["status"] == "applied" and fields["applied_at"] is not None
    b.update_workflow.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_all_changes_equals_the_candidate(user):
    stored = await _stored()
    with _Backend(stored) as b:
        proposal = await _propose(b)
        candidate = b.create_proposal.call_args.kwargs["candidate_json"]
        response = await _apply(user, proposal, [c.change_id for c in proposal.changes])

    assert response.saved is True, response
    saved = b.saved_payload
    assert compute_changes(saved, candidate) == []
    new_id = _node_id(candidate, "Book demo")
    new_node = next(n for n in saved["nodes"] if n["id"] == new_id)
    qualify_pos = next(n for n in saved["nodes"] if n["data"]["name"] == "Qualify")[
        "position"
    ]
    # Placed next to its incoming neighbour, not at the origin.
    assert new_node["position"] == {
        "x": qualify_pos["x"] + 400.0,
        "y": qualify_pos["y"] + 200.0,
    }


@pytest.mark.asyncio
async def test_apply_stale_proposal_is_refused_and_marked(user):
    stored = await _stored()
    with _Backend(stored) as b:
        proposal = await _propose(b)
        b.draft.workflow_json["nodes"][0]["data"]["prompt"] = "Changed in the canvas"
        response = await _apply(user, proposal, [proposal.changes[0].change_id])
    assert response.saved is False
    assert response.error_code == "stale"
    assert response.proposal.status == ProposalStatus.stale
    assert b.update_proposal.call_args.kwargs == {"status": "stale"}
    b.save.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_requires_dependencies(user):
    stored = await _stored()
    with _Backend(stored) as b:
        proposal = await _propose(b)
        edge = next(
            c for c in proposal.changes if c.op == ChangeOp.add and c.target == "edge"
        )
        response = await _apply(user, proposal, [edge.change_id])
    assert response.saved is False
    assert response.error_code == "missing_dependency"
    assert edge.depends_on[0] in response.error
    b.save.assert_not_awaited()
    b.update_proposal.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_node_removal_requires_its_edge_removals(user):
    stored = await _stored(PROPOSED_CODE)
    with _Backend(stored) as b:
        proposal = await _propose(b, code=BASE_CODE)  # removes "Book demo"
        removal = next(
            c
            for c in proposal.changes
            if c.op == ChangeOp.remove and c.target == "node"
        )
        assert len(removal.depends_on) == 2
        response = await _apply(user, proposal, [removal.change_id])
    assert response.error_code == "missing_dependency"
    for dep in removal.depends_on:
        assert dep in response.error


@pytest.mark.asyncio
async def test_apply_partial_selection_that_breaks_the_graph(user):
    stored = await _stored()
    with _Backend(stored) as b:
        proposal = await _propose(b)
        removal = next(c for c in proposal.changes if c.op == ChangeOp.remove)
        # Removing Qualify -> Goodbye without the new path strands Goodbye.
        response = await _apply(user, proposal, [removal.change_id])
    assert response.saved is False
    assert response.error_code == "graph_validation"
    assert response.error
    b.save.assert_not_awaited()
    assert response.proposal.status == ProposalStatus.pending


@pytest.mark.asyncio
async def test_apply_honours_expected_draft_version(user):
    stored = await _stored()
    with _Backend(stored) as b:
        proposal = await _propose(b)
        accepted = [f"n:{_node_id(stored, 'Qualify')}"]
        conflict = await _apply(user, proposal, accepted, expected_draft_version=99)
        assert conflict.error_code == "draft_conflict"
        b.save.assert_not_awaited()
        ok = await _apply(
            user,
            proposal,
            accepted,
            expected_draft_version=compute_definition_hash(stored),
        )
    assert ok.saved is True


@pytest.mark.asyncio
async def test_double_apply_is_refused(user):
    stored = await _stored()
    with _Backend(stored) as b:
        proposal = await _propose(b)
        accepted = [f"n:{_node_id(stored, 'Qualify')}"]
        first = await _apply(user, proposal, accepted)
        second = await _apply(user, proposal, accepted)
    assert first.saved is True
    assert second.saved is False
    assert second.error_code == "already_applied"
    assert b.save.await_count == 1


@pytest.mark.asyncio
async def test_apply_rejects_empty_or_unknown_selection(user):
    stored = await _stored()
    with _Backend(stored) as b:
        proposal = await _propose(b)
        empty = await _apply(user, proposal, [])
        unknown = await _apply(user, proposal, ["n:does-not-exist"])
    assert empty.error_code == "validation_error"
    assert unknown.error_code == "validation_error"
    assert "n:does-not-exist" in unknown.error
    b.save.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_unknown_or_other_org_proposal_is_not_found(user):
    with _Backend(await _stored()) as b:
        proposal = await _propose(b)
        response = await proposals.apply_proposal(
            proposal_uuid=proposal.proposal_uuid,
            organization_id=ORG + 1,
            user=user,
            request=ApplyProposalRequest(accepted_change_ids=["x"]),
        )
    assert response.error_code == "not_found"


@pytest.mark.asyncio
async def test_apply_accepted_rename_renames_the_workflow(user):
    stored = await _stored()
    renamed = BASE_CODE.replace('name: "Lead flow"', 'name: "Lead flow v2"')
    with _Backend(stored) as b:
        proposal = await _propose(b, code=renamed)
        assert [c.change_id for c in proposal.changes] == ["w:name"]
        response = await _apply(user, proposal, ["w:name"])
    assert response.saved is True
    assert b.update_workflow.call_args.kwargs["name"] == "Lead flow v2"
    assert b.update_workflow.call_args.kwargs["organization_id"] == ORG


# ─── discard ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_discard_then_apply(user):
    with _Backend(await _stored()) as b:
        proposal = await _propose(b)
        discarded = await proposals.discard_proposal(
            proposal_uuid=proposal.proposal_uuid, organization_id=ORG
        )
        again = await proposals.discard_proposal(
            proposal_uuid=proposal.proposal_uuid, organization_id=ORG
        )
        response = await _apply(user, proposal, [proposal.changes[0].change_id])
        missing = await proposals.discard_proposal(
            proposal_uuid="nope", organization_id=ORG
        )
    assert discarded.status == ProposalStatus.discarded
    assert again.status == ProposalStatus.discarded
    # Discard is a conditional pending -> discarded claim (never overrides
    # an `applying` claim); the second discard is a no-op.
    assert b.models[proposal.proposal_uuid].status == "discarded"
    b.update_proposal.assert_not_awaited()
    assert response.error_code == "discarded"
    assert missing is None


# ─── apply: create ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_apply_create_creates_and_publishes_v1(user):
    with _Backend(None) as b:
        proposal = await _propose(b, code=BASE_CODE, ctx=_ctx(None))
        created = _FakeWorkflow(id=55, name="Lead flow")
        stored_v1: dict = {}

        async def fake_create(name, payload, user_id, organization_id):
            stored_v1.update(copy.deepcopy(payload))
            created.released_definition = SimpleNamespace(
                workflow_json=copy.deepcopy(payload), version_number=1
            )
            return created

        async def get_workflow(workflow_id, organization_id=None):
            return created if (workflow_id, organization_id) == (55, ORG) else None

        with (
            patch.object(
                db_client, "create_workflow", AsyncMock(side_effect=fake_create)
            ) as create,
            patch.object(
                db_client, "get_workflow", AsyncMock(side_effect=get_workflow)
            ),
            patch.object(db_client, "sync_triggers_for_workflow", AsyncMock()),
            patch.object(sdk_authoring, "capture_event") as capture,
        ):
            response = await _apply(user, proposal, [])

    assert response.saved is True, response
    assert response.workflow_id == 55
    assert response.draft_hash == compute_definition_hash(stored_v1)
    assert response.proposal.status == ProposalStatus.applied
    assert response.proposal.workflow_id == 55
    args = create.call_args.args
    assert args[0] == "Lead flow" and args[2] == user.id and args[3] == ORG
    assert capture.call_args.kwargs["properties"]["source"] == "copilot"
    fields = b.update_proposal.call_args.kwargs
    assert fields["workflow_id"] == 55 and fields["status"] == "applied"


@pytest.mark.asyncio
async def test_apply_create_failure_keeps_proposal_pending(user):
    with _Backend(None) as b:
        proposal = await _propose(b, code=BASE_CODE, ctx=_ctx(None))
        with (
            patch.object(db_client, "create_workflow", AsyncMock()) as create,
            patch.object(
                sdk_authoring,
                "extract_trigger_paths",
                return_value=["taken"],
            ),
        ):
            b.assert_paths.side_effect = TriggerPathConflictError(["taken"])
            response = await _apply(user, proposal, [])
    assert response.saved is False
    assert response.error_code == "trigger_path_conflict"
    create.assert_not_awaited()
    b.update_proposal.assert_not_awaited()


# ─── mapping + drift ─────────────────────────────────────────────────────


def test_proposal_from_model_reads_session_uuid_from_relationship():
    model = SimpleNamespace(
        proposal_uuid="p",
        session=SimpleNamespace(session_uuid="s-9"),
        kind="edit",
        workflow_id=3,
        summary="x",
        base_hash="h",
        changes=[],
        status="stale",
        workflow_name=None,
        created_at=None,
    )
    proposal = proposals.proposal_from_model(model)
    assert proposal.session_uuid == "s-9"
    assert proposal.status == ProposalStatus.stale
    assert (
        proposals.proposal_from_model(model, session_uuid="other").session_uuid
        == "other"
    )


def test_create_error_codes_are_documented_on_the_mcp_tool():
    source = Path(sdk_authoring.__file__).read_text(encoding="utf-8")
    start = source.index("async def create_workflow_from_code")
    body = source[start:]
    codes = set(re.findall(r'_create_error\(\s*"([a-z_]+)"', body))
    codes |= set(
        re.findall(
            r'_error_result\(\s*"([a-z_]+)"',
            source[
                source.index("async def _parse_stage") : source.index(
                    "async def validate_workflow_payload"
                )
            ],
        )
    )
    codes |= {"parse_error", "validation_error"}
    assert codes == set(create_workflow_module.CREATE_WORKFLOW_ERROR_CODES)
    doc = create_workflow_module.create_workflow.__doc__ or ""
    assert not sorted(c for c in codes if f"`{c}`" not in doc)


@pytest.mark.asyncio
async def test_concurrent_apply_saves_once(user):
    """Two applies at once: the atomic claim lets exactly one save."""
    import asyncio

    stored = await _stored()
    with _Backend(stored) as db:
        proposal = await _propose(db)
        ids = [c.change_id for c in proposal.changes]
        first, second = await asyncio.gather(
            _apply(user, proposal, ids), _apply(user, proposal, ids)
        )
        outcomes = sorted([first.saved, second.saved])
        assert outcomes == [False, True]
        loser = first if not first.saved else second
        assert loser.error_code == "already_applied"
        assert db.save.await_count == 1


@pytest.mark.asyncio
async def test_failed_apply_releases_the_claim(user):
    """A refused apply (missing dependency) leaves the proposal pending."""
    stored = await _stored()
    with _Backend(stored) as db:
        proposal = await _propose(db)
        ids = [c.change_id for c in proposal.changes]
        bad = await _apply(user, proposal, ["does-not-exist"])
        assert not bad.saved
        assert db.models[proposal.proposal_uuid].status == "pending"
        ok = await _apply(user, proposal, ids)
        assert ok.saved
