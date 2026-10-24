"""Copilot milestone 2c: settings changes, new tools, KB attach, and the
build -> agent context transfer.

The settings service (`api/services/copilot/settings.py`, a parallel lane)
is replaced by a small fake module with the agreed interface, so these
tests pin the proposals / tools integration only. The graph side runs the
real Node TS validator like `test_copilot_proposals.py`, whose `_Backend`
fake is reused.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

import api.services.copilot as copilot_pkg
from api.db import db_client
from api.routes import copilot as routes
from api.services import tool_management
from api.services.copilot import brief as brief_service
from api.services.copilot import proposals, tools
from api.services.copilot.contracts import (
    ApplyProposalRequest,
    BriefChangeStatus,
    ChangeItem,
    ChangeOp,
    ChangeTarget,
    Proposal,
    ProposalKind,
    ProposalStatus,
    ToolContext,
    WorkflowBrief,
)
from api.services.copilot.diff import align_ids, compute_changes
from api.services.workflow import sdk_authoring
from api.services.workflow.doctor import llm as doctor_llm
from api.services.workflow.dto import sanitize_workflow_definition
from api.services.workflow.sdk_authoring import compute_definition_hash
from api.tests.test_copilot_proposals import (
    BASE_CODE,
    ORG,
    PROPOSED_CODE,
    WORKFLOW_ID,
    _Backend,
    _FakeDraft,
    _FakeWorkflow,
    _node_id,
    _stored,
)

needs_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="node binary not available"
)

BOOK_TOOL = {
    "name": "book_demo",
    "description": "Book a demo slot for the caller.",
    "definition": {
        "type": "http_api",
        "config": {"method": "POST", "url": "https://api.example.com/book"},
    },
}

TOOL_CODE = BASE_CODE.replace(
    'agentNode({ name: "Qualify", prompt: "Ask budget." })',
    'agentNode({ name: "Qualify", prompt: "Ask budget.", '
    'tool_uuids: ["tool:book_demo"] })',
)
assert TOOL_CODE != BASE_CODE

KB_CODE = BASE_CODE.replace(
    'agentNode({ name: "Qualify", prompt: "Ask budget." })',
    'agentNode({ name: "Qualify", prompt: "Ask budget.", '
    'document_uuids: ["doc-pricing-1"] })',
)
assert KB_CODE != BASE_CODE

BASE_CONFIGS = {"max_call_duration": 300, "turn_stop_strategy": "transcription"}
BASE_VARS = {"company": "Dograh"}


# ─── fake settings service (the agreed settings.py interface) ────────────


class _SettingsValidationError(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class _SettingsPlan:
    changes: list
    candidate_configurations: dict
    candidate_template_variables: dict


def _hash(configurations, template_variables) -> str:
    raw = json.dumps([configurations or {}, template_variables or {}], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


META_KEY = "__copilot_settings_meta__"
REAL_SECRET = "sk-live-REALSECRET-0000000000000000"
OVERRIDE = "model_configuration_v2_override"


def _get(path: str, configs: dict, tvars: dict) -> Any:
    if path.startswith("models."):
        return (configs.get(OVERRIDE) or {}).get(path.split(".", 1)[1])
    if path.startswith("template_variables."):
        return tvars.get(path.split(".", 1)[1])
    return configs.get(path)


def _set(path: str, value: Any, configs: dict, tvars: dict) -> None:
    if path.startswith("models."):
        # Like the real service: the compiled override carries real keys.
        section = {**(value or {}), "api_key": REAL_SECRET}
        configs.setdefault(OVERRIDE, {})[path.split(".", 1)[1]] = section
        return
    if path.startswith("template_variables."):
        key = path.split(".", 1)[1]
        if value is None:
            tvars.pop(key, None)
        else:
            tvars[key] = value
    else:
        configs[path] = value


def _plan(*, workflow, organization_id, patch, working_configurations, **kwargs):
    configs = copy.deepcopy(working_configurations)
    tvars = copy.deepcopy(kwargs["working_template_variables"])
    errors = [f"{p} is not a setting" for p in patch.values if p.startswith("bad")]
    if errors:
        raise _SettingsValidationError(errors)
    changes = []
    for path, value in patch.values.items():
        before = _get(path, configs, tvars)
        if before == value:
            continue
        _set(path, value, configs, tvars)
        changes.append(
            ChangeItem(
                change_id=f"s:{path}",
                target=ChangeTarget.workflow,
                op=ChangeOp.update,
                ref_id=f"s:{path}",
                label=f"Setting: {path} {before} → {value}",
                before={"value": before},
                after={"value": value},
            )
        )
    if changes:
        configs[META_KEY] = {"stored_key": REAL_SECRET}
    return _SettingsPlan(changes, configs, tvars)


def _merge(
    *,
    current_configurations,
    current_template_variables,
    plan_changes,
    candidate_configurations,
    candidate_template_variables,
    accepted_change_ids,
):
    configs = copy.deepcopy(current_configurations)
    tvars = copy.deepcopy(current_template_variables)
    for change in plan_changes:
        if change.change_id not in accepted_change_ids:
            continue
        path = change.ref_id[2:]
        if path.startswith("models."):
            service = path.split(".", 1)[1]
            section = candidate_configurations[OVERRIDE][service]
            configs.setdefault(OVERRIDE, {})[service] = copy.deepcopy(section)
            continue
        _set(
            path,
            _get(path, candidate_configurations, candidate_template_variables),
            configs,
            tvars,
        )
    configs.pop(META_KEY, None)
    return configs, tvars


async def _load_working_settings(workflow):
    draft = await db_client.get_draft_version(workflow.id)
    source = draft if draft is not None else workflow
    return (
        copy.deepcopy(getattr(source, "workflow_configurations", None) or {}),
        copy.deepcopy(getattr(source, "template_context_variables", None) or {}),
    )


@pytest.fixture
def fake_settings(monkeypatch):
    module = types.ModuleType("api.services.copilot.settings")
    module.SettingsValidationError = _SettingsValidationError
    module.SettingsPlan = _SettingsPlan
    module.settings_hash = _hash
    module.plan_settings_changes = _plan
    module.merge_accepted_settings = _merge
    module.load_working_settings = _load_working_settings
    module.calls = []

    async def get_settings_view(*, workflow, organization_id):
        module.calls.append(("view", workflow.id, organization_id))
        return {"settings": {"max_call_duration": {"value": 300}}}

    def list_model_options(*, organization_id):
        module.calls.append(("models", organization_id))
        return {"tts": [{"provider": "elevenlabs"}]}

    async def list_voices(*, organization_id, provider, user):
        module.calls.append(("voices", organization_id, provider, user))
        if provider == "nope":
            raise _SettingsValidationError(["Unknown TTS provider nope"])
        return [{"id": "v1", "name": "Calm"}]

    module.get_settings_view = get_settings_view
    module.list_model_options = list_model_options
    module.list_voices = list_voices
    monkeypatch.setitem(sys.modules, "api.services.copilot.settings", module)
    monkeypatch.setattr(copilot_pkg, "settings", module, raising=False)
    return module


# ─── backend with settings + tools ───────────────────────────────────────


class _M2cBackend(_Backend):
    """`_Backend` plus draft settings, the settings-aware save and tools."""

    def __init__(
        self,
        stored: dict | None,
        *,
        configs: dict | None = None,
        tvars: dict | None = None,
        existing_tools: list | None = None,
    ):
        super().__init__(stored)
        if self.draft is not None:
            self.draft.workflow_configurations = copy.deepcopy(
                BASE_CONFIGS if configs is None else configs
            )
            self.draft.template_context_variables = copy.deepcopy(
                BASE_VARS if tvars is None else tvars
            )
        self.existing_tools = existing_tools or []
        self.created_tools: list[tuple[Any, Any, str]] = []
        self.archive = AsyncMock(return_value=True)
        self.create_tool = AsyncMock(side_effect=self._create_tool)
        self._patches += [
            patch.object(
                db_client,
                "get_tools_for_organization",
                AsyncMock(side_effect=lambda org, **kw: list(self.existing_tools)),
            ),
            patch.object(db_client, "archive_tool", self.archive),
            patch.object(tool_management, "create_tool_for_user", self.create_tool),
        ]

    async def _save(
        self,
        *,
        workflow_id,
        workflow_definition,
        workflow_configurations=None,
        template_context_variables=None,
    ):
        old = self.draft
        new = _FakeDraft(
            version_number=old.version_number,
            workflow_json=copy.deepcopy(
                workflow_definition
                if workflow_definition is not None
                else old.workflow_json
            ),
        )
        new.workflow_configurations = copy.deepcopy(
            workflow_configurations
            if workflow_configurations is not None
            else old.workflow_configurations
        )
        new.template_context_variables = copy.deepcopy(
            template_context_variables
            if template_context_variables is not None
            else old.template_context_variables
        )
        new.id = 501
        self.draft = new
        return new

    async def _create_tool(self, request, user, *, source):
        uuid = f"tool-uuid-{len(self.created_tools) + 1}"
        self.created_tools.append((request, user, source))
        return SimpleNamespace(tool_uuid=uuid, name=request.name)


def _ctx(workflow_id: int | None = WORKFLOW_ID, **kw) -> ToolContext:
    return ToolContext(
        session_uuid="sess-1",
        organization_id=ORG,
        user_id=1,
        workflow_id=workflow_id,
        **kw,
    )


@pytest.fixture
def user():
    return SimpleNamespace(id=1, selected_organization_id=ORG, provider_id="prov-1")


async def _apply(user, proposal, accepted, **kwargs):
    return await proposals.apply_proposal(
        proposal_uuid=proposal.proposal_uuid,
        organization_id=ORG,
        user=user,
        request=ApplyProposalRequest(accepted_change_ids=accepted, **kwargs),
    )


def _ids(proposal: Proposal) -> list[str]:
    return [c.change_id for c in proposal.changes]


# ─── propose: settings ───────────────────────────────────────────────────


@needs_node
async def test_graph_only_proposal_is_unchanged(fake_settings):
    """Regression: code only -> the pre-M2c ids, labels, order and hash, and
    no M2c columns passed to the DB client."""
    stored = await _stored()
    with _M2cBackend(stored) as b:
        outcome = await proposals.create_proposal(
            code=PROPOSED_CODE, summary="Add demo booking", ctx=_ctx()
        )
    assert outcome.ok is True, outcome.content
    kwargs = b.create_proposal.call_args.kwargs
    for key in ("settings_candidate", "settings_base_hash", "new_tools"):
        assert key not in kwargs
    validated = await sdk_authoring.parse_and_validate_workflow_code(
        code=PROPOSED_CODE, organization_id=ORG, previous_json=stored
    )
    candidate = align_ids(validated.payload, stored)
    expected = compute_changes(
        sanitize_workflow_definition(stored),
        candidate,
        before_name="Lead flow",
        after_name=validated.name,
    )
    assert outcome.proposal.changes == expected
    assert outcome.proposal.base_hash == compute_definition_hash(stored)
    assert outcome.proposal.settings_base_hash is None
    assert kwargs["code"] == PROPOSED_CODE
    assert fake_settings.calls == []


@needs_node
async def test_settings_only_proposal(fake_settings):
    stored = await _stored()
    with _M2cBackend(stored) as b:
        outcome = await proposals.create_proposal(
            code=None,
            summary="Longer calls",
            ctx=_ctx(),
            settings={"max_call_duration": 600, "template_variables.city": "Pune"},
        )
    assert outcome.ok is True, outcome.content
    proposal = outcome.proposal
    assert _ids(proposal) == ["s:max_call_duration", "s:template_variables.city"]
    assert all(c.target == ChangeTarget.workflow for c in proposal.changes)
    assert proposal.changes[0].label == "Setting: max_call_duration 300 → 600"
    assert proposal.base_hash is None
    assert proposal.settings_base_hash == _hash(BASE_CONFIGS, BASE_VARS)
    kwargs = b.create_proposal.call_args.kwargs
    assert kwargs["code"] == "" and kwargs["candidate_json"] == {}
    assert kwargs["settings_candidate"] == {
        "configurations": {
            **BASE_CONFIGS,
            "max_call_duration": 600,
            META_KEY: {"stored_key": REAL_SECRET},
        },
        "template_variables": {**BASE_VARS, "city": "Pune"},
    }
    assert "new_tools" not in kwargs
    assert outcome.summary == "Proposed 2 changes"
    b.save.assert_not_awaited()


@needs_node
async def test_settings_validation_error_goes_back_to_the_model(fake_settings):
    stored = await _stored()
    with _M2cBackend(stored) as b:
        outcome = await proposals.create_proposal(
            code=PROPOSED_CODE,
            summary="s",
            ctx=_ctx(),
            settings={"bad_path": 1, "max_call_duration": 600},
        )
    assert outcome.ok is False
    assert outcome.content["error_code"] == "validation_error"
    assert outcome.content["errors"] == ["bad_path is not a setting"]
    assert "get_workflow_settings" in outcome.content["hint"]
    b.create_proposal.assert_not_awaited()


@needs_node
async def test_settings_matching_current_values_are_no_changes(fake_settings):
    stored = await _stored()
    with _M2cBackend(stored) as b:
        outcome = await proposals.create_proposal(
            code=None, summary="s", ctx=_ctx(), settings={"max_call_duration": 300}
        )
    assert outcome.ok is False
    assert outcome.content["error_code"] == "no_changes"
    b.create_proposal.assert_not_awaited()


@needs_node
async def test_mixed_proposal_orders_nodes_edges_settings_tools(fake_settings):
    stored = await _stored()
    code = PROPOSED_CODE.replace(
        'prompt: "Offer a demo slot." }',
        'prompt: "Offer a demo slot.", tool_uuids: ["tool:book_demo"] }',
    )
    with _M2cBackend(stored) as b:
        outcome = await proposals.create_proposal(
            code=code,
            summary="Demo booking",
            ctx=_ctx(),
            settings={"max_call_duration": 600},
            new_tools=[BOOK_TOOL],
        )
    assert outcome.ok is True, outcome.content
    targets = [c.target.value for c in outcome.proposal.changes]
    assert targets == sorted(targets, key=["node", "edge", "workflow", "tool"].index), (
        targets
    )
    assert targets.count("workflow") == 1 and targets[-1] == "tool"
    book = next(c for c in outcome.proposal.changes if c.label.endswith('"Book demo"'))
    assert book.depends_on == ["t:book_demo"]
    tool_change = outcome.proposal.changes[-1]
    assert tool_change.change_id == tool_change.ref_id == "t:book_demo"
    assert tool_change.op == ChangeOp.add
    assert tool_change.label == (
        "Create tool: book_demo (HTTP POST https://api.example.com/book)"
    )
    kwargs = b.create_proposal.call_args.kwargs
    assert kwargs["base_hash"] == compute_definition_hash(stored)
    assert [t["name"] for t in kwargs["new_tools"]] == ["book_demo"]
    assert kwargs["settings_base_hash"] == _hash(BASE_CONFIGS, BASE_VARS)


# ─── propose: new tools ──────────────────────────────────────────────────


@needs_node
async def test_unknown_tool_placeholder_is_rejected(fake_settings):
    stored = await _stored()
    with _M2cBackend(stored) as b:
        outcome = await proposals.create_proposal(
            code=TOOL_CODE, summary="s", ctx=_ctx()
        )
    assert outcome.ok is False
    assert outcome.content["error_code"] == "validation_error"
    assert '"tool:book_demo" in node "Qualify"' in outcome.content["error"]
    b.create_proposal.assert_not_awaited()


@needs_node
async def test_unknown_placeholder_rejected_when_building_new(fake_settings):
    with _M2cBackend(None) as b:
        outcome = await proposals.create_proposal(
            code=TOOL_CODE, summary="s", ctx=_ctx(None)
        )
    assert outcome.ok is False
    assert outcome.content["error_code"] == "validation_error"
    b.create_proposal.assert_not_awaited()


@pytest.mark.parametrize(
    "tool, existing, needle",
    [
        (BOOK_TOOL, [SimpleNamespace(name="Book_Demo", status="active")], "exists"),
        (
            {
                **BOOK_TOOL,
                "definition": {
                    "type": "http_api",
                    "config": {
                        "method": "POST",
                        "url": "https://api.example.com/book",
                        "headers": {"Authorization": "Bearer sk-live"},
                    },
                },
            },
            [],
            "looks like a secret",
        ),
        ({"name": "x"}, [], "new_tools[0]"),
    ],
)
async def test_invalid_new_tools_are_rejected(fake_settings, tool, existing, needle):
    stored = {"nodes": [], "edges": []}
    with _M2cBackend(stored, existing_tools=existing) as b:
        outcome = await proposals.create_proposal(
            code=None, summary="s", ctx=_ctx(), new_tools=[tool]
        )
    assert outcome.ok is False
    assert outcome.content["error_code"] == "validation_error"
    assert needle in outcome.content["error"], outcome.content
    b.create_proposal.assert_not_awaited()


async def test_duplicate_new_tool_names_and_foreign_credentials_are_rejected(
    fake_settings,
):
    with _M2cBackend({"nodes": [], "edges": []}) as b:
        dup = await proposals.create_proposal(
            code=None,
            summary="s",
            ctx=_ctx(),
            new_tools=[BOOK_TOOL, {**BOOK_TOOL, "name": "BOOK_DEMO"}],
        )
        with patch.object(
            db_client, "get_credential_by_uuid", AsyncMock(return_value=None)
        ):
            foreign = await proposals.create_proposal(
                code=None,
                summary="s",
                ctx=_ctx(),
                new_tools=[
                    {
                        **BOOK_TOOL,
                        "definition": {
                            "type": "http_api",
                            "config": {
                                "method": "POST",
                                "url": "https://api.example.com/book",
                                "credential_uuid": "cred-other-org",
                            },
                        },
                    }
                ],
            )
    assert dup.ok is False and "Duplicate" in dup.content["error"]
    assert foreign.ok is False and "cred-other-org" in foreign.content["error"]
    b.create_proposal.assert_not_awaited()


async def test_new_tools_refused_in_self_improve(fake_settings):
    ctx = _ctx(mode="self_improve", iteration_id=5)
    with _M2cBackend({"nodes": [], "edges": []}) as b:
        outcome = await proposals.create_proposal(
            code=None, summary="s", ctx=ctx, new_tools=[BOOK_TOOL]
        )
        with patch.object(
            db_client, "get_proposal_for_iteration", AsyncMock(return_value=None)
        ):
            via_tool = await tools.execute_tool(
                "propose_changes",
                {"summary": "s", "new_tools": [BOOK_TOOL]},
                ctx,
            )
    assert outcome.ok is False and outcome.content["error_code"] == "not_allowed"
    assert via_tool.ok is False and via_tool.content["error_code"] == "not_allowed"
    b.create_proposal.assert_not_awaited()


async def test_settings_and_tools_need_an_existing_workflow(fake_settings):
    with _M2cBackend(None) as b:
        settings = await proposals.create_proposal(
            code=BASE_CODE, summary="s", ctx=_ctx(None), settings={"a": 1}
        )
        new_tools = await proposals.create_proposal(
            code=BASE_CODE, summary="s", ctx=_ctx(None), new_tools=[BOOK_TOOL]
        )
    for outcome in (settings, new_tools):
        assert outcome.ok is False and outcome.content["error_code"] == "not_allowed"
    b.create_proposal.assert_not_awaited()


# ─── apply: settings ─────────────────────────────────────────────────────


async def _settings_proposal(b, settings):
    outcome = await proposals.create_proposal(
        code=None, summary="s", ctx=_ctx(), settings=settings
    )
    assert outcome.ok is True, outcome.content
    return outcome.proposal


@needs_node
async def test_settings_only_apply_leaves_workflow_json_untouched(fake_settings, user):
    stored = await _stored()
    with _M2cBackend(stored) as b:
        proposal = await _settings_proposal(
            b, {"max_call_duration": 600, "template_variables.city": "Pune"}
        )
        response = await _apply(user, proposal, _ids(proposal))

    assert response.saved is True, response
    kwargs = b.save.call_args.kwargs
    assert kwargs["workflow_definition"] is None
    assert kwargs["workflow_configurations"] == {
        **BASE_CONFIGS,
        "max_call_duration": 600,
    }
    assert kwargs["template_context_variables"] == {**BASE_VARS, "city": "Pune"}
    assert b.draft.workflow_json == stored
    assert response.draft_hash == compute_definition_hash(stored)
    fields = b.update_proposal.call_args.kwargs
    assert fields["accepted_change_ids"] == _ids(proposal)
    assert fields["applied_definition_id"] == 501
    b.update_workflow.assert_not_awaited()


@needs_node
async def test_apply_merges_only_accepted_settings_over_a_concurrent_graph_edit(
    fake_settings, user
):
    stored = await _stored()
    with _M2cBackend(stored) as b:
        proposal = await _settings_proposal(
            b, {"max_call_duration": 600, "turn_stop_strategy": "turn_analyzer"}
        )
        # The user edits the graph meanwhile: a settings-only proposal does
        # not touch the graph, so it is not stale and keeps the user's edit.
        edited = copy.deepcopy(stored)
        edited["nodes"][0]["data"]["prompt"] = "Hello there!"
        b.draft.workflow_json = edited
        response = await _apply(user, proposal, ["s:turn_stop_strategy"])

    assert response.saved is True, response
    assert b.draft.workflow_json == edited
    assert b.draft.workflow_configurations == {
        "max_call_duration": 300,
        "turn_stop_strategy": "turn_analyzer",
    }
    fields = b.update_proposal.call_args.kwargs
    assert fields["accepted_change_ids"] == ["s:turn_stop_strategy"]


@needs_node
async def test_stale_settings_are_refused(fake_settings, user):
    stored = await _stored()
    with _M2cBackend(stored) as b:
        proposal = await _settings_proposal(b, {"max_call_duration": 600})
        b.draft.workflow_configurations = {**BASE_CONFIGS, "max_call_duration": 900}
        response = await _apply(user, proposal, _ids(proposal))
    assert response.saved is False and response.error_code == "stale"
    assert b.models[proposal.proposal_uuid].status == "stale"
    b.save.assert_not_awaited()


@needs_node
async def test_mixed_apply_saves_graph_and_settings_once(fake_settings, user):
    stored = await _stored()
    qualify = _node_id(stored, "Qualify")
    with _M2cBackend(stored) as b:
        outcome = await proposals.create_proposal(
            code=PROPOSED_CODE,
            summary="s",
            ctx=_ctx(),
            settings={"max_call_duration": 600},
        )
        proposal = outcome.proposal
        response = await _apply(user, proposal, [f"n:{qualify}", "s:max_call_duration"])
    assert response.saved is True, response
    assert b.save.await_count == 1
    kwargs = b.save.call_args.kwargs
    by_id = {n["id"]: n for n in kwargs["workflow_definition"]["nodes"]}
    assert by_id[qualify]["data"]["prompt"] == "Ask budget and timeline."
    assert kwargs["workflow_configurations"]["max_call_duration"] == 600
    # Untouched template variables are not rewritten.
    assert kwargs["template_context_variables"] == BASE_VARS


# ─── apply: new tools ────────────────────────────────────────────────────


async def _tool_proposal(b, code=TOOL_CODE):
    outcome = await proposals.create_proposal(
        code=code, summary="s", ctx=_ctx(), new_tools=[BOOK_TOOL]
    )
    assert outcome.ok is True, outcome.content
    return outcome.proposal


@needs_node
async def test_new_tool_is_created_and_substituted(fake_settings, user):
    stored = await _stored()
    qualify = _node_id(stored, "Qualify")
    with _M2cBackend(stored) as b:
        proposal = await _tool_proposal(b)
        changes = {c.change_id: c for c in proposal.changes}
        assert changes[f"n:{qualify}"].depends_on == ["t:book_demo"]

        missing = await _apply(user, proposal, [f"n:{qualify}"])
        assert missing.error_code == "missing_dependency"
        assert b.create_tool.await_count == 0

        response = await _apply(user, proposal, _ids(proposal))

    assert response.saved is True, response
    ((request, acting, source),) = b.created_tools
    assert request.name == "book_demo" and acting is user and source == "copilot"
    saved = b.save.call_args.kwargs["workflow_definition"]
    node = next(n for n in saved["nodes"] if n["id"] == qualify)
    assert node["data"]["tool_uuids"] == ["tool-uuid-1"]
    assert "tool:" not in json.dumps(saved)
    assert response.draft_hash == compute_definition_hash(saved)
    fields = b.update_proposal.call_args.kwargs
    assert fields["accepted_change_ids"] == _ids(proposal)
    b.archive.assert_not_awaited()


async def test_tools_only_apply_creates_without_saving(fake_settings, user):
    stored = {"nodes": [], "edges": []}
    with _M2cBackend(stored) as b:
        proposal = await _tool_proposal(b, code=None)
        assert _ids(proposal) == ["t:book_demo"]
        response = await _apply(user, proposal, ["t:book_demo"])
    assert response.saved is True, response
    assert len(b.created_tools) == 1
    b.save.assert_not_awaited()
    assert b.models[proposal.proposal_uuid].status == "applied"


@needs_node
async def test_created_tools_are_archived_when_the_save_fails(fake_settings, user):
    stored = await _stored()
    with _M2cBackend(stored) as b:
        proposal = await _tool_proposal(b)
        conflict = await _apply(
            user, proposal, _ids(proposal), expected_draft_version="not-the-hash"
        )
        assert conflict.saved is False and conflict.error_code == "draft_conflict"
        b.archive.assert_awaited_once_with("tool-uuid-1", ORG)
        # The claim is released: the proposal can be applied again.
        assert b.models[proposal.proposal_uuid].status == "pending"

        b.save.side_effect = RuntimeError("db down")
        with pytest.raises(RuntimeError):
            await _apply(user, proposal, _ids(proposal))
        assert b.archive.await_args_list[-1].args == ("tool-uuid-2", ORG)
        assert b.models[proposal.proposal_uuid].status == "pending"


@needs_node
async def test_tool_create_failure_archives_earlier_tools(fake_settings, user):
    stored = await _stored()
    second = {**BOOK_TOOL, "name": "check_slot"}
    with _M2cBackend(stored) as b:
        outcome = await proposals.create_proposal(
            code=None, summary="s", ctx=_ctx(), new_tools=[BOOK_TOOL, second]
        )
        proposal = outcome.proposal
        calls = {"n": 0}

        async def flaky(request, acting, *, source):
            calls["n"] += 1
            if calls["n"] == 2:
                raise tool_management.ToolManagementError(
                    "credential_not_found", "Credential gone"
                )
            return SimpleNamespace(tool_uuid="tool-ok")

        b.create_tool.side_effect = flaky
        response = await _apply(user, proposal, _ids(proposal))
    assert response.saved is False and response.error_code == "credential_not_found"
    b.archive.assert_awaited_once_with("tool-ok", ORG)
    assert b.models[proposal.proposal_uuid].status == "pending"
    b.save.assert_not_awaited()


# ─── KB attach (existing documents by uuid) ──────────────────────────────


@needs_node
async def test_kb_attach_via_document_uuids_applies(fake_settings, user):
    stored = await _stored()
    qualify = _node_id(stored, "Qualify")
    with _M2cBackend(stored) as b:
        outcome = await proposals.create_proposal(
            code=KB_CODE, summary="Attach pricing doc", ctx=_ctx()
        )
        assert outcome.ok is True, outcome.content
        (change,) = outcome.proposal.changes
        assert [d.field for d in change.field_diffs] == ["document_uuids"]
        response = await _apply(user, outcome.proposal, [change.change_id])
    assert response.saved is True, response
    saved = b.save.call_args.kwargs["workflow_definition"]
    node = next(n for n in saved["nodes"] if n["id"] == qualify)
    assert node["data"]["document_uuids"] == ["doc-pricing-1"]


# ─── tools.py: specs and settings tools ──────────────────────────────────


def _spec(ctx, name):
    return next((s for s in tools.get_tool_specs(ctx) if s.name == name), None)


def test_settings_tools_visibility_per_mode():
    build = {s.name for s in tools.get_tool_specs(_ctx(None))}
    assert "get_workflow_settings" not in build
    assert {"list_model_options", "list_voices"} <= build
    edit = {s.name for s in tools.get_tool_specs(_ctx())}
    assert {"get_workflow_settings", "list_model_options", "list_voices"} <= edit
    si = {s.name for s in tools.get_tool_specs(_ctx(mode="self_improve"))}
    assert {"get_workflow_settings", "list_model_options", "list_voices"} <= si
    for name in ("get_workflow_settings", "list_model_options", "list_voices"):
        assert _spec(_ctx(), name).read_only is True


def test_propose_changes_schema_per_mode():
    jsonschema = pytest.importorskip("jsonschema")
    build = _spec(_ctx(None), "propose_changes")
    assert build.input_schema == tools._PROPOSE_CHANGES_SPEC.input_schema
    chat = _spec(_ctx(), "propose_changes")
    assert chat.input_schema["required"] == ["summary"]
    assert {"code", "settings", "new_tools"} <= set(chat.input_schema["properties"])
    assert "models.tts" in chat.description and "tool:<name>" in chat.description
    si = _spec(_ctx(mode="self_improve"), "propose_changes")
    assert "settings" in si.input_schema["properties"]
    assert "new_tools" not in si.input_schema["properties"]
    for spec in (build, chat, si):
        jsonschema.Draft202012Validator.check_schema(spec.input_schema)
    validator = jsonschema.Draft202012Validator(chat.input_schema)
    validator.validate({"summary": "s", "new_tools": [BOOK_TOOL]})
    validator.validate({"summary": "s", "settings": {"max_call_duration": 600}})


async def test_get_workflow_settings_is_bound_to_the_session(fake_settings):
    workflow = SimpleNamespace(id=WORKFLOW_ID)
    with patch.object(db_client, "get_workflow", AsyncMock(return_value=workflow)):
        outcome = await tools.execute_tool("get_workflow_settings", {}, _ctx())
        hidden = await tools.execute_tool("get_workflow_settings", {}, _ctx(None))
    assert outcome.ok is True
    assert outcome.content == {"settings": {"max_call_duration": {"value": 300}}}
    assert fake_settings.calls[0] == ("view", WORKFLOW_ID, ORG)
    assert hidden.ok is False and hidden.content["error_code"] == "not_allowed"


async def test_model_and_voice_tools(fake_settings):
    acting = SimpleNamespace(id=1, selected_organization_id=ORG)
    with patch.object(db_client, "get_user_by_id", AsyncMock(return_value=acting)):
        models = await tools.execute_tool("list_model_options", {}, _ctx(None))
        voices = await tools.execute_tool(
            "list_voices", {"provider": "elevenlabs"}, _ctx()
        )
        bad = await tools.execute_tool("list_voices", {"provider": "nope"}, _ctx())
        missing = await tools.execute_tool("list_voices", {}, _ctx())
    assert models.ok is True and models.content["tts"][0]["provider"] == "elevenlabs"
    assert voices.ok is True and voices.summary == "Listed 1 elevenlabs voices"
    call = fake_settings.calls[1]
    assert call[:3] == ("voices", ORG, "elevenlabs")
    assert call[3].selected_organization_id == ORG
    assert bad.ok is False and bad.content["error_code"] == "validation_error"
    assert "Unknown TTS provider" in bad.content["error"]
    assert missing.ok is False


async def test_propose_changes_passes_settings_and_new_tools():
    sentinel = SimpleNamespace(ok=True)
    with patch.object(
        tools.proposals, "create_proposal", AsyncMock(return_value=sentinel)
    ) as create:
        await tools.execute_tool(
            "propose_changes",
            {
                "summary": "s",
                "settings": {"max_call_duration": 600},
                "new_tools": [BOOK_TOOL],
            },
            _ctx(),
        )
        nothing = await tools.execute_tool("propose_changes", {"summary": "s"}, _ctx())
        build = await tools.execute_tool(
            "propose_changes",
            {"code": "src", "summary": "s", "settings": {"a": 1}},
            _ctx(None),
        )
    create.assert_awaited_once_with(
        code=None,
        summary="s",
        ctx=_ctx(),
        targets=[],
        settings={"max_call_duration": 600},
        new_tools=[BOOK_TOOL],
    )
    assert nothing.ok is False and "at least one" in nothing.content["error"]
    assert build.ok is False and build.content["error_code"] == "not_allowed"


# ─── context transfer (build -> agent) ───────────────────────────────────


async def _apply_create(b, user):
    outcome = await proposals.create_proposal(
        code=BASE_CODE, summary="s", ctx=_ctx(None)
    )
    assert outcome.ok is True, outcome.content
    created = _FakeWorkflow(id=55, name="Lead flow")

    async def fake_create(name, payload, user_id, organization_id):
        created.released_definition = SimpleNamespace(
            workflow_json=copy.deepcopy(payload), version_number=1, id=77
        )
        return created

    async def get_workflow(workflow_id, organization_id=None):
        return created if (workflow_id, organization_id) == (55, ORG) else None

    with (
        patch.object(db_client, "create_workflow", AsyncMock(side_effect=fake_create)),
        patch.object(db_client, "get_workflow", AsyncMock(side_effect=get_workflow)),
        patch.object(db_client, "sync_triggers_for_workflow", AsyncMock()),
        patch.object(sdk_authoring, "capture_event"),
    ):
        response = await _apply(user, outcome.proposal, [])
    await proposals.wait_for_background_tasks()
    return response


@needs_node
async def test_create_apply_rebinds_the_session_and_drafts_the_brief(user):
    rebind = AsyncMock(return_value=SimpleNamespace(session_uuid="sess-1"))
    draft = AsyncMock(return_value=None)
    with (
        _M2cBackend(None) as b,
        patch.object(brief_service, "draft_brief_from_chats", draft),
    ):
        with patch.object(db_client, "update_copilot_session", rebind):
            response = await _apply_create(b, user)
    assert response.saved is True and response.workflow_id == 55
    rebind.assert_awaited_once_with("sess-1", ORG, workflow_id=55)
    draft.assert_awaited_once_with(
        workflow_id=55, organization_id=ORG, session_uuid="sess-1"
    )


@needs_node
@pytest.mark.parametrize("failing", ["rebind", "draft"])
async def test_context_transfer_failures_never_fail_the_create(user, failing):
    rebind = AsyncMock(return_value=SimpleNamespace(session_uuid="sess-1"))
    draft = AsyncMock(return_value=None)
    if failing == "rebind":
        rebind.side_effect = RuntimeError("db hiccup")
    else:
        draft.side_effect = RuntimeError("LLM down")
    with (
        _M2cBackend(None) as b,
        patch.object(brief_service, "draft_brief_from_chats", draft),
    ):
        with patch.object(db_client, "update_copilot_session", rebind):
            response = await _apply_create(b, user)
    assert response.saved is True and response.workflow_id == 55
    assert response.proposal.status == ProposalStatus.applied
    if failing == "rebind":
        draft.assert_not_awaited()


# ─── DB: migration columns, rebind listing, scoped brief draft ───────────


async def _org_with_workflow(db, slug):
    user, _ = await db.get_or_create_user_by_provider_id(f"{slug}_user")
    org, _ = await db.get_or_create_organization_by_provider_id(f"{slug}_org", user.id)
    await db.update_user_selected_organization(user.id, org.id)
    workflow = await db.create_workflow(
        name=f"{slug} workflow",
        workflow_definition={"nodes": [], "edges": []},
        user_id=user.id,
        organization_id=org.id,
    )
    return user, org, workflow


async def test_proposal_columns_round_trip(db_session):
    user, org, workflow = await _org_with_workflow(db_session, "m2c_cols")
    session = await db_session.create_copilot_session(
        organization_id=org.id, user_id=user.id, workflow_id=workflow.id
    )
    legacy = await db_session.create_copilot_proposal(
        session_id=session.id,
        organization_id=org.id,
        workflow_id=workflow.id,
        kind=ProposalKind.edit,
        summary="graph only",
        base_hash="h",
        candidate_json={},
        code="src",
        changes=[],
    )
    assert legacy.settings_candidate is None
    assert legacy.settings_base_hash is None and legacy.new_tools == []

    row = await db_session.create_copilot_proposal(
        session_id=session.id,
        organization_id=org.id,
        workflow_id=workflow.id,
        kind=ProposalKind.edit,
        summary="settings",
        base_hash=None,
        candidate_json={},
        code="",
        changes=[],
        settings_candidate={"configurations": {"a": 1}, "template_variables": {}},
        settings_base_hash="sh",
        new_tools=[BOOK_TOOL],
    )
    assert row.settings_candidate["configurations"] == {"a": 1}
    assert row.settings_base_hash == "sh" and row.new_tools[0]["name"] == "book_demo"
    assert proposals.proposal_from_model(row).settings_base_hash == "sh"
    updated = await db_session.update_copilot_proposal(
        row.proposal_uuid, org.id, settings_base_hash="sh2", new_tools=[]
    )
    assert updated.settings_base_hash == "sh2" and updated.new_tools == []


async def test_rebound_build_session_is_listed_and_drafts_from_it_only(
    db_session, monkeypatch
):
    user, org, workflow = await _org_with_workflow(db_session, "m2c_rebind")
    build = await db_session.create_copilot_session(
        organization_id=org.id, user_id=user.id, workflow_id=None
    )
    await db_session.update_copilot_session(
        build.session_uuid,
        org.id,
        title="Build",
        transcript=[
            {"role": "user", "content": "Build a demo booker. Never quote prices."},
            {"role": "assistant", "content": "Here is a plan."},
        ],
    )
    other = await db_session.create_copilot_session(
        organization_id=org.id, user_id=user.id, workflow_id=workflow.id
    )
    await db_session.update_copilot_session(
        other.session_uuid,
        org.id,
        transcript=[{"role": "user", "content": "OTHER CHAT TEXT"}],
    )
    seen = {}

    async def fake_structured(*, system, user_content, output_model, **kwargs):
        seen["user_content"] = user_content
        return WorkflowBrief(goal="Book demos", must_never=["Quote prices"]), {}

    monkeypatch.setattr(doctor_llm, "claude_structured", fake_structured)
    proposal = Proposal(
        proposal_uuid="p",
        session_uuid=build.session_uuid,
        kind=ProposalKind.create,
        summary="s",
    )
    await proposals._carry_build_context(proposal, org.id, workflow.id)
    await proposals.wait_for_background_tasks()

    async def allowed(organization_id):
        return None

    monkeypatch.setattr(routes, "_require_ai", allowed)
    listed = await routes.list_copilot_sessions(workflow.id, user)
    assert build.session_uuid in [s.session_uuid for s in listed]
    rebound = await db_session.get_copilot_session(build.session_uuid, org.id)
    assert rebound.workflow_id == workflow.id

    assert "Never quote prices." in seen["user_content"]
    assert "OTHER CHAT TEXT" not in seen["user_content"]
    current = await brief_service.get_brief(workflow.id, org.id)
    assert current.version == 0
    assert current.pending_change.status == BriefChangeStatus.pending
    assert current.pending_change.proposed.must_never == ["Quote prices"]

    # Default behaviour unchanged: without session_uuid every chat is used.
    await brief_service.draft_brief_from_chats(
        workflow_id=workflow.id, organization_id=org.id
    )
    assert "OTHER CHAT TEXT" in seen["user_content"]


async def test_scoped_draft_ignores_a_session_of_another_workflow(db_session):
    user, org, workflow = await _org_with_workflow(db_session, "m2c_scope")
    stray = await db_session.create_copilot_session(
        organization_id=org.id, user_id=user.id, workflow_id=None
    )
    await db_session.update_copilot_session(
        stray.session_uuid, org.id, transcript=[{"role": "user", "content": "hi"}]
    )
    with pytest.raises(brief_service.BriefDraftUnavailable) as info:
        await brief_service.draft_brief_from_chats(
            workflow_id=workflow.id,
            organization_id=org.id,
            session_uuid=stray.session_uuid,
        )
    assert info.value.reason == "no_chat_history"


# ─── secrets never leave the server (settings_candidate is server-side) ──


@needs_node
async def test_no_secret_reaches_the_model_ui_or_transcript(fake_settings, user):
    from api.services.copilot.agent import tool_result_text

    stored = await _stored()
    settings = {
        "models.tts": {"provider": "elevenlabs", "voice": "v1"},
        "max_call_duration": 600,
    }
    with _M2cBackend(stored) as b:
        outcome = await proposals.create_proposal(
            code=PROPOSED_CODE, summary="s", ctx=_ctx(), settings=settings
        )
        assert outcome.ok is True, outcome.content
        model = b.models[outcome.proposal.proposal_uuid]
        # Stored server-side (needed for apply) ...
        assert REAL_SECRET in json.dumps(model.settings_candidate)
        with patch.object(
            proposals, "_validate_model_override", AsyncMock(return_value=[])
        ):
            response = await _apply(user, outcome.proposal, _ids(outcome.proposal))
    assert response.saved is True, response
    # ... but never in what the model, the UI (SSE / GET) or the transcript see.
    exposed = [
        json.dumps(outcome.content, default=str),
        outcome.summary,
        tool_result_text(outcome),
        outcome.proposal.model_dump_json(),
        proposals.proposal_from_model(model).model_dump_json(),
        response.model_dump_json(),
    ]
    for text in exposed:
        assert REAL_SECRET not in text
        assert META_KEY not in text
    # The merged draft keeps the real key and drops the private meta key.
    assert b.draft.workflow_configurations[OVERRIDE]["tts"]["api_key"] == REAL_SECRET
    assert META_KEY not in b.draft.workflow_configurations


@needs_node
async def test_model_change_runs_the_live_key_check(fake_settings, user):
    stored = await _stored()
    check = AsyncMock(return_value=["Invalid ElevenLabs API key"])
    with _M2cBackend(stored) as b:
        outcome = await proposals.create_proposal(
            code=None,
            summary="s",
            ctx=_ctx(),
            settings={
                "models.tts": {"provider": "elevenlabs", "voice": "v1"},
                "max_call_duration": 600,
            },
        )
        proposal = outcome.proposal
        with patch.object(proposals, "_validate_model_override", check):
            only_duration = await _apply(user, proposal, ["s:max_call_duration"])
            assert only_duration.saved is True
            check.assert_not_awaited()

            outcome = await proposals.create_proposal(
                code=None,
                summary="s",
                ctx=_ctx(),
                settings={"models.tts": {"provider": "elevenlabs", "voice": "v2"}},
            )
            failed = await _apply(user, outcome.proposal, _ids(outcome.proposal))
    assert failed.saved is False and failed.error_code == "validation_error"
    assert "Invalid ElevenLabs API key" in failed.error
    merged, org, acting = check.await_args.args
    assert merged[OVERRIDE]["tts"]["voice"] == "v2" and org == ORG and acting is user
    assert b.models[outcome.proposal.proposal_uuid].status == "pending"


async def test_validate_model_override_uses_the_route_validator():
    from api.services.configuration import check_validity

    dograh = {
        "version": 2,
        "mode": "dograh",
        "dograh": {"api_key": "dgr-real-key", "voice": "default"},
    }
    user = SimpleNamespace(provider_id="prov-1")
    assert await proposals._validate_model_override({}, ORG, user) == []
    with patch.object(
        check_validity.UserConfigurationValidator,
        "validate",
        AsyncMock(side_effect=ValueError("bad key")),
    ) as validate:
        errors = await proposals._validate_model_override({OVERRIDE: dograh}, ORG, user)
    assert errors == ["bad key"]
    assert validate.await_args.kwargs == {
        "organization_id": ORG,
        "created_by": "prov-1",
    }
    masked = {**dograh, "dograh": {"api_key": "****-key", "voice": "default"}}
    with patch.object(
        check_validity.UserConfigurationValidator, "validate", AsyncMock()
    ):
        assert await proposals._validate_model_override({OVERRIDE: masked}, ORG, user)


async def test_voice_catalog_outage_is_a_plain_tool_error(fake_settings):
    from api.errors.mps import MPSUnavailableError

    async def down(**kwargs):
        raise MPSUnavailableError("get_voices", status_code=503)

    fake_settings.list_voices = down
    acting = SimpleNamespace(id=1, selected_organization_id=ORG)
    with patch.object(db_client, "get_user_by_id", AsyncMock(return_value=acting)):
        outcome = await tools.execute_tool(
            "list_voices", {"provider": "elevenlabs"}, _ctx()
        )
    assert outcome.ok is False and outcome.content["error_code"] == "unavailable"


# ─── against the real settings service (lane S) ──────────────────────────


@needs_node
async def test_real_settings_service_settings_only_round_trip(user):
    from api.services.copilot import settings as real

    stored = await _stored()
    configs = {"max_call_duration": 300}
    with _M2cBackend(stored, configs=configs, tvars={"company": "Dograh"}) as b:
        bad = await proposals.create_proposal(
            code=None, summary="s", ctx=_ctx(), settings={"no_such_setting": 1}
        )
        outcome = await proposals.create_proposal(
            code=None,
            summary="s",
            ctx=_ctx(),
            settings={
                "max_call_duration": 600,
                "turn_stop_strategy": "turn_analyzer",
                "template_variables.city": "Pune",
            },
        )
        assert outcome.ok is True, outcome.content
        proposal = outcome.proposal
        assert all(c.change_id.startswith("s:") for c in proposal.changes)
        assert proposal.settings_base_hash == real.settings_hash(
            configs, {"company": "Dograh"}
        )
        accepted = [
            c.change_id
            for c in proposal.changes
            if c.change_id != "s:turn_stop_strategy"
        ]
        response = await _apply(user, proposal, accepted)
    assert bad.ok is False and bad.content["error_code"] == "validation_error"
    assert "unknown setting" in bad.content["error"]
    assert response.saved is True, response
    assert b.draft.workflow_json == stored
    assert b.draft.workflow_configurations["max_call_duration"] == 600
    assert b.draft.workflow_configurations.get("turn_stop_strategy") != "turn_analyzer"
    assert b.draft.template_context_variables == {"company": "Dograh", "city": "Pune"}
    assert real.CANDIDATE_META_KEY not in json.dumps(b.draft.workflow_configurations)
