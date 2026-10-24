"""Tests for `api.services.workflow.sdk_authoring` (the lifted MCP save path).

Regression contract (plan R5 / D22):
- `test_mcp_save_workflow.py` passes unchanged (the MCP tool is now a thin
  wrapper around this service).
- One test per check added on top of the old MCP pipeline, each with its
  named `error_code`.
- A masked-secret round trip: project -> save -> stored keys unchanged.
- A golden comparison: real workflow TS (fixtures/sdk_authoring) through a
  frozen copy of the pre-lift MCP tool body and through the new service,
  equal except for the listed intentional differences.

DB access is mocked the same way `test_mcp_save_workflow.py` does it; the
real Node TS validator runs end-to-end.
"""

from __future__ import annotations

import copy
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError as PydanticValidationError

from api.db import db_client
from api.db.agent_trigger_client import TriggerPathConflictError
from api.mcp_server.tools import save_workflow as save_workflow_module
from api.mcp_server.tools._workflow_projection import (
    select_workflow_projection_source,
)
from api.mcp_server.ts_bridge import TsBridgeError, generate_code, parse_code
from api.services.configuration.masking import mask_workflow_definition
from api.services.workflow import sdk_authoring
from api.services.workflow.dto import ReactFlowDTO
from api.services.workflow.errors import ItemKind
from api.services.workflow.layout import reconcile_positions
from api.services.workflow.sdk_authoring import (
    WorkflowNotFoundError,
    compute_definition_hash,
    save_workflow_draft_from_code,
)
from api.services.workflow.tool_name_validation import (
    validate_workflow_tool_name_collisions,
)
from api.services.workflow.trigger_paths import validate_trigger_paths
from api.services.workflow.workflow_graph import WorkflowGraph

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node binary not available"
)

FIXTURES = Path(__file__).parent / "fixtures" / "sdk_authoring"
GOLDEN_FIXTURES = ["wf5.ts", "wf7_live.ts", "wf7_proposed.ts"]

# Real keys whose masks are what wf5.ts carries (`****...t3st`, `****...n0vm`).
REAL_TUNER_KEY = "tuner-live-key-000000000000000000000000t3st"
REAL_NOVEUM_KEY = "noveum-live-key-000000000000000n0vm"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# ─── Frozen pre-lift reference (api/mcp_server/tools/save_workflow.py) ────
# Copied verbatim from the tool body before the lift, minus the MCP auth
# call (the user is passed in). Do not "fix" this function: it is the
# golden baseline the new service is compared against.


async def _legacy_previous_workflow_json(workflow: Any) -> dict[str, Any] | None:
    source = await select_workflow_projection_source(workflow)
    return source.payload


def _legacy_error_result(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"saved": False, "error_code": code, "error": message, **extra}


def _legacy_format_errors(errors: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for e in errors:
        loc = ""
        line = e.get("line")
        col = e.get("column")
        if line is not None:
            loc = f" (line {line}" + (f", col {col}" if col is not None else "") + ")"
        parts.append(f"{e.get('message', '')}{loc}")
    return "\n".join(parts)


async def _legacy_save_workflow(user: Any, workflow_id: int, code: str) -> dict:
    workflow = await db_client.get_workflow(
        workflow_id, organization_id=user.selected_organization_id
    )
    if not workflow:
        raise LookupError(f"Workflow {workflow_id} not found")

    try:
        parsed = await parse_code(code)
    except TsBridgeError as e:
        return _legacy_error_result("bridge_error", str(e))

    if not parsed.get("ok"):
        stage = parsed.get("stage", "parse")
        errs = parsed.get("errors") or []
        code_key = "parse_error" if stage == "parse" else "validation_error"
        return _legacy_error_result(code_key, _legacy_format_errors(errs), errors=errs)

    payload = parsed["workflow"]
    new_name = (parsed.get("workflowName") or "").strip()

    payload = reconcile_positions(
        payload, await _legacy_previous_workflow_json(workflow)
    )
    trigger_path_issues = validate_trigger_paths(payload)
    if trigger_path_issues:
        return _legacy_error_result(
            "validation_error",
            "\n".join(issue.message for issue in trigger_path_issues),
        )

    try:
        dto = ReactFlowDTO.model_validate(payload)
    except PydanticValidationError as e:
        return _legacy_error_result("schema_validation", str(e))

    try:
        WorkflowGraph(dto)
    except (ValueError, Exception) as e:
        return _legacy_error_result("graph_validation", str(e))

    tool_name_errors = await validate_workflow_tool_name_collisions(
        payload,
        user.selected_organization_id,
    )
    if tool_name_errors:
        return _legacy_error_result(
            "graph_validation",
            "\n".join(error["message"] for error in tool_name_errors),
            errors=tool_name_errors,
        )

    name_changed = bool(new_name) and new_name != workflow.name
    if name_changed:
        await db_client.update_workflow(
            workflow_id=workflow_id,
            name=new_name,
            workflow_definition=None,
            template_context_variables=None,
            workflow_configurations=None,
            organization_id=user.selected_organization_id,
        )

    draft = await db_client.save_workflow_draft(
        workflow_id=workflow_id,
        workflow_definition=payload,
    )

    return {
        "saved": True,
        "workflow_id": workflow_id,
        "version_number": draft.version_number,
        "status": draft.status,
        "node_count": len(payload["nodes"]),
        "edge_count": len(payload["edges"]),
        "name": new_name or workflow.name,
        "renamed": name_changed,
    }


# ─── Fakes & fixtures ────────────────────────────────────────────────────


@dataclass
class _FakeDraft:
    version_number: int = 2
    status: str = "draft"
    workflow_json: dict | None = None


@dataclass
class _FakeWorkflow:
    id: int = 1
    organization_id: int = 1
    name: str = "test"
    released_definition: Any = None
    workflow_definition: dict | None = None
    current_definition: Any = None
    extra: dict = field(default_factory=dict)


@pytest.fixture
def user() -> MagicMock:
    u = MagicMock()
    u.selected_organization_id = 1
    u.id = 1
    return u


class _Backends:
    """Patches the db_client singleton for both the legacy and new path."""

    def __init__(self, workflow: _FakeWorkflow, stored_draft: _FakeDraft | None):
        self.workflow = workflow
        self.stored_draft = stored_draft
        self.save = AsyncMock(return_value=_FakeDraft())
        self.update = AsyncMock(return_value=workflow)
        self.assert_paths = AsyncMock(return_value=None)
        self._patches = [
            patch.object(db_client, "get_workflow", AsyncMock(return_value=workflow)),
            patch.object(
                db_client, "get_draft_version", AsyncMock(return_value=stored_draft)
            ),
            patch.object(db_client, "save_workflow_draft", self.save),
            patch.object(db_client, "update_workflow", self.update),
            patch.object(
                db_client, "assert_trigger_paths_available", self.assert_paths
            ),
            patch.object(db_client, "get_tools_by_uuids", AsyncMock(return_value=[])),
        ]

    def __enter__(self):
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()

    @property
    def saved_payload(self) -> dict:
        return self.save.call_args.kwargs["workflow_definition"]


def _backends(
    previous: dict | None = None, name: str = "test", version_number: int = 3
) -> _Backends:
    draft = (
        _FakeDraft(version_number=version_number, workflow_json=copy.deepcopy(previous))
        if previous is not None
        else None
    )
    return _Backends(_FakeWorkflow(name=name), draft)


def _valid_code(name: str = "test") -> str:
    return f'''import {{ Workflow }} from "@dograh/sdk";
import {{ startCall, endCall }} from "@dograh/sdk/typed";

const wf = new Workflow({{ name: "{name}" }});

const greeting = wf.addTyped(startCall({{ name: "greeting", prompt: "Hi!" }}));
const done     = wf.addTyped(endCall({{ name: "done", prompt: "Bye." }}));

wf.edge(greeting, done, {{ label: "done", condition: "conversation complete" }});
'''


def _start_end_payload(**start_data: Any) -> dict:
    return {
        "nodes": [
            {
                "id": "start-1",
                "type": "startCall",
                "position": {"x": 0, "y": 0},
                "data": {"name": "Start", "prompt": "Greet.", **start_data},
            },
            {
                "id": "end-1",
                "type": "endCall",
                "position": {"x": 0, "y": 200},
                "data": {"name": "End", "prompt": "Bye."},
            },
        ],
        "edges": [
            {
                "id": "e-1",
                "source": "start-1",
                "target": "end-1",
                "data": {"label": "done", "condition": "conversation complete"},
            }
        ],
    }


async def _save_with_payload(user, payload: dict, **kwargs) -> dict:
    """Run the service with the parser stubbed to return `payload`."""
    return await save_workflow_draft_from_code(
        user=user,
        organization_id=1,
        workflow_id=1,
        code="ignored",
        parse_code_fn=AsyncMock(
            return_value={"ok": True, "workflowName": "test", "workflow": payload}
        ),
        reconcile_positions_fn=lambda p, _prev: p,
        **kwargs,
    )


async def _new_save(user, code: str, **kwargs) -> dict:
    return await save_workflow_draft_from_code(
        user=user, organization_id=1, workflow_id=1, code=code, **kwargs
    )


async def _parsed(code: str) -> dict:
    result = await parse_code(code)
    assert result.get("ok"), result
    return result["workflow"]


def _with_real_secrets(definition: dict) -> dict:
    out = copy.deepcopy(definition)
    for node in out["nodes"]:
        if node["type"] == "tuner":
            node["data"]["tuner_api_key"] = REAL_TUNER_KEY
        if node["type"] == "noveum":
            node["data"]["noveum_api_key"] = REAL_NOVEUM_KEY
    return out


def _secret_values(definition: dict) -> dict[str, str]:
    out = {}
    for node in definition["nodes"]:
        for key, value in node.get("data", {}).items():
            if key.endswith("api_key"):
                out[f"{node['id']}.{key}"] = value
    return out


# ─── Added check: sanitize ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sanitize_strips_ui_only_fields_before_save(user):
    payload = _start_end_payload(invalid=True, validationMessage="stale")
    payload["edges"][0]["data"]["invalid"] = True
    with _backends() as b:
        result = await _save_with_payload(user, payload)

    assert result["saved"] is True
    saved = b.saved_payload
    assert "invalid" not in saved["nodes"][0]["data"]
    assert "validationMessage" not in saved["nodes"][0]["data"]
    assert "invalid" not in saved["edges"][0]["data"]
    assert saved["nodes"][0]["data"]["prompt"] == "Greet."


# ─── Added check: ensure_trigger_paths ───────────────────────────────────


@pytest.mark.asyncio
async def test_trigger_without_path_gets_minted_uuid(user):
    payload = _start_end_payload()
    payload["nodes"].append(
        {
            "id": "trigger-1",
            "type": "trigger",
            "position": {"x": 0, "y": 400},
            "data": {"name": "API Trigger"},
        }
    )
    with _backends() as b:
        result = await _save_with_payload(user, payload)

    assert result["saved"] is True
    trigger = next(n for n in b.saved_payload["nodes"] if n["type"] == "trigger")
    assert re.fullmatch(r"[0-9a-f-]{36}", trigger["data"]["trigger_path"])
    b.assert_paths.assert_awaited_once_with(
        trigger_paths=[trigger["data"]["trigger_path"]], exclude_workflow_id=1
    )


# ─── Added check: trigger path owned by another workflow ─────────────────


@pytest.mark.asyncio
async def test_trigger_path_conflict_rejected(user):
    payload = _start_end_payload()
    payload["nodes"].append(
        {
            "id": "trigger-1",
            "type": "trigger",
            "position": {"x": 0, "y": 400},
            "data": {"name": "API Trigger", "trigger_path": "support_west"},
        }
    )
    with _backends() as b:
        b.assert_paths.side_effect = TriggerPathConflictError(["support_west"])
        result = await _save_with_payload(user, payload)

    assert result["saved"] is False
    assert result["error_code"] == "trigger_path_conflict"
    assert result["trigger_paths"] == ["support_west"]
    assert "support_west" in result["error"]
    b.save.assert_not_awaited()
    b.update.assert_not_awaited()


# ─── Added check: node instance limits ───────────────────────────────────


@pytest.mark.asyncio
async def test_node_instance_validation_rejected(user):
    # WorkflowGraph enforces the same limits today, so it is stubbed out to
    # prove the added check is a real second gate (the UI route's check).
    payload = _start_end_payload()
    payload["nodes"].append(
        {
            "id": "start-2",
            "type": "startCall",
            "position": {"x": 200, "y": 0},
            "data": {"name": "Start 2", "prompt": "Greet again."},
        }
    )
    with _backends() as b, patch.object(sdk_authoring, "WorkflowGraph"):
        result = await _save_with_payload(user, payload)

    assert result["saved"] is False
    assert result["error_code"] == "node_instance_validation"
    assert "at most" in result["error"]
    assert result["errors"][0]["kind"] == ItemKind.workflow
    b.save.assert_not_awaited()
    b.update.assert_not_awaited()


# ─── Added check: transition tool names ──────────────────────────────────


@pytest.mark.asyncio
async def test_transition_name_validation_rejected(user):
    payload = _start_end_payload()
    payload["nodes"].append(
        {
            "id": "end-2",
            "type": "endCall",
            "position": {"x": 200, "y": 200},
            "data": {"name": "End 2", "prompt": "Bye."},
        }
    )
    payload["edges"].append(
        {
            "id": "e-2",
            "source": "start-1",
            "target": "end-2",
            "data": {"label": "done", "condition": "other"},
        }
    )
    with (
        _backends() as b,
        patch.object(sdk_authoring, "WorkflowGraph"),
        patch.object(
            sdk_authoring,
            "validate_workflow_tool_name_collisions",
            AsyncMock(return_value=[]),
        ),
    ):
        result = await _save_with_payload(user, payload)

    assert result["saved"] is False
    assert result["error_code"] == "transition_name_validation"
    assert 'Transition tool name "done" is duplicated' in result["error"]
    b.save.assert_not_awaited()


# ─── Added check: masked-secret merge (round trip) ───────────────────────


@pytest.mark.asyncio
async def test_masked_secret_round_trip_keeps_stored_keys(user):
    """stored (real keys) -> masked projection -> TS -> save -> keys unchanged."""
    stored = _with_real_secrets(await _parsed(_fixture("wf5.ts")))
    projected_code = await generate_code(
        mask_workflow_definition(stored), workflow_name="Customer Support - inbound"
    )
    assert REAL_TUNER_KEY not in projected_code
    assert REAL_NOVEUM_KEY not in projected_code

    with _backends(previous=stored, name="Customer Support - inbound") as b:
        result = await _new_save(user, projected_code)

    assert result["saved"] is True
    assert _secret_values(b.saved_payload) == _secret_values(stored)


@pytest.mark.asyncio
async def test_masked_secret_fixture_restores_stored_keys(user):
    """The real wf5 projection (masked keys) does not overwrite real keys."""
    stored = _with_real_secrets(await _parsed(_fixture("wf5.ts")))
    with _backends(previous=stored, name="Customer Support - inbound") as b:
        result = await _new_save(user, _fixture("wf5.ts"))
    assert result["saved"] is True
    assert _secret_values(b.saved_payload) == _secret_values(stored)


@pytest.mark.asyncio
async def test_new_unmasked_secret_replaces_stored_key(user):
    stored = _with_real_secrets(await _parsed(_fixture("wf5.ts")))
    code = _fixture("wf5.ts").replace(
        '"***************************************t3st"', '"brand-new-tuner-key"'
    )
    with _backends(previous=stored, name="Customer Support - inbound") as b:
        result = await _new_save(user, code)
    assert result["saved"] is True
    tuner = next(n for n in b.saved_payload["nodes"] if n["type"] == "tuner")
    assert tuner["data"]["tuner_api_key"] == "brand-new-tuner-key"


# ─── Optimistic lock ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_optimistic_lock_conflict_on_stale_hash(user):
    previous = await _parsed(_valid_code())
    with _backends(previous=previous) as b:
        result = await _new_save(
            user, _valid_code(), expected_draft_version="0" * 64 + "stale"
        )
    assert result["saved"] is False
    assert result["error_code"] == "draft_conflict"
    assert result["current_definition_hash"] == compute_definition_hash(previous)
    assert result["current_version_number"] == 3
    b.save.assert_not_awaited()
    b.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_optimistic_lock_conflict_on_stale_version_number(user):
    previous = await _parsed(_valid_code())
    with _backends(previous=previous, version_number=4) as b:
        result = await _new_save(user, _valid_code(), expected_draft_version=3)
    assert result["error_code"] == "draft_conflict"
    b.save.assert_not_awaited()


@pytest.mark.asyncio
async def test_optimistic_lock_passes_when_current(user):
    previous = await _parsed(_valid_code())
    with _backends(previous=previous, version_number=4) as b:
        by_hash = await _new_save(
            user,
            _valid_code(),
            expected_draft_version=compute_definition_hash(previous),
        )
        by_number = await _new_save(user, _valid_code(), expected_draft_version=4)
    assert by_hash["saved"] is True
    assert by_number["saved"] is True
    assert b.save.await_count == 2


# ─── Hash ────────────────────────────────────────────────────────────────


def test_definition_hash_is_deterministic_and_order_independent():
    a = {"nodes": [{"id": "1", "data": {"b": 2, "a": 1}}], "edges": []}
    b = {"edges": [], "nodes": [{"data": {"a": 1, "b": 2}, "id": "1"}]}
    assert compute_definition_hash(a) == compute_definition_hash(b)
    assert compute_definition_hash(a) == compute_definition_hash(copy.deepcopy(a))
    assert re.fullmatch(r"[0-9a-f]{64}", compute_definition_hash(a))
    c = copy.deepcopy(a)
    c["nodes"][0]["data"]["a"] = 3
    assert compute_definition_hash(a) != compute_definition_hash(c)
    assert compute_definition_hash(None) == compute_definition_hash({})


# ─── Not found ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_workflow_raises_not_found(user):
    with patch.object(db_client, "get_workflow", AsyncMock(return_value=None)):
        with pytest.raises(WorkflowNotFoundError):
            await _new_save(user, _valid_code())


# ─── Error codes stay documented ─────────────────────────────────────────


def test_every_service_error_code_is_documented_on_the_mcp_tool():
    source = Path(sdk_authoring.__file__).read_text(encoding="utf-8")
    codes = set(re.findall(r'_error_result\(\s*"([a-z_]+)"', source))
    codes |= set(re.findall(r'"(parse_error|validation_error)"', source))
    assert codes, "regex broke"
    mcp_codes = codes - {"draft_conflict"}  # MCP never sends a version
    assert mcp_codes == set(save_workflow_module.SAVE_WORKFLOW_ERROR_CODES)
    doc = save_workflow_module.save_workflow.__doc__ or ""
    assert not sorted(c for c in mcp_codes if f"`{c}`" not in doc)


# ─── Golden: old MCP path vs new service on real workflows ───────────────
#
# Intentional differences (everything else must be identical):
#   1. masked secrets are restored to the stored real keys;
#   2. non-spec node/edge data is stripped (no-op for parser output);
#   3. trigger nodes without a path get a minted UUID (none in fixtures);
#   4. new rejections: node_instance_validation, transition_name_validation,
#      trigger_path_conflict, draft_conflict (none trigger on these inputs).


def _strip_intentional(definition: dict) -> dict:
    out = copy.deepcopy(definition)
    for node in out["nodes"]:
        for key in list(node.get("data", {})):
            if key.endswith("api_key"):
                node["data"][key] = "<secret>"
    return out


async def _run_both(user, code: str, previous: dict | None, name: str):
    with _backends(previous=previous, name=name) as b:
        old = await _legacy_save_workflow(user, 1, code)
        old_payload = copy.deepcopy(b.saved_payload) if b.save.await_count else None
        old_update = b.update.call_args
        b.save.reset_mock()
        b.update.reset_mock()
        new = await _new_save(user, code)
        new_payload = copy.deepcopy(b.saved_payload) if b.save.await_count else None
        new_update = b.update.call_args
    return old, new, old_payload, new_payload, old_update, new_update


@pytest.mark.asyncio
@pytest.mark.parametrize("fixture", GOLDEN_FIXTURES)
@pytest.mark.parametrize("previous_kind", ["none", "live"])
async def test_golden_old_vs_new_on_real_workflows(user, fixture, previous_kind):
    code = _fixture(fixture)
    if previous_kind == "none":
        previous = None
    elif fixture == "wf5.ts":
        previous = _with_real_secrets(await _parsed(code))
    else:
        previous = await _parsed(_fixture("wf7_live.ts"))

    old, new, old_payload, new_payload, old_update, new_update = await _run_both(
        user, code, previous, name="stored-name"
    )

    assert old["saved"] is True
    assert new == old
    assert new_update == old_update  # rename path identical
    if previous is not None and fixture == "wf5.ts":
        # Intentional difference 1: real keys survive in the new path only.
        assert _secret_values(new_payload) == _secret_values(previous)
        assert _secret_values(old_payload) != _secret_values(previous)
    else:
        assert _secret_values(new_payload) == _secret_values(old_payload)
    assert _strip_intentional(new_payload) == _strip_intentional(old_payload)


def _broken_variants(code: str) -> dict[str, str]:
    return {
        "parse_error": code + "\nfunction evil() { return 1; }\n",
        "validation_error": code.replace("prompt:", "promt:", 1),
        # Dropping every edge strands the agent nodes (no incoming edge).
        "graph_validation": re.sub(r"wf\.edge\([^;]*\);\s*", "", code, flags=re.S),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("fixture", GOLDEN_FIXTURES)
async def test_golden_rejections_match_old_path(user, fixture):
    for label, code in _broken_variants(_fixture(fixture)).items():
        old, new, *_ = await _run_both(user, code, previous=None, name="x")
        assert new == old, f"{fixture}/{label} diverged"
        assert old["saved"] is False, f"{fixture}/{label} was accepted"
        assert old["error_code"] == label
        assert new["error_code"] == label


# ─── MCP wrapper delegates ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mcp_wrapper_delegates_to_service(user):
    sentinel = {"saved": True, "sentinel": 1}
    with (
        patch.object(
            save_workflow_module,
            "authenticate_mcp_request",
            AsyncMock(return_value=user),
        ),
        patch.object(
            save_workflow_module,
            "save_workflow_draft_from_code",
            AsyncMock(return_value=sentinel),
        ) as svc,
    ):
        result = await save_workflow_module.save_workflow(workflow_id=7, code="c")
    assert result is sentinel
    kwargs = svc.call_args.kwargs
    assert kwargs["workflow_id"] == 7
    assert kwargs["organization_id"] == 1
    assert kwargs["code"] == "c"
    assert "expected_draft_version" not in kwargs


async def test_projection_masks_integration_secrets(monkeypatch):
    """get_workflow_code must never send raw integration keys to MCP clients."""
    from types import SimpleNamespace

    from api.mcp_server.tools import _workflow_projection as projection

    secret = "sk-live-REALSECRET1234567890"
    payload = {
        "nodes": [
            {
                "id": "1",
                "type": "tuner",
                "position": {"x": 0, "y": 0},
                "data": {"name": "Tuner", "tuner_api_key": secret},
            }
        ],
        "edges": [],
    }
    captured = {}

    async def fake_generate(workflow, workflow_name=""):
        captured["workflow"] = workflow
        return "code"

    async def fake_source(workflow):
        return projection.WorkflowProjectionSource(
            payload=payload, version="draft", version_number=1
        )

    monkeypatch.setattr(projection, "generate_code", fake_generate)
    monkeypatch.setattr(projection, "select_workflow_projection_source", fake_source)
    await projection.project_workflow_to_sdk_view(SimpleNamespace(name="wf"))
    assert secret not in str(captured["workflow"])
    assert payload["nodes"][0]["data"]["tuner_api_key"] == secret  # source untouched
