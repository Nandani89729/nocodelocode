"""Save LLM/SDK-authored TypeScript as a workflow draft (or a new workflow).

Shared by the MCP `save_workflow` / `create_workflow` tools and in-product
authoring flows (the Agent Doctor "Save as draft" action, the Agent
Copilot). The MCP tools are thin auth wrappers around
`save_workflow_draft_from_code` and `create_workflow_from_code`.

Pipeline (each rejection returns `saved: False` plus an `error_code`):

    1. Parse + spec-validate via the Node TS validator (AST-only, the code
       is never executed)                         -> `bridge_error`,
                                                     `parse_error`,
                                                     `validation_error`
    2. Reconcile node positions, check trigger-path format
                                                  -> `validation_error`
    3. Wire-format DTO check                      -> `schema_validation`
    4. Graph invariants + custom-tool name clashes -> `graph_validation`
    5. Checks shared with the UI save route (`PUT /workflow/{id}`):
       sanitize (strip non-spec node/edge data), mint missing trigger paths,
       node instance limits                       -> `node_instance_validation`
       transition tool names                      -> `transition_name_validation`
       trigger paths owned by another workflow    -> `trigger_path_conflict`
    6. Optional optimistic lock                   -> `draft_conflict`
    7. Restore real secrets where the projected code carried masked ones
       (`merge_workflow_api_keys`), rename if needed, save the draft.

Steps 1-5 are `parse_and_validate_workflow_code` (no writes), so callers
that only need a validated candidate (the copilot's proposals) run exactly
the checks a save runs. Steps 2b-5 alone are `validate_workflow_payload`
(for JSON that was not parsed from code, e.g. a merged proposal); steps 6-7
are `save_validated_draft`.

Steps 1-4 and the result payload are byte-for-byte the pre-existing MCP
behaviour; step 5-7 are the intentional additions. Every check that
existed before still runs first, so inputs rejected before keep the same
`error_code`.

`create_workflow_from_code` is the pre-existing MCP `create_workflow`
pipeline (steps 1-4, trigger-path conflict, persist as published v1),
unchanged; its failures carry `created: False`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger
from pydantic import ValidationError as PydanticValidationError

from api.db import db_client
from api.db.agent_trigger_client import TriggerPathConflictError
from api.enums import PostHogEvent
from api.services.configuration.masking import merge_workflow_api_keys
from api.services.posthog_client import capture_event
from api.services.workflow.draft_validation import (
    node_instance_validation_errors,
    transition_tool_name_validation_errors,
)
from api.services.workflow.dto import ReactFlowDTO, sanitize_workflow_definition
from api.services.workflow.layout import reconcile_positions
from api.services.workflow.tool_name_validation import (
    validate_workflow_tool_name_collisions,
)
from api.services.workflow.trigger_paths import (
    ensure_trigger_paths,
    extract_trigger_paths,
    validate_trigger_paths,
)
from api.services.workflow.workflow_graph import WorkflowGraph

ParseCodeFn = Callable[[str], Awaitable[dict[str, Any]]]
ReconcileFn = Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]]


class WorkflowNotFoundError(LookupError):
    """The workflow does not exist in the caller's organization."""

    def __init__(self, workflow_id: int):
        super().__init__(f"Workflow {workflow_id} not found")
        self.workflow_id = workflow_id


@dataclass
class ValidatedWorkflow:
    """Outcome of `parse_and_validate_workflow_code` / `validate_workflow_payload`.

    On success `payload` is the canonical workflow JSON (positions
    reconciled, sanitized, trigger paths minted) and `error` is None. On
    rejection `error` is the `{"saved": False, "error_code", "error", ...}`
    dict the save path returns.
    """

    payload: dict[str, Any] | None = None
    #: `new Workflow({ name })` from the source ("" when absent / not parsed)
    name: str = ""
    error: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def compute_definition_hash(workflow_json: dict | None) -> str:
    """Stable sha256 of a workflow definition (canonical JSON).

    Used as the optimistic-lock token: callers read the working copy,
    remember its hash, and pass it back as `expected_draft_version`.
    """
    canonical = json.dumps(
        workflow_json or {},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _error_result(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"saved": False, "error_code": code, "error": message, **extra}


def _create_error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    # Separate helper (not `_error_result`) so the save-path error-code drift
    # check, which scans this module for `_error_result("...")`, stays exact.
    return {"created": False, "error_code": code, "error": message, **extra}


def _as_create_error(save_error: dict[str, Any]) -> dict[str, Any]:
    return {
        "created": False,
        **{k: v for k, v in save_error.items() if k != "saved"},
    }


def _format_errors(errors: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for e in errors:
        loc = ""
        line = e.get("line")
        col = e.get("column")
        if line is not None:
            loc = f" (line {line}" + (f", col {col}" if col is not None else "") + ")"
        parts.append(f"{e.get('message', '')}{loc}")
    return "\n".join(parts)


def _version_matches(expected: str | int, source: Any) -> bool:
    """An int is compared to the working copy's version_number; a string to
    its definition hash (draft rows are updated in place, so the version
    number alone does not change between saves)."""
    if isinstance(expected, bool):
        return False
    if isinstance(expected, int):
        return source.version_number == expected
    expected_str = str(expected).strip()
    if expected_str.isdigit():
        return source.version_number == int(expected_str)
    return compute_definition_hash(source.payload) == expected_str


# ─── Validation stages (no writes) ───────────────────────────────────────


async def _parse_stage(
    code: str, parse_code_fn: ParseCodeFn | None
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Step 1. Returns `(parsed, None)` or `(None, error_result)`."""
    from api.mcp_server.ts_bridge import TsBridgeError, parse_code

    _parse = parse_code_fn or parse_code
    try:
        parsed = await _parse(code)
    except TsBridgeError as e:
        logger.warning(f"ts_bridge failure: {e}")
        return None, _error_result("bridge_error", str(e))

    if not parsed.get("ok"):
        stage = parsed.get("stage", "parse")
        errs = parsed.get("errors") or []
        code_key = "parse_error" if stage == "parse" else "validation_error"
        return None, _error_result(code_key, _format_errors(errs), errors=errs)
    return parsed, None


async def _structural_checks(
    payload: dict[str, Any], organization_id: int
) -> dict[str, Any] | None:
    """Steps 2b-4 (shared with `create_workflow_from_code`)."""
    trigger_path_issues = validate_trigger_paths(payload)
    if trigger_path_issues:
        return _error_result(
            "validation_error",
            "\n".join(issue.message for issue in trigger_path_issues),
        )

    # 2. Pydantic shape check (defence in depth — parser is spec-driven).
    try:
        dto = ReactFlowDTO.model_validate(payload)
    except PydanticValidationError as e:
        return _error_result("schema_validation", str(e))

    # 3. Graph-level semantic validation (start-node count, edge shape).
    try:
        WorkflowGraph(dto)
    except (ValueError, Exception) as e:  # WorkflowGraph raises ValueError
        return _error_result("graph_validation", str(e))

    tool_name_errors = await validate_workflow_tool_name_collisions(
        payload,
        organization_id,
    )
    if tool_name_errors:
        return _error_result(
            "graph_validation",
            "\n".join(error["message"] for error in tool_name_errors),
            errors=tool_name_errors,
        )
    return None


async def validate_workflow_payload(
    payload: dict[str, Any],
    *,
    organization_id: int,
    exclude_workflow_id: int | None = None,
) -> ValidatedWorkflow:
    """Steps 2b-5 on workflow JSON whose positions are already set.

    Returns the canonical (sanitized, trigger paths minted) payload or the
    first rejection. `exclude_workflow_id` is the workflow being saved, so
    its own trigger paths do not count as taken. Read-only.
    """
    error = await _structural_checks(payload, organization_id)
    if error:
        return ValidatedWorkflow(error=error)

    # 5. Checks shared with the UI save route (PUT /workflow/{id}).
    payload = sanitize_workflow_definition(payload)
    payload = ensure_trigger_paths(payload)

    instance_errors = node_instance_validation_errors(payload)
    if instance_errors:
        return ValidatedWorkflow(
            error=_error_result(
                "node_instance_validation",
                "\n".join(error["message"] for error in instance_errors),
                errors=instance_errors,
            )
        )

    transition_errors = transition_tool_name_validation_errors(payload)
    if transition_errors:
        return ValidatedWorkflow(
            error=_error_result(
                "transition_name_validation",
                "\n".join(error["message"] for error in transition_errors),
                errors=transition_errors,
            )
        )

    trigger_paths = extract_trigger_paths(payload)
    if trigger_paths:
        try:
            await db_client.assert_trigger_paths_available(
                trigger_paths=trigger_paths,
                exclude_workflow_id=exclude_workflow_id,
            )
        except TriggerPathConflictError as e:
            return ValidatedWorkflow(
                error=_error_result(
                    "trigger_path_conflict",
                    "Trigger path is already in use by another workflow: "
                    + ", ".join(e.trigger_paths)
                    + ". Choose another path or leave it empty to get a unique one.",
                    trigger_paths=e.trigger_paths,
                )
            )

    return ValidatedWorkflow(payload=payload)


async def parse_and_validate_workflow_code(
    *,
    code: str,
    organization_id: int,
    previous_json: dict[str, Any] | None,
    exclude_workflow_id: int | None = None,
    parse_code_fn: ParseCodeFn | None = None,
    reconcile_positions_fn: ReconcileFn | None = None,
) -> ValidatedWorkflow:
    """Steps 1-5 without saving: parse SDK TypeScript into canonical JSON.

    `previous_json` is the working copy positions are reconciled against
    (None for a new workflow). Secrets are NOT merged here (that needs node
    ids that match the working copy); `save_validated_draft` merges them.
    """
    _reconcile = reconcile_positions_fn or reconcile_positions

    parsed, error = await _parse_stage(code, parse_code_fn)
    if error:
        return ValidatedWorkflow(error=error)

    payload = parsed["workflow"]
    name = (parsed.get("workflowName") or "").strip()

    # 1b. Reconcile node positions against the previously-stored workflow.
    # The parser drops positions by design (LLMs don't place nodes well).
    payload = _reconcile(payload, previous_json)

    validated = await validate_workflow_payload(
        payload,
        organization_id=organization_id,
        exclude_workflow_id=exclude_workflow_id,
    )
    validated.name = name
    return validated


# ─── Writes ──────────────────────────────────────────────────────────────


async def save_validated_draft(
    *,
    workflow: Any,
    organization_id: int,
    source: Any,
    payload: dict[str, Any] | None,
    new_name: str = "",
    expected_draft_version: str | int | None = None,
    include_definition_id: bool = False,
    workflow_configurations: dict[str, Any] | None = None,
    template_context_variables: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Steps 6-7: lock check, secret merge, rename, save as the draft.

    `payload` must come from `validate_workflow_payload` (or
    `parse_and_validate_workflow_code`); `source` is the working copy
    (`select_workflow_projection_source`) it was validated against.
    `include_definition_id` adds the saved draft's definition id
    (`definition_id`) to the result; off for the MCP tool, whose output
    shape is a regression contract.

    `workflow_configurations` / `template_context_variables` (the Agent
    Copilot's settings changes) are written in the same draft update; None
    leaves them unchanged (the MCP tools never pass them). `payload=None`
    saves settings only and leaves the draft's workflow JSON untouched.
    """
    workflow_id = workflow.id
    previous_json = source.payload

    # 6. Optimistic lock: refuse to overwrite a working copy that changed
    # since the caller read it.
    if expected_draft_version is not None and not _version_matches(
        expected_draft_version, source
    ):
        return _error_result(
            "draft_conflict",
            "The workflow changed since it was read. Fetch the latest version, "
            "re-apply your edits and save again.",
            current_version_number=source.version_number,
            current_definition_hash=compute_definition_hash(previous_json),
        )

    # 7a. Projected code carries masked secrets; never let a mask overwrite
    # the stored real key.
    if payload is not None:
        payload = merge_workflow_api_keys(payload, previous_json)

    # 7b. If the `new Workflow({ name })` in the edited source differs from
    # the stored name, rename the workflow. Name is a workflow-level field
    # (not versioned), so this takes effect immediately.
    name_changed = bool(new_name) and new_name != workflow.name
    if name_changed:
        await db_client.update_workflow(
            workflow_id=workflow_id,
            name=new_name,
            workflow_definition=None,
            template_context_variables=None,
            workflow_configurations=None,
            organization_id=organization_id,
        )

    # 7c. Save as a new draft (existing published version stays intact).
    # Settings are passed only when given, so the call (and the MCP save)
    # is unchanged for graph-only saves.
    settings_kwargs: dict[str, Any] = {}
    if workflow_configurations is not None:
        settings_kwargs["workflow_configurations"] = workflow_configurations
    if template_context_variables is not None:
        settings_kwargs["template_context_variables"] = template_context_variables
    draft = await db_client.save_workflow_draft(
        workflow_id=workflow_id,
        workflow_definition=payload,
        **settings_kwargs,
    )

    counted = payload if payload is not None else (previous_json or {})
    result = {
        "saved": True,
        "workflow_id": workflow_id,
        "version_number": draft.version_number,
        "status": draft.status,
        "node_count": len(counted.get("nodes") or []),
        "edge_count": len(counted.get("edges") or []),
        "name": new_name or workflow.name,
        "renamed": name_changed,
    }
    if include_definition_id:
        result["definition_id"] = getattr(draft, "id", None)
    return result


async def save_workflow_draft_from_code(
    *,
    user: Any,
    organization_id: int,
    workflow_id: int,
    code: str,
    expected_draft_version: str | int | None = None,
    parse_code_fn: ParseCodeFn | None = None,
    reconcile_positions_fn: ReconcileFn | None = None,
) -> dict[str, Any]:
    """Parse SDK TypeScript and save the resulting workflow as a draft.

    Raises `WorkflowNotFoundError` when the workflow is not in
    `organization_id`. Every other failure is returned as
    `{"saved": False, "error_code": ..., "error": ...}`.

    `expected_draft_version` (optional) is either the working copy's
    `version_number` (int) or its `compute_definition_hash` (str); a
    mismatch returns `error_code="draft_conflict"` without writing.

    `parse_code_fn` / `reconcile_positions_fn` let callers supply the
    parser and position reconciler (the MCP wrapper passes its own module
    references so existing tests can patch them there).
    """
    workflow = await db_client.get_workflow(
        workflow_id, organization_id=organization_id
    )
    if not workflow:
        raise WorkflowNotFoundError(workflow_id)

    # The working copy the caller projected from (draft > published > legacy).
    # Used for position reconciliation, the optimistic lock and secret merge.
    source = await select_working_copy(workflow)

    validated = await parse_and_validate_workflow_code(
        code=code,
        organization_id=organization_id,
        previous_json=source.payload,
        exclude_workflow_id=workflow_id,
        parse_code_fn=parse_code_fn,
        reconcile_positions_fn=reconcile_positions_fn,
    )
    if not validated.ok:
        return validated.error

    return await save_validated_draft(
        workflow=workflow,
        organization_id=organization_id,
        source=source,
        payload=validated.payload,
        new_name=validated.name,
        expected_draft_version=expected_draft_version,
    )


async def select_working_copy(workflow: Any) -> Any:
    """The working copy (draft > published > legacy) every authoring path
    reads, diffs and saves against (`WorkflowProjectionSource`)."""
    # Imported here, not at module level: api.mcp_server's package init
    # imports the save_workflow tool, which imports this module (cycle).
    from api.mcp_server.tools._workflow_projection import (
        select_workflow_projection_source,
    )

    return await select_workflow_projection_source(workflow)


async def create_workflow_from_code(
    *,
    user: Any,
    organization_id: int,
    code: str,
    parse_code_fn: ParseCodeFn | None = None,
    reconcile_positions_fn: ReconcileFn | None = None,
    event_source: str = "mcp",
) -> dict[str, Any]:
    """Parse SDK TypeScript and create a new workflow, published as v1.

    The MCP `create_workflow` pipeline (its docstring documents every
    `error_code`). Failures return `{"created": False, "error_code", "error"}`.
    `event_source` tags the PostHog WORKFLOW_CREATED event.
    """
    _reconcile = reconcile_positions_fn or reconcile_positions

    # 1. Parse + spec-validate via the Node TS validator.
    parsed, error = await _parse_stage(code, parse_code_fn)
    if error:
        return _as_create_error(error)

    payload = parsed["workflow"]
    name = (parsed.get("workflowName") or "").strip()
    if not name:
        return _create_error(
            "missing_name",
            'Workflow name is required. Add `new Workflow({ name: "..." })` to the source.',
        )

    # 1b. New workflow — no prior version to reconcile against; layout
    # places new nodes adjacent to their first incoming neighbor.
    payload = _reconcile(payload, None)

    # 2-3. Trigger-path format, DTO, graph and tool-name checks.
    error = await _structural_checks(payload, organization_id)
    if error:
        return _as_create_error(error)

    # 4. Reject upfront if any trigger path collides with another workflow's
    # trigger in this org so we don't leave an orphan workflow record.
    trigger_paths = extract_trigger_paths(payload)
    if trigger_paths:
        try:
            await db_client.assert_trigger_paths_available(
                trigger_paths=trigger_paths,
            )
        except TriggerPathConflictError as e:
            return _create_error(
                "trigger_path_conflict", str(e), trigger_paths=e.trigger_paths
            )

    # 5. Persist as a new workflow with v1 published.
    workflow = await db_client.create_workflow(
        name,
        payload,
        user.id,
        organization_id,
    )

    capture_event(
        distinct_id=str(user.provider_id),
        event=PostHogEvent.WORKFLOW_CREATED,
        properties={
            "workflow_id": workflow.id,
            "workflow_name": workflow.name,
            "source": event_source,
            "organization_id": organization_id,
        },
    )

    if trigger_paths:
        await db_client.sync_triggers_for_workflow(
            workflow_id=workflow.id,
            organization_id=organization_id,
            trigger_paths=trigger_paths,
        )

    return {
        "created": True,
        "workflow_id": workflow.id,
        "name": workflow.name,
        "status": workflow.status,
        "version_number": 1,
        "node_count": len(payload["nodes"]),
        "edge_count": len(payload["edges"]),
    }
