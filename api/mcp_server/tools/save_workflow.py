"""MCP tool that accepts LLM-authored SDK TypeScript and saves it as a draft.

Thin wrapper: authenticates the MCP caller, then delegates to
`api.services.workflow.sdk_authoring.save_workflow_draft_from_code`, which
owns the parse -> validate -> save pipeline (see that module for the full
flow and every `error_code`).

Each failure path returns an `error_code`. Those codes and their meanings
are documented in the `save_workflow` docstring (the description shipped
to the LLM via `tools/list`); keep the two in sync —
`test_mcp_instructions_drift.py` enforces it. All LLM-facing errors
include file:line:column where available so the LLM can correct its code
directly.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

# `db_client` is re-exported so `patch("...save_workflow.db_client.<method>")`
# targets keep resolving; the service uses the same singleton.
from api.db import db_client  # noqa: F401
from api.mcp_server.auth import authenticate_mcp_request
from api.mcp_server.tracing import traced_tool
from api.mcp_server.ts_bridge import parse_code
from api.services.workflow.layout import reconcile_positions
from api.services.workflow.sdk_authoring import (
    WorkflowNotFoundError,
    save_workflow_draft_from_code,
)

# Every `error_code` this tool can return (`draft_conflict` is excluded:
# the tool never passes `expected_draft_version`). Kept here, next to the
# docstring that documents them, so the description drift check in
# `test_mcp_instructions_drift.py` (which scans this module) and
# `test_sdk_authoring.py` (which scans the service) both see them.
SAVE_WORKFLOW_ERROR_CODES = frozenset(
    {
        "parse_error",
        "validation_error",
        "schema_validation",
        "graph_validation",
        "node_instance_validation",
        "transition_name_validation",
        "trigger_path_conflict",
        "bridge_error",
    }
)


@traced_tool
async def save_workflow(workflow_id: int, code: str) -> dict[str, Any]:
    """Parse SDK TypeScript and save the resulting workflow as a draft.

    `code` is TypeScript source using `@dograh/sdk`. Fetch the current
    code first via `get_workflow_code(workflow_id)`, edit it, then pass
    the full updated source here.

    Example code:
        import { Workflow } from "@dograh/sdk";
        import { startCall, endCall } from "@dograh/sdk/typed";

        const wf = new Workflow({ name: "lead_qualification" });
        const greeting = wf.addTyped(startCall({ name: "Greeting", prompt: "Hi!" }));
        const done     = wf.addTyped(endCall({ name: "Done", prompt: "Bye." }));
        wf.edge(greeting, done, { label: "done", condition: "conversation complete" });

    On success the draft version is saved; the published version is
    untouched.

    On failure the result has `saved: false`, a machine-readable
    `error_code`, and a human-readable `error` (with file:line:column
    where the problem is locatable). Resubmit the full corrected source —
    patches are not accepted. Possible `error_code` values:
    - `parse_error` — disallowed construct or malformed TypeScript.
    - `validation_error` — node data failed spec validation (unknown
      field, missing required, wrong type, option out of range).
    - `schema_validation` — wire-format (DTO) rejection; rare.
    - `graph_validation` — structural rule broken (e.g. no start node,
      unreachable node, edge to/from the wrong node type).
    - `node_instance_validation` — too many nodes of a type that allows
      only a limited number (e.g. more than one start node).
    - `transition_name_validation` — two edges from the same node produce
      the same transition tool name; give them distinct labels.
    - `trigger_path_conflict` — an API Trigger path is already used by
      another workflow; pick another or leave it empty to get a unique one.
    - `bridge_error` — internal/transient; retry once, then surface it.

    Masked secrets (`****abcd`) copied from `get_workflow_code` are kept
    as the stored real keys; send a new unmasked value to replace one.
    """
    user = await authenticate_mcp_request()

    try:
        return await save_workflow_draft_from_code(
            user=user,
            organization_id=user.selected_organization_id,
            workflow_id=workflow_id,
            code=code,
            # Resolved from this module at call time so tests (and callers)
            # patching `save_workflow.parse_code` / `.reconcile_positions`
            # keep working after the lift into services.
            parse_code_fn=parse_code,
            reconcile_positions_fn=reconcile_positions,
        )
    except WorkflowNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
