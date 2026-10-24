"""Copilot proposals: validate SDK code into a reviewable diff, apply a subset.

`create_proposal` (the `propose_changes` tool) parses and validates the
model's SDK TypeScript through the same pipeline a save uses
(`sdk_authoring.parse_and_validate_workflow_code`) but never writes the
workflow. It stores the canonical candidate JSON and its `ChangeItem`s.

`apply_proposal` (the user's "Apply" in the review UI) merges ONLY the
accepted changes into the current working copy, re-validates the result
and saves it as the draft through `sdk_authoring.save_validated_draft`
(the MCP save path), or — for a new workflow — creates it through
`sdk_authoring.create_workflow_from_code` (published as v1, exactly like
the MCP `create_workflow` tool). Publishing an edit stays manual.

M2c: a proposal may also carry settings changes (`settings`, dotted path ->
value, planned by `api/services/copilot/settings.py`; ChangeItems
`target=workflow`, ref_id "s:<path>") and new reusable tools (`new_tools`,
`CreateToolRequest` shapes; ChangeItems `target=tool`, ref_id "t:<name>").
Node code references a new tool as the placeholder "tool:<name>" in
`tool_uuids`; on apply accepted tools are created first (as the acting
user, through the MCP `create_tool` service), placeholders are replaced by
their uuids, accepted settings are merged, and ONE draft save writes the
graph and the settings. If that save fails the created tools are archived.
Graph-only proposals behave exactly as before (same change ids, labels,
order, base_hash and staleness check).

Every lookup is scoped to the caller's organization.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import re
from datetime import UTC, datetime
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, ValidationError

from api.db import db_client
from api.services.configuration.masking import merge_workflow_api_keys
from api.services.copilot.contracts import (
    NEW_TOOL_PLACEHOLDER_PREFIX,
    SETTINGS_REF_PREFIX,
    TOOL_REF_PREFIX,
    ApplyProposalRequest,
    ApplyProposalResponse,
    ChangeItem,
    ChangeOp,
    ChangeTarget,
    Hypothesis,
    Proposal,
    ProposalKind,
    ProposalStatus,
    SettingsPatch,
    ToolContext,
    ToolOutcome,
)
from api.services.copilot.diff import align_ids, compute_changes
from api.services.workflow.dto import sanitize_workflow_definition
from api.services.workflow.layout import reconcile_positions
from api.services.workflow.sdk_authoring import (
    compute_definition_hash,
    create_workflow_from_code,
    parse_and_validate_workflow_code,
    save_validated_draft,
    select_working_copy,
    validate_workflow_payload,
)

_MISSING_NAME_MESSAGE = (
    'Workflow name is required. Add `new Workflow({ name: "..." })` to the source.'
)
_FIX_HINT = (
    "Fix the code and call propose_changes again with the full corrected "
    "source (patches are not accepted)."
)
_SETTINGS_FIX_HINT = (
    "Fix `settings` (paths and allowed values from get_workflow_settings) and "
    "call propose_changes again."
)


# ─── model <-> contract ──────────────────────────────────────────────────


def _value(raw: Any) -> Any:
    return getattr(raw, "value", raw)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _session_uuid(model: Any) -> str:
    direct = getattr(model, "session_uuid", None)
    if direct:
        return str(direct)
    session = getattr(model, "session", None)
    nested = getattr(session, "session_uuid", None) if session is not None else None
    return str(nested) if nested else ""


def _hypotheses(raw: Any) -> dict[str, Hypothesis]:
    out: dict[str, Hypothesis] = {}
    for change_id, value in (raw or {}).items():
        try:
            out[str(change_id)] = Hypothesis.model_validate(value)
        except Exception:  # noqa: BLE001 - skip a malformed stored link
            logger.warning(f"copilot: malformed hypothesis for change {change_id}")
    return out


def proposal_from_model(model: Any, *, session_uuid: str | None = None) -> Proposal:
    """Map a stored copilot proposal row to the `Proposal` contract.

    The row stores `session_id`; the contract carries `session_uuid`, read
    from `model.session_uuid` / `model.session.session_uuid` when the model
    exposes it, or passed explicitly.
    """
    return Proposal(
        proposal_uuid=str(model.proposal_uuid),
        session_uuid=session_uuid or _session_uuid(model),
        kind=ProposalKind(_value(model.kind)),
        workflow_id=model.workflow_id,
        summary=model.summary or "",
        base_hash=model.base_hash,
        settings_base_hash=getattr(model, "settings_base_hash", None),
        changes=[ChangeItem.model_validate(c) for c in (model.changes or [])],
        status=ProposalStatus(_value(model.status)),
        workflow_name=getattr(model, "workflow_name", None),
        created_at=_iso(getattr(model, "created_at", None)),
        iteration_id=getattr(model, "iteration_id", None),
        accepted_change_ids=[
            str(c) for c in (getattr(model, "accepted_change_ids", None) or [])
        ],
        applied_definition_id=getattr(model, "applied_definition_id", None),
        hypotheses=_hypotheses(getattr(model, "hypotheses", None)),
    )


# ─── hypotheses (Self-improve next-calls check, M2b O3) ──────────────────


class ProposalTarget(BaseModel):
    """`propose_changes(targets=[...])`: a Doctor cluster a change fixes.

    `node_name` names the changed node (or "edge:<source>-><target>") the
    target explains; it defaults to `fix_target`.
    """

    model_config = {"extra": "forbid"}

    node_name: Optional[str] = None
    cluster_id: Optional[str] = None
    title: str
    category: Optional[str] = None
    fix_target: Optional[str] = None

    def match_key(self) -> str:
        return _norm_name(self.node_name or self.fix_target or "")

    def hypothesis(self) -> Hypothesis:
        return Hypothesis(
            cluster_id=self.cluster_id,
            title=self.title,
            category=self.category,
            fix_target=self.fix_target or self.node_name,
        )


def _norm_name(text: str) -> str:
    text = " ".join(str(text).split()).strip().strip("\"'").lower()
    return re.sub(r"\s*->\s*", "->", text)


def _node_names(*workflows: dict | None) -> dict[str, str]:
    names: dict[str, str] = {}
    for workflow in workflows:
        for node in (workflow or {}).get("nodes") or []:
            name = (node.get("data") or {}).get("name")
            if name:
                names[str(node.get("id"))] = str(name)
    return names


def _change_keys(change: ChangeItem, names: dict[str, str]) -> set[str]:
    """Names a target may use to point at this change."""
    view = change.after or change.before or {}
    if change.target == ChangeTarget.node:
        name = (view.get("data") or {}).get("name") or names.get(change.ref_id)
        return {_norm_name(name)} if name else set()
    if change.target == ChangeTarget.edge:
        source = names.get(str(view.get("source")), str(view.get("source")))
        target = names.get(str(view.get("target")), str(view.get("target")))
        keys = {_norm_name(f"edge:{source}->{target}")}
        label = (view.get("data") or {}).get("label")
        if label:
            keys.add(_norm_name(f"edge:{label}"))
        return keys
    return {"workflow"}


def map_hypotheses(
    changes: list[ChangeItem],
    targets: list[ProposalTarget],
    names: dict[str, str],
) -> tuple[dict[str, Hypothesis], list[ProposalTarget]]:
    """change_id -> Hypothesis for each change a target names (first match
    wins), plus the targets that matched no change."""
    hypotheses: dict[str, Hypothesis] = {}
    used: set[int] = set()
    for change in changes:
        keys = _change_keys(change, names)
        for index, target in enumerate(targets):
            if target.match_key() and target.match_key() in keys:
                hypotheses[change.change_id] = target.hypothesis()
                used.add(index)
                break
    unmatched = [t for i, t in enumerate(targets) if i not in used]
    return hypotheses, unmatched


# ─── create (propose_changes tool) ───────────────────────────────────────


def _rejected(
    error_code: str, error: str, *, errors: Any = None, hint: str = _FIX_HINT
) -> ToolOutcome:
    content: dict[str, Any] = {"error_code": error_code, "error": error}
    if errors:
        content["errors"] = errors
    if hint:
        content["hint"] = hint
    return ToolOutcome(
        ok=False, content=content, summary=f"Proposal rejected ({error_code})"
    )


def _rejected_from_save_error(error: dict[str, Any]) -> ToolOutcome:
    extra = {
        k: v for k, v in error.items() if k not in ("saved", "error_code", "error")
    }
    return _rejected(
        error["error_code"], error.get("error", ""), errors=extra.get("errors")
    )


def _model_view(
    proposal: Proposal, unmatched: list[ProposalTarget] | None = None
) -> dict[str, Any]:
    """What the model sees: labels and ids only (no node data, no secrets)."""
    changes = []
    for c in proposal.changes:
        item: dict[str, Any] = {
            "change_id": c.change_id,
            "label": c.label,
            "depends_on": c.depends_on,
        }
        if c.change_id in proposal.hypotheses:
            item["fixes"] = proposal.hypotheses[c.change_id].title
        changes.append(item)
    view: dict[str, Any] = {
        "proposal_uuid": proposal.proposal_uuid,
        "kind": proposal.kind.value,
        "summary": proposal.summary,
        "workflow_name": proposal.workflow_name,
        "changes": changes,
        "next": (
            "The proposal is shown to the user for review. Nothing is saved "
            "until they apply it. Briefly tell them what you proposed and wait."
        ),
    }
    if unmatched:
        view["unmatched_targets"] = [t.model_dump(exclude_none=True) for t in unmatched]
        view["targets_note"] = (
            "These targets matched no changed node (match is on node_name, or "
            "fix_target when node_name is omitted). The proposal is stored; "
            "mention the unlinked problems in your summary."
        )
    return view


# ─── settings and new tools (M2c) ────────────────────────────────────────


class _ProposalRejected(Exception):
    """Internal: a proposal part failed; carries the outcome for the model."""

    def __init__(self, outcome: ToolOutcome):
        super().__init__(str(outcome.content))
        self.outcome = outcome


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _settings_service() -> Any:
    # Imported lazily: settings.py pulls in the model-configuration stack.
    from api.services.copilot import settings as settings_service

    return settings_service


def _settings_errors(exc: Exception) -> list[str]:
    return [str(e) for e in (getattr(exc, "errors", None) or [str(exc)])]


async def working_settings(workflow: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """The working copy's raw (workflow_configurations,
    template_context_variables), WITH secrets: server-side only."""
    return await _maybe_await(_settings_service().load_working_settings(workflow))


async def _validate_model_override(
    configurations: dict[str, Any], organization_id: int, user: Any
) -> list[str]:
    """The live key check `PUT /workflow/{id}` runs when the model override
    changes (UserConfigurationValidator). Returns error messages."""
    from api.schemas.ai_model_configuration import OrganizationAIModelConfigurationV2
    from api.services.configuration.ai_model_configuration import (
        WORKFLOW_MODEL_CONFIGURATION_V2_OVERRIDE_KEY,
        check_for_masked_keys_in_ai_model_configuration_v2,
        compile_ai_model_configuration_v2,
    )
    from api.services.configuration.check_validity import UserConfigurationValidator

    raw = (configurations or {}).get(WORKFLOW_MODEL_CONFIGURATION_V2_OVERRIDE_KEY)
    if not raw:
        return []  # override cleared: the organization default applies
    try:
        override = OrganizationAIModelConfigurationV2.model_validate(raw)
        check_for_masked_keys_in_ai_model_configuration_v2(override)
        await UserConfigurationValidator().validate(
            compile_ai_model_configuration_v2(override),
            organization_id=organization_id,
            created_by=getattr(user, "provider_id", None),
        )
    except (ValidationError, ValueError) as e:
        return [str(e)]
    return []


def is_settings_change(change: ChangeItem) -> bool:
    return change.ref_id.startswith(SETTINGS_REF_PREFIX)


def is_graph_change(change: ChangeItem) -> bool:
    """Node, edge and workflow-name changes (what `code` produced)."""
    return change.target != ChangeTarget.tool and not is_settings_change(change)


async def _plan_settings(
    *, workflow: Any, organization_id: int, settings: dict[str, Any]
) -> tuple[list[ChangeItem], dict[str, Any] | None, str | None]:
    """(settings changes, settings_candidate, settings_base_hash)."""
    service = _settings_service()
    try:
        patch = SettingsPatch(values=dict(settings))
    except ValidationError as e:
        raise _ProposalRejected(
            _rejected(
                "validation_error",
                "`settings` must be an object of dotted path -> value.",
                errors=e.errors(include_url=False),
                hint=_SETTINGS_FIX_HINT,
            )
        ) from e
    configurations, template_variables = await working_settings(workflow)
    base_hash = service.settings_hash(configurations, template_variables)
    try:
        plan = await _maybe_await(
            service.plan_settings_changes(
                workflow=workflow,
                organization_id=organization_id,
                patch=patch,
                working_configurations=copy.deepcopy(configurations),
                working_template_variables=copy.deepcopy(template_variables),
            )
        )
    except service.SettingsValidationError as e:
        errors = _settings_errors(e)
        raise _ProposalRejected(
            _rejected(
                "validation_error",
                "The settings are not valid: " + "; ".join(errors),
                errors=errors,
                hint=_SETTINGS_FIX_HINT,
            )
        ) from e
    changes = [
        c if isinstance(c, ChangeItem) else ChangeItem.model_validate(c)
        for c in plan.changes or []
    ]
    if not changes:
        return [], None, None
    candidate = {
        "configurations": plan.candidate_configurations,
        "template_variables": plan.candidate_template_variables,
    }
    return changes, candidate, base_hash


#: http_api header names that carry secrets; those belong in a credential.
_SECRET_HEADER = re.compile(
    r"authorization|api[-_]?key|token|secret|password|passwd|cookie|"
    r"x-auth|bearer|signature",
    re.IGNORECASE,
)


def _secret_problems(request: Any) -> list[str]:
    definition = request.definition.model_dump()
    config = definition.get("config") or {}
    headers = config.get("headers") or {}
    problems = [
        f'Header "{name}" looks like a secret. Remove it and reference an '
        "existing credential_uuid from list_credentials instead (users create "
        "credentials in the UI)."
        for name in headers
        if _SECRET_HEADER.search(str(name))
    ]
    url = str(config.get("url") or "")
    if re.search(r"://[^/@\s]+:[^/@\s]+@", url) or re.search(
        r"[?&](api[-_]?key|token|secret|password|access_token)=", url, re.IGNORECASE
    ):
        problems.append(
            "The URL embeds credentials. Use a credential_uuid from "
            "list_credentials instead."
        )
    return problems


def _tool_label(request: Any) -> str:
    definition = request.definition.model_dump()
    config = definition.get("config") or {}
    category = str(request.category)
    if category == "http_api":
        method = str(config.get("method") or "").upper()
        detail = f"HTTP {method} {config.get('url') or ''}".strip()
    elif category == "mcp":
        detail = f"MCP {config.get('url') or ''}".strip()
    else:
        detail = category.replace("_", " ")
    return f"Create tool: {request.name} ({detail})"


def tool_change_id(name: str) -> str:
    return f"{TOOL_REF_PREFIX}{name}"


async def _plan_new_tools(
    raw_tools: list[Any], organization_id: int
) -> tuple[list[Any], list[ChangeItem]]:
    """Validate `new_tools` like the MCP `create_tool` (schema, credential and
    destination ownership) plus unique names; nothing is created here."""
    from api.schemas.tool import CreateToolRequest
    from api.services.tool_management import (
        ToolManagementError,
        validate_tool_references,
    )

    errors: list[str] = []
    requests: list[Any] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_tools):
        label = f"new_tools[{index}]"
        if not isinstance(raw, dict):
            errors.append(f"{label}: must be an object (create_tool request shape).")
            continue
        try:
            request = CreateToolRequest.model_validate(raw)
        except ValidationError as e:
            name = raw.get("name")
            errors.append(
                f"{label}{f' ({name!r})' if name else ''}: "
                f"{e.errors(include_url=False)}"
            )
            continue
        name = request.name
        if not name.strip() or name != name.strip():
            errors.append(f"{label}: `name` must be non-empty without outer spaces.")
            continue
        if name.lower() in seen:
            errors.append(f'Duplicate new tool name "{name}".')
            continue
        seen.add(name.lower())
        errors.extend(f'Tool "{name}": {p}' for p in _secret_problems(request))
        requests.append(request)

    if not errors and requests:
        existing = await db_client.get_tools_for_organization(organization_id)
        taken = {
            str(t.name).strip().lower(): t.name
            for t in existing
            if _value(getattr(t, "status", "active")) != "archived"
        }
        for request in requests:
            if request.name.lower() in taken:
                errors.append(
                    f'A tool named "{taken[request.name.lower()]}" already exists. '
                    "Attach it by its uuid from list_tools, or choose another name."
                )
                continue
            try:
                await validate_tool_references(
                    request.definition.model_dump(), organization_id=organization_id
                )
            except ToolManagementError as e:
                errors.append(f'Tool "{request.name}": {e.message}')

    if errors:
        raise _ProposalRejected(
            _rejected(
                "validation_error",
                "The new tools are not valid: " + "; ".join(errors),
                errors=errors,
                hint=(
                    "Fix `new_tools` (create_tool request shape; credentials "
                    "only by credential_uuid from list_credentials) and call "
                    "propose_changes again."
                ),
            )
        )

    changes = [
        ChangeItem(
            change_id=tool_change_id(r.name),
            target=ChangeTarget.tool,
            op=ChangeOp.add,
            ref_id=tool_change_id(r.name),
            label=_tool_label(r),
            after=r.model_dump(mode="json"),
        )
        for r in requests
    ]
    return requests, changes


def _placeholders(workflow: dict | None) -> dict[str, list[str]]:
    """node id -> new-tool names referenced as "tool:<name>" in tool_uuids."""
    found: dict[str, list[str]] = {}
    for node in (workflow or {}).get("nodes") or []:
        data = node.get("data") or {}
        for ref in data.get("tool_uuids") or []:
            if isinstance(ref, str) and ref.startswith(NEW_TOOL_PLACEHOLDER_PREFIX):
                names = found.setdefault(str(node.get("id")), [])
                name = ref[len(NEW_TOOL_PLACEHOLDER_PREFIX) :]
                if name not in names:
                    names.append(name)
    return found


def _link_placeholders(
    graph_changes: list[ChangeItem],
    candidate: dict | None,
    tool_names: set[str],
) -> None:
    """Node changes using "tool:<name>" depend on "t:<name>"; an unknown
    placeholder rejects the proposal."""
    used = _placeholders(candidate)
    names = _node_names(candidate)
    unknown = [
        f'"{NEW_TOOL_PLACEHOLDER_PREFIX}{name}" in node "{names.get(nid, nid)}"'
        for nid, refs in used.items()
        for name in refs
        if name not in tool_names
    ]
    if unknown:
        raise _ProposalRejected(
            _rejected(
                "validation_error",
                "Unknown tool placeholder: "
                + ", ".join(unknown)
                + '. A "tool:<name>" placeholder must name a tool in `new_tools`; '
                "attach existing tools by their uuid from list_tools.",
            )
        )
    for change in graph_changes:
        if change.target != ChangeTarget.node or change.op == ChangeOp.remove:
            continue
        for name in used.get(change.ref_id, []):
            dep = tool_change_id(name)
            if dep not in change.depends_on:
                change.depends_on.append(dep)


def substitute_tool_placeholders(
    workflow: dict, tool_uuids: dict[str, str]
) -> dict[str, Any]:
    """Replace "tool:<name>" in node tool_uuids with the created tools' uuids."""
    out = copy.deepcopy(workflow)
    for node in out.get("nodes") or []:
        data = node.get("data") or {}
        refs = data.get("tool_uuids")
        if not isinstance(refs, list):
            continue
        replaced: list[Any] = []
        for ref in refs:
            if isinstance(ref, str) and ref.startswith(NEW_TOOL_PLACEHOLDER_PREFIX):
                name = ref[len(NEW_TOOL_PLACEHOLDER_PREFIX) :]
                if name not in tool_uuids:
                    raise ValueError(
                        f'Node "{data.get("name") or node.get("id")}" uses the new '
                        f'tool "{name}", which is not being created.'
                    )
                ref = tool_uuids[name]
            if ref not in replaced:
                replaced.append(ref)
        data["tool_uuids"] = replaced
    return out


async def create_proposal(
    *,
    code: str | None,
    summary: str,
    ctx: ToolContext,
    targets: list[ProposalTarget] | None = None,
    settings: dict[str, Any] | None = None,
    new_tools: list[Any] | None = None,
) -> ToolOutcome:
    """Validate SDK code / settings / new tools into a stored, reviewable
    proposal (no workflow write).

    edit (ctx.workflow_id set): diff against the working copy
    (draft > published > legacy). create: every node/edge is an `add`.
    Validation failures return ok=False with the MCP save/create
    `error_code` and text so the model can fix and re-propose.

    `targets` link changes to Doctor clusters (`Proposal.hypotheses`, see
    `map_hypotheses`). In self_improve mode the proposal is stored with
    `ctx.iteration_id`; a second proposal for the round is `already_proposed`.

    M2c: `settings` (dotted path -> value) and `new_tools` (CreateToolRequest
    shapes) need an existing workflow; `new_tools` is refused in
    self_improve. Changes are ordered nodes, edges, workflow name, settings,
    tools. With `code` only, the result is exactly the pre-M2c proposal.
    """
    targets = list(targets or [])
    settings = dict(settings or {})
    raw_tools = list(new_tools or [])
    self_improve = ctx.mode == "self_improve"
    if self_improve and ctx.iteration_id is None:
        return _rejected(
            "validation_error", "This Self-improve round has no iteration.", hint=""
        )
    if code is not None and not isinstance(code, str):
        return _rejected("validation_error", "`code` must be non-empty SDK source.")
    has_code = isinstance(code, str) and bool(code.strip())
    if not has_code and not settings and not raw_tools:
        return _rejected("validation_error", "`code` must be non-empty SDK source.")
    if raw_tools and self_improve:
        return _rejected(
            "not_allowed",
            "New tools cannot be created in a Self-improve round (no human is "
            "present to set up credentials). Mention the missing tool in your "
            "summary instead.",
            hint="",
        )
    if (settings or raw_tools) and ctx.workflow_id is None:
        return _rejected(
            "not_allowed",
            "Settings and new tools need an existing workflow. Propose the new "
            "workflow's code first; after the user creates it, this chat "
            "continues on the new agent.",
            hint="",
        )
    summary = (summary or "").strip() or "Proposed changes"

    session = await db_client.get_copilot_session(ctx.session_uuid, ctx.organization_id)
    if session is None:
        return _rejected("not_found", "Copilot session not found.", hint="")

    extra: dict[str, Any] = {}
    stored_code = code if has_code else ""
    if ctx.workflow_id is not None:
        workflow = await db_client.get_workflow(
            ctx.workflow_id, organization_id=ctx.organization_id
        )
        if workflow is None:
            return _rejected(
                "not_found", f"Workflow {ctx.workflow_id} not found.", hint=""
            )
        source = await select_working_copy(workflow)
        working = source.payload

        graph_changes: list[ChangeItem] = []
        candidate: dict[str, Any] = {}
        base_hash: str | None = None
        workflow_name = workflow.name
        names = _node_names(working)
        if has_code:
            validated = await parse_and_validate_workflow_code(
                code=code,
                organization_id=ctx.organization_id,
                previous_json=working,
                exclude_workflow_id=workflow.id,
            )
            if not validated.ok:
                return _rejected_from_save_error(validated.error)

            # Canonical candidate, as the save path would store it: ids aligned
            # with the working copy, then masked secrets restored.
            candidate = align_ids(validated.payload, working)
            candidate = merge_workflow_api_keys(candidate, working)
            # Diff against the sanitized working copy: UI-only data a canvas save
            # left behind (e.g. `invalid`) is not a change the model made.
            graph_changes = compute_changes(
                sanitize_workflow_definition(working),
                candidate,
                before_name=workflow.name or "",
                after_name=validated.name or None,
            )
            if not graph_changes and not settings and not raw_tools:
                return _rejected(
                    "no_changes",
                    "The code produces no changes to the current workflow.",
                    hint="Edit the code from get_workflow_code, then propose again.",
                )
            base_hash = compute_definition_hash(working)
            workflow_name = validated.name or workflow.name
            names = _node_names(working, candidate)

        try:
            tool_requests, tool_changes = (
                await _plan_new_tools(raw_tools, ctx.organization_id)
                if raw_tools
                else ([], [])
            )
            if has_code:
                _link_placeholders(
                    graph_changes, candidate, {r.name for r in tool_requests}
                )
            settings_changes, settings_candidate, settings_base_hash = (
                await _plan_settings(
                    workflow=workflow,
                    organization_id=ctx.organization_id,
                    settings=settings,
                )
                if settings
                else ([], None, None)
            )
        except _ProposalRejected as rejected:
            return rejected.outcome

        if not graph_changes:
            # Settings / tools only: the graph is not part of this proposal
            # (no graph staleness check, the save leaves workflow_json alone).
            candidate, base_hash, stored_code = {}, None, ""
            workflow_name = workflow.name
        changes = graph_changes + settings_changes + tool_changes
        if not changes:
            return _rejected(
                "no_changes",
                "The proposal produces no changes: the code, settings and tools "
                "match the current workflow.",
                hint=(
                    "Read get_workflow_code / get_workflow_settings, then "
                    "propose again."
                ),
            )
        if settings_changes:
            extra["settings_candidate"] = settings_candidate
            extra["settings_base_hash"] = settings_base_hash
        if tool_requests:
            extra["new_tools"] = [r.model_dump(mode="json") for r in tool_requests]
        kind = ProposalKind.edit
        workflow_id = workflow.id
    else:
        validated = await parse_and_validate_workflow_code(
            code=code,
            organization_id=ctx.organization_id,
            previous_json=None,
        )
        if not validated.ok:
            return _rejected_from_save_error(validated.error)
        if not validated.name:
            return _rejected("missing_name", _MISSING_NAME_MESSAGE)
        candidate = validated.payload
        changes = compute_changes(None, candidate)
        try:
            # A new workflow cannot create tools: any placeholder is unknown.
            _link_placeholders(changes, candidate, set())
        except _ProposalRejected as rejected:
            return rejected.outcome
        kind = ProposalKind.create
        base_hash = None
        workflow_id = None
        workflow_name = validated.name
        names = _node_names(candidate)

    hypotheses, unmatched = map_hypotheses(changes, targets, names)
    if hypotheses:
        extra["hypotheses"] = hypotheses
    if self_improve:
        extra["iteration_id"] = ctx.iteration_id
    try:
        model = await db_client.create_copilot_proposal(
            session_id=session.id,
            organization_id=ctx.organization_id,
            workflow_id=workflow_id,
            kind=kind.value,
            summary=summary,
            base_hash=base_hash,
            candidate_json=candidate,
            code=stored_code,
            changes=[c.model_dump(mode="json") for c in changes],
            workflow_name=workflow_name,
            **extra,
        )
    except ValueError as e:
        if self_improve and "already has a proposal" in str(e):
            return _rejected(
                "already_proposed",
                "This Self-improve round already has its proposal. Do not "
                "propose again; end the round with your summary.",
                hint="",
            )
        raise
    proposal = proposal_from_model(model, session_uuid=ctx.session_uuid)
    noun = "change" if len(changes) == 1 else "changes"
    return ToolOutcome(
        ok=True,
        content=_model_view(proposal, unmatched),
        summary=f"Proposed {len(changes)} {noun}",
        proposal=proposal,
    )


# ─── apply ───────────────────────────────────────────────────────────────


def _failure(
    error_code: str,
    error: str,
    *,
    proposal: Proposal | None = None,
    workflow_id: int | None = None,
) -> ApplyProposalResponse:
    return ApplyProposalResponse(
        saved=False,
        workflow_id=workflow_id,
        error_code=error_code,
        error=error,
        proposal=proposal,
    )


def _missing_dependencies(
    changes: dict[str, ChangeItem], accepted: set[str]
) -> list[ChangeItem]:
    """Changes that must also be accepted (transitive closure)."""
    missing: dict[str, ChangeItem] = {}
    stack = list(accepted)
    seen = set(accepted)
    while stack:
        item = changes[stack.pop()]
        for dep in item.depends_on:
            if dep in seen or dep not in changes:
                continue
            seen.add(dep)
            missing[dep] = changes[dep]
            stack.append(dep)
    return list(missing.values())


def _edge_key(edge: dict) -> tuple:
    return (
        edge.get("source"),
        edge.get("target"),
        str((edge.get("data") or {}).get("label") or "").strip(),
    )


def merge_accepted_changes(
    working: dict | None,
    candidate: dict,
    changes: list[ChangeItem],
) -> dict[str, Any]:
    """Working copy + ONLY `changes` (already filtered to the accepted ones).

    Node updates take the candidate's type and data but keep the working
    node's position and other UI fields; added nodes are placed with
    `reconcile_positions`; every untouched node keeps its position.
    """
    merged = copy.deepcopy(working or {"nodes": [], "edges": []})
    merged.setdefault("nodes", [])
    merged.setdefault("edges", [])
    cand_nodes = {str(n.get("id")): n for n in candidate.get("nodes") or []}
    cand_edges = {str(e.get("id")): e for e in candidate.get("edges") or []}

    added_node_ids: set[str] = set()
    for change in changes:
        ref = change.ref_id
        if change.target == ChangeTarget.node:
            if change.op == ChangeOp.remove:
                merged["nodes"] = [
                    n for n in merged["nodes"] if str(n.get("id")) != ref
                ]
                continue
            cand = cand_nodes.get(ref)
            if cand is None:
                raise ValueError(f"Proposal is missing node {ref}")
            if change.op == ChangeOp.add:
                node = copy.deepcopy(cand)
                node["position"] = {"x": 0.0, "y": 0.0}
                merged["nodes"].append(node)
                added_node_ids.add(ref)
            else:
                for node in merged["nodes"]:
                    if str(node.get("id")) == ref:
                        node["type"] = cand.get("type")
                        node["data"] = copy.deepcopy(cand.get("data") or {})
                        break
                else:
                    raise ValueError(f"Node {ref} is no longer in the workflow")
        elif change.target == ChangeTarget.edge:
            if change.op == ChangeOp.remove:
                merged["edges"] = [
                    e for e in merged["edges"] if str(e.get("id")) != ref
                ]
                continue
            after = change.after or {}
            cand = cand_edges.get(ref)
            if cand is None or _edge_key(cand) != _edge_key(after):
                cand = next(
                    (
                        e
                        for e in candidate.get("edges") or []
                        if _edge_key(e) == _edge_key(after)
                    ),
                    None,
                )
            if cand is None:
                raise ValueError(f"Proposal is missing edge {ref}")
            if change.op == ChangeOp.add:
                merged["edges"].append(copy.deepcopy(cand))
            else:
                for edge in merged["edges"]:
                    if str(edge.get("id")) == ref:
                        edge["source"] = cand.get("source")
                        edge["target"] = cand.get("target")
                        edge["data"] = copy.deepcopy(cand.get("data") or {})
                        break
                else:
                    raise ValueError(f"Edge {ref} is no longer in the workflow")
        # Workflow-level changes (the name) are applied by the caller.

    if added_node_ids:
        kept = {
            str(n.get("id")): copy.deepcopy(n.get("position"))
            for n in merged["nodes"]
            if str(n.get("id")) not in added_node_ids
        }
        merged = reconcile_positions(merged, working)
        for node in merged["nodes"]:
            nid = str(node.get("id"))
            if nid in kept and kept[nid] is not None:
                node["position"] = kept[nid]
    return merged


async def _mark(proposal_uuid: str, organization_id: int, **fields: Any) -> Any:
    return await db_client.update_copilot_proposal(
        proposal_uuid, organization_id, **fields
    )


async def apply_proposal(
    *,
    proposal_uuid: str,
    organization_id: int,
    user: Any,
    request: ApplyProposalRequest,
) -> ApplyProposalResponse:
    """Apply the accepted changes of a pending proposal.

    edit: merge into the working copy and save as the draft (optimistic
    lock via `request.expected_draft_version`). create: create the
    workflow from the stored code (published as v1, like MCP create).
    """
    model = await db_client.get_copilot_proposal(proposal_uuid, organization_id)
    if model is None:
        return _failure("not_found", "Proposal not found.")
    proposal = proposal_from_model(model)

    if proposal.status == ProposalStatus.applying:
        return _failure(
            "already_applied",
            "This proposal is already being applied.",
            proposal=proposal,
            workflow_id=proposal.workflow_id,
        )
    if proposal.status != ProposalStatus.pending:
        code = {
            ProposalStatus.applied: "already_applied",
            ProposalStatus.discarded: "discarded",
            ProposalStatus.stale: "stale",
        }[proposal.status]
        return _failure(
            code,
            f"This proposal is {proposal.status.value}; ask the copilot again.",
            proposal=proposal,
            workflow_id=proposal.workflow_id,
        )

    # Claim the proposal before saving (pending -> applying, M2b O2): the
    # status check above and the save are not atomic, so without this two
    # concurrent applies could both save. `applying` is not "reviewed".
    claimed = await db_client.claim_copilot_proposal(
        proposal.proposal_uuid,
        organization_id,
        from_status=ProposalStatus.pending.value,
        to_status=ProposalStatus.applying.value,
    )
    if not claimed:
        return _failure(
            "already_applied",
            "This proposal is already being applied.",
            proposal=proposal,
            workflow_id=proposal.workflow_id,
        )

    response: ApplyProposalResponse | None = None
    try:
        if proposal.kind == ProposalKind.create:
            response = await _apply_create(model, proposal, organization_id, user)
        else:
            response = await _apply_edit(
                model, proposal, organization_id, request, user
            )
    finally:
        # Release the claim when nothing was saved (stale keeps its own status).
        if response is None or (not response.saved and response.error_code != "stale"):
            await db_client.claim_copilot_proposal(
                proposal.proposal_uuid,
                organization_id,
                from_status=ProposalStatus.applying.value,
                to_status=ProposalStatus.pending.value,
            )
    if response.saved or response.error_code == "stale":
        # Committed applied / stale: a Self-improve round is now reviewed.
        await _mark_round_reviewed(proposal, organization_id)
    return response


async def _mark_round_reviewed(proposal: Proposal, organization_id: int) -> None:
    """D4: a round's proposal applied / discarded / stale starts the next
    window. Best effort: the proposal status already records the review."""
    if proposal.iteration_id is None or proposal.workflow_id is None:
        return
    try:
        await db_client.mark_reviewed(
            workflow_id=proposal.workflow_id, organization_id=organization_id
        )
    except Exception:  # noqa: BLE001 - never fail the user's apply/discard
        logger.exception(
            f"copilot: mark_reviewed failed for proposal {proposal.proposal_uuid}"
        )


async def _apply_create(
    model: Any, proposal: Proposal, organization_id: int, user: Any
) -> ApplyProposalResponse:
    result = await create_workflow_from_code(
        user=user,
        organization_id=organization_id,
        code=model.code,
        event_source="copilot",
    )
    if not result.get("created"):
        return _failure(
            result.get("error_code") or "validation_error",
            result.get("error") or "Could not create the workflow.",
            proposal=proposal,
        )

    workflow_id = result["workflow_id"]
    draft_hash = None
    definition_id = None
    workflow = await db_client.get_workflow(
        workflow_id, organization_id=organization_id
    )
    if workflow is not None:
        draft_hash = compute_definition_hash(
            (await select_working_copy(workflow)).payload
        )
        definition_id = getattr(
            getattr(workflow, "released_definition", None), "id", None
        )
    fields: dict[str, Any] = {
        "status": ProposalStatus.applied.value,
        "applied_at": datetime.now(UTC),
        "workflow_id": workflow_id,
        # create is all-or-nothing
        "accepted_change_ids": [c.change_id for c in proposal.changes],
    }
    if isinstance(definition_id, int):
        fields["applied_definition_id"] = definition_id
    updated = await _mark(proposal.proposal_uuid, organization_id, **fields)
    await _carry_build_context(proposal, organization_id, workflow_id)
    return ApplyProposalResponse(
        saved=True,
        workflow_id=workflow_id,
        draft_hash=draft_hash,
        proposal=_after_update(updated, proposal, ProposalStatus.applied, workflow_id),
    )


# ─── build -> agent context transfer (M2c A) ─────────────────────────────

#: Background brief drafts (strong refs so they are not garbage-collected).
_BACKGROUND_TASKS: set[asyncio.Task] = set()


async def _carry_build_context(
    proposal: Proposal, organization_id: int, workflow_id: int
) -> None:
    """After "Create workflow": rebind the build session to the new workflow
    (its Copilot resumes the same chat, now in edit mode) and draft the
    brief from that chat in the background (a pending BriefChange the user
    confirms). Never fails the create."""
    if not proposal.session_uuid:
        return
    try:
        rebound = await db_client.update_copilot_session(
            proposal.session_uuid, organization_id, workflow_id=workflow_id
        )
    except Exception:  # noqa: BLE001 - the workflow exists; keep the create
        logger.exception(
            f"copilot: could not rebind session {proposal.session_uuid} to "
            f"workflow {workflow_id}"
        )
        return
    if rebound is None:
        return
    task = asyncio.create_task(
        _draft_brief_for_session(proposal.session_uuid, organization_id, workflow_id)
    )
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


async def _draft_brief_for_session(
    session_uuid: str, organization_id: int, workflow_id: int
) -> Any:
    """Best effort: one LLM call over the build chat -> pending BriefChange."""
    from api.services.copilot import brief as brief_service

    try:
        return await brief_service.draft_brief_from_chats(
            workflow_id=workflow_id,
            organization_id=organization_id,
            session_uuid=session_uuid,
        )
    except brief_service.BriefDraftUnavailable as e:
        logger.info(
            f"copilot: no brief drafted for workflow {workflow_id} after create "
            f"({e.reason})"
        )
    except Exception:  # noqa: BLE001 - never surfaces; the user can draft later
        logger.exception(
            f"copilot: brief draft after create failed for workflow {workflow_id}"
        )
    return None


async def wait_for_background_tasks() -> None:
    """Await pending background brief drafts (tests, graceful shutdown)."""
    while _BACKGROUND_TASKS:
        await asyncio.gather(*list(_BACKGROUND_TASKS), return_exceptions=True)


def _after_update(
    updated: Any,
    proposal: Proposal,
    status: ProposalStatus,
    workflow_id: int | None = None,
) -> Proposal:
    if updated is not None:
        try:
            return proposal_from_model(
                updated, session_uuid=proposal.session_uuid or None
            )
        except Exception:  # noqa: BLE001 - fall back to the known state
            logger.warning("copilot: could not map updated proposal row")
    changes: dict[str, Any] = {"status": status}
    if workflow_id is not None:
        changes["workflow_id"] = workflow_id
    return proposal.model_copy(update=changes)


async def _stale(
    proposal: Proposal, organization_id: int, workflow_id: int | None
) -> ApplyProposalResponse:
    updated = await _mark(
        proposal.proposal_uuid, organization_id, status=ProposalStatus.stale.value
    )
    return _failure(
        "stale",
        "The workflow changed since this proposal was made. Ask the "
        "copilot to propose the changes again.",
        proposal=_after_update(updated, proposal, ProposalStatus.stale),
        workflow_id=workflow_id,
    )


def _apply_user(user: Any, organization_id: int) -> Any:
    """The applying user pinned to the proposal's organization (the tool
    service scopes by `selected_organization_id`)."""
    if getattr(user, "selected_organization_id", None) == organization_id:
        return user
    from api.services.copilot.tools import _OrgScopedUser

    return _OrgScopedUser(user, organization_id)


async def _create_accepted_tools(
    model: Any,
    selected_tools: list[ChangeItem],
    organization_id: int,
    user: Any,
) -> dict[str, str]:
    """Create the accepted new tools through the MCP `create_tool` service,
    as the applying user. Returns name -> tool_uuid. On a failure the tools
    created so far are archived and the error is re-raised."""
    from api.schemas.tool import CreateToolRequest
    from api.services.tool_management import create_tool_for_user

    stored = {
        str(raw.get("name")): raw
        for raw in (getattr(model, "new_tools", None) or [])
        if isinstance(raw, dict)
    }
    acting = _apply_user(user, organization_id)
    created: dict[str, str] = {}
    try:
        for change in selected_tools:
            name = change.ref_id[len(TOOL_REF_PREFIX) :]
            raw = stored.get(name)
            if raw is None:
                raise ValueError(f'Proposal is missing the new tool "{name}"')
            request = CreateToolRequest.model_validate(raw)
            tool = await create_tool_for_user(request, acting, source="copilot")
            created[name] = tool.tool_uuid
    except BaseException:
        await _archive_tools(created, organization_id)
        raise
    return created


async def _archive_tools(created: dict[str, str], organization_id: int) -> None:
    """Best effort: never leave tools from a failed apply behind."""
    for name, tool_uuid in created.items():
        try:
            await db_client.archive_tool(tool_uuid, organization_id)
        except Exception:  # noqa: BLE001 - log and keep archiving the rest
            logger.exception(
                f"copilot: could not archive tool {name} ({tool_uuid}) after a "
                "failed apply"
            )


async def _apply_edit(
    model: Any,
    proposal: Proposal,
    organization_id: int,
    request: ApplyProposalRequest,
    user: Any = None,
) -> ApplyProposalResponse:
    workflow_id = proposal.workflow_id
    workflow = (
        await db_client.get_workflow(workflow_id, organization_id=organization_id)
        if workflow_id is not None
        else None
    )
    if workflow is None:
        return _failure("not_found", "Workflow not found.", proposal=proposal)

    source = await select_working_copy(workflow)
    working = source.payload
    # Graph staleness exactly as before; settings/tools-only proposals carry
    # no base_hash (they do not touch the graph).
    has_graph = proposal.base_hash is not None or not any(
        not is_graph_change(c) for c in proposal.changes
    )
    if has_graph and compute_definition_hash(working) != proposal.base_hash:
        return await _stale(proposal, organization_id, workflow_id)
    current_settings: tuple[dict[str, Any], dict[str, Any]] | None = None
    if proposal.settings_base_hash is not None:
        current_settings = await working_settings(workflow)
        if (
            _settings_service().settings_hash(*current_settings)
            != proposal.settings_base_hash
        ):
            return await _stale(proposal, organization_id, workflow_id)

    by_id = {c.change_id: c for c in proposal.changes}
    accepted = list(dict.fromkeys(request.accepted_change_ids))
    if not accepted:
        return _failure(
            "validation_error",
            "Select at least one change to apply.",
            proposal=proposal,
            workflow_id=workflow_id,
        )
    unknown = [cid for cid in accepted if cid not in by_id]
    if unknown:
        return _failure(
            "validation_error",
            "Unknown change ids: " + ", ".join(unknown),
            proposal=proposal,
            workflow_id=workflow_id,
        )
    missing = _missing_dependencies(by_id, set(accepted))
    if missing:
        return _failure(
            "missing_dependency",
            "These changes must also be accepted: "
            + "; ".join(f"{c.change_id} ({c.label})" for c in missing),
            proposal=proposal,
            workflow_id=workflow_id,
        )

    # Apply in the proposal's order (nodes, edges, workflow, settings, tools).
    accepted_set = set(accepted)
    selected = [c for c in proposal.changes if c.change_id in accepted_set]
    selected_graph = [c for c in selected if is_graph_change(c)]
    selected_settings = [c for c in selected if is_settings_change(c)]
    selected_tools = [c for c in selected if c.target == ChangeTarget.tool]

    # 1. Graph: merge and validate (new-tool placeholders are plain strings to
    # the validators; they are re-validated with real uuids below).
    merged: dict[str, Any] | None = None
    if selected_graph:
        try:
            merged = merge_accepted_changes(
                working, model.candidate_json or {}, selected_graph
            )
        except ValueError as e:
            return _failure(
                "validation_error", str(e), proposal=proposal, workflow_id=workflow_id
            )
        validated = await validate_workflow_payload(
            merged,
            organization_id=organization_id,
            exclude_workflow_id=workflow_id,
        )
        if not validated.ok:
            return _failure(
                validated.error["error_code"],
                validated.error.get("error") or "The selected changes are not valid.",
                proposal=proposal,
                workflow_id=workflow_id,
            )
        merged = validated.payload

    # 2. Settings: only the accepted paths, over the CURRENT settings.
    configurations: dict[str, Any] | None = None
    template_variables: dict[str, Any] | None = None
    if selected_settings:
        service = _settings_service()
        if current_settings is None:
            current_settings = await working_settings(workflow)
        candidate = getattr(model, "settings_candidate", None) or {}
        try:
            configurations, template_variables = await _maybe_await(
                service.merge_accepted_settings(
                    current_configurations=copy.deepcopy(current_settings[0]),
                    current_template_variables=copy.deepcopy(current_settings[1]),
                    plan_changes=[c for c in proposal.changes if is_settings_change(c)],
                    candidate_configurations=candidate.get("configurations") or {},
                    candidate_template_variables=(
                        candidate.get("template_variables") or {}
                    ),
                    accepted_change_ids=[c.change_id for c in selected_settings],
                )
            )
        except service.SettingsValidationError as e:
            return _failure(
                "validation_error",
                "The selected settings are not valid: "
                + "; ".join(_settings_errors(e)),
                proposal=proposal,
                workflow_id=workflow_id,
            )
        if any(
            c.change_id.startswith(f"{SETTINGS_REF_PREFIX}models")
            for c in selected_settings
        ):
            errors = await _validate_model_override(
                configurations, organization_id, user
            )
            if errors:
                return _failure(
                    "validation_error",
                    "The selected model settings are not valid: " + "; ".join(errors),
                    proposal=proposal,
                    workflow_id=workflow_id,
                )

    # 3. Tools first (the graph needs their uuids), then ONE save.
    created: dict[str, str] = {}
    if selected_tools:
        from api.services.tool_management import ToolManagementError

        try:
            created = await _create_accepted_tools(
                model, selected_tools, organization_id, user
            )
        except ToolManagementError as e:
            return _failure(
                e.error_code or "validation_error",
                f"Could not create the tool: {e.message}",
                proposal=proposal,
                workflow_id=workflow_id,
            )
        except (ValidationError, ValueError) as e:
            return _failure(
                "validation_error",
                f"Could not create the tool: {e}",
                proposal=proposal,
                workflow_id=workflow_id,
            )

    result: dict[str, Any] | None = None
    try:
        if merged is not None and _placeholders(merged):
            merged = substitute_tool_placeholders(merged, created)
            # Real uuids: tool-name collisions are checked now.
            validated = await validate_workflow_payload(
                merged,
                organization_id=organization_id,
                exclude_workflow_id=workflow_id,
            )
            if not validated.ok:
                await _archive_tools(created, organization_id)
                return _failure(
                    validated.error["error_code"],
                    validated.error.get("error")
                    or "The selected changes are not valid.",
                    proposal=proposal,
                    workflow_id=workflow_id,
                )
            merged = validated.payload

        new_name = ""
        for change in selected_graph:
            if change.target == ChangeTarget.workflow and change.after:
                new_name = str(change.after.get("name") or "").strip()

        if merged is not None or configurations is not None:
            result = await save_validated_draft(
                workflow=workflow,
                organization_id=organization_id,
                source=source,
                payload=merged,
                new_name=new_name,
                expected_draft_version=request.expected_draft_version,
                include_definition_id=True,
                workflow_configurations=configurations,
                template_context_variables=template_variables,
            )
            if not result.get("saved"):
                await _archive_tools(created, organization_id)
                return _failure(
                    result.get("error_code") or "validation_error",
                    result.get("error") or "Could not save the draft.",
                    proposal=proposal,
                    workflow_id=workflow_id,
                )
    except BaseException:
        if result is None or not result.get("saved"):
            await _archive_tools(created, organization_id)
        raise

    # `save_validated_draft` stores `merged` (secrets merged in place), so its
    # hash is the new working copy's hash; settings/tools-only keep the graph.
    draft_hash = compute_definition_hash(merged if merged is not None else working)
    fields: dict[str, Any] = {
        "status": ProposalStatus.applied.value,
        "applied_at": datetime.now(UTC),
        "accepted_change_ids": [c.change_id for c in selected],
    }
    if result is not None and isinstance(result.get("definition_id"), int):
        fields["applied_definition_id"] = result["definition_id"]
    updated = await _mark(proposal.proposal_uuid, organization_id, **fields)
    return ApplyProposalResponse(
        saved=True,
        workflow_id=workflow_id,
        draft_hash=draft_hash,
        proposal=_after_update(updated, proposal, ProposalStatus.applied),
    )


# ─── discard ─────────────────────────────────────────────────────────────


async def discard_proposal(
    *, proposal_uuid: str, organization_id: int
) -> Proposal | None:
    """pending -> discarded. None when the proposal is not in the org; a
    proposal that is no longer pending is returned unchanged."""
    model = await db_client.get_copilot_proposal(proposal_uuid, organization_id)
    if model is None:
        return None
    proposal = proposal_from_model(model)
    if proposal.status != ProposalStatus.pending:
        return proposal
    # Conditional (pending only): never discard a proposal an apply has
    # just claimed (`applying`).
    claimed = await db_client.claim_copilot_proposal(
        proposal_uuid,
        organization_id,
        from_status=ProposalStatus.pending.value,
        to_status=ProposalStatus.discarded.value,
    )
    current = await db_client.get_copilot_proposal(proposal_uuid, organization_id)
    if not claimed:
        return proposal_from_model(current) if current is not None else proposal
    await _mark_round_reviewed(proposal, organization_id)
    return _after_update(current, proposal, ProposalStatus.discarded)
