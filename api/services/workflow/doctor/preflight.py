"""Static workflow preflight (Layer 0, teardown change D).

Checks that run before any call analysis or replay:

- every node reachable from the start node can reach an End Call node
  (or a node holding a call-ending tool: end_call / transfer); without an end
  path simulations loop
- tools a node's prompt refers to are attached to that node (best effort:
  tool name / generated function name mentioned in the prompt text)
- no transition tool-name collisions (reuses ``workflow_graph`` and
  ``tool_name_validation`` validators; no duplicate logic)
- the raw definition parses as a ``ReactFlowDTO`` (``graph_invalid`` otherwise)
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any, Iterable, Optional

from loguru import logger
from pydantic import ValidationError

from api.enums import ToolCategory
from api.services.workflow.doctor.contracts import PreflightIssue
from api.services.workflow.dto import NodeType, ReactFlowDTO
from api.services.workflow.tools.custom_tool import custom_tool_function_name
from api.services.workflow.workflow_graph import (
    validate_unique_transition_tool_names,
)

#: Tool categories whose invocation ends (or hands off) the call.
CALL_ENDING_TOOL_CATEGORIES = frozenset(
    {
        ToolCategory.END_CALL.value,
        ToolCategory.TRANSFER_CALL.value,
        ToolCategory.TRANSFER_AGENT.value,
    }
)

#: Node types that are not part of the conversation graph.
_NON_CONVERSATIONAL_TYPES = frozenset(
    {
        NodeType.globalNode.value,
        NodeType.trigger.value,
        NodeType.webhook.value,
        NodeType.qa.value,
    }
)

_MIN_TOOL_NAME_LEN = 4


def _nodes(workflow_json: dict) -> list[dict]:
    nodes = (workflow_json or {}).get("nodes")
    return [n for n in nodes if isinstance(n, dict)] if isinstance(nodes, list) else []


def _edges(workflow_json: dict) -> list[dict]:
    edges = (workflow_json or {}).get("edges")
    return [e for e in edges if isinstance(e, dict)] if isinstance(edges, list) else []


def _data(node: dict) -> dict:
    data = node.get("data")
    return data if isinstance(data, dict) else {}


def _name(node: dict) -> str:
    return _data(node).get("name") or str(node.get("id"))


def _tool_attr(tool: Any, attr: str) -> Any:
    if isinstance(tool, dict):
        return tool.get(attr)
    value = getattr(tool, attr, None)
    return getattr(value, "value", value)


# ---------------------------------------------------------------------------
# Individual checks (pure; unit-testable without a DB)
# ---------------------------------------------------------------------------


def check_graph_parses(workflow_json: dict) -> list[PreflightIssue]:
    try:
        ReactFlowDTO.model_validate(workflow_json)
    except ValidationError as exc:
        first = exc.errors()[:3]
        detail = "; ".join(
            f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('msg')}"
            for e in first
        )
        return [
            PreflightIssue(
                check="graph_invalid",
                detail=f"Workflow definition does not validate: {detail}",
            )
        ]
    return []


def check_end_call_reachability(
    workflow_json: dict,
    call_ending_tool_uuids: Optional[set[str]] = None,
) -> list[PreflightIssue]:
    """Every conversational node reachable from the start must reach an end."""
    call_ending_tool_uuids = call_ending_tool_uuids or set()
    nodes = {
        str(n.get("id")): n
        for n in _nodes(workflow_json)
        if n.get("type") not in _NON_CONVERSATIONAL_TYPES
    }
    if not nodes:
        return [PreflightIssue(check="graph_invalid", detail="Workflow has no nodes.")]

    out_adj: dict[str, set[str]] = {nid: set() for nid in nodes}
    in_adj: dict[str, set[str]] = {nid: set() for nid in nodes}
    for e in _edges(workflow_json):
        src, dst = str(e.get("source")), str(e.get("target"))
        if src in nodes and dst in nodes:
            out_adj[src].add(dst)
            in_adj[dst].add(src)

    terminals = {
        nid
        for nid, n in nodes.items()
        if n.get("type") == NodeType.endNode.value
        or call_ending_tool_uuids.intersection(_data(n).get("tool_uuids") or [])
    }
    starts = [
        nid for nid, n in nodes.items() if n.get("type") == NodeType.startNode.value
    ]
    if not terminals:
        return [
            PreflightIssue(
                check="end_call_unreachable",
                detail="Workflow has no End Call node and no call-ending tool, "
                "so no conversation path can end the call.",
            )
        ]

    def _bfs(seeds: Iterable[str], adj: dict[str, set[str]]) -> set[str]:
        seen = set(seeds)
        queue = deque(seen)
        while queue:
            cur = queue.popleft()
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    reachable = _bfs(starts, out_adj) if starts else set(nodes)
    can_end = _bfs(terminals, in_adj)
    issues = []
    for nid in sorted(reachable - can_end, key=lambda i: _name(nodes[i])):
        issues.append(
            PreflightIssue(
                check="end_call_unreachable",
                node_name=_name(nodes[nid]),
                detail=f'No path from "{_name(nodes[nid])}" reaches an End Call node; '
                "a conversation that gets here can never end on its own.",
            )
        )
    return issues


def _mentions(prompt: str, needle: str) -> bool:
    if len(needle) < _MIN_TOOL_NAME_LEN:
        return False
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(needle) + r"(?![A-Za-z0-9_])"
    return re.search(pattern, prompt, flags=re.IGNORECASE) is not None


def check_tools_in_prompts(
    workflow_json: dict, tools: Iterable[Any]
) -> list[PreflightIssue]:
    """Best effort: a prompt that names a tool should have that tool attached.

    ``tools`` are the organization's tools (ORM rows or dicts with
    ``tool_uuid``, ``name``, ``category``). Matching uses the generated
    function name (``custom_tool_function_name``) and, for multi-word names,
    the display name. Global-node tools are available everywhere, so a tool
    attached to the global node counts as attached.
    """
    tools = list(tools or [])
    if not tools:
        return []
    nodes = _nodes(workflow_json)
    global_tools: set[str] = set()
    for n in nodes:
        if n.get("type") == NodeType.globalNode.value:
            global_tools.update(_data(n).get("tool_uuids") or [])

    # A node's own transition names (edge labels) share the tool-call
    # namespace: "end_call" in a prompt usually means the transition, not an
    # unattached "End Call" tool, so those names never count as mentions.
    outgoing: dict[str, set[str]] = {}
    for e in _edges(workflow_json):
        label = (e.get("data") or {}).get("label")
        if isinstance(label, str) and label:
            outgoing.setdefault(str(e.get("source")), set()).add(label.lower())

    issues: list[PreflightIssue] = []
    for n in nodes:
        if n.get("type") in _NON_CONVERSATIONAL_TYPES:
            continue
        data = _data(n)
        prompt = data.get("prompt") or ""
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        attached = set(data.get("tool_uuids") or []) | global_tools
        transitions = outgoing.get(str(n.get("id")), set())
        for tool in tools:
            uuid = _tool_attr(tool, "tool_uuid")
            name = _tool_attr(tool, "name") or ""
            if not uuid or uuid in attached or not name:
                continue
            fn_name = custom_tool_function_name(name)
            # Distinctive names only: the function name (snake case) or a
            # multi-word display name; a single dictionary word ("transfer")
            # is too ambiguous to flag.
            needles = {fn_name} if "_" in fn_name else set()
            if " " in name.strip():
                needles.add(name.strip())
            needles = {nd for nd in needles if nd.lower() not in transitions}
            if any(_mentions(prompt, needle) for needle in needles):
                issues.append(
                    PreflightIssue(
                        check="tool_not_attached",
                        node_name=_name(n),
                        detail=f'The prompt of "{_name(n)}" refers to tool "{name}" '
                        "but the tool is not attached to this node.",
                    )
                )
    return issues


def check_transition_collisions(workflow_json: dict) -> list[PreflightIssue]:
    """Reuse ``validate_unique_transition_tool_names`` on the raw definition."""
    nodes = {str(n.get("id")): _name(n) for n in _nodes(workflow_json)}
    transitions = []
    for e in _edges(workflow_json):
        data = e.get("data") if isinstance(e.get("data"), dict) else {}
        label = data.get("label")
        if isinstance(label, str) and label and e.get("id") is not None:
            transitions.append((str(e["id"]), str(e.get("source")), label))
    edge_source = {edge_id: src for edge_id, src, _ in transitions}
    return [
        PreflightIssue(
            check="tool_name_collision",
            node_name=nodes.get(edge_source.get(err["id"] or "", ""), None),
            detail=err["message"],
        )
        for err in validate_unique_transition_tool_names(transitions)
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run_preflight(
    workflow_json: dict,
    organization_id: int,
    *,
    tools: Optional[list[Any]] = None,
) -> list[PreflightIssue]:
    """Run all static checks. ``tools`` defaults to the org's active tools."""
    from api.db import db_client
    from api.services.workflow.tool_name_validation import (
        validate_workflow_tool_name_collisions,
    )

    if tools is None:
        try:
            tools = await db_client.get_tools_for_organization(organization_id)
        except Exception as exc:  # preflight must not fail on a tool lookup
            logger.warning(f"Doctor preflight: tool lookup failed: {exc}")
            tools = []

    call_ending = {
        _tool_attr(t, "tool_uuid")
        for t in tools
        if _tool_attr(t, "category") in CALL_ENDING_TOOL_CATEGORIES
    }

    issues: list[PreflightIssue] = []
    issues.extend(check_graph_parses(workflow_json))
    issues.extend(check_end_call_reachability(workflow_json, call_ending))
    issues.extend(check_tools_in_prompts(workflow_json, tools))
    issues.extend(check_transition_collisions(workflow_json))

    try:
        node_names = {str(n.get("id")): _name(n) for n in _nodes(workflow_json)}
        # Edge errors carry the edge id; attribute them to the source node.
        for e in _edges(workflow_json):
            node_names.setdefault(
                str(e.get("id")), node_names.get(str(e.get("source")), "")
            )
        for err in await validate_workflow_tool_name_collisions(
            workflow_json, organization_id
        ):
            issues.append(
                PreflightIssue(
                    check="tool_name_collision",
                    node_name=node_names.get(err["id"] or "") or None,
                    detail=err["message"],
                )
            )
    except Exception as exc:
        logger.warning(f"Doctor preflight: custom tool collision check failed: {exc}")
    return issues
