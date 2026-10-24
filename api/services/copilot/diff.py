"""Reviewable diff between a workflow working copy and a copilot candidate.

Two steps, both pure (no I/O):

1. `align_ids(after_json, before_json)` — the SDK parser numbers nodes in
   declaration order ("1", "2", ...) and names edges "<source>-<target>",
   so ids in parsed code say nothing about which stored node a candidate
   node is. Alignment gives every candidate node the id of the working-copy
   node it corresponds to (same type + name; or, for a rename, the only
   remaining node of that type when it is single-instance, keeps its prompt
   or has a similar name) and every candidate edge the id of the
   matching stored edge, and gives genuinely new nodes/edges ids that do
   not collide with the working copy. Must run before secrets are merged
   (`merge_workflow_api_keys` matches nodes by id) and before the diff.

2. `compute_changes(before_json, after_json)` — one `ChangeItem` per added,
   updated or removed node / edge (plus the workflow name). UI-only fields
   (position, measured, selected, dragging, width, height, ...) never count:
   nodes compare `type` + `data`, edges `source` + `target` + `data`.
"""

from __future__ import annotations

import copy
import re
from difflib import SequenceMatcher
from typing import Any, Iterable

from api.services.configuration.masking import mask_workflow_definition
from api.services.copilot.contracts import (
    ChangeItem,
    ChangeOp,
    ChangeTarget,
    FieldDiff,
)

WORKFLOW_NAME_CHANGE_ID = "w:name"


# ─── helpers ─────────────────────────────────────────────────────────────


def node_change_id(node_id: str) -> str:
    return f"n:{node_id}"


def edge_change_id(edge_id: str) -> str:
    return f"e:{edge_id}"


def _natural_key(value: str) -> tuple:
    parts = re.split(r"(\d+)", str(value))
    return tuple((0, int(p)) if p.isdigit() else (1, p) for p in parts if p != "")


def _nodes(wf: dict | None) -> list[dict]:
    return [n for n in (wf or {}).get("nodes") or [] if isinstance(n, dict)]


def _edges(wf: dict | None) -> list[dict]:
    return [e for e in (wf or {}).get("edges") or [] if isinstance(e, dict)]


def _data(item: dict) -> dict:
    """Node/edge data with unset (None) values dropped: a key the parser
    leaves out and a key stored as null mean the same thing."""
    raw = item.get("data")
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if v is not None}


def _name(node: dict | None) -> str:
    if not node:
        return ""
    return str((node.get("data") or {}).get("name") or "").strip()


def _label(edge: dict) -> str:
    return str((edge.get("data") or {}).get("label") or "").strip()


def _node_view(node: dict) -> dict[str, Any]:
    return {"id": node.get("id"), "type": node.get("type"), "data": _data(node)}


def _edge_view(edge: dict) -> dict[str, Any]:
    return {
        "id": edge.get("id"),
        "source": edge.get("source"),
        "target": edge.get("target"),
        "data": _data(edge),
    }


def _masked_node(node: dict) -> dict:
    """Never put a real integration secret into a reviewable change."""
    masked = mask_workflow_definition({"nodes": [copy.deepcopy(node)]}) or {}
    return (masked.get("nodes") or [node])[0]


def _numeric_ids(ids: Iterable[str]) -> list[int]:
    out = []
    for value in ids:
        try:
            out.append(int(str(value)))
        except ValueError:
            continue
    return out


# ─── 1. id alignment ─────────────────────────────────────────────────────

_RENAME_NAME_SIMILARITY = 0.6


def _looks_renamed(
    before: dict, after: dict, before_nodes: list[dict], after_nodes: list[dict]
) -> bool:
    """An unmatched before/after pair of one type is the same node renamed
    when the type is single-instance on both sides (start, global, QA...),
    the prompt is unchanged, or the names are close."""
    node_type = before.get("type")
    if (
        sum(1 for n in before_nodes if n.get("type") == node_type) == 1
        and sum(1 for n in after_nodes if n.get("type") == node_type) == 1
    ):
        return True
    before_prompt = (before.get("data") or {}).get("prompt")
    if before_prompt and before_prompt == (after.get("data") or {}).get("prompt"):
        return True
    ratio = SequenceMatcher(None, _name(before).lower(), _name(after).lower()).ratio()
    return ratio >= _RENAME_NAME_SIMILARITY


def align_ids(after_json: dict, before_json: dict | None) -> dict:
    """Return a copy of `after_json` whose node/edge ids match `before_json`
    wherever they denote the same node/edge (see module docstring)."""
    after = copy.deepcopy(after_json)
    after_nodes = _nodes(after)
    before_nodes = _nodes(before_json)

    id_map: dict[str, str] = {}
    unmatched_before = list(before_nodes)
    unmatched_after: list[dict] = []

    # Pass 1: same type and same (non-empty) name, in order.
    for node in after_nodes:
        key = (node.get("type"), _name(node))
        match = None
        if key[1]:
            match = next(
                (b for b in unmatched_before if (b.get("type"), _name(b)) == key),
                None,
            )
        if match is not None:
            unmatched_before.remove(match)
            id_map[str(node.get("id"))] = str(match.get("id"))
        else:
            unmatched_after.append(node)

    # Pass 2: exactly one unmatched node of a type on each side that looks
    # like the same node renamed; anything more ambiguous stays add + remove.
    still_unmatched: list[dict] = []
    for node in unmatched_after:
        node_type = node.get("type")
        before_same = [b for b in unmatched_before if b.get("type") == node_type]
        after_same = [a for a in unmatched_after if a.get("type") == node_type]
        if (
            len(before_same) == 1
            and len(after_same) == 1
            and _looks_renamed(before_same[0], node, before_nodes, after_nodes)
        ):
            unmatched_before.remove(before_same[0])
            id_map[str(node.get("id"))] = str(before_same[0].get("id"))
        else:
            still_unmatched.append(node)

    # New nodes: the UI's scheme (max numeric id + 1), never a stored id.
    taken = {str(b.get("id")) for b in before_nodes} | set(id_map.values())
    next_id = max(_numeric_ids(taken), default=0) + 1
    for node in still_unmatched:
        while str(next_id) in taken:
            next_id += 1
        id_map[str(node.get("id"))] = str(next_id)
        taken.add(str(next_id))
        next_id += 1

    for node in after_nodes:
        node["id"] = id_map.get(str(node.get("id")), node.get("id"))

    # Edges: remap endpoints, then reuse the stored edge's id.
    after_edges = _edges(after)
    for edge in after_edges:
        edge["source"] = id_map.get(str(edge.get("source")), edge.get("source"))
        edge["target"] = id_map.get(str(edge.get("target")), edge.get("target"))

    before_edges = list(_edges(before_json))
    remaining = list(before_edges)
    pending: list[dict] = []
    assigned: dict[int, str] = {}
    for edge in after_edges:
        key = (edge.get("source"), edge.get("target"), _label(edge))
        match = next(
            (
                b
                for b in remaining
                if (b.get("source"), b.get("target"), _label(b)) == key
            ),
            None,
        )
        if match is not None:
            remaining.remove(match)
            assigned[id(edge)] = str(match.get("id"))
        else:
            pending.append(edge)
    # Same endpoints, changed label: the same edge, updated.
    unresolved: list[dict] = []
    for edge in pending:
        ends = (edge.get("source"), edge.get("target"))
        before_same = [
            b for b in remaining if (b.get("source"), b.get("target")) == ends
        ]
        after_same = [a for a in pending if (a.get("source"), a.get("target")) == ends]
        if len(before_same) == 1 and len(after_same) == 1:
            remaining.remove(before_same[0])
            assigned[id(edge)] = str(before_same[0].get("id"))
        else:
            unresolved.append(edge)

    taken_edges = {str(b.get("id")) for b in before_edges} | set(assigned.values())
    for edge in unresolved:
        base = f"{edge.get('source')}-{edge.get('target')}"
        candidate = base
        suffix = 2
        while candidate in taken_edges:
            candidate = f"{base}-{suffix}"
            suffix += 1
        assigned[id(edge)] = candidate
        taken_edges.add(candidate)

    for edge in after_edges:
        edge["id"] = assigned[id(edge)]
    return after


# ─── 2. changes ──────────────────────────────────────────────────────────


def _field_diffs(
    before: dict, after: dict, extra: Iterable[str] = ()
) -> list[FieldDiff]:
    diffs: list[FieldDiff] = []
    for field in extra:
        if before.get(field) != after.get(field):
            diffs.append(
                FieldDiff(field=field, before=before.get(field), after=after.get(field))
            )
    before_data, after_data = before.get("data") or {}, after.get("data") or {}
    for key in sorted(set(before_data) | set(after_data)):
        if before_data.get(key) != after_data.get(key):
            diffs.append(
                FieldDiff(
                    field=key, before=before_data.get(key), after=after_data.get(key)
                )
            )
    return diffs


def _match_edges(
    before_edges: list[dict], after_edges: list[dict]
) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    """Pairs (before, after) by id, then by (source, target, label)."""
    before_by_id = {str(e.get("id")): e for e in before_edges}
    pairs: list[tuple[dict, dict]] = []
    used_before: set[str] = set()
    leftover_after: list[dict] = []
    for edge in after_edges:
        match = before_by_id.get(str(edge.get("id")))
        if match is not None and str(match.get("id")) not in used_before:
            pairs.append((match, edge))
            used_before.add(str(match.get("id")))
        else:
            leftover_after.append(edge)

    added: list[dict] = []
    for edge in leftover_after:
        key = (edge.get("source"), edge.get("target"), _label(edge))
        match = next(
            (
                b
                for b in before_edges
                if str(b.get("id")) not in used_before
                and (b.get("source"), b.get("target"), _label(b)) == key
            ),
            None,
        )
        if match is not None:
            pairs.append((match, edge))
            used_before.add(str(match.get("id")))
        else:
            added.append(edge)
    removed = [e for e in before_edges if str(e.get("id")) not in used_before]
    return pairs, added, removed


def compute_changes(
    before_json: dict | None,
    after_json: dict,
    *,
    before_name: str | None = None,
    after_name: str | None = None,
) -> list[ChangeItem]:
    """Changes turning `before_json` into `after_json` (None = new workflow).

    Nodes match by id, edges by id then (source, target, label). Pass the
    stored and proposed workflow names to also get a workflow-level rename.
    Order: nodes, then edges (each by id), then workflow-level changes.
    Secret node fields are masked in `before` / `after` / `field_diffs`.
    """
    before_nodes = {str(n.get("id")): n for n in _nodes(before_json)}
    after_nodes = {str(n.get("id")): n for n in _nodes(after_json)}
    before_names = {nid: _name(n) or nid for nid, n in before_nodes.items()}
    after_names = {nid: _name(n) or nid for nid, n in after_nodes.items()}

    def node_label(node_id: str) -> str:
        return after_names.get(node_id) or before_names.get(node_id) or node_id

    added_nodes = sorted(set(after_nodes) - set(before_nodes), key=_natural_key)
    removed_nodes = sorted(set(before_nodes) - set(after_nodes), key=_natural_key)
    common_nodes = sorted(set(before_nodes) & set(after_nodes), key=_natural_key)

    node_items: list[ChangeItem] = []
    for nid in added_nodes:
        view = _node_view(_masked_node(after_nodes[nid]))
        node_items.append(
            ChangeItem(
                change_id=node_change_id(nid),
                target=ChangeTarget.node,
                op=ChangeOp.add,
                ref_id=nid,
                label=f'Add {view["type"]} node "{after_names[nid]}"',
                after=view,
            )
        )
    for nid in common_nodes:
        before_view = _node_view(before_nodes[nid])
        after_view = _node_view(after_nodes[nid])
        if before_view["type"] == after_view["type"] and (
            before_view["data"] == after_view["data"]
        ):
            continue
        masked_before = _node_view(_masked_node(before_nodes[nid]))
        masked_after = _node_view(_masked_node(after_nodes[nid]))
        diffs = _field_diffs(masked_before, masked_after, extra=("type",))
        # A secret replaced by a different secret masks to the same string;
        # still report the field, with masked values.
        changed = {
            k
            for k in set(before_view["data"]) | set(after_view["data"])
            if before_view["data"].get(k) != after_view["data"].get(k)
        }
        for key in sorted(changed - {d.field for d in diffs}):
            diffs.append(
                FieldDiff(
                    field=key,
                    before=masked_before["data"].get(key),
                    after=masked_after["data"].get(key),
                )
            )
        fields = ", ".join(d.field for d in diffs)
        old_name, new_name = before_names[nid], after_names[nid]
        title = (
            f'"{new_name}"' if old_name == new_name else f'"{old_name}" → "{new_name}"'
        )
        node_items.append(
            ChangeItem(
                change_id=node_change_id(nid),
                target=ChangeTarget.node,
                op=ChangeOp.update,
                ref_id=nid,
                label=f"Update node {title} ({fields})",
                before=masked_before,
                after=masked_after,
                field_diffs=diffs,
            )
        )
    for nid in removed_nodes:
        view = _node_view(_masked_node(before_nodes[nid]))
        node_items.append(
            ChangeItem(
                change_id=node_change_id(nid),
                target=ChangeTarget.node,
                op=ChangeOp.remove,
                ref_id=nid,
                label=f'Remove {view["type"]} node "{before_names[nid]}"',
                before=view,
            )
        )

    # Edges.
    pairs, added_edges, removed_edges = _match_edges(
        _edges(before_json), _edges(after_json)
    )

    def edge_title(edge: dict) -> str:
        title = f'"{node_label(str(edge.get("source")))}" → "{node_label(str(edge.get("target")))}"'
        label = _label(edge)
        return f"{title} [{label}]" if label else title

    def endpoint_deps(edge: dict) -> list[str]:
        return [
            node_change_id(str(edge.get(end)))
            for end in ("source", "target")
            if str(edge.get(end)) in added_nodes
        ]

    edge_items: list[ChangeItem] = []
    for edge in added_edges:
        eid = str(edge.get("id"))
        edge_items.append(
            ChangeItem(
                change_id=edge_change_id(eid),
                target=ChangeTarget.edge,
                op=ChangeOp.add,
                ref_id=eid,
                label=f"Add edge {edge_title(edge)}",
                after=_edge_view(edge),
                depends_on=sorted(set(endpoint_deps(edge)), key=_natural_key),
            )
        )
    for before_edge, after_edge in pairs:
        before_view = _edge_view(before_edge)
        after_view = _edge_view(after_edge)
        if all(before_view[k] == after_view[k] for k in ("source", "target", "data")):
            continue
        ref = str(before_edge.get("id"))
        diffs = _field_diffs(before_view, after_view, extra=("source", "target"))
        edge_items.append(
            ChangeItem(
                change_id=edge_change_id(ref),
                target=ChangeTarget.edge,
                op=ChangeOp.update,
                ref_id=ref,
                label=f"Update edge {edge_title(after_edge)} "
                f"({', '.join(d.field for d in diffs)})",
                before=before_view,
                after={**after_view, "id": ref},
                field_diffs=diffs,
                depends_on=sorted(set(endpoint_deps(after_edge)), key=_natural_key),
            )
        )
    for edge in removed_edges:
        eid = str(edge.get("id"))
        edge_items.append(
            ChangeItem(
                change_id=edge_change_id(eid),
                target=ChangeTarget.edge,
                op=ChangeOp.remove,
                ref_id=eid,
                label=f"Remove edge {edge_title(edge)}",
                before=_edge_view(edge),
            )
        )
    edge_items.sort(key=lambda item: _natural_key(item.ref_id))

    # A node removal needs every change that detaches an edge from it.
    for item in node_items:
        if item.op != ChangeOp.remove:
            continue
        deps = []
        for edge_item in edge_items:
            if edge_item.op == ChangeOp.add:
                continue
            before = edge_item.before or {}
            touches = item.ref_id in (
                str(before.get("source")),
                str(before.get("target")),
            )
            if touches:
                deps.append(edge_item.change_id)
        item.depends_on = sorted(set(deps), key=_natural_key)

    node_items.sort(key=lambda item: _natural_key(item.ref_id))
    items = node_items + edge_items

    if (
        before_json is not None
        and after_name
        and before_name is not None
        and after_name.strip() != (before_name or "").strip()
    ):
        items.append(
            ChangeItem(
                change_id=WORKFLOW_NAME_CHANGE_ID,
                target=ChangeTarget.workflow,
                op=ChangeOp.update,
                ref_id="workflow",
                label=f'Rename workflow "{before_name}" → "{after_name.strip()}"',
                before={"name": before_name},
                after={"name": after_name.strip()},
                field_diffs=[
                    FieldDiff(
                        field="name", before=before_name, after=after_name.strip()
                    )
                ],
            )
        )
    return items
