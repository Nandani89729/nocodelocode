"""`api.services.copilot.diff`: id alignment and reviewable change items."""

from __future__ import annotations

import copy

from api.services.copilot.contracts import ChangeOp, ChangeTarget
from api.services.copilot.diff import align_ids, compute_changes


def _node(node_id, name, node_type="agentNode", x=0, y=0, **data):
    return {
        "id": node_id,
        "type": node_type,
        "position": {"x": x, "y": y},
        "data": {"name": name, "prompt": f"{name} prompt", **data},
    }


def _edge(source, target, label, edge_id=None, **data):
    return {
        "id": edge_id or f"{source}-{target}",
        "source": source,
        "target": target,
        "data": {"label": label, "condition": f"when {label}", **data},
    }


def _base() -> dict:
    return {
        "nodes": [
            _node("1", "Greeting", "startCall", 0, 0),
            _node("2", "Qualify", "agentNode", 400, 200),
            _node("3", "Goodbye", "endCall", 800, 400),
        ],
        "edges": [_edge("1", "2", "qualify"), _edge("2", "3", "done")],
        "viewport": {"x": 0, "y": 0, "zoom": 1},
    }


def _by_id(changes):
    return {c.change_id: c for c in changes}


# ─── compute_changes ─────────────────────────────────────────────────────


def test_identical_workflows_have_no_changes():
    assert compute_changes(_base(), _base()) == []


def test_positions_and_ui_fields_are_ignored():
    after = _base()
    for node in after["nodes"]:
        node["position"] = {"x": 999, "y": -5}
        node["measured"] = {"width": 10, "height": 10}
        node["selected"] = True
        node["dragging"] = True
    after["edges"][0]["selected"] = True
    after["edges"][0]["animated"] = True
    after["viewport"] = {"x": 50, "y": 50, "zoom": 2}
    assert compute_changes(_base(), after) == []


def test_none_valued_data_equals_missing_key():
    after = _base()
    after["nodes"][1]["data"]["tool_uuids"] = None
    assert compute_changes(_base(), after) == []


def test_node_update_lists_changed_fields_and_both_sides():
    after = _base()
    after["nodes"][1]["data"]["prompt"] = "New qualify prompt"
    after["nodes"][1]["data"]["tool_uuids"] = ["t-1"]
    (change,) = compute_changes(_base(), after)
    assert change.change_id == "n:2"
    assert change.target == ChangeTarget.node and change.op == ChangeOp.update
    assert [d.field for d in change.field_diffs] == ["prompt", "tool_uuids"]
    assert change.field_diffs[0].before == "Qualify prompt"
    assert change.field_diffs[0].after == "New qualify prompt"
    assert change.label == 'Update node "Qualify" (prompt, tool_uuids)'
    assert "position" not in change.before and "position" not in change.after


def test_node_type_change_is_an_update():
    after = _base()
    after["nodes"][1]["type"] = "otherNode"
    (change,) = compute_changes(_base(), after)
    assert change.op == ChangeOp.update
    assert change.field_diffs[0].field == "type"


def test_node_rename_label_shows_both_names():
    after = _base()
    after["nodes"][1]["data"]["name"] = "Qualify lead"
    (change,) = compute_changes(_base(), after)
    assert change.label == 'Update node "Qualify" → "Qualify lead" (name)'


def test_added_node_and_edges_depend_on_the_node():
    after = _base()
    after["nodes"].append(_node("4", "Book demo"))
    after["edges"].append(_edge("2", "4", "book"))
    changes = compute_changes(_base(), after)
    by_id = _by_id(changes)
    assert by_id["n:4"].op == ChangeOp.add
    assert by_id["n:4"].label == 'Add agentNode node "Book demo"'
    assert by_id["n:4"].before is None
    edge = by_id["e:2-4"]
    assert edge.op == ChangeOp.add
    assert edge.depends_on == ["n:4"]
    assert edge.label == 'Add edge "Qualify" → "Book demo" [book]'


def test_edge_between_two_new_nodes_depends_on_both():
    after = _base()
    after["nodes"] += [_node("4", "A"), _node("5", "B")]
    after["edges"] += [_edge("2", "4", "a"), _edge("4", "5", "b")]
    by_id = _by_id(compute_changes(_base(), after))
    assert by_id["e:4-5"].depends_on == ["n:4", "n:5"]
    assert by_id["e:2-4"].depends_on == ["n:4"]


def test_removed_node_depends_on_removal_of_its_edges():
    before = _base()
    before["nodes"].append(_node("4", "Transfer"))
    before["edges"] += [_edge("2", "4", "transfer"), _edge("4", "3", "after")]
    by_id = _by_id(compute_changes(before, _base()))
    node = by_id["n:4"]
    assert node.op == ChangeOp.remove
    assert node.depends_on == ["e:2-4", "e:4-3"]
    assert by_id["e:2-4"].op == ChangeOp.remove
    assert by_id["e:2-4"].depends_on == []
    assert node.label == 'Remove agentNode node "Transfer"'


def test_removed_node_depends_on_edge_rewired_away_from_it():
    before = _base()
    before["nodes"].append(_node("4", "Transfer"))
    before["edges"].append(_edge("4", "3", "after", edge_id="e-x"))
    after = _base()
    after["edges"].append(_edge("1", "3", "after", edge_id="e-x"))
    by_id = _by_id(compute_changes(before, after))
    assert by_id["e:e-x"].op == ChangeOp.update
    assert by_id["n:4"].depends_on == ["e:e-x"]


def test_edge_update_on_condition_change():
    after = _base()
    after["edges"][1]["data"]["condition"] = "caller is done"
    (change,) = compute_changes(_base(), after)
    assert change.change_id == "e:2-3"
    assert change.op == ChangeOp.update
    assert [d.field for d in change.field_diffs] == ["condition"]
    assert change.label == 'Update edge "Qualify" → "Goodbye" [done] (condition)'


def test_edges_with_different_ids_match_on_source_target_label():
    after = _base()
    after["edges"][0]["id"] = "ui-generated-edge"
    assert compute_changes(_base(), after) == []

    after["edges"][0]["data"]["condition"] = "changed"
    (change,) = compute_changes(_base(), after)
    # Referenced by the working copy's id so apply can find it.
    assert change.ref_id == "1-2"
    assert change.after["id"] == "1-2"


def test_create_kind_is_all_adds():
    changes = compute_changes(None, _base())
    assert [c.change_id for c in changes] == ["n:1", "n:2", "n:3", "e:1-2", "e:2-3"]
    assert all(c.op == ChangeOp.add for c in changes)


def test_ordering_is_nodes_then_edges_by_natural_id():
    before = {"nodes": [], "edges": []}
    after = {
        "nodes": [_node("10", "J"), _node("2", "B"), _node("1", "A")],
        "edges": [_edge("2", "10", "x"), _edge("1", "2", "y")],
    }
    ids = [c.change_id for c in compute_changes(before, after)]
    assert ids == ["n:1", "n:2", "n:10", "e:1-2", "e:2-10"]
    # Deterministic.
    assert ids == [c.change_id for c in compute_changes(before, copy.deepcopy(after))]


def test_workflow_rename_is_a_workflow_change():
    changes = compute_changes(
        _base(), _base(), before_name="Old name", after_name="New name"
    )
    (change,) = changes
    assert change.change_id == "w:name"
    assert change.target == ChangeTarget.workflow
    assert change.after == {"name": "New name"}
    assert (
        compute_changes(_base(), _base(), before_name="Same", after_name="Same") == []
    )
    assert compute_changes(_base(), _base(), before_name="Same", after_name=None) == []


def test_secrets_are_masked_in_change_items():
    before = _base()
    before["nodes"].append(
        _node("4", "Tuner", "tuner", tuner_api_key="real-key-0000000000abcd")
    )
    after = copy.deepcopy(before)
    after["nodes"][3]["data"]["tuner_api_key"] = "brand-new-key-000000wxyz"
    (change,) = compute_changes(before, after)
    dumped = change.model_dump_json()
    assert "real-key" not in dumped and "brand-new-key" not in dumped
    assert change.field_diffs[0].field == "tuner_api_key"

    added = compute_changes(None, after)
    assert "brand-new-key" not in "".join(c.model_dump_json() for c in added)


# ─── align_ids ───────────────────────────────────────────────────────────


def _parsed_like(workflow: dict) -> dict:
    """Renumber like the SDK parser: nodes "1".. in order, edges "s-t"."""
    out = copy.deepcopy(workflow)
    mapping = {n["id"]: str(i + 1) for i, n in enumerate(out["nodes"])}
    for node in out["nodes"]:
        node["id"] = mapping[node["id"]]
        node["position"] = {"x": 0, "y": 0}
    for edge in out["edges"]:
        edge["source"] = mapping[edge["source"]]
        edge["target"] = mapping[edge["target"]]
        edge["id"] = f"{edge['source']}-{edge['target']}"
    return out


def _ui_ids(workflow: dict) -> dict:
    out = copy.deepcopy(workflow)
    mapping = {n["id"]: f"dnd_{n['id']}" for n in out["nodes"]}
    for node in out["nodes"]:
        node["id"] = mapping[node["id"]]
    for edge in out["edges"]:
        edge["source"] = mapping[edge["source"]]
        edge["target"] = mapping[edge["target"]]
        edge["id"] = f"xy-edge__{edge['source']}-{edge['target']}"
    return out


def test_align_maps_parsed_ids_onto_stored_ids():
    stored = _ui_ids(_base())
    parsed = _parsed_like(stored)
    aligned = align_ids(parsed, stored)
    assert [n["id"] for n in aligned["nodes"]] == [n["id"] for n in stored["nodes"]]
    assert [e["id"] for e in aligned["edges"]] == [e["id"] for e in stored["edges"]]
    assert compute_changes(stored, aligned) == []
    assert parsed["nodes"][0]["id"] == "1"  # input not mutated


def test_align_reordered_declarations_still_match_by_name():
    stored = _ui_ids(_base())
    parsed = _parsed_like(stored)
    parsed["nodes"].reverse()
    assert compute_changes(stored, align_ids(parsed, stored)) == []


def test_align_new_node_gets_fresh_non_colliding_id():
    stored = _base()
    after = copy.deepcopy(stored)
    after["nodes"].append(_node("9", "Book demo"))
    after["edges"].append(_edge("2", "9", "book"))
    parsed = _parsed_like(after)  # new node parsed as "4"
    stored["nodes"].append(_node("4", "Old unrelated"))  # "4" taken in store
    stored["edges"].append(_edge("2", "4", "old"))
    aligned = align_ids(parsed, stored)
    new_node = next(n for n in aligned["nodes"] if n["data"]["name"] == "Book demo")
    assert new_node["id"] == "5"
    by_id = _by_id(compute_changes(stored, aligned))
    assert by_id["n:5"].op == ChangeOp.add
    assert by_id["n:4"].op == ChangeOp.remove
    assert by_id["e:2-5"].depends_on == ["n:5"]


def test_align_single_renamed_node_of_a_type_is_an_update():
    stored = _base()
    after = copy.deepcopy(stored)
    after["nodes"][2]["data"]["name"] = "Farewell"
    aligned = align_ids(_parsed_like(after), stored)
    (change,) = compute_changes(stored, aligned)
    assert change.change_id == "n:3" and change.op == ChangeOp.update


def test_align_renamed_agent_node_with_same_prompt_is_an_update():
    stored = _base()
    stored["nodes"].append(_node("4", "Other agent"))
    stored["edges"].append(_edge("2", "4", "other"))
    after = copy.deepcopy(stored)
    after["nodes"][1]["data"]["name"] = "Lead qualification"  # prompt kept
    aligned = align_ids(_parsed_like(after), stored)
    (change,) = compute_changes(stored, aligned)
    assert change.change_id == "n:2" and change.op == ChangeOp.update


def test_align_edge_with_changed_label_keeps_stored_id():
    stored = _ui_ids(_base())
    after = copy.deepcopy(stored)
    after["edges"][1]["data"]["label"] = "finished"
    aligned = align_ids(_parsed_like(after), stored)
    (change,) = compute_changes(stored, aligned)
    assert change.ref_id == stored["edges"][1]["id"]
    assert change.op == ChangeOp.update
