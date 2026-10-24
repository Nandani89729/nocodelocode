"""Doctor static preflight (Layer 0, teardown D)."""

from unittest.mock import AsyncMock, patch

from api.services.workflow.doctor.preflight import (
    check_end_call_reachability,
    check_tools_in_prompts,
    check_transition_collisions,
    run_preflight,
)


def node(id_, type_, name, prompt="Be helpful.", tool_uuids=None):
    data = {"name": name, "prompt": prompt}
    if tool_uuids is not None:
        data["tool_uuids"] = tool_uuids
    return {"id": id_, "type": type_, "position": {"x": 0, "y": 0}, "data": data}


def edge(id_, src, dst, label):
    return {
        "id": id_,
        "source": src,
        "target": dst,
        "data": {"label": label, "condition": f"when {label}"},
    }


def wf(nodes, edges):
    return {"nodes": nodes, "edges": edges}


def test_workflow_without_end_call_is_flagged():
    w = wf(
        [node("1", "startCall", "Start"), node("2", "agentNode", "Qualify")],
        [edge("e1", "1", "2", "qualify")],
    )
    issues = check_end_call_reachability(w)
    assert [i.check for i in issues] == ["end_call_unreachable"]
    assert "no End Call node" in issues[0].detail


def test_dead_end_branch_is_flagged_per_node():
    w = wf(
        [
            node("1", "startCall", "Start"),
            node("2", "agentNode", "Qualify"),
            node("3", "agentNode", "Loop A"),
            node("4", "agentNode", "Loop B"),
            node("5", "endCall", "End"),
            node("g", "globalNode", "Global"),
        ],
        [
            edge("e1", "1", "2", "qualify"),
            edge("e2", "2", "5", "done"),
            edge("e3", "2", "3", "loop"),
            edge("e4", "3", "4", "next"),
            edge("e5", "4", "3", "back"),
        ],
    )
    issues = check_end_call_reachability(w)
    assert sorted(i.node_name for i in issues) == ["Loop A", "Loop B"]


def test_call_ending_tool_counts_as_an_end():
    w = wf(
        [
            node("1", "startCall", "Start", tool_uuids=["t-end"]),
            node("2", "agentNode", "Qualify"),
            node("3", "endCall", "End"),
        ],
        [edge("e1", "1", "3", "finish")],
    )
    assert check_end_call_reachability(w, {"t-end"}) == []


def test_tool_named_in_prompt_but_not_attached():
    tools = [
        {"tool_uuid": "t1", "name": "Book Appointment", "category": "http_api"},
        {"tool_uuid": "t2", "name": "transfer", "category": "transfer_call"},
    ]
    w = wf(
        [
            node(
                "1",
                "startCall",
                "Start",
                prompt="When they agree, call book_appointment. You may transfer.",
            ),
            node(
                "2",
                "agentNode",
                "Booking",
                prompt="Use Book Appointment to confirm.",
                tool_uuids=["t1"],
            ),
        ],
        [],
    )
    issues = check_tools_in_prompts(w, tools)
    assert [(i.check, i.node_name) for i in issues] == [("tool_not_attached", "Start")]


def test_transition_collisions_reuse_existing_validator():
    w = wf(
        [
            node("1", "startCall", "Start"),
            node("2", "agentNode", "A"),
            node("3", "endCall", "End"),
        ],
        [edge("e1", "1", "2", "Next Step"), edge("e2", "1", "3", "next-step")],
    )
    issues = check_transition_collisions(w)
    assert len(issues) == 2
    assert {i.check for i in issues} == {"tool_name_collision"}
    assert issues[0].node_name == "Start"


async def test_run_preflight_combines_checks_without_db():
    w = wf(
        [node("1", "startCall", "Start"), node("2", "agentNode", "Qualify")],
        [edge("e1", "1", "2", "qualify")],
    )
    with patch(
        "api.services.workflow.tool_name_validation.validate_workflow_tool_name_collisions",
        AsyncMock(return_value=[]),
    ):
        issues = await run_preflight(w, organization_id=1, tools=[])
    assert "end_call_unreachable" in {i.check for i in issues}


def test_transition_named_like_a_tool_is_not_a_missing_tool():
    """wf7 false positive: "end_call" in a prompt is the node's own transition,
    not the org's unattached "End Call" tool (function name end_call)."""
    tools = [{"tool_uuid": "t1", "name": "End Call", "category": "end_call"}]
    w = wf(
        [
            node("1", "startCall", "Welcome", prompt="If they say bye, call end_call."),
            node("2", "agentNode", "Other", prompt="Then call end_call."),
            node("3", "endCall", "Goodbye"),
        ],
        [edge("e1", "1", "3", "end_call")],
    )
    issues = check_tools_in_prompts(w, tools)
    # Node 1 has the transition; node 2 does not, so it is still flagged.
    assert [i.node_name for i in issues] == ["Other"]
