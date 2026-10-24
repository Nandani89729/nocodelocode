"""Doctor timeline compaction and Layer 2 latency (artifact exclusion)."""

from api.services.workflow.doctor.latency import compute_latency
from api.services.workflow.doctor.timeline import (
    build_run_timeline,
    build_timeline,
    compact_event,
    parse_tool_result,
)


def ev(type_, turn, node="Main", **payload):
    return {"type": type_, "turn": turn, "node_name": node, "payload": payload}


def breakdown(turn, total, node="Main", fn_calls=None):
    return ev(
        "rtf-latency-breakdown",
        turn,
        node,
        total_secs=total,
        user_turn_secs=0.3,
        ttfb=[
            {"processor": "DeepgramSTTService#0", "duration_secs": 0.2},
            {"processor": "OpenAILLMService#0", "duration_secs": 0.9},
            {"processor": "CartesiaTTSService#0", "duration_secs": 0.25},
        ],
        contributions=[],
        function_calls=fn_calls or [],
    )


def test_tool_result_parses_json_and_python_repr():
    assert parse_tool_result('{"total_results": 0}') == {"total_results": 0}
    assert parse_tool_result("{'status': 'ok'}") == {"status": "ok"}
    assert parse_tool_result("plain text") == "plain text"


def test_compact_event_kinds_and_interim_user_dropped():
    assert (
        compact_event(ev("rtf-user-transcription", 1, text="hi", final=False)) is None
    )
    user = compact_event(
        ev("rtf-user-transcription", 1, text="hi", final=True, turn_interrupted=True)
    )
    assert user["kind"] == "user" and user["turn_interrupted"] is True
    assert compact_event(ev("rtf-bot-started-speaking", 1)) is None
    err = compact_event(
        ev("rtf-pipeline-error", 2, error="boom", fatal=True, processor="STT")
    )
    assert err == {
        "turn": 2,
        "node": "Main",
        "t": None,
        "kind": "error",
        "error": "boom",
        "fatal": True,
        "processor": "STT",
    }
    tr = compact_event(
        ev(
            "rtf-node-transition",
            3,
            previous_node_name="A",
            node_name="B",
        )
    )
    assert tr["kind"] == "transition" and tr["from"] == "A" and tr["to"] == "B"


def test_latency_artifacts_excluded_on_tool_and_end_call_turns():
    events = [
        # turn 1: normal reply
        ev("rtf-user-transcription", 1, text="hello", final=True),
        ev("rtf-bot-text", 1, text="Hi there"),
        breakdown(1, 1.2),
        # turn 2: ends in a transition tool call, no bot speech after it
        ev("rtf-user-transcription", 2, text="book me", final=True),
        ev("rtf-function-call-start", 2, function_name="go_to_booking"),
        ev("rtf-function-call-end", 2, function_name="go_to_booking", result="ok"),
        ev("rtf-latency-measured", 2, latency_seconds=95.0),
        # turn 3: KB tool then bot answers -> real latency, kept
        ev("rtf-user-transcription", 3, text="price?", final=True, node="Booking"),
        ev(
            "rtf-function-call-start",
            3,
            node="Booking",
            function_name="retrieve_from_knowledge_base",
        ),
        ev(
            "rtf-function-call-end",
            3,
            node="Booking",
            function_name="retrieve_from_knowledge_base",
            result='{"total_results": 2}',
        ),
        ev("rtf-bot-text", 3, node="Booking", text="It is 10 dollars"),
        breakdown(3, 2.0, node="Booking"),
        # turn 4: end_call even with speech
        ev("rtf-user-transcription", 4, text="bye", final=True, node="Booking"),
        ev("rtf-bot-text", 4, node="Booking", text="Goodbye"),
        ev("rtf-function-call-start", 4, node="Booking", function_name="end_call"),
        breakdown(4, 135.0, node="Booking"),
    ]
    timeline = build_timeline(events)
    excluded = {(e["turn"], e["kind"]) for e in timeline if e.get("excluded")}
    assert excluded == {(2, "latency"), (4, "latency_breakdown")}
    kb_result = next(
        e for e in timeline if e["kind"] == "tool_result" and e["turn"] == 3
    )
    assert kb_result["result"] == {"total_results": 2}

    rows = {r.node_name: r for r in compute_latency([timeline])}
    assert rows["Main"].turns == 1 and rows["Main"].p90_s == 1.2
    assert rows["Booking"].turns == 1 and rows["Booking"].p90_s == 2.0
    assert rows["Booking"].llm_ttfb_p50_s == 0.9
    assert rows["Booking"].stt_p50_s == 0.2
    assert rows["Booking"].tts_ttfb_p50_s == 0.25


def test_latency_falls_back_to_measured_for_legacy_runs():
    events = [
        ev("rtf-user-transcription", 1, text="hello", final=True),
        ev("rtf-bot-text", 1, text="Hi"),
        ev("rtf-latency-measured", 1, latency_seconds=1.5),
        ev("rtf-user-transcription", 2, text="and?", final=True),
        ev("rtf-bot-text", 2, text="More"),
        ev("rtf-latency-measured", 2, latency_seconds=2.5),
    ]
    (row,) = compute_latency([build_timeline(events)])
    assert row.turns == 2 and row.p50_s == 2.5 and row.p90_s == 2.5
    assert row.llm_ttfb_p50_s is None


def test_build_run_timeline_handles_json_strings():
    run = {
        "id": 7,
        "created_at": None,
        "definition_id": 3,
        "mode": "twilio",
        "logs": '{"realtime_feedback_events": [{"type": "rtf-bot-text", "turn": 0, '
        '"node_name": "Start", "payload": {"text": "Hi"}}]}',
        "usage_info": "{}",
        "gathered_context": {"mapped_call_disposition": "XFER"},
        "initial_context": {"name": "Ann"},
    }
    view = build_run_timeline(run)
    assert view["run_id"] == 7 and view["has_events"] is True
    assert view["disposition"] == "XFER"
    assert view["timeline"][0]["text"] == "Hi"
    assert "logs" not in view


def test_latency_skips_waiting_turns():
    """wf7 run 16: a 57.7s 'latency' was the caller waiting on a hung agent
    (services took ~3s). It is a Layer 1 problem, not processing latency."""
    from api.services.workflow.doctor.latency import compute_latency

    fast = {
        "kind": "latency_breakdown",
        "node": "Welcome",
        "turn": "1",
        "total": 2.8,
        "user_turn": 0.8,
        "ttfb": {"DograhLLMService#1": 1.3, "DograhTTSService#1": 0.7},
    }
    hung = {
        "kind": "latency_breakdown",
        "node": "Welcome",
        "turn": "5",
        "total": 57.7,
        "user_turn": 1.0,
        "ttfb": {"DograhLLMService#1": 1.3, "DograhTTSService#1": 0.7},
    }
    hung_fallback = {"kind": "latency", "node": "Welcome", "turn": "5", "secs": 57.7}
    rows = compute_latency([[fast, hung, hung_fallback]])
    assert len(rows) == 1
    assert rows[0].turns == 1 and rows[0].p90_s == 2.8
