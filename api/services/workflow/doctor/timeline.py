"""Compact per-run timelines for the Agent Doctor (single implementation, Q3).

Ported from the call-doctor skill's ``export_runs.py`` (``compact_event``).
The skill script is expected to call the API instead of duplicating this.

A timeline is a list of small dicts, one per meaningful realtime-feedback
event, each tagged with ``turn`` and ``node``::

    {"kind": "user" | "bot" | "tool_call" | "tool_result" | "transition"
             | "latency" | "latency_breakdown" | "error", "turn": 3,
     "node": "Check Work", "t": "<iso timestamp>", ...kind specific fields}

Latency entries on turns that end in a tool call (no bot speech after it)
or that invoke ``end_call`` are marked ``excluded: True``: the measured
interval there runs until the *next* speech, which produces the 50-135s
artifacts seen in real logs. Layer 2 skips excluded entries.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Iterable, Optional

from pipecat.utils.enums import RealtimeFeedbackType

from api.services.pipecat.realtime_feedback_events import (
    LATENCY_BREAKDOWN_EVENT_TYPE,
)

_USER = RealtimeFeedbackType.USER_TRANSCRIPTION.value
_BOT = RealtimeFeedbackType.BOT_TEXT.value
_FN_START = RealtimeFeedbackType.FUNCTION_CALL_START.value
_FN_END = RealtimeFeedbackType.FUNCTION_CALL_END.value
_TRANSITION = RealtimeFeedbackType.NODE_TRANSITION.value
_LATENCY = RealtimeFeedbackType.LATENCY_MEASURED.value
_ERROR = RealtimeFeedbackType.PIPELINE_ERROR.value

#: Per-turn LLM usage event (not emitted yet; Layer 0's prompt-load check
#: reads it when present, otherwise reports insufficient data).
LLM_USAGE_EVENT_TYPE = "rtf-llm-usage"

#: Function names that end the call; latency on their turn is an artifact.
CALL_ENDING_FUNCTION_MARKERS = ("end_call", "endcall", "hang_up", "hangup")

#: Timeline kinds that make up the conversation proper (used to decide how a
#: turn ends).
CONVERSATIONAL_KINDS = ("user", "bot", "tool_call", "tool_result", "transition")


def as_obj(value: Any) -> Any:
    """JSON columns can arrive as already-parsed objects or as strings."""
    if isinstance(value, (str, bytes)):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def parse_tool_result(result: Any) -> Any:
    """Tool results are JSON (new runs) or a Python repr (older runs)."""
    if not isinstance(result, str):
        return result
    try:
        return json.loads(result)
    except ValueError:
        pass
    try:
        return ast.literal_eval(result)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return result


def _round(value: Any, ndigits: int = 3) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def compact_event(ev: dict) -> Optional[dict]:
    """Map one realtime-feedback event to a compact timeline entry.

    Returns None for UI-only events (speaking markers, mute, ttfb-metric
    which duplicates the breakdown).
    """
    if not isinstance(ev, dict):
        return None
    t = ev.get("type", "")
    p = ev.get("payload") or {}
    base = {
        "turn": ev.get("turn"),
        "node": ev.get("node_name"),
        "t": ev.get("timestamp") or p.get("timestamp"),
    }
    if t == _USER:
        # Interim transcripts are noise; only final ones (or legacy events
        # without the flag) are kept.
        if p.get("final") is False:
            return None
        out: dict[str, Any] = {"kind": "user", "text": p.get("text")}
        if p.get("turn_interrupted"):
            out["turn_interrupted"] = True
    elif t == _BOT:
        out = {"kind": "bot", "text": p.get("text")}
        if p.get("turn_interrupted"):
            out["turn_interrupted"] = True
    elif t == _FN_START:
        out = {
            "kind": "tool_call",
            "fn": p.get("function_name"),
            "args": parse_tool_result(p.get("arguments")),
        }
    elif t == _FN_END:
        out = {
            "kind": "tool_result",
            "fn": p.get("function_name"),
            "result": parse_tool_result(p.get("result")),
        }
    elif t == _TRANSITION:
        out = {
            "kind": "transition",
            "from": p.get("previous_node_name"),
            "to": p.get("node_name"),
        }
    elif t == _LATENCY:
        out = {"kind": "latency", "secs": _round(p.get("latency_seconds")) or 0.0}
    elif t == LATENCY_BREAKDOWN_EVENT_TYPE:
        out = {
            "kind": "latency_breakdown",
            "total": _round(p.get("total_secs")) or 0.0,
            "user_turn": _round(p.get("user_turn_secs")),
            "ttfb": {
                m.get("processor", "?"): _round(m.get("duration_secs"))
                for m in p.get("ttfb") or []
                if isinstance(m, dict)
            },
            "parts": [
                {
                    "key": c.get("key"),
                    "owner": c.get("owner"),
                    "secs": _round(c.get("duration_secs")),
                }
                for c in p.get("contributions") or []
                if isinstance(c, dict)
            ],
            "fn_calls": [
                {"fn": f.get("function_name"), "secs": _round(f.get("duration_secs"))}
                for f in p.get("function_calls") or []
                if isinstance(f, dict)
            ],
        }
    elif t == _ERROR:
        out = {
            "kind": "error",
            "error": p.get("error"),
            "fatal": bool(p.get("fatal")),
            "processor": p.get("processor"),
        }
    elif t == LLM_USAGE_EVENT_TYPE:
        out = {
            "kind": "llm_usage",
            "prompt_tokens": p.get("prompt_tokens"),
            "completion_tokens": p.get("completion_tokens"),
        }
    else:
        return None
    return {**base, **out}


def is_call_ending_function(name: Optional[str]) -> bool:
    lowered = (name or "").lower()
    return any(marker in lowered for marker in CALL_ENDING_FUNCTION_MARKERS)


def artifact_turns(timeline: list[dict]) -> set:
    """Turns whose latency must be excluded from Layer 2.

    A turn qualifies when it invokes a call-ending function, or when its last
    conversational entry is a tool call/result/transition (no bot speech after
    the tool call), so the latency interval spans until the next speech.
    """
    by_turn: dict[Any, list[dict]] = {}
    for entry in timeline:
        if entry.get("turn") is None:
            continue
        by_turn.setdefault(entry["turn"], []).append(entry)

    excluded: set = set()
    for turn, entries in by_turn.items():
        conversational = [e for e in entries if e["kind"] in CONVERSATIONAL_KINDS]
        if any(
            e["kind"] == "tool_call" and is_call_ending_function(e.get("fn"))
            for e in conversational
        ):
            excluded.add(turn)
            continue
        if any(
            is_call_ending_function(f.get("fn"))
            for e in entries
            if e["kind"] == "latency_breakdown"
            for f in e.get("fn_calls") or []
        ):
            excluded.add(turn)
            continue
        tool_positions = [
            i for i, e in enumerate(conversational) if e["kind"] == "tool_call"
        ]
        if not tool_positions:
            continue
        after_last_tool = conversational[tool_positions[-1] + 1 :]
        if not any(e["kind"] == "bot" for e in after_last_tool):
            excluded.add(turn)
    return excluded


def build_timeline(events: Iterable[dict]) -> list[dict]:
    """Compact events and mark latency artifacts as ``excluded``."""
    timeline = [c for c in (compact_event(e) for e in events or []) if c]
    excluded = artifact_turns(timeline)
    for entry in timeline:
        if entry["kind"] in ("latency", "latency_breakdown") and (
            entry.get("turn") in excluded
        ):
            entry["excluded"] = True
    return timeline


def run_events(run: dict) -> list[dict]:
    """Realtime-feedback events of a run row (``logs`` may be a JSON string)."""
    logs = as_obj(run.get("logs")) or {}
    if isinstance(logs, list):
        return logs
    if not isinstance(logs, dict):
        return []
    events = logs.get("realtime_feedback_events") or []
    return events if isinstance(events, list) else []


def build_run_timeline(run: dict) -> dict:
    """Compact view of one run row from ``get_runs_with_logs_for_workflow``.

    Raw logs are dropped here so the worker never holds 100 runs of raw
    events (P2): everything downstream works off this dict.
    """
    gathered = as_obj(run.get("gathered_context")) or {}
    initial = as_obj(run.get("initial_context")) or {}
    usage = as_obj(run.get("usage_info")) or {}
    events = run_events(run)
    created_at = run.get("created_at")
    return {
        "run_id": run.get("id"),
        "created_at": created_at.isoformat()
        if hasattr(created_at, "isoformat")
        else created_at,
        "definition_id": run.get("definition_id"),
        "mode": run.get("mode"),
        "has_events": bool(events),
        "disposition": gathered.get("mapped_call_disposition")
        or gathered.get("call_disposition")
        if isinstance(gathered, dict)
        else None,
        "initial_context": initial if isinstance(initial, dict) else {},
        "extracted_variables": gathered.get("extracted_variables")
        if isinstance(gathered, dict)
        else None,
        "gathered_context": gathered if isinstance(gathered, dict) else {},
        "usage_info": usage if isinstance(usage, dict) else {},
        "timeline": build_timeline(events),
    }


def first_bot_text(timeline: list[dict]) -> Optional[str]:
    for entry in timeline:
        if entry["kind"] == "bot" and (entry.get("text") or "").strip():
            return entry["text"]
    return None


def format_timeline_for_llm(timeline: list[dict]) -> str:
    """Render a timeline as compact lines for Claude (one entry per line)."""
    lines: list[str] = []
    for e in timeline:
        prefix = f"[turn {e.get('turn')} | {e.get('node') or '?'}]"
        kind = e["kind"]
        if kind in ("user", "bot"):
            flag = " (interrupted)" if e.get("turn_interrupted") else ""
            lines.append(f"{prefix} {kind}{flag}: {e.get('text')}")
        elif kind == "tool_call":
            args = json.dumps(e.get("args"), ensure_ascii=False, default=str)
            lines.append(f"{prefix} tool_call {e.get('fn')}({args})")
        elif kind == "tool_result":
            result = json.dumps(e.get("result"), ensure_ascii=False, default=str)
            if len(result) > 1500:
                result = result[:1500] + "…[truncated]"
            lines.append(f"{prefix} tool_result {e.get('fn')}: {result}")
        elif kind == "transition":
            lines.append(f"{prefix} transition {e.get('from')} -> {e.get('to')}")
        elif kind == "error":
            fatal = "FATAL " if e.get("fatal") else ""
            lines.append(
                f"{prefix} {fatal}pipeline_error ({e.get('processor')}): {e.get('error')}"
            )
        elif kind == "latency" and not e.get("excluded"):
            lines.append(f"{prefix} latency {e.get('secs')}s")
        # latency_breakdown is Layer 2 material; omitted to save tokens.
    return "\n".join(lines)
