"""Layer 2: latency per node from compact run timelines.

Per (run, turn) the ``latency_breakdown`` total is used; turns without a
breakdown (legacy runs) fall back to ``latency`` (rtf-latency-measured).
Entries marked ``excluded`` by ``timeline.build_timeline`` (turns ending in a
tool call or end_call) are skipped: their interval runs to the next speech.
Breakdowns whose total exceeds the measured service time (STT/user turn +
LLM + TTS TTFB) by more than ``IDLE_GAP_SECONDS`` are skipped too: that time
was spent waiting (a hang or an idle reminder), which Layer 1 reports as a
behavior problem, not processing latency.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

from api.services.workflow.doctor.contracts import LatencyRow

#: Unexplained seconds in a breakdown above which the turn counts as waiting.
IDLE_GAP_SECONDS = 5.0


def _is_waiting(entry: dict) -> bool:
    measured = sum(float(v) for v in (entry.get("ttfb") or {}).values() if v)
    measured += float(entry.get("user_turn") or 0)
    return measured > 0 and float(entry["total"]) - measured > IDLE_GAP_SECONDS


def percentile(values: list[float], q: float) -> Optional[float]:
    """Nearest-rank percentile (same rule as the call-doctor export)."""
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 3)


def _service_kind(processor: str) -> Optional[str]:
    upper = (processor or "").upper()
    if "STT" in upper:
        return "stt"
    if "TTS" in upper:
        return "tts"
    if "LLM" in upper:
        return "llm"
    return None


def compute_latency(timelines: Iterable[list[dict]]) -> list[LatencyRow]:
    """p50/p90 user→bot latency per node, plus per-service TTFB p50.

    ``timelines`` are the ``timeline`` lists of compact run views. Rows are
    sorted by p90 descending (slowest step first).
    """
    totals: dict[str, list[float]] = defaultdict(list)
    ttfb: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for timeline in timelines:
        breakdown_turns: dict[tuple, Optional[dict]] = {}
        measured_turns: dict[tuple, dict] = {}
        for entry in timeline:
            if entry.get("excluded"):
                continue
            key = (entry.get("node") or "?", entry.get("turn"))
            if entry.get("kind") == "latency_breakdown" and entry.get("total"):
                if _is_waiting(entry):
                    # Also drop the measured fallback for this turn.
                    breakdown_turns.setdefault(key, None)
                    continue
                breakdown_turns.setdefault(key, entry)
            elif entry.get("kind") == "latency" and entry.get("secs"):
                measured_turns.setdefault(key, entry)

        for (node, _), entry in breakdown_turns.items():
            if entry is None:
                continue
            totals[node].append(float(entry["total"]))
            stt_seen = False
            for proc, secs in (entry.get("ttfb") or {}).items():
                kind = _service_kind(proc)
                if kind and secs is not None:
                    ttfb[node][kind].append(float(secs))
                    stt_seen = stt_seen or kind == "stt"
            if not stt_seen and entry.get("user_turn") is not None:
                # STT finalisation is folded into the user turn when the STT
                # service reports no TTFB of its own.
                ttfb[node]["stt"].append(float(entry["user_turn"]))
        for key, entry in measured_turns.items():
            if key in breakdown_turns:
                continue
            totals[key[0]].append(float(entry["secs"]))

    rows = [
        LatencyRow(
            node_name=node,
            turns=len(values),
            p50_s=percentile(values, 0.5) or 0.0,
            p90_s=percentile(values, 0.9) or 0.0,
            llm_ttfb_p50_s=percentile(ttfb[node].get("llm", []), 0.5),
            stt_p50_s=percentile(ttfb[node].get("stt", []), 0.5),
            tts_ttfb_p50_s=percentile(ttfb[node].get("tts", []), 0.5),
        )
        for node, values in totals.items()
        if values
    ]
    rows.sort(key=lambda r: r.p90_s, reverse=True)
    return rows
