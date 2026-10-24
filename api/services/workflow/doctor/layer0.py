"""Layer 0 per-run platform checks: did the pipeline run as designed?

Checks (design doc "Doctor check definitions"):

- **greeting**: first bot text vs the start node greeting rendered with the
  call variables; normalised similarity < ``L0_GREETING_SIMILARITY_MIN``
  flags (a generic "Hello! How can I help you today?" means the node prompt
  never loaded). Skipped when the start node has no text greeting.
- **prompt load**: first-turn LLM prompt tokens vs the rendered start-node
  prompt token estimate (len/4); below ``L0_PROMPT_TOKEN_RATIO_MIN`` flags
  ``system_prompt_missing``.
- **pipeline errors**: any fatal ``rtf-pipeline-error`` or ≥2 from the same
  processor.

Runs missing the inputs a check needs (legacy runs) are counted in
``insufficient_data[check]``; missing data is never reported as a failure.
Inputs are the compact run views from ``timeline.build_run_timeline``.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from typing import Any, Optional

from api.services.workflow.doctor.contracts import (
    L0_GREETING_SIMILARITY_MIN,
    L0_PROMPT_TOKEN_RATIO_MIN,
    Layer0Flag,
    Layer0Summary,
)
from api.services.workflow.doctor.timeline import first_bot_text
from api.services.workflow.dto import NodeType
from api.utils.template_renderer import render_template

GREETING = "greeting_mismatch"
PROMPT = "system_prompt_missing"
ERRORS = "pipeline_errors"

#: Timeline kind of a per-turn LLM usage record (timeline.LLM_USAGE_EVENT_TYPE).
LLM_USAGE_EVENT_KINDS = ("llm_usage",)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node_by_type(workflow_json: dict, node_type: str) -> Optional[dict]:
    for n in (workflow_json or {}).get("nodes") or []:
        if isinstance(n, dict) and n.get("type") == node_type:
            return n
    return None


def _data(node: Optional[dict]) -> dict:
    data = (node or {}).get("data")
    return data if isinstance(data, dict) else {}


def call_context(run: dict) -> dict:
    """Variables the engine renders templates with (initial context merged)."""
    initial = run.get("initial_context") or {}
    gathered = run.get("gathered_context") or {}
    return {
        **initial,
        "initial_context": initial,
        "gathered_context": gathered,
    }


def normalise(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def greeting_similarity(expected: str, actual: str) -> float:
    """Similarity of the bot's first line to the expected greeting.

    The first bot text may continue past the greeting (greeting + LLM
    sentence), so the best of full-text and same-length-prefix is used.
    """
    exp, act = normalise(expected), normalise(actual)
    if not exp or not act:
        return 0.0
    full = difflib.SequenceMatcher(None, exp, act).ratio()
    prefix = difflib.SequenceMatcher(None, exp, act[: len(exp) + 5]).ratio()
    return max(full, prefix)


def expected_greeting(run: dict, workflow_json: dict) -> Optional[str]:
    """Rendered start greeting for this run, or None if there is none."""
    context = call_context(run)
    override = (run.get("initial_context") or {}).get("greeting_override")
    if isinstance(override, dict):
        if override.get("type") == "text" and override.get("text"):
            return str(render_template(override["text"], context))
        if override.get("type") == "audio":
            return None
    start = _data(_node_by_type(workflow_json, NodeType.startNode.value))
    if start.get("greeting_type") == "audio":
        return None
    greeting = start.get("greeting")
    if not isinstance(greeting, str) or not greeting.strip():
        return None
    return str(render_template(greeting, context))


def expected_prompt_tokens(run: dict, workflow_json: dict) -> Optional[int]:
    """Token estimate (len/4) of the rendered start prompt (+ global prompt)."""
    context = call_context(run)
    start = _data(_node_by_type(workflow_json, NodeType.startNode.value))
    prompt = start.get("prompt") or ""
    if start.get("add_global_prompt", True):
        global_prompt = _data(
            _node_by_type(workflow_json, NodeType.globalNode.value)
        ).get("prompt")
        if global_prompt:
            prompt = f"{global_prompt}\n{prompt}"
    if not prompt.strip():
        return None
    rendered = str(render_template(prompt, context))
    return max(1, len(rendered) // 4)


def first_turn_prompt_tokens(run: dict) -> Optional[int]:
    """LLM prompt tokens of the first turn, if the run recorded them.

    ``usage_info`` today aggregates tokens per processor|||model for the whole
    call, which cannot answer "did the first turn carry the system prompt".
    Supported per-turn shapes (any of):
      - ``usage_info["llm_turns"][0]["prompt_tokens"]``
      - ``usage_info["first_turn_llm"]["prompt_tokens"]``
      - a timeline entry of kind ``llm_usage`` with ``prompt_tokens``
    Anything else → None (insufficient data).
    """
    usage = run.get("usage_info") or {}
    turns = usage.get("llm_turns") or usage.get("per_turn_llm")
    if isinstance(turns, list) and turns and isinstance(turns[0], dict):
        value = turns[0].get("prompt_tokens")
        if isinstance(value, (int, float)):
            return int(value)
    first = usage.get("first_turn_llm")
    if isinstance(first, dict) and isinstance(first.get("prompt_tokens"), (int, float)):
        return int(first["prompt_tokens"])
    for entry in run.get("timeline") or []:
        if entry.get("kind") in LLM_USAGE_EVENT_KINDS and isinstance(
            entry.get("prompt_tokens"), (int, float)
        ):
            return int(entry["prompt_tokens"])
    return None


# ---------------------------------------------------------------------------
# Checks: each returns (flag | None, insufficient: bool)
# ---------------------------------------------------------------------------


def check_greeting(run: dict, workflow_json: dict) -> tuple[Optional[Layer0Flag], bool]:
    expected = expected_greeting(run, workflow_json)
    if expected is None:
        return None, False  # skipped: nothing to compare against
    actual = first_bot_text(run.get("timeline") or [])
    if actual is None:
        return None, True
    score = greeting_similarity(expected, actual)
    if score < L0_GREETING_SIMILARITY_MIN:
        return (
            Layer0Flag(
                run_id=run["run_id"],
                check=GREETING,
                detail=f'First bot line "{actual[:160]}" does not match the start '
                f'greeting "{expected[:160]}" (similarity {score:.2f}); the start '
                "node prompt may not have loaded.",
            ),
            False,
        )
    return None, False


def check_prompt_load(
    run: dict, workflow_json: dict
) -> tuple[Optional[Layer0Flag], bool]:
    actual = first_turn_prompt_tokens(run)
    expected = expected_prompt_tokens(run, workflow_json)
    if actual is None or expected is None:
        return None, actual is None
    ratio = actual / expected
    if ratio < L0_PROMPT_TOKEN_RATIO_MIN:
        return (
            Layer0Flag(
                run_id=run["run_id"],
                check=PROMPT,
                detail=f"First LLM turn used {actual} prompt tokens; the start node "
                f"prompt alone is ~{expected} tokens ({ratio:.0%}). The system "
                "prompt likely did not load.",
            ),
            False,
        )
    return None, False


def check_pipeline_errors(run: dict) -> tuple[Optional[Layer0Flag], bool]:
    if not run.get("has_events", True):
        return None, True
    errors = [e for e in run.get("timeline") or [] if e.get("kind") == "error"]
    if not errors:
        return None, False
    fatal = [e for e in errors if e.get("fatal")]
    by_proc = Counter(e.get("processor") or "unknown" for e in errors)
    repeated = {p: c for p, c in by_proc.items() if c >= 2}
    if not fatal and not repeated:
        return None, False
    parts = []
    if fatal:
        parts.append(
            f"{len(fatal)} fatal ({fatal[0].get('processor') or 'unknown'}: "
            f"{str(fatal[0].get('error'))[:160]})"
        )
    for proc, count in repeated.items():
        parts.append(f"{count}× {proc}")
    return (
        Layer0Flag(
            run_id=run["run_id"],
            check=ERRORS,
            detail="Pipeline errors: " + "; ".join(parts),
        ),
        False,
    )


def run_layer0(
    runs: list[dict],
    definitions: dict[Any, dict],
    fallback_workflow_json: Optional[dict] = None,
) -> Layer0Summary:
    """Run all Layer 0 checks over compact run views.

    ``definitions`` maps definition_id → workflow_json; runs whose definition
    is unknown use ``fallback_workflow_json``.
    """
    flags: list[Layer0Flag] = []
    insufficient: Counter = Counter()
    for run in runs:
        wf = definitions.get(run.get("definition_id")) or fallback_workflow_json or {}
        if not run.get("has_events", True):
            for check in (GREETING, PROMPT, ERRORS):
                insufficient[check] += 1
            continue
        for check, (flag, lacking) in (
            (GREETING, check_greeting(run, wf)),
            (PROMPT, check_prompt_load(run, wf)),
            (ERRORS, check_pipeline_errors(run)),
        ):
            if flag:
                flags.append(flag)
            if lacking:
                insufficient[check] += 1
    return Layer0Summary(
        runs_checked=len(runs),
        runs_flagged=len({f.run_id for f in flags}),
        flags=flags,
        insufficient_data=dict(insufficient),
    )
