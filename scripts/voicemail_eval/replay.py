"""Replay a harvested voicemail dataset through both classifiers and compare.

Calls are slow and cost money; replay is free. Every scenario call yields one
stored transcript, and this script runs those transcripts through the current
gpt-4.1 classifier and through Jev as many times as we like -- which is where
prompt iteration actually happens, without redialling anyone.

    export OPENROUTER_API_KEY=sk-or-v1-...
    python -m scripts.voicemail_eval.replay \
        --dataset scripts/voicemail_eval/dataset.jsonl \
        --output scripts/voicemail_eval/results.json

One key drives both arms: gpt-4.1 goes through OpenRouter's chat endpoint, Jev
through OpenRouter's Decisions API.

Fidelity matters more than convenience here. Both arms reuse the exact artefacts
the live pipeline uses -- ``VoicemailDetector.DEFAULT_SYSTEM_PROMPT`` and its
``"CONVERSATION" in response`` matching for the incumbent,
``typesafe_jev_llm.CLASSIFICATION_CRITERIA`` and ``_confidence_of`` for Jev. If
this file ever paraphrases either of them, the comparison stops describing
production and the numbers become decoration.
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp

from api.services.pipecat.typesafe_jev_llm import (
    CLASSIFICATION_CRITERIA,
    OPENROUTER_BASE_URL,
    _KEYWORD_BY_OPTION,
    _confidence_of,
    build_payload,
)
from api.services.pipecat.voicemail_timing import describe_timing
from pipecat.extensions.voicemail.voicemail_detector import VoicemailDetector

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_BASELINE_MODEL = "openai/gpt-4.1"
DEFAULT_JEV_MODEL = "typesafe/jev-1.13"

#: The incumbent prompt asks for exactly one word. Capping keeps a
#: limited-credit OpenRouter account from 402-ing on the reserved window.
BASELINE_MAX_TOKENS = 16

#: Thresholds swept to find where holding beats guessing.
THRESHOLD_SWEEP = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9)

#: carrier collapses to voicemail in the live pipeline, so score against the
#: two-way decision the detector actually makes.
BINARY = {"human": "human", "voicemail": "voicemail", "carrier": "voicemail"}


class Verdict:
    """One classifier's answer for one scenario."""

    def __init__(
        self,
        label: str | None,
        latency_s: float,
        confidence: float | None = None,
        raw: Any = None,
    ):
        self.label = label  # "human" | "voicemail" | None (declined / failed)
        self.latency_s = latency_s
        self.confidence = confidence
        self.raw = raw


def _fail_on_account_error(arm: str, status: int, body: Any) -> None:
    """Abort the run on auth or credit errors.

    Every later call would fail the same way, and each failure would be scored
    as a held case, quietly turning an empty account into a cautious classifier.
    """
    if status in (401, 402, 403):
        raise SystemExit(
            f"{arm}: HTTP {status} from OpenRouter, stopping the run. Check the key "
            f"and its credit balance. {str(body)[:200]}"
        )


async def classify_baseline(
    session: aiohttp.ClientSession, utterances: list[str], *, model: str
) -> Verdict:
    """The incumbent: gpt-4.1 with the live prompt and the live string match."""
    messages = [
        # role "developer" mirrors how VoicemailDetector seeds its context.
        {"role": "developer", "content": VoicemailDetector.DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(utterances)},
    ]
    started = time.perf_counter()
    async with session.post(
        OPENROUTER_CHAT_URL,
        json={
            "model": model,
            "messages": messages,
            "temperature": 0,
            # The prompt demands a single word, so a tight cap cannot change
            # which keyword comes back. Without it OpenRouter reserves the
            # model's full window up front and rejects the request on a
            # limited-credit account with a 402.
            "max_tokens": BASELINE_MAX_TOKENS,
        },
    ) as resp:
        body = await resp.json()
    latency = time.perf_counter() - started

    if resp.status != 200:
        _fail_on_account_error("baseline", resp.status, body)
        # Surfaced rather than swallowed: a 429 here would otherwise look
        # identical to "the model declined to classify".
        print(f"  !! baseline HTTP {resp.status}: {str(body)[:160]}", file=sys.stderr)
        return Verdict(None, latency, raw=body)

    text = (body["choices"][0]["message"]["content"] or "").upper()
    # Exactly the precedence ClassificationProcessor._process_classification uses.
    if "CONVERSATION" in text:
        return Verdict("human", latency, raw=text)
    if "VOICEMAIL" in text:
        return Verdict("voicemail", latency, raw=text)
    return Verdict(None, latency, raw=text)


async def classify_jev(
    session: aiohttp.ClientSession,
    utterances: list[str],
    *,
    model: str,
    criteria: dict[str, str],
    timing: dict | None = None,
) -> Verdict:
    """Jev through OpenRouter's Decisions API, same question as the live service.

    ``timing`` is a ``describe_timing`` view of the callee's speech; when given,
    it travels in the state next to the transcript.
    """
    payload = build_payload(utterances, model=model, timing=timing)
    payload["questions"]["answered_by"]["criteria"] = criteria
    started = time.perf_counter()
    async with session.post(OPENROUTER_BASE_URL, json=payload) as resp:
        body = await resp.json()
    latency = time.perf_counter() - started

    if resp.status != 200:
        _fail_on_account_error("jev", resp.status, body)
        # Surfaced, and never scored as "held": an API error is not Jev
        # declining to classify.
        print(f"  !! jev HTTP {resp.status}: {str(body)[:160]}", file=sys.stderr)
        return Verdict(None, latency, raw=body)

    answer = (body.get("answers") or {}).get("answered_by") or {}
    option = answer.get("choice")
    confidence = _confidence_of(answer)
    if option not in _KEYWORD_BY_OPTION:
        return Verdict(None, latency, confidence, raw=body)
    # Keep the raw three-way option; collapse only when scoring.
    return Verdict(BINARY[option], latency, confidence, raw=option)


def _pct(n: int, d: int) -> str:
    return f"{(100.0 * n / d):.0f}%" if d else "n/a"


def _wilson(k: int, n: int, z: float = 1.96) -> str:
    """95% Wilson interval for k successes in n, as '91-97%'."""
    if not n:
        return "n/a"
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return f"{100 * max(0.0, centre - half):.0f}-{100 * min(1.0, centre + half):.0f}%"


#: Above this many records, per-scenario tables list only the rows some
#: classifier got wrong.
ROWS_ALL_UP_TO = 40


def _summarise(
    name: str, records: list[dict], verdicts: list[Verdict], threshold: float | None
) -> dict:
    correct = declined = wrong = 0
    for rec, v in zip(records, verdicts):
        expected = BINARY[rec["expected_label"]]
        label = v.label
        if (
            threshold is not None
            and v.confidence is not None
            and v.confidence < threshold
        ):
            label = None  # held: below the gate, no verdict emitted
        if label is None:
            declined += 1
        elif label == expected:
            correct += 1
        else:
            wrong += 1
    lat = [v.latency_s for v in verdicts if v.latency_s]
    return {
        "name": name,
        "n": len(records),
        "correct": correct,
        "wrong": wrong,
        "declined": declined,
        "accuracy_of_answered": _pct(correct, correct + wrong),
        "ci95": _wilson(correct, correct + wrong),
        "p50_ms": round(1000 * statistics.median(lat)) if lat else None,
        "p95_ms": round(1000 * sorted(lat)[max(0, int(len(lat) * 0.95) - 1)])
        if lat
        else None,
    }


def _print_summary(rows: list[dict]) -> None:
    print()
    print(
        f"{'classifier':<28}{'n':>4}{'right':>7}{'wrong':>7}{'held':>6}{'acc*':>7}"
        f"{'95% CI':>10}{'p50':>8}{'p95':>8}"
    )
    print("-" * 85)
    for r in rows:
        print(
            f"{r['name']:<28}{r['n']:>4}{r['correct']:>7}{r['wrong']:>7}"
            f"{r['declined']:>6}{r['accuracy_of_answered']:>7}{r['ci95']:>10}"
            f"{(str(r['p50_ms']) + 'ms') if r['p50_ms'] else '-':>8}"
            f"{(str(r['p95_ms']) + 'ms') if r['p95_ms'] else '-':>8}"
        )
    print(
        "* accuracy is over answered cases only; held cases are neither right nor wrong."
    )


def _print_per_scenario(
    records: list[dict], base: list[Verdict], jev: list[Verdict]
) -> None:
    print()
    print(f"{'id':<8}{'expected':<11}{'live':<11}{'gpt-4.1':<11}{'jev':<11}{'conf':>6}")
    print("-" * 60)
    only_wrong = len(records) > ROWS_ALL_UP_TO
    for rec, b, j in zip(records, base, jev):
        expected = BINARY[rec["expected_label"]]
        if only_wrong and b.label in (expected, None) and j.label == expected:
            continue
        mark = lambda got: (got or "-") + ("" if got in (expected, None) else " X")  # noqa: E731
        conf = f"{j.confidence:.2f}" if j.confidence is not None else "-"
        print(
            f"{rec['scenario_id']:<8}{expected:<11}{rec['detector_verdict'] or '-':<11}"
            f"{mark(b.label):<11}{mark(j.label):<11}{conf:>6}"
        )
    print("X marks a disagreement with the known label. With n~20 these individual")
    print("rows carry more signal than the percentages above.")


def _by_category(
    records: list[dict], arms: list[tuple[str, list[Verdict]]]
) -> list[dict]:
    """Right/answered per synthetic category, for each arm. Empty for real calls."""
    cats = sorted({r["category"] for r in records if r.get("category")})
    rows = []
    for cat in cats:
        idx = [i for i, r in enumerate(records) if r.get("category") == cat]
        row: dict[str, Any] = {"category": cat, "n": len(idx)}
        for name, verdicts in arms:
            s = _summarise(name, [records[i] for i in idx], [verdicts[i] for i in idx], None)
            row[name] = f"{s['correct']}/{s['correct'] + s['wrong']}"
        rows.append(row)
    if rows:
        names = [n for n, _ in arms]
        print()
        print(f"{'category':<14}{'n':>5}" + "".join(f"{n:>14}" for n in names))
        print("-" * (19 + 14 * len(names)))
        for row in rows:
            print(
                f"{row['category']:<14}{row['n']:>5}"
                + "".join(f"{row[n]:>14}" for n in names)
            )
    return rows


def _print_timing_per_scenario(
    records: list[dict], plain: list[Verdict], timed: list[Verdict], views: list[dict]
) -> None:
    print()
    print(
        f"{'id':<8}{'expected':<11}{'jev':<13}{'jev+timing':<13}{'conf':>6}"
        f"{'onset':>7}{'first':>7}{'bursts':>8}{'1st-half':>10}"
    )
    print("-" * 83)
    only_wrong = len(records) > ROWS_ALL_UP_TO
    for rec, p, t, v in zip(records, plain, timed, views):
        expected = BINARY[rec["expected_label"]]
        if only_wrong and p.label == expected and t.label == expected:
            continue
        mark = lambda got: (got or "-") + ("" if got == expected else " X")  # noqa: E731
        conf = f"{t.confidence:.2f}" if t.confidence is not None else "-"
        onset = v["first_speech_after_seconds"]
        half = v["share_of_speech_in_first_half"]
        print(
            f"{rec['scenario_id']:<8}{expected:<11}{mark(p.label):<13}"
            f"{mark(t.label):<13}{conf:>6}"
            f"{(f'{onset}s' if onset is not None else '-'):>7}"
            f"{v['first_speech_lasted_seconds']:>6}s"
            f"{v['separate_bursts_of_speech']:>8}"
            f"{(half if half is not None else '-'):>10}"
        )


def _cross_check(records: list[dict], base: list[Verdict]) -> None:
    """The most important check: does replay reproduce what production did?

    Every number this script prints is downstream of the assumption that
    replaying a stored transcript reproduces the live classifier's decision. If
    it does not, the harness is measuring something other than the pipeline.
    """
    comparable = [
        (r, v)
        for r, v in zip(records, base)
        if v.label is not None and r.get("detector_verdict")
    ]
    if not comparable:
        if not any(r.get("detector_verdict") for r in records):
            return  # synthetic data: there is no live decision to compare with
        print("\n[cross-check] no comparable runs")
        return
    agree = sum(1 for r, v in comparable if r["detector_verdict"] == v.label)
    print(
        f"\n[cross-check] replayed gpt-4.1 matches the live detector on "
        f"{agree}/{len(comparable)} runs ({_pct(agree, len(comparable))})"
    )
    if agree < len(comparable):
        drift = [
            r["scenario_id"] for r, v in comparable if r["detector_verdict"] != v.label
        ]
        print(f"              diverged on: {drift}")
        print("              investigate before trusting anything above -- either the")
        print(
            "              replay differs from live, or the model is non-deterministic."
        )


async def main(args: argparse.Namespace) -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY first.", file=sys.stderr)
        return 2

    dataset = Path(args.dataset)
    if not dataset.exists():
        print(f"dataset not found: {dataset} (run harvest.py first)", file=sys.stderr)
        return 2
    records = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [r for r in records if r.get("opening_utterances")]
    if not records:
        print("no records with captured speech", file=sys.stderr)
        return 2

    print(f"-> replaying {len(records)} record(s) from {dataset}")

    headers = {"Authorization": f"Bearer {api_key}"}
    results: dict[str, Any] = {"n": len(records), "arms": []}

    # Sequential on purpose: concurrent requests contend and make the latency
    # numbers meaningless, which is the mistake the TypeSafe cookbook calls out.
    async with aiohttp.ClientSession(headers=headers) as session:
        # --skip-baseline: gpt-4.1's answers do not change while only the Jev
        # question is being iterated, so re-asking it wastes time and credit.
        base_verdicts = (
            [Verdict(None, 0.0) for _ in records]
            if args.skip_baseline
            else [
                await classify_baseline(
                    session, r["opening_utterances"], model=args.baseline_model
                )
                for r in records
            ]
        )
        jev_verdicts = [
            await classify_jev(
                session,
                r["opening_utterances"],
                model=args.jev_model,
                criteria=CLASSIFICATION_CRITERIA,
            )
            for r in records
        ]

        rows = [
            *(
                []
                if args.skip_baseline
                else [_summarise("gpt-4.1 (incumbent)", records, base_verdicts, None)]
            ),
            _summarise(f"jev (no gate)", records, jev_verdicts, None),
            _summarise(
                f"jev @ conf>={args.threshold}", records, jev_verdicts, args.threshold
            ),
        ]
        _print_summary(rows)
        _print_per_scenario(records, base_verdicts, jev_verdicts)
        _cross_check(records, base_verdicts)
        results["arms"] = rows
        results["by_category"] = _by_category(
            records,
            ([] if args.skip_baseline else [("gpt-4.1", base_verdicts)])
            + [("jev", jev_verdicts)],
        )

        # Transcript plus callee speech timing (timing.py must have run).
        window = str(args.timing_window)
        timed = [
            (i, r)
            for i, r in enumerate(records)
            if args.timing and window in (r.get("timing") or {})
        ]
        if timed:
            views = [
                describe_timing(r["timing"][window], 1000.0 * args.timing_window)
                for _, r in timed
            ]
            timed_verdicts = [
                await classify_jev(
                    session,
                    r["opening_utterances"],
                    model=args.jev_model,
                    criteria=CLASSIFICATION_CRITERIA,
                    timing=view,
                )
                for (_, r), view in zip(timed, views)
            ]
            subset = [r for _, r in timed]
            plain = [jev_verdicts[i] for i, _ in timed]
            timing_rows = [
                _summarise("jev (same subset)", subset, plain, None),
                _summarise(f"jev + timing@{window}s", subset, timed_verdicts, None),
                _summarise(
                    f"jev + timing @ conf>={args.threshold}",
                    subset,
                    timed_verdicts,
                    args.threshold,
                ),
            ]
            print(f"\n== with callee speech timing ({window}s window) ==")
            _print_summary(timing_rows)
            _print_timing_per_scenario(subset, plain, timed_verdicts, views)
            results["timing_arms"] = timing_rows
            results["timing_by_category"] = _by_category(
                subset, [("jev", plain), ("jev+timing", timed_verdicts)]
            )
            results["timing_per_scenario"] = [
                {
                    "scenario_id": r["scenario_id"],
                    "expected": BINARY[r["expected_label"]],
                    "jev": p.label,
                    "jev_timing": t.label,
                    "jev_timing_option": t.raw if isinstance(t.raw, str) else None,
                    "jev_timing_confidence": t.confidence,
                    "timing": view,
                }
                for r, p, t, view in zip(subset, plain, timed_verdicts, views)
            ]
        elif args.timing:
            print(
                f"\n(no records carry {window}s timing; run scripts.voicemail_eval.timing"
                " to add the jev + timing arm)"
            )

        # Where does holding start to beat guessing?
        print()
        print(f"{'threshold':<12}{'right':>7}{'wrong':>7}{'held':>6}")
        print("-" * 32)
        sweep = []
        for t in THRESHOLD_SWEEP:
            s = _summarise(f"t={t}", records, jev_verdicts, t)
            sweep.append(
                {"threshold": t, **{k: s[k] for k in ("correct", "wrong", "declined")}}
            )
            print(f"{t:<12}{s['correct']:>7}{s['wrong']:>7}{s['declined']:>6}")
        results["threshold_sweep"] = sweep

        # How early could we have committed?
        if args.truncations:
            print()
            print(f"{'window':<12}{'right':>7}{'wrong':>7}{'held':>6}{'no speech':>11}")
            print("-" * 43)
            trunc = []
            for cutoff in sorted(records[0].get("opening_by_seconds", {}), key=float):
                subset, empty = [], 0
                for r in records:
                    utts = r["opening_by_seconds"].get(cutoff) or []
                    if not utts:
                        empty += 1
                        continue
                    subset.append((r, utts))
                if not subset:
                    continue
                vs = [
                    await classify_jev(
                        session,
                        utts,
                        model=args.jev_model,
                        criteria=CLASSIFICATION_CRITERIA,
                    )
                    for _, utts in subset
                ]
                s = _summarise(
                    f"jev@{cutoff}s", [r for r, _ in subset], vs, args.threshold
                )
                trunc.append(
                    {
                        "seconds": float(cutoff),
                        "no_speech": empty,
                        **{k: s[k] for k in ("correct", "wrong", "declined")},
                    }
                )
                print(
                    f"{cutoff + 's':<12}{s['correct']:>7}{s['wrong']:>7}{s['declined']:>6}{empty:>11}"
                )
            results["truncation_sweep"] = trunc

    results["per_scenario"] = [
        {
            "scenario_id": r["scenario_id"],
            "expected": BINARY[r["expected_label"]],
            "live_detector": r["detector_verdict"],
            "gpt41": b.label,
            "jev": j.label,
            "jev_option": j.raw if isinstance(j.raw, str) else None,
            "jev_confidence": j.confidence,
        }
        for r, b, j in zip(records, base_verdicts, jev_verdicts)
    ]

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nOK wrote {out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="scripts/voicemail_eval/dataset.jsonl")
    parser.add_argument(
        "--output", default=None, help="Write the full result JSON here."
    )
    parser.add_argument("--baseline-model", default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--jev-model", default=DEFAULT_JEV_MODEL)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument(
        "--timing-window",
        type=float,
        default=5.0,
        help="Which timing.py window to offer Jev, in seconds.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Do not re-ask gpt-4.1; for iterating on the Jev question alone.",
    )
    parser.add_argument(
        "--no-timing",
        dest="timing",
        action="store_false",
        help="Skip the jev + timing arm.",
    )
    parser.add_argument(
        "--no-truncations",
        dest="truncations",
        action="store_false",
        help="Skip the how-early-could-we-commit sweep (saves one call per window).",
    )
    sys.exit(asyncio.run(main(parser.parse_args())) or 0)
