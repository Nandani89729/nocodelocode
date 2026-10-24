"""Harvest scripted voicemail-eval calls into a replay dataset.

Reads the runs produced by the scenarios in ``scenarios.md`` and writes one
JSONL record per call: the transcript the classifier would have seen, what the
live detector actually decided, and the label we know is true because we acted
it out.

Two sources, because the calls and the validation data live in different places:

    # Cloud calls (the real dataset)
    export DOGRAH_API_KEY=...
    python -m scripts.voicemail_eval.harvest \
        --source api --base-url https://app.dograh.com \
        --runs scripts/voicemail_eval/runs.csv \
        --out scripts/voicemail_eval/dataset.jsonl

    # Local database (for validating this script against real event JSON)
    set -a && source api/.env && set +a
    python -m scripts.voicemail_eval.harvest \
        --source db --runs scripts/voicemail_eval/runs.csv \
        --out /tmp/local.jsonl

``--runs`` is a two-column CSV, ``scenario_id,run_id``, filled in as the calls
are placed. Scenario labels are read from ``scenarios.md``.

On reading the detector's verdict: use ``gathered_context["call_status"]``, not
``call_disposition``. ``voicemail_detected`` is *also* one of the default
``CallDispositionOption`` codes, so the post-call LLM classifier can assign it
to a call the detector never flagged. ``call_status`` is written only by
``PipecatEngine.end_call_with_reason`` and is never LLM-generated, so it is the
only field that answers "what did the detector decide".
"""

import argparse
import asyncio
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from api.utils.transcript import generate_transcript_text
from pipecat.utils.enums import EndTaskReason, RealtimeFeedbackType

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCENARIOS_MD = Path(__file__).resolve().parent / "scenarios.md"

#: Prefixes of the openings we truncate to, answering "how early could we
#: safely have committed?" during replay.
TRUNCATION_SECONDS = (1.0, 2.0, 3.0)

#: Call statuses that mean "the platform concluded a machine answered".
#:
#: Only the first is in our EndTaskReason enum. ``voicemail_no_message`` and
#: ``machine_timeout`` appear in cloud data but exist nowhere in this branch,
#: so the deployed build is ahead of this checkout. They are treated as machine
#: outcomes here because they cannot describe a live person; if they turn out
#: to mean something narrower, this set is the one line to correct.
MACHINE_CALL_STATUSES = frozenset(
    {
        EndTaskReason.VOICEMAIL_DETECTED.value,
        "voicemail_no_message",
        "machine_timeout",
    }
)


#: Twilio returns AnsweredBy as free text and the repo has no enum for it.
#: With MachineDetection=Enable (what the provider sends) the practical set is
#: human / machine_start / fax / unknown, but bucket defensively.
def bucket_answered_by(value: str | None) -> str | None:
    if not value:
        return None
    v = str(value).strip().lower()
    if v.startswith("machine"):
        return "voicemail"
    if v == "human":
        return "human"
    return v  # fax, unknown, or something new — keep it visible


def parse_scenarios(path: Path = SCENARIOS_MD) -> dict[str, str]:
    """Map scenario id -> expected label from the scenarios table."""
    if not path.exists():
        raise SystemExit(f"scenarios file not found: {path}")
    labels: dict[str, str] = {}
    row = re.compile(
        r"^\|\s*(?P<id>[a-z]+-\d+)\s*\|\s*(?P<label>human|voicemail|carrier)\s*\|"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        m = row.match(line.strip())
        if m:
            labels[m.group("id")] = m.group("label")
    if not labels:
        raise SystemExit(f"no scenario rows parsed from {path}")
    return labels


def read_run_map(path: Path) -> list[tuple[str, int]]:
    """Read the scenario_id,run_id CSV produced while placing calls."""
    pairs: list[tuple[str, int]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.reader(f):
            if not raw or raw[0].strip().startswith("#"):
                continue
            if raw[0].strip() == "scenario_id":  # header
                continue
            if len(raw) < 2:
                continue
            pairs.append((raw[0].strip(), int(raw[1].strip())))
    return pairs


def _sorted_events(logs: dict | None) -> list[dict]:
    events = (logs or {}).get("realtime_feedback_events") or []
    # Persisted events are already timestamp-ordered by InMemoryLogsBuffer, but
    # sorting again is free and makes this robust to hand-edited fixtures.
    return sorted(events, key=lambda e: e.get("timestamp") or "")


def _final_user_utterances(events: Iterable[dict]) -> list[tuple[str, str]]:
    """Return (timestamp, text) for each final user transcription."""
    out: list[tuple[str, str]] = []
    for e in events:
        if e.get("type") != RealtimeFeedbackType.USER_TRANSCRIPTION.value:
            continue
        payload = e.get("payload") or {}
        if payload.get("final") is not True:
            continue
        text = (payload.get("text") or "").strip()
        if text:
            out.append((e.get("timestamp") or "", text))
    return out


def _seconds_between(start_iso: str, end_iso: str) -> float | None:
    from datetime import datetime

    try:
        return (
            datetime.fromisoformat(end_iso) - datetime.fromisoformat(start_iso)
        ).total_seconds()
    except (ValueError, TypeError):
        return None


def build_record(
    *,
    scenario_id: str,
    expected_label: str,
    run_id: int,
    workflow_id: int | None,
    gathered_context: dict | None,
    usage_info: dict | None,
    logs: dict | None,
    include_events: bool,
) -> dict[str, Any]:
    events = _sorted_events(logs)
    utterances = _final_user_utterances(events)
    opening = [text for _, text in utterances]

    call_status = (gathered_context or {}).get("call_status")
    detector_said_voicemail = call_status in MACHINE_CALL_STATUSES

    # Prefixes of the opening, so replay can ask how early a verdict was safe.
    by_seconds: dict[str, list[str]] = {}
    if utterances:
        t0 = utterances[0][0]
        for cutoff in TRUNCATION_SECONDS:
            kept = [
                text
                for ts, text in utterances
                if (_seconds_between(t0, ts) or 0.0) <= cutoff
            ]
            by_seconds[str(cutoff)] = kept

    record = {
        "scenario_id": scenario_id,
        "expected_label": expected_label,
        "run_id": run_id,
        "workflow_id": workflow_id,
        # What the live detector decided. call_status only — see module docstring.
        "detector_verdict": "voicemail" if detector_said_voicemail else "human",
        "call_status": call_status,
        # Twilio's own opinion, when AMD was enabled. Frequently absent.
        "answered_by_raw": (gathered_context or {}).get("answered_by"),
        "answered_by": bucket_answered_by((gathered_context or {}).get("answered_by")),
        # The classifier sees user turns only; the branch's system prompt is
        # not part of the state. Mirrors typesafe_jev_llm._transcript_from.
        "opening_utterances": opening,
        "opening_by_seconds": by_seconds,
        # Full rendered transcript, for a human reading the dataset.
        "transcript": generate_transcript_text(events),
        "call_duration_seconds": (usage_info or {}).get("call_duration_seconds"),
        "num_events": len(events),
    }
    if include_events:
        record["events"] = events
    return record


async def load_from_db(run_ids: list[int]) -> dict[int, dict]:
    from api.db import db_client

    out: dict[int, dict] = {}
    for run_id in run_ids:
        run = await db_client.get_workflow_run_by_id(run_id)
        if run is None:
            print(f"  ! run {run_id} not found", file=sys.stderr)
            continue
        out[run_id] = {
            "workflow_id": run.workflow_id,
            "gathered_context": run.gathered_context,
            "usage_info": run.usage_info,
            "logs": run.logs,
        }
    return out


async def load_from_api(
    run_pairs: list[tuple[str, int]], *, base_url: str, api_key: str, workflow_id: int
) -> dict[int, dict]:
    import aiohttp

    out: dict[int, dict] = {}
    # api/services/auth/depends.py authenticates scripts via X-API-Key; the
    # Authorization header is reserved for Stack Auth bearer tokens.
    headers = {"X-API-Key": api_key}
    url_tpl = f"{base_url.rstrip('/')}/api/v1/workflow/{workflow_id}/runs/{{run_id}}"
    async with aiohttp.ClientSession(headers=headers) as session:
        for _, run_id in run_pairs:
            async with session.get(url_tpl.format(run_id=run_id)) as resp:
                if resp.status != 200:
                    print(
                        f"  ! run {run_id}: HTTP {resp.status} {await resp.text()}",
                        file=sys.stderr,
                    )
                    continue
                body = await resp.json()
            out[run_id] = {
                "workflow_id": body.get("workflow_id"),
                "gathered_context": body.get("gathered_context"),
                "usage_info": body.get("usage_info"),
                "logs": body.get("logs"),
            }
    return out


async def main(args: argparse.Namespace) -> int:
    labels = parse_scenarios()
    pairs = read_run_map(Path(args.runs))
    if not pairs:
        print(f"no runs listed in {args.runs}", file=sys.stderr)
        return 2

    unknown = [sid for sid, _ in pairs if sid not in labels]
    if unknown:
        print(f"unknown scenario ids (not in scenarios.md): {unknown}", file=sys.stderr)
        return 2

    print(f"-> {len(pairs)} run(s) from {args.runs}, source={args.source}")

    if args.source == "db":
        fetched = await load_from_db([rid for _, rid in pairs])
    else:
        if not args.api_key:
            print("--source api needs DOGRAH_API_KEY or --api-key", file=sys.stderr)
            return 2
        if not args.workflow_id:
            print("--source api needs --workflow-id", file=sys.stderr)
            return 2
        fetched = await load_from_api(
            pairs,
            base_url=args.base_url,
            api_key=args.api_key,
            workflow_id=args.workflow_id,
        )

    records = []
    for scenario_id, run_id in pairs:
        raw = fetched.get(run_id)
        if raw is None:
            continue
        records.append(
            build_record(
                scenario_id=scenario_id,
                expected_label=labels[scenario_id],
                run_id=run_id,
                workflow_id=raw.get("workflow_id"),
                gathered_context=raw.get("gathered_context"),
                usage_info=raw.get("usage_info"),
                logs=raw.get("logs"),
                include_events=args.with_events,
            )
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    silent = [r["scenario_id"] for r in records if not r["opening_utterances"]]
    labelled = [r for r in records if r["answered_by"]]

    print(f"OK wrote {len(records)} record(s) to {out_path}")
    print(
        f"  detector said voicemail : {sum(r['detector_verdict'] == 'voicemail' for r in records)}"
    )
    print(
        f"  expected voicemail      : {sum(r['expected_label'] == 'voicemail' for r in records)}"
    )
    print(f"  with Twilio answered_by : {len(labelled)}")
    if silent:
        print(f"  ! no user speech captured in: {silent}")
    return 0


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", choices=("api", "db"), default="api")
    parser.add_argument("--runs", default="scripts/voicemail_eval/runs.csv")
    parser.add_argument("--out", default="scripts/voicemail_eval/dataset.jsonl")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DOGRAH_BASE_URL", "https://app.dograh.com"),
    )
    parser.add_argument("--api-key", default=os.environ.get("DOGRAH_API_KEY"))
    parser.add_argument("--workflow-id", type=int, default=None)
    parser.add_argument(
        "--with-events",
        action="store_true",
        help="Embed the raw event list in each record. Large; only needed to re-derive fields.",
    )
    sys.exit(asyncio.run(main(parser.parse_args())) or 0)
