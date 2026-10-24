"""Add callee speech-timing features to a harvested voicemail dataset.

For each record, fetches the callee-only track ``recordings/{run_id}/user.wav``,
runs Silero VAD over the opening seconds, and stores the temporal features from
``api.services.pipecat.voicemail_timing`` under ``record["timing"]``, keyed by
window length. ``replay.py`` then offers them to Jev alongside the transcript.

    export DOGRAH_API_KEY=...
    python -m scripts.voicemail_eval.timing \
        --dataset scripts/voicemail_eval/dataset.jsonl

Audio is cached under ``--audio-dir``, so re-running with different windows
does not download again. The dataset file is rewritten in place.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import aiohttp

from api.services.pipecat.voicemail_timing import (
    read_wav_mono,
    speech_segments,
    temporal_features,
)

#: The paper's two windows: 5 s was its best balance, 3 s trades ~2 points of
#: accuracy for a decision two seconds sooner.
DEFAULT_WINDOWS_S = (3.0, 5.0)

#: With ``--anchor speech``, the window opens this long before the first
#: speech, a typical carrier routing delay.
ONSET_LEAD_MS = 500.0


async def fetch_user_audio(
    session: aiohttp.ClientSession, base_url: str, run_id: int, dest: Path
) -> bool:
    """Download the callee track for ``run_id`` into ``dest`` via a signed URL."""
    key = f"recordings/{run_id}/user.wav"
    async with session.get(
        f"{base_url.rstrip('/')}/api/v1/s3/signed-url", params={"key": key}
    ) as resp:
        if resp.status != 200:
            print(
                f"  ! run {run_id}: signed URL HTTP {resp.status} {await resp.text()}",
                file=sys.stderr,
            )
            return False
        url = (await resp.json())["url"]

    # The signed URL carries its own auth; the API key must not ride along to
    # the object store.
    async with aiohttp.ClientSession() as plain:
        async with plain.get(url) as resp:
            if resp.status != 200:
                print(f"  ! run {run_id}: audio HTTP {resp.status}", file=sys.stderr)
                return False
            dest.write_bytes(await resp.read())
    return True


async def main(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset)
    records = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    windows_ms = [1000.0 * w for w in args.windows]

    missing = [
        r for r in records if not (audio_dir / f"{r['run_id']}_user.wav").exists()
    ]
    if missing:
        if not args.api_key:
            print("set DOGRAH_API_KEY or pass --api-key", file=sys.stderr)
            return 2
        async with aiohttp.ClientSession(headers={"X-API-Key": args.api_key}) as s:
            for r in missing:
                await fetch_user_audio(
                    s, args.base_url, r["run_id"], audio_dir / f"{r['run_id']}_user.wav"
                )

    print(
        f"{'id':<8}{'label':<11}{'win':>5}{'onset':>7}{'first':>7}{'segs':>6}{'1st-half':>10}"
    )
    print("-" * 54)
    done = 0
    for r in records:
        path = audio_dir / f"{r['run_id']}_user.wav"
        if not path.exists():
            r.pop("timing", None)
            continue
        pcm, rate = read_wav_mono(path.read_bytes())
        if args.anchor == "speech":
            # Scripted callers wait for the agent's opener, so their speech
            # starts seconds after answer. Re-zero on the first speech to keep
            # the shape features; first_onset_ms then carries no signal.
            segs = speech_segments(pcm, rate)
            if segs:
                t0 = max(0.0, segs[0][0] - ONSET_LEAD_MS)
                segs = [(s - t0, e - t0) for s, e in segs]
        else:
            segs = speech_segments(pcm, rate, window_ms=max(windows_ms))
        r["timing_anchor"] = args.anchor
        r["timing"] = {str(w / 1000.0): temporal_features(segs, w) for w in windows_ms}
        done += 1
        for w in windows_ms:
            f = r["timing"][str(w / 1000.0)]
            print(
                f"{r['scenario_id']:<8}{r['expected_label']:<11}{w / 1000:>4.0f}s"
                f"{f['first_onset_ms'] / 1000:>6.1f}s{f['first_seg_ms'] / 1000:>6.1f}s"
                f"{int(f['num_segments']):>6}{f['speech_first_half_ratio']:>10.2f}"
            )

    with dataset.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nOK timing added to {done}/{len(records)} record(s) in {dataset}")
    return 0 if done else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="scripts/voicemail_eval/dataset.jsonl")
    parser.add_argument("--audio-dir", default="scripts/voicemail_eval/audio")
    parser.add_argument(
        "--windows", type=float, nargs="+", default=list(DEFAULT_WINDOWS_S)
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DOGRAH_BASE_URL", "https://app.dograh.com"),
    )
    parser.add_argument("--api-key", default=os.environ.get("DOGRAH_API_KEY"))
    parser.add_argument(
        "--anchor",
        choices=("answer", "speech"),
        default="answer",
        help="Start the window at call answer (as live) or at the callee's first speech.",
    )
    sys.exit(asyncio.run(main(parser.parse_args())) or 0)
