# Voicemail eval — Phase 0

Decide whether TypeSafe's Jev should replace the gpt-4.1 classifier inside
Pipecat's `VoicemailDetector`, by measuring both on calls where we know the
right answer.

## Why this exists, and what it can't tell you

We cannot mine production for this: voicemail detection only runs on outbound
telephony, and the local database holds nothing but inbound browser sessions.
So we make the data — place calls, answer them ourselves, act out a scripted
scenario. Ground truth is then exact, because the person answering knows what
they were pretending to be. No labelling pass, no arguing with Twilio's AMD.

**This is a smoke test with latency numbers, not an accuracy verdict.** At n=20,
18/20 and 19/20 are indistinguishable. Use it to answer "does Jev work in our
pipeline, is it faster, and does it fail anywhere gpt-4.1 doesn't" — then gate
the actual rollout on real call volume.

Two things make twenty calls go further than twenty calls should:

- **The scenarios are adversarial on purpose.** Clean voicemail greetings are the
  case both classifiers already pass. `scenarios.md` spends twelve of its twenty
  rows on traps, noise and genuine ambiguity.
- **Record once, replay many.** Calls are slow; replay is free. Prompt iteration
  happens against the stored transcripts, so trying a new Jev phrasing costs
  cents and seconds instead of a redial.

## The loop

### 1. Place the calls

Pick a workflow with `voicemail_detection.enabled = true` on current defaults
(gpt-4.1). Work through `scenarios.md` top to bottom, reading each opening line
**as written**. Record the run id against the scenario id in `runs.csv`:

```csv
scenario_id,run_id
vm-01,4821
hu-01,4822
```

### 2. Harvest

```bash
export DOGRAH_API_KEY=...
python -m scripts.voicemail_eval.harvest \
    --source api --base-url https://app.dograh.com --workflow-id 123 \
    --runs scripts/voicemail_eval/runs.csv \
    --out scripts/voicemail_eval/dataset.jsonl
```

Writes one JSONL record per call: the user turns the classifier saw, the full
transcript, what the live detector decided, Twilio's `answered_by` if AMD was on,
and 1s/2s/3s prefixes of the opening.

It warns if a call captured no user speech — rerun those, an empty transcript
tests nothing.

### 3. Replay and compare

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
python -m scripts.voicemail_eval.replay \
    --dataset scripts/voicemail_eval/dataset.jsonl \
    --output scripts/voicemail_eval/results.json
```

Prints a summary table, a per-scenario table, a confidence-threshold sweep, and
a how-early-could-we-commit sweep over the truncated openings.

**Read the per-scenario table first.** At this sample size *which* cases each
classifier fails is far more informative than the percentages — one model
tripping on `hd-01` ("Hello, you've reached me") and the other on `am-02` tells
you something the aggregate hides.

### 4. Iterate the Jev question

`CLASSIFICATION_CRITERIA` in `api/services/pipecat/typesafe_jev_llm.py` is
currently long and listy, inherited from the gpt-4.1 prompt it replaced.
OpenRouter's Jev Lab recipes use a much terser house style — a direct question
plus one short contrasting sentence per option — and TypeSafe's own jaggedness
page warns that Jev reads instructions literally. Edit the criteria, re-run
step 3, keep whichever wins. No redialling.

## The check that matters most

`replay.py` prints a `[cross-check]` line: how often the replayed gpt-4.1 verdict
matches what the live detector actually recorded for that same call.

Everything else in the output assumes replaying a stored transcript reproduces
the live decision. If that line is not near-perfect, the harness is measuring
something other than the pipeline, and no number above it means anything. Treat
a low cross-check as a stop-and-investigate, not a footnote.

## Validating the tooling without placing a call

`harvest.py --source db` reads the local Postgres directly. The seven local runs
are inbound browser sessions with no voicemail, so they prove nothing about
accuracy — but they exercise event ordering and transcript extraction against
genuine event JSON:

```bash
set -a && source api/.env && set +a
printf 'scenario_id,run_id\nhu-01,1\nhu-02,2\n' > /tmp/local_runs.csv
python -m scripts.voicemail_eval.harvest \
    --source db --runs /tmp/local_runs.csv --out /tmp/local.jsonl
```

## Notes on the data

- **Detector verdict is read from `gathered_context["call_status"]`, not
  `call_disposition`.** `voicemail_detected` is also one of the default
  `CallDispositionOption` codes, so the post-call LLM classifier can assign it to
  a call the detector never flagged. `call_status` is written only by
  `PipecatEngine.end_call_with_reason` and is never LLM-generated.
- **`answered_by` is free text.** Twilio has no enum and the repo defines none.
  With `MachineDetection=Enable` expect `human` / `machine_start` / `fax` /
  `unknown`, plus absent when AMD is off for that org. `bucket_answered_by`
  treats anything starting `machine` as voicemail and passes the rest through
  visibly rather than silently coercing it.
- **`carrier` collapses to voicemail when scoring**, because that is what the
  live pipeline does. Jev still reports it separately, which is worth keeping an
  eye on for future retry logic — "not in service" and "left a voicemail" deserve
  different campaign behaviour.
