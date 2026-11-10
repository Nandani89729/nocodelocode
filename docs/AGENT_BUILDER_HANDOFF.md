# Handoff: Dograh Agent Builder

## What this is
A fork of Dograh (open-source voice-AI platform, upstream `dograh-hq/dograh`, synced to v1.47.0) with three features that let a voice agent improve itself from its own call logs:

1. **Agent Doctor**: analyses a workflow's recent calls and produces a report of problems. Each problem is attributed to a fixable node, edge, knowledge base or config.
2. **Agent Copilot**: a chat panel inside the workflow editor. It reads the workflow, plans, and produces a *proposal* (a per-node diff). The user applies all of it or part of it, and the result is saved as a draft. It also edits Settings, creates tools and attaches knowledge-base docs. "Build with Copilot" drafts a new agent from a brief.
3. **Self-improve**: an automatic loop. When enough new real calls accumulate, it runs the Doctor on them, then the Copilot proposes fixes once. The user reviews, applies or discards, and then the next round starts.

Core rule throughout: **every finding must map to an edit.**

Repo: `github.com/Nandani89729/nocodelocode` (imported from `github.com/RuchiketJadhav/dograh-agent-builder`, branch `agent-builder` at `674e2b2a`).
Other branches: `main`, `feat/call-analysis-logging` (base of this work), `feat/futureagi-integration` (older base; rebase before using), `fix/windows-dev-script-parity`, `fix/windows-setup-script`.

## State of the data
The EC2 box this ran on has been terminated. The database (workflows, call runs, reports), the call recordings and all `.env` keys are gone. Start with an empty DB and new keys. The test agent used was workflow 7, "Dograh Support - Inbound (Call Doctor)". It must be rebuilt if needed.

## Setup
```bash
git clone --recurse-submodules https://github.com/RuchiketJadhav/dograh-agent-builder.git
cd dograh-agent-builder && git switch agent-builder
cp api/.env.example api/.env && cp ui/.env.example ui/.env    # fill in keys
(cd api/mcp_server/ts_validator && npm ci)    # REQUIRED, see gotchas
docker compose -f docker-compose-local.yaml up -d             # postgres, redis, minio
bash scripts/start_services_dev.sh                            # API + arq workers, runs alembic
cd ui && npm install && npm run dev                           # http://localhost:3000
```
Tests: `set -a; source api/.env.test; set +a; python -m pytest api/tests/<file> -q`. Postgres-backed tests need the compose services up. At last count there were about 740 backend tests and 140 UI tests (vitest), all green.

## Keys and switches
- `ANTHROPIC_API_KEY`: the default and calibrated provider for both Doctor and Copilot.
- `DOCTOR_LLM_PROVIDER=openai`, plus `OPENAI_API_KEY` and optional `DOCTOR_OPENAI_MODEL` (default `gpt-5.5`): a temporary switch, used because no Anthropic key was available. Both `doctor/llm.py` and `copilot/llm.py` read it. Remove it once an Anthropic key exists, then re-run the judge eval.
- The Doctor's AI stages (layer1, clusters, rules) run only if a key exists **and** the org opted in via `PUT /doctor/settings` (org config key `doctor_ai_opt_in`). Preflight, Layer 0 and latency always run.

## Code map (all under `api/` unless noted)
**Doctor**: `services/workflow/doctor/`. `contracts.py` is the seam between modules.
- Pipeline (`analysis.py`): preflight → layer0 (platform checks) → layer1 (LLM finds problems in batches) → clusters → layer2 (`latency.py`, per node) → rules. Every stage and every batch is checkpointed into `report.result`, so it is resumable. A failed stage leaves the report `partial`.
- Problem classes: hallucination, knowledge gap, workflow/routing error, STT mishearing, turn-taking, then latency.
- `redaction.py` strips PII before transcripts go to the LLM. `verify.py` checks a proposed fix by replaying it in text chat.
- Jobs: `tasks/doctor.py` (`analyze_workflow_calls`), `tasks/doctor_verify.py` (`verify_doctor_fix`). Migration `8c2d5e7f1a43`.
- Routes: `routes/doctor.py` (`/workflow/{id}/doctor/reports`, `/doctor/reports/{uuid}[/resume]`, `/doctor/settings`), `routes/doctor_verify.py` (`/doctor/reports/{uuid}/clusters/{cluster_id}/propose`, `/doctor/verifications/{id}/start|save|discard`, `/workflow/{id}/text-chat/regenerate`).
- UI: `ui/src/components/doctor/`, `ui/src/lib/doctor/`, page `ui/src/app/workflow/[workflowId]/doctor/[reportUuid]/`.
- Judge eval: `tests/evals/doctor_judge/` (`labels.json` holds human labels). Passed 92.5% on gpt-5.5 with 0 critical false passes, majority of 3 votes. **Not yet run on Claude.**

**Copilot**: `services/copilot/`. `contracts.py` is the seam.
- `agent.py` runs one turn: user message → loop of model call + tool call until done or the step cap. It streams events (`text_delta`, `tool_call`, `tool_result`, `plan`, `proposal`) over SSE. Every turn ends with exactly one `message_done` or `error`. The transcript is persisted after each step.
- `tools.py` reuses the Dograh MCP tools, acting as the user. `proposals.py` and `diff.py` produce per-node diffs. Applying a subset saves a draft, never a publish; the apply claim is atomic (`claim_copilot_proposal`).
- `settings.py` and `services/configuration/model_catalog.py` handle Settings edits. Secrets stay server-side in `copilot_proposals.settings_candidate`. New tools use `tool:<name>` placeholders and are archived if the save fails.
- `brief.py` holds the workflow brief (the business requirements). Users confirm or dismiss brief changes.
- Migrations `4e9a1c7d3b25`, `c3e8a5f27d14`. Routes `routes/copilot.py` (`/copilot/sessions…`, `/copilot/proposals/{uuid}/apply|discard`, `/workflow/{id}/brief…`). UI `ui/src/components/copilot/`, `ui/src/lib/copilot/`.

**Self-improve**: `services/self_improve/`. `contracts.py` is the seam.
- `iteration.py` runs one round (arq job `self_improve_iteration` in `tasks/self_improve.py`). A round runs queued → diagnosing (Doctor on the exact run_ids) → proposing (Copilot `mode=self_improve`, proposes once) → ready / no_changes / failed. Two failures in a row pause the loop. `no_changes` doubles the call window (max 4×).
- The trigger is in the workflow completion job: real calls only, AI on, loop not paused. The next round waits until the previous one is reviewed. A sweep re-enqueues stuck rounds.
- `carryover.py` tells the next round what earlier rounds fixed. Migration `b7d2f4a91c60`. Routes `/workflow/{id}/self-improve[/run|/resume|/iterations]`. UI `ui/src/components/self-improve/`.

**Logging changes (from `feat/call-analysis-logging`), needed by the Doctor**:
- pipecat `on_latency_breakdown` is persisted as `rtf-latency-breakdown`.
- `turn_interrupted` is added to transcript events.
- Tool results are saved as JSON, not `str()`.

**Also in the commit**:
- TypeSafe Jev voicemail detector (`services/pipecat/typesafe_jev_llm.py`, `voicemail_timing.py`) and its eval in `scripts/voicemail_eval/`. Accuracy: gpt-4.1 93%, reworded Jev 90%; Jev hangs up on far fewer humans but misses more voicemails. The timing features were tested and dropped.
- `ts_bridge.py` Windows fix: the subprocess falls back to a thread under `uvicorn --reload`.
- `tzdata` added to requirements.

## Gotchas
- **`npm ci` in `api/mcp_server/ts_validator`**: git doesn't carry its `node_modules`. Without it, every Copilot proposal and MCP save fails with `Cannot find package 'typescript'`.
- **MinIO image**: `quay.io/minio/minio` now returns 401 and `dl.min.io` returns 410. Use a cached image or swap in another S3-compatible image.
- Dograh's MCP (`/api/v1/mcp`, `X-API-Key`) can create and save workflows but **cannot read call runs**. The Doctor reads Postgres directly.
- The repo has no ruff config, but `ruff format` is the house style for new files.
- The hey-api UI client hasn't been regenerated for the new routes.

## Open items, highest value first
1. **Never browser-tested end to end.** Copilot, Doctor UI and Self-improve all pass unit tests and live API smoke runs, but none has been clicked through in the UI. Start here.
2. Get an Anthropic key, drop `DOCTOR_LLM_PROVIDER=openai`, and re-run the judge eval on Claude.
3. Doctor gate failed 0/2 on wf7. Its 8.1k-char Global prompt held all the pricing, so prompt edits on one node couldn't stop it answering. Proposed fix: "knowledge placement" edits that move facts from the Global node to the node that needs them, then verify both turns.
4. Security bug that predates this work: `mask_workflow_configurations` doesn't mask `voicemail_detection.api_key`, and the save side needs a matching merge.
5. Self-improve eval (D5/T18) is not built yet.
6. The retention cron that calls `verify.expire_report_candidates` is not wired yet.
7. Undecided: how to handle call screeners. The proposal is to answer "state your name and reason", wait on "please hold" up to 60 s, and hang up otherwise.

Accepted risks:
- D26: editing was built before replay was proven.
- D28: the 0/10 keep rule.
- D31: expectations are not frozen.

## Candidate upstream PRs
These are small and self-contained:
- The logging changes.
- The `ts_bridge` Windows fix.
- `tzdata`.
- The `npm ci` setup doc.
- Two Windows script-fix branches.
