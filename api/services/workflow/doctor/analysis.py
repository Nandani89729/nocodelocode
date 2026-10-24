"""Agent Doctor orchestrator: one report = preflight → Layer 0 → Layer 1
batches → clusters → Layer 2 → rules.

- Every stage (and every Layer 1 batch) is checkpointed into
  ``report.result`` (R3/D20). A resumed job skips done stages and done
  batches and analyses the same run ids.
- A failing stage does not stop independent stages; the report then ends
  ``partial`` with finished stages kept, and "Resume" reruns only the rest.
- AI stages (layer1, clusters, rules) run only when the deployment has an
  Anthropic key **and** the org opted in (A5, enforced here, server-side).
  Preflight, Layer 0 and Layer 2 always run.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from loguru import logger

from api.db import db_client
from api.services.workflow.doctor.contracts import (
    FEW_CALLS_THRESHOLD,
    LAYER1_BATCH_SIZE,
    STAGE_ORDER,
    ReportResult,
    ReportStatus,
    StageName,
    StageState,
)
from api.services.workflow.doctor.latency import compute_latency
from api.services.workflow.doctor.layer0 import run_layer0
from api.services.workflow.doctor.layer1 import (
    analyse_batch,
    build_system_prompt,
    cluster_findings,
)
from api.services.workflow.doctor.llm import anthropic_key_configured
from api.services.workflow.doctor.preflight import run_preflight
from api.services.workflow.doctor.redaction import redact_run
from api.services.workflow.doctor.rules import extract_rules
from api.services.workflow.doctor.timeline import as_obj, build_run_timeline

#: organization_configurations key; value True (or {"enabled": true}) opts the
#: org in to sending transcripts to Anthropic. Off by default.
DOCTOR_AI_OPT_IN_KEY = "doctor_ai_opt_in"

AI_STAGES = frozenset({StageName.layer1, StageName.clusters, StageName.rules})

STAGE_LABELS = {
    StageName.preflight: "workflow checks",
    StageName.layer0: "platform checks",
    StageName.layer1: "problems",
    StageName.clusters: "problem grouping",
    StageName.layer2: "latency",
    StageName.rules: "rule inventory",
}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def opted_in(value: Any) -> bool:
    """Interpret the stored doctor_ai_opt_in configuration value."""
    if isinstance(value, dict):
        value = value.get("enabled", value.get("value"))
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


async def resolve_ai_access(organization_id: int) -> tuple[bool, Optional[str]]:
    """(ai_enabled, ai_disabled_reason). Key first, then org opt-in."""
    if not anthropic_key_configured():
        return False, "no_key"
    value = await db_client.get_configuration_value(
        organization_id, DOCTOR_AI_OPT_IN_KEY, False
    )
    if not opted_in(value):
        return False, "not_opted_in"
    return True, None


def _workflow_json(definition: Any) -> Optional[dict]:
    if definition is None:
        return None
    wf = as_obj(_get(definition, "workflow_json"))
    return wf if isinstance(wf, dict) and wf.get("nodes") else None


async def _load_definitions(workflow: Any) -> tuple[dict, dict]:
    """(definition_id → workflow_json, the definition the workflow serves now)."""
    definitions: dict[int, dict] = {}
    try:
        for version in await db_client.get_workflow_versions(_get(workflow, "id")):
            wf = _workflow_json(version)
            if wf:
                definitions[_get(version, "id")] = wf
    except Exception as exc:
        logger.warning(f"Doctor: could not load workflow versions: {exc}")
    current = (
        _workflow_json(_get(workflow, "released_definition"))
        or _workflow_json(_get(workflow, "current_definition"))
        or as_obj(_get(workflow, "workflow_definition"))
        or {}
    )
    for rel in ("released_definition", "current_definition"):
        d = _get(workflow, rel)
        if d is not None and _workflow_json(d):
            definitions.setdefault(_get(d, "id"), _workflow_json(d))
    return definitions, current


class _Analysis:
    def __init__(self, report_uuid: str, organization_id: int, result: ReportResult):
        self.report_uuid = report_uuid
        self.organization_id = organization_id
        self.result = result
        self.views: dict[int, dict] = {}
        self.definitions: dict[int, dict] = {}
        self.current_json: dict = {}
        self.usage: dict[str, int] = {}

    async def checkpoint(self, **kwargs: Any) -> None:
        await db_client.update_analysis_report(
            self.report_uuid,
            self.organization_id,
            result=self.result.model_dump(mode="json"),
            **kwargs,
        )

    def add_usage(self, usage: dict) -> None:
        for k, v in (usage or {}).items():
            if isinstance(v, (int, float)):
                self.usage[k] = self.usage.get(k, 0) + int(v)

    # ------------------------------------------------------------ stages
    async def stage_preflight(self) -> None:
        self.result.preflight = await run_preflight(
            self.current_json, self.organization_id
        )

    async def stage_layer0(self) -> None:
        self.result.layer0 = run_layer0(
            [self.views[rid] for rid in self.result.run_ids if rid in self.views],
            self.definitions,
            self.current_json,
        )

    async def stage_layer1(self) -> None:
        state = self.result.stages[StageName.layer1]
        run_ids = self.result.run_ids
        batches = [
            run_ids[i : i + LAYER1_BATCH_SIZE]
            for i in range(0, len(run_ids), LAYER1_BATCH_SIZE)
        ]
        state.total_batches = len(batches)
        used_defs = {
            d: self.definitions[d]
            for d in self.result.definition_ids
            if d in self.definitions
        } or {"current": self.current_json}
        system_prompt = build_system_prompt(used_defs)
        flags: dict[int, list[str]] = {}
        for flag in self.result.layer0.flags if self.result.layer0 else []:
            flags.setdefault(flag.run_id, []).append(f"{flag.check}: {flag.detail}")

        for idx, batch in enumerate(batches):
            if idx in state.completed_batches:
                continue
            redacted = [
                redact_run(self.views[rid]) for rid in batch if rid in self.views
            ]
            try:
                findings, usage = await analyse_batch(redacted, system_prompt, flags)
            except Exception as exc:
                raise RuntimeError(f"batch {idx + 1}/{len(batches)}: {exc}") from exc
            self.add_usage(usage)
            self.result.findings_by_batch[idx] = findings
            state.completed_batches.append(idx)
            await self.checkpoint()

    async def stage_clusters(self) -> None:
        findings = [
            f
            for idx in sorted(self.result.findings_by_batch)
            for f in self.result.findings_by_batch[idx]
        ]
        clusters, passed, usage = await cluster_findings(
            findings, set(self.result.run_ids)
        )
        self.add_usage(usage)
        self.result.clusters = clusters
        self.result.passed_checks = passed

    async def stage_layer2(self) -> None:
        self.result.latency = compute_latency(
            self.views[rid]["timeline"]
            for rid in self.result.run_ids
            if rid in self.views
        )

    async def stage_rules(self) -> None:
        rules, usage = await extract_rules(self.current_json)
        self.add_usage(usage)
        self.result.rules = rules


async def _load_runs(analysis: _Analysis, workflow_id: int, params: dict) -> None:
    query: dict[str, Any] = {
        "limit": int(params.get("limit") or 50),
        "filters": params.get("filters"),
    }
    if params.get("run_ids"):
        # A Self-improve round: exactly these runs (limit/filters ignored).
        query["run_ids"] = [int(rid) for rid in params["run_ids"]]
    rows = await db_client.get_runs_with_logs_for_workflow(
        workflow_id, analysis.organization_id, **query
    )
    # Compact immediately and drop raw logs (P2).
    views = {}
    for row in rows:
        view = build_run_timeline(row)
        views[view["run_id"]] = view
    del rows
    result = analysis.result
    if result.run_ids:
        # Resume: analyse exactly the runs the first attempt picked.
        result.run_ids = [rid for rid in result.run_ids if rid in views]
    else:
        result.run_ids = list(views)
    analysis.views = {rid: views[rid] for rid in result.run_ids}
    result.definition_ids = sorted(
        {v["definition_id"] for v in analysis.views.values() if v["definition_id"]}
    )
    result.few_calls = len(result.run_ids) < FEW_CALLS_THRESHOLD


def _status_of(report: Any) -> str:
    status = _get(report, "status")
    return getattr(status, "value", status)


async def run_analysis(report_uuid: str, organization_id: int) -> Optional[str]:
    """Run (or resume) one Doctor report. Returns the final status value."""
    report = await db_client.get_analysis_report(report_uuid, organization_id)
    if report is None:
        logger.warning(
            f"Doctor report {report_uuid} not found for org {organization_id}"
        )
        return None
    if _status_of(report) == ReportStatus.completed.value:
        return ReportStatus.completed.value

    result = ReportResult.model_validate(as_obj(_get(report, "result")) or {})
    for stage in STAGE_ORDER:
        result.stages.setdefault(stage, StageState())
    analysis = _Analysis(report_uuid, organization_id, result)
    params = as_obj(_get(report, "params")) or {}
    workflow_id = _get(report, "workflow_id")

    try:
        workflow = await db_client.get_workflow(
            workflow_id, organization_id=organization_id
        )
        if workflow is None:
            raise LookupError(f"workflow {workflow_id} not found")
        analysis.definitions, analysis.current_json = await _load_definitions(workflow)
        await _load_runs(analysis, workflow_id, params)
        result.ai_enabled, result.ai_disabled_reason = await resolve_ai_access(
            organization_id
        )
        await analysis.checkpoint(status=ReportStatus.running, error="")
    except Exception as exc:
        logger.exception(f"Doctor report {report_uuid}: setup failed")
        await db_client.update_analysis_report(
            report_uuid,
            organization_id,
            status=ReportStatus.failed,
            error=f"Could not start the analysis: {exc}",
        )
        return ReportStatus.failed.value

    handlers = {
        StageName.preflight: analysis.stage_preflight,
        StageName.layer0: analysis.stage_layer0,
        StageName.layer1: analysis.stage_layer1,
        StageName.clusters: analysis.stage_clusters,
        StageName.layer2: analysis.stage_layer2,
        StageName.rules: analysis.stage_rules,
    }
    first_error: Optional[str] = None
    try:
        for stage in STAGE_ORDER:
            state = result.stages[stage]
            if state.status == "done":
                continue
            if stage in AI_STAGES and not result.ai_enabled:
                state.status, state.error = "skipped", None
                await analysis.checkpoint()
                continue
            if (
                stage == StageName.clusters
                and result.stages[StageName.layer1].status != "done"
            ):
                state.status, state.error = "pending", "waiting for problems stage"
                first_error = first_error or "problem grouping waits for problems"
                continue
            state.status, state.error = "running", None
            await analysis.checkpoint()
            try:
                await handlers[stage]()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(
                    f"Doctor report {report_uuid}: stage {stage.value} failed"
                )
                state.status, state.error = "failed", str(exc)[:500]
                first_error = first_error or (
                    f"Analysis stopped at {STAGE_LABELS[stage]} ({str(exc)[:300]})"
                )
            else:
                state.status = "done"
            await analysis.checkpoint()
    except asyncio.CancelledError:
        # Job timeout: keep what finished so Resume can pick up from here.
        for state in result.stages.values():
            if state.status == "running":
                state.status, state.error = "failed", "timed out"
        await analysis.checkpoint(
            status=ReportStatus.partial, error="Analysis timed out; resume to finish."
        )
        raise

    if analysis.usage:
        logger.info(f"Doctor report {report_uuid}: Claude usage {analysis.usage}")
        merged = dict(result.claude_usage)
        for key, value in analysis.usage.items():
            merged[key] = merged.get(key, 0) + value
        result.claude_usage = merged
    final = ReportStatus.partial if first_error else ReportStatus.completed
    await analysis.checkpoint(status=final, error=first_error or "")
    return final.value
