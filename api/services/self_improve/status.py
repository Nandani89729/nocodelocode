"""Self-improve status line and round views (DS-1, DS-2, DS-3, DS-10).

State precedence (first match wins)::

    no_key ─► ai_disabled ─► off ─► paused ─► waiting ─► running ─► collecting

``waiting`` = a ready round whose proposal is not reviewed yet (D4);
``running`` = a round queued / diagnosing / proposing.
"""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger

from api.db import db_client
from api.services.self_improve.contracts import (
    ACTIVE_ITERATION_STATUSES,
    DEFAULT_EVERY_N_CALLS,
    FixCarryover,
    IterationResponse,
    IterationStatus,
    SelfImproveSettings,
    SelfImproveState,
    SelfImproveStatus,
)
from api.services.workflow.doctor.analysis import resolve_ai_access
from api.services.workflow.doctor.contracts import ReportResult, StageName
from api.services.workflow.doctor.timeline import as_obj

_ACTIVE = frozenset(s.value for s in ACTIVE_ITERATION_STATUSES)

#: Rounds read to find the current one (D4 keeps at most one open round,
#: always the newest).
STATUS_ROUNDS_LIMIT = 5


def _plain(value: Any) -> Any:
    return getattr(value, "value", value)


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def count_problems(result: Any) -> Optional[int]:
    """Actionable problems in a Doctor report: clusters seen in more than
    one call that are not platform issues. None until problem grouping is
    done (DS-3 "Writing fixes for N problems")."""
    try:
        parsed = ReportResult.model_validate(as_obj(result) or {})
    except Exception:
        return None
    stage = parsed.stages.get(StageName.clusters)
    if getattr(stage, "status", None) != "done":
        return None
    return sum(
        1 for c in parsed.clusters if not c.watch_only and c.category != "platform"
    )


async def load_problem_count(iteration: Any) -> Optional[int]:
    report_uuid = getattr(iteration, "report_uuid", None)
    if not report_uuid:
        return None
    try:
        report = await db_client.get_analysis_report(
            str(report_uuid), iteration.organization_id
        )
    except Exception as exc:
        logger.warning(f"Self-improve: report lookup failed: {exc}")
        return None
    return count_problems(report.result) if report is not None else None


async def iteration_with_problems(iteration: Any) -> IterationResponse:
    """``iteration_to_response`` plus ``problem_count`` from its report."""
    return iteration_to_response(
        iteration, problem_count=await load_problem_count(iteration)
    )


def iteration_to_response(
    iteration: Any, problem_count: Optional[int] = None
) -> IterationResponse:
    carryover: list[FixCarryover] = []
    for raw in as_obj(getattr(iteration, "carryover", None)) or []:
        try:
            carryover.append(FixCarryover.model_validate(raw))
        except Exception as exc:
            logger.warning(f"Self-improve: bad carryover item skipped: {exc}")
    report_uuid = getattr(iteration, "report_uuid", None)
    proposal_uuid = getattr(iteration, "proposal_uuid", None)
    return IterationResponse(
        iteration_id=iteration.id,
        number=iteration.number,
        status=IterationStatus(_plain(iteration.status)),
        run_ids=[int(r) for r in as_obj(getattr(iteration, "run_ids", None)) or []],
        report_uuid=str(report_uuid) if report_uuid else None,
        proposal_uuid=str(proposal_uuid) if proposal_uuid else None,
        reviewed=bool(getattr(iteration, "reviewed", False)),
        carryover=carryover,
        problem_count=problem_count,
        summary=getattr(iteration, "summary", None) or None,
        error=getattr(iteration, "error", None) or None,
        created_at=_iso(getattr(iteration, "created_at", None)) or "",
        updated_at=_iso(getattr(iteration, "updated_at", None)),
    )


def is_active(iteration: Any) -> bool:
    return _plain(iteration.status) in _ACTIVE


def is_waiting(iteration: Any) -> bool:
    """A ready round the user has not applied / discarded yet (D4)."""
    return _plain(iteration.status) == IterationStatus.ready.value and not bool(
        getattr(iteration, "reviewed", False)
    )


def current_iteration(rounds: list[Any]) -> Optional[Any]:
    """DS-1: the running or waiting round, else the latest (newest first)."""
    for iteration in rounds:
        if is_active(iteration) or is_waiting(iteration):
            return iteration
    return rounds[0] if rounds else None


def compute_state(
    *,
    ai_disabled_reason: Optional[str],
    ai_enabled: bool,
    settings: Any,
    rounds: list[Any],
) -> SelfImproveState:
    if not ai_enabled:
        if ai_disabled_reason == "no_key":
            return SelfImproveState.no_key
        return SelfImproveState.ai_disabled
    if settings is None or not settings.enabled:
        return SelfImproveState.off
    if settings.paused:
        return SelfImproveState.paused
    if any(is_waiting(it) for it in rounds):
        return SelfImproveState.waiting
    if any(is_active(it) for it in rounds):
        return SelfImproveState.running
    return SelfImproveState.collecting


def _brief_has_content(row: Any) -> bool:
    if row is None:
        return False
    brief = as_obj(getattr(row, "brief", None))
    if not isinstance(brief, dict):
        return bool(brief)
    return any((v.strip() if isinstance(v, str) else v) for v in brief.values() if v)


async def build_status(workflow_id: int, organization_id: int) -> SelfImproveStatus:
    """Assemble GET /workflow/{id}/self-improve. The caller has already
    checked that the workflow belongs to the org."""
    settings = await db_client.get_self_improve(workflow_id, organization_id)
    ai_enabled, reason = await resolve_ai_access(organization_id)
    rounds = await db_client.list_iterations(
        workflow_id, organization_id, limit=STATUS_ROUNDS_LIMIT
    )
    collected = (
        await db_client.count_collected_runs(workflow_id, organization_id)
        if settings is not None
        else 0
    )
    brief = await db_client.get_workflow_brief(workflow_id, organization_id)

    every_n = getattr(settings, "every_n_calls", None) or DEFAULT_EVERY_N_CALLS
    multiplier = getattr(settings, "window_multiplier", None) or 1
    current = current_iteration(rounds)
    return SelfImproveStatus(
        workflow_id=workflow_id,
        settings=SelfImproveSettings(
            enabled=bool(getattr(settings, "enabled", False)),
            every_n_calls=every_n,
        ),
        state=compute_state(
            ai_disabled_reason=reason,
            ai_enabled=ai_enabled,
            settings=settings,
            rounds=rounds,
        ),
        collected=collected,
        window_size=every_n * multiplier,
        window_multiplier=multiplier,
        paused_reason=(
            getattr(settings, "paused_reason", None)
            if getattr(settings, "paused", False)
            else None
        ),
        to_review=sum(1 for it in rounds if is_waiting(it)),
        current_iteration=(await iteration_with_problems(current) if current else None),
        has_brief=_brief_has_content(brief),
    )
