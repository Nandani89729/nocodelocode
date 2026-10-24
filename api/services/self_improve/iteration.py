"""One Self-improve round (RI-6, O4, O5, D6).

Runs inside the arq job ``self_improve_iteration`` (api/tasks/self_improve.py)
under the Doctor's admission permit. Every stage is persisted on the round,
so a retried or swept job resumes where the last attempt stopped::

 completion job ──(real conversation? AI on? not paused)──► self_improve_runs += run
       │ window full (N × multiplier) AND previous round reviewed
       ▼
  iteration[k] queued ──► diagnosing (Doctor on exact run_ids) ──► proposing (copilot mode=self_improve,
       │                         │ fail                              bound to report k, propose once)
       │                         ▼                                         │
       │                      failed ──(2 in a row)──► paused (Resume)     ├─► ready ──(user applies/discards/stale)──► reviewed
       │                                                                   └─► no_changes ──► multiplier ×2 (max 4) ──► reviewed
       └─ sweep re-enqueues rounds stuck past job timeout

Resume rules:
- diagnosing: the round's report is created once (``report_id`` stored) and
  ``run_analysis`` resumes it from its checkpoints; a completed report is
  reused as is.
- proposing: the carry-over is stored when the stage starts; a proposal the
  round already made (``get_proposal_for_iteration``) is reused, so the
  copilot proposes at most once per round.
- ready / no_changes / failed are terminal: the job returns immediately.
  The terminal status is written *before* ``mark_round_outcome`` so a retry
  never applies the D6 spend guard twice.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from loguru import logger

from api.db import db_client
from api.services.self_improve.carryover import build_carryover, carryover_line
from api.services.self_improve.contracts import (
    FixCarryover,
    IterationStatus,
)
from api.services.self_improve.status import count_problems
from api.services.workflow.doctor.analysis import resolve_ai_access, run_analysis
from api.services.workflow.doctor.contracts import (
    CreateReportRequest,
    ReportResult,
    StageName,
)
from api.services.workflow.doctor.timeline import as_obj

TERMINAL_STATUSES = frozenset(
    {
        IterationStatus.ready.value,
        IterationStatus.no_changes.value,
        IterationStatus.failed.value,
    }
)

#: How many earlier rounds the copilot message summarises.
PREVIOUS_ROUNDS_IN_MESSAGE = 5

USER_LEFT_MESSAGE = (
    "The person who turned on Self-improve is no longer in this organization. "
    "Turn Self-improve off and on again to run rounds as you."
)
AI_OFF_MESSAGES = {
    "no_key": "This server has no AI key, so the calls could not be analysed.",
    "not_opted_in": "AI analysis is not enabled for this organization.",
}
BUSY_MESSAGE = "The Doctor was busy for too long. The next round will try again."


class RoundFailed(Exception):
    """A round cannot continue; the message is shown to the user as is."""


def _plain(value: Any) -> Any:
    return getattr(value, "value", value)


def _run_ids(iteration: Any) -> list[int]:
    return [int(r) for r in as_obj(iteration.run_ids) or []]


async def _run_self_improve_turn(**kwargs: Any) -> tuple[Optional[str], str]:
    """Lane B's copilot entry point, imported lazily (it pulls in the LLM
    stack). Returns ``(proposal_uuid | None, summary_text)``."""
    from api.services.copilot import agent

    return await agent.run_self_improve_turn(**kwargs)


# ---------------------------------------------------------------------------
# Ending a round
# ---------------------------------------------------------------------------


async def _finish(
    iteration: Any,
    *,
    status: IterationStatus,
    outcome: str,
    reason: Optional[str] = None,
    **fields: Any,
) -> Optional[str]:
    """Write the terminal status, then apply the D6 outcome exactly once.

    The status write is the guard: it only happens while the round is still
    non-terminal (re-read here), and a retry of a terminal round returns
    before reaching this function.
    """
    fresh = await db_client.get_iteration(iteration.id, iteration.organization_id)
    if fresh is None:
        return None
    if _plain(fresh.status) in TERMINAL_STATUSES:
        return _plain(fresh.status)
    await db_client.update_iteration(
        iteration.id, iteration.organization_id, status=status, **fields
    )
    try:
        await db_client.mark_round_outcome(
            workflow_id=iteration.workflow_id,
            organization_id=iteration.organization_id,
            outcome=outcome,
            reason=reason,
        )
    except Exception:
        logger.exception(
            f"Self-improve: could not record outcome {outcome} "
            f"for iteration {iteration.id}"
        )
    return status.value


async def fail_iteration(
    iteration_id: int, organization_id: int, message: str
) -> Optional[str]:
    """Mark a round failed with a plain message (no-op when already ended)."""
    iteration = await db_client.get_iteration(iteration_id, organization_id)
    if iteration is None:
        return None
    return await _finish(
        iteration,
        status=IterationStatus.failed,
        outcome="failed",
        reason=message,
        error=message,
    )


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


async def _resolve_user(settings: Any, organization_id: int) -> Any:
    """Rounds run as the user who turned Self-improve on (RI-5)."""
    user_id = getattr(settings, "enabled_by_user_id", None)
    if user_id is None:
        raise RoundFailed(USER_LEFT_MESSAGE)
    if not await db_client.is_user_member_of_organization(user_id, organization_id):
        raise RoundFailed(USER_LEFT_MESSAGE)
    user = await db_client.get_user_by_id(user_id)
    if user is None:
        raise RoundFailed(USER_LEFT_MESSAGE)
    return user


async def _ensure_report(iteration: Any, settings: Any) -> Any:
    """The round's Doctor report (created once, on exactly its run ids)."""
    organization_id = iteration.organization_id
    if iteration.report_id and iteration.report_uuid:
        report = await db_client.get_analysis_report(
            str(iteration.report_uuid), organization_id
        )
        if report is not None:
            return report
    run_ids = _run_ids(iteration)
    if not run_ids:
        raise RoundFailed("This round has no calls to analyse.")
    params = CreateReportRequest(limit=len(run_ids), run_ids=run_ids)
    report = await db_client.create_analysis_report(
        organization_id=organization_id,
        workflow_id=iteration.workflow_id,
        created_by=getattr(settings, "enabled_by_user_id", None),
        params=params.model_dump(mode="json"),
    )
    ai_enabled, reason = await resolve_ai_access(organization_id)
    initial = ReportResult(ai_enabled=ai_enabled, ai_disabled_reason=reason)
    report = (
        await db_client.update_analysis_report(
            str(report.report_uuid),
            organization_id,
            result=initial.model_dump(mode="json"),
        )
        or report
    )
    await db_client.update_iteration(iteration.id, organization_id, report_id=report.id)
    return report


async def _diagnose(iteration: Any, settings: Any) -> tuple[Any, ReportResult]:
    """Run (or resume) the Doctor inline; the report must have problem
    grouping done (O4) or the round fails."""
    organization_id = iteration.organization_id
    ai_enabled, reason = await resolve_ai_access(organization_id)
    if not ai_enabled:
        raise RoundFailed(AI_OFF_MESSAGES.get(reason or "", AI_OFF_MESSAGES["no_key"]))
    if _plain(iteration.status) != IterationStatus.diagnosing.value:
        await db_client.update_iteration(
            iteration.id, organization_id, status=IterationStatus.diagnosing
        )
    report = await _ensure_report(iteration, settings)
    report_uuid = str(report.report_uuid)
    await run_analysis(report_uuid, organization_id)

    report = await db_client.get_analysis_report(report_uuid, organization_id)
    if report is None:
        raise RoundFailed("The round's analysis report was deleted.")
    result = ReportResult.model_validate(as_obj(report.result) or {})
    clusters = result.stages.get(StageName.clusters)
    if getattr(clusters, "status", None) != "done":
        detail = (report.error or "").strip()
        raise RoundFailed(
            "The analysis did not finish grouping problems"
            + (f": {detail}" if detail else ".")
        )
    return report, result


async def _carry_over(
    iteration: Any, report: Any, result: ReportResult
) -> list[FixCarryover]:
    """Next-calls check for earlier accepted changes; fills ``since_fix`` on
    the report's matching clusters. Never fails the round."""
    try:
        lines = await build_carryover(
            workflow_id=iteration.workflow_id,
            organization_id=iteration.organization_id,
            number=iteration.number,
            run_ids=_run_ids(iteration),
            result=result,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception(f"Self-improve: carry-over failed for {iteration.id}")
        return []
    if any(c.since_fix for c in result.clusters):
        try:
            await db_client.update_analysis_report(
                str(report.report_uuid),
                iteration.organization_id,
                result=result.model_dump(mode="json"),
            )
        except Exception:
            logger.exception(f"Self-improve: could not save since_fix ({iteration.id})")
    return lines


def _previous_round_line(it: Any) -> str:
    status = _plain(it.status)
    if status == IterationStatus.no_changes.value:
        return f"Round {it.number}: no changes needed."
    if status == IterationStatus.failed.value:
        return f"Round {it.number}: failed ({it.error or 'no reason recorded'})."
    if status == IterationStatus.ready.value:
        proposal_status = _plain(getattr(it, "proposal_status", None))
        text = {
            "applied": "the user applied its proposal",
            "discarded": "the user discarded its proposal",
            "stale": "its proposal went stale (the workflow changed before review)",
        }.get(proposal_status, "its proposal is still waiting for review")
        summary = f" Proposal: {it.summary}" if getattr(it, "summary", None) else ""
        return f"Round {it.number}: {text}.{summary}"
    return f"Round {it.number}: {status}."


def build_round_message(
    *,
    iteration: Any,
    report_uuid: str,
    result: ReportResult,
    carryover: list[FixCarryover],
    previous: list[Any],
) -> str:
    """The system-generated message for the round's copilot turn."""
    calls = len(result.run_ids or _run_ids(iteration))
    problems = count_problems(result.model_dump(mode="json")) or 0
    lines = [
        f"Self-improve round {iteration.number}.",
        f"The Doctor analysed {calls} real calls of this workflow. Its report "
        f"(report_uuid {report_uuid}) is bound to this round: read it with "
        f"get_doctor_report. It found {len(result.clusters)} problem groups, "
        f"{problems} of them actionable (seen in more than one call and not a "
        f"platform issue).",
        "Read the brief with get_workflow_brief and respect it.",
    ]
    if carryover:
        lines.append("")
        lines.append("Since earlier fixes:")
        lines.extend(f"- {carryover_line(c)}" for c in carryover)
    if previous:
        lines.append("")
        lines.append("Previous rounds:")
        lines.extend(f"- {_previous_round_line(it)}" for it in previous)
    lines.append("")
    lines.append(
        "Propose minimal fixes for the most important actionable problems in "
        "one proposal, with the problem each change targets; or propose "
        "nothing and say why if no change is worth making."
    )
    return "\n".join(lines)


async def _propose(
    iteration: Any,
    user: Any,
    report: Any,
    result: ReportResult,
    carryover: list[FixCarryover],
) -> Optional[str]:
    organization_id = iteration.organization_id
    summary: Optional[str] = None
    proposal = await db_client.get_proposal_for_iteration(iteration.id, organization_id)
    if proposal is None:
        session = await db_client.get_or_create_self_improve_session(
            organization_id=organization_id,
            workflow_id=iteration.workflow_id,
            user_id=user.id,
        )
        rounds = await db_client.list_iterations(
            iteration.workflow_id,
            organization_id,
            limit=PREVIOUS_ROUNDS_IN_MESSAGE + 1,
        )
        previous = [it for it in rounds if it.number < iteration.number][
            :PREVIOUS_ROUNDS_IN_MESSAGE
        ]
        report_uuid = str(report.report_uuid)
        text = build_round_message(
            iteration=iteration,
            report_uuid=report_uuid,
            result=result,
            carryover=carryover,
            previous=list(reversed(previous)),
        )
        try:
            proposal_uuid, summary = await _run_self_improve_turn(
                session_uuid=str(session.session_uuid),
                organization_id=organization_id,
                user=user,
                iteration_id=iteration.id,
                report_uuid=report_uuid,
                text=text,
            )
        except Exception as exc:
            # copilot.agent.SelfImproveTurnError: the turn failed before
            # proposing; its message is already user-facing.
            if type(exc).__name__ == "SelfImproveTurnError":
                raise RoundFailed(
                    f"The Copilot could not write fixes: {str(exc)[:300]}"
                ) from exc
            raise
        if proposal_uuid:
            proposal = await db_client.get_proposal_for_iteration(
                iteration.id, organization_id
            ) or await db_client.get_copilot_proposal(proposal_uuid, organization_id)
            if proposal is None:
                raise RoundFailed("The proposed changes could not be found.")
    summary = (summary or "").strip() or None

    if proposal is not None:
        return await _finish(
            iteration,
            status=IterationStatus.ready,
            outcome="changes",
            proposal_id=proposal.id,
            summary=summary or (getattr(proposal, "summary", None) or None),
        )
    calls = len(result.run_ids or _run_ids(iteration))
    return await _finish(
        iteration,
        status=IterationStatus.no_changes,
        outcome="no_changes",
        summary=summary or f"No changes needed in {calls} calls.",
    )


async def run_iteration(iteration_id: int, organization_id: int) -> Optional[str]:
    """Run (or resume) one round. Returns its final status value, or None
    when the round does not exist in the org."""
    iteration = await db_client.get_iteration(iteration_id, organization_id)
    if iteration is None:
        logger.warning(f"Self-improve iteration {iteration_id} not found")
        return None
    status = _plain(iteration.status)
    if status in TERMINAL_STATUSES:
        return status

    try:
        settings = await db_client.get_self_improve(
            iteration.workflow_id, organization_id
        )
        if settings is None:
            raise RoundFailed("Self-improve is not set up for this workflow.")
        user = await _resolve_user(settings, organization_id)

        if status in (IterationStatus.queued.value, IterationStatus.diagnosing.value):
            report, result = await _diagnose(iteration, settings)
            carryover = await _carry_over(iteration, report, result)
            iteration = await db_client.update_iteration(
                iteration.id,
                organization_id,
                status=IterationStatus.proposing,
                carryover=carryover,
            )
            if iteration is None:
                return None
        else:  # proposing: resume with the stored report + carry-over
            report = await db_client.get_analysis_report(
                str(iteration.report_uuid), organization_id
            )
            if report is None:
                raise RoundFailed("The round's analysis report was deleted.")
            result = ReportResult.model_validate(as_obj(report.result) or {})
            carryover = [
                FixCarryover.model_validate(c)
                for c in as_obj(iteration.carryover) or []
            ]
        return await _propose(iteration, user, report, result, carryover)
    except asyncio.CancelledError:
        # Job timeout / shutdown: keep the stage; the sweep re-enqueues it.
        raise
    except RoundFailed as exc:
        logger.info(f"Self-improve iteration {iteration_id} failed: {exc}")
        return await fail_iteration(iteration_id, organization_id, str(exc))
    except Exception as exc:
        logger.exception(f"Self-improve iteration {iteration_id}: unexpected failure")
        return await fail_iteration(
            iteration_id,
            organization_id,
            f"Unexpected error: {str(exc)[:300]}",
        )
