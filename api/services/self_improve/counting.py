"""Which completed runs count toward Self-improve, and starting a round.

Called from the completion job (``api/tasks/workflow_completion.py``) once
per finished run (RI-3: only that run's logs are read, never history):

    completed run ─► voice mode? (telephony/web channel, D3)
                  ─► Self-improve on and not paused for the workflow?
                  ─► caller spoke >= MIN_USER_TURNS times? (D3)
                  ─► org AI access (key + opt-in, RI-5)?
                  ─► count_run_for_self_improve (row lock + dedupe, RI-2/O7)
                  ─► claim_window_for_iteration ─► enqueue the round job
                     with a deterministic job id (arq dedupe, RI-2)

The same claim + enqueue is used by "Run now", by a settings change that
shrinks N below the calls already collected (DS-10) and by Resume.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from loguru import logger

from api.db import db_client
from api.enums import WORKFLOW_RUN_MODES_BY_CHANNEL, WorkflowRunChannel
from api.services.self_improve.contracts import MIN_USER_TURNS
from api.services.workflow.doctor.analysis import resolve_ai_access
from api.services.workflow.doctor.timeline import build_timeline, run_events

#: Run modes that are a voice conversation (phone or web), D3. Text chat and
#: Doctor replays (``doctor_replay``) are not in either channel.
VOICE_RUN_MODES: frozenset[str] = frozenset(
    WORKFLOW_RUN_MODES_BY_CHANNEL[WorkflowRunChannel.TELEPHONY.value]
    + WORKFLOW_RUN_MODES_BY_CHANNEL[WorkflowRunChannel.WEB.value]
)


def iteration_job_id(workflow_id: int, number: int) -> str:
    """Deterministic arq job id: one job per round (RI-2)."""
    return f"self-improve-{workflow_id}-{number}"


def _plain(value: Any) -> Any:
    return getattr(value, "value", value)


def is_voice_mode(mode: Any) -> bool:
    return _plain(mode) in VOICE_RUN_MODES


def count_user_turns(timeline: Iterable[dict]) -> int:
    """Final caller utterances with text (interim transcripts are dropped by
    ``build_timeline`` already)."""
    return sum(
        1
        for entry in timeline or []
        if entry.get("kind") == "user" and str(entry.get("text") or "").strip()
    )


def caller_spoke(logs: Any) -> bool:
    """True when the run's own logs hold >= MIN_USER_TURNS caller turns."""
    timeline = build_timeline(run_events({"logs": logs}))
    return count_user_turns(timeline) >= MIN_USER_TURNS


def _is_on(settings: Any) -> bool:
    return bool(
        settings is not None
        and getattr(settings, "enabled", False)
        and not getattr(settings, "paused", False)
    )


async def ai_access_ok(organization_id: int) -> bool:
    ai_enabled, _reason = await resolve_ai_access(organization_id)
    return ai_enabled


async def enqueue_iteration(iteration: Any) -> None:
    """Enqueue the round job. A duplicate job id is a silent no-op in arq."""
    # Imported lazily: api.tasks.arq imports every task module at load time.
    from api.tasks.arq import enqueue_job
    from api.tasks.function_names import FunctionNames

    await enqueue_job(
        FunctionNames.SELF_IMPROVE_ITERATION,
        iteration.id,
        iteration.organization_id,
        _job_id=iteration_job_id(iteration.workflow_id, iteration.number),
    )


async def claim_and_enqueue(
    workflow_id: int, organization_id: int, *, force: bool = False
) -> Optional[Any]:
    """Create the next round if one may start, and enqueue it.

    Returns the new round or None. An enqueue failure is logged, not raised:
    the round stays ``queued`` and the stuck-round sweep re-enqueues it (O5).
    """
    iteration = await db_client.claim_window_for_iteration(
        workflow_id=workflow_id, organization_id=organization_id, force=force
    )
    if iteration is None:
        return None
    logger.info(
        f"Self-improve: round {iteration.number} (iteration {iteration.id}) "
        f"queued for workflow {workflow_id} with {len(iteration.run_ids or [])} calls"
    )
    try:
        await enqueue_iteration(iteration)
    except Exception as exc:
        logger.error(
            f"Self-improve: could not enqueue iteration {iteration.id}; "
            f"the sweep will retry: {exc}"
        )
    return iteration


async def count_completed_run(workflow_run_id: int) -> Optional[Any]:
    """Count one completed run toward its workflow's window; start a round
    when the window fills. Returns the started round, else None."""
    run = await db_client.get_workflow_run_by_id(workflow_run_id)
    if run is None or not getattr(run, "is_completed", False):
        return None
    if not is_voice_mode(getattr(run, "mode", None)):
        return None
    workflow = getattr(run, "workflow", None)
    organization_id = getattr(workflow, "organization_id", None)
    workflow_id = getattr(run, "workflow_id", None)
    if organization_id is None or workflow_id is None:
        return None

    # Cheapest check first: most workflows never turn Self-improve on.
    settings = await db_client.get_self_improve(workflow_id, organization_id)
    if not _is_on(settings):
        return None
    if not caller_spoke(getattr(run, "logs", None)):
        return None
    if not await ai_access_ok(organization_id):
        return None

    counted, collected = await db_client.count_run_for_self_improve(
        workflow_id=workflow_id, organization_id=organization_id, run_id=run.id
    )
    if counted:
        logger.debug(
            f"Self-improve: run {run.id} counted for workflow {workflow_id} "
            f"({collected} collected)"
        )
    # Also after a dedupe (a retried completion job): the earlier attempt may
    # have counted the run and then failed before claiming the round.
    return await claim_and_enqueue(workflow_id, organization_id)
