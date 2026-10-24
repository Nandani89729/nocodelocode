"""arq job + sweep for Self-improve rounds (RI-6, O5).

A round runs the Doctor and one copilot turn inside one Doctor admission
permit (``run_with_admission``), so Self-improve shares the Doctor's global
and per-org caps. Register in ``WorkerSettings.functions`` as::

    func(
        self_improve_iteration,
        name=FunctionNames.SELF_IMPROVE_ITERATION,
        timeout=DOCTOR_JOB_TIMEOUT_SECONDS,
        max_tries=DOCTOR_ADMISSION_MAX_TRIES + 1,
        keep_result=0,
    )

``keep_result=0``: arq refuses a job id while a result for it is stored, and
the job id is deterministic per round. Without a stored result the sweep can
re-enqueue a round whose job timed out; the job itself is idempotent (a
round that already ended returns at once).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

from loguru import logger

from api.db import db_client
from api.services.self_improve.contracts import (
    MAX_ROUND_AGE_SECONDS,
    STUCK_ROUND_SECONDS,
)
from api.services.self_improve.counting import enqueue_iteration
from api.services.self_improve.iteration import (
    BUSY_MESSAGE,
    TERMINAL_STATUSES,
    fail_iteration,
    run_iteration,
)
from api.tasks.doctor import run_with_admission

#: Shown when the sweep gives up on a round that never finished.
ROUND_TIMED_OUT_MESSAGE = (
    "The round kept timing out and was stopped. The next round runs after more calls."
)


def _plain(value: Any) -> Any:
    return getattr(value, "value", value)


async def self_improve_iteration(
    ctx: dict, iteration_id: int, organization_id: int
) -> Optional[str]:
    """Run one Self-improve round once a Doctor permit is free."""
    iteration = await db_client.get_iteration(iteration_id, organization_id)
    if iteration is None:
        logger.warning(f"Self-improve iteration {iteration_id} not found")
        return None
    if _plain(iteration.status) in TERMINAL_STATUSES:
        # Already ended (a duplicate or swept job): don't take a permit.
        return _plain(iteration.status)

    async def work() -> Optional[str]:
        return await run_iteration(iteration_id, organization_id)

    async def exhausted() -> Optional[str]:
        return await fail_iteration(iteration_id, organization_id, BUSY_MESSAGE)

    return await run_with_admission(
        ctx, organization_id, f"self-improve:{iteration_id}", work, exhausted
    )


async def sweep_stuck_self_improve_rounds(ctx: Any = None) -> int:
    """Re-enqueue rounds stuck in queued/diagnosing/proposing past the job
    timeout (worker restart, lost job). Same deterministic job id, so a job
    that is still queued or running is not duplicated. Returns how many
    enqueues were attempted."""
    try:
        stuck = await db_client.list_stuck_iterations(
            STUCK_ROUND_SECONDS, max_age_seconds=MAX_ROUND_AGE_SECONDS
        )
    except Exception as exc:
        logger.error(f"Self-improve sweep: could not list stuck rounds: {exc}")
        return 0
    attempted = 0
    for item in stuck:
        if item.get("expired"):
            try:
                await fail_iteration(
                    item["id"], item["organization_id"], ROUND_TIMED_OUT_MESSAGE
                )
            except Exception as exc:
                logger.error(
                    f"Self-improve sweep: could not fail iteration {item['id']}: {exc}"
                )
            continue
        try:
            await enqueue_iteration(
                SimpleNamespace(
                    id=item["id"],
                    organization_id=item["organization_id"],
                    workflow_id=item["workflow_id"],
                    number=item["number"],
                )
            )
            attempted += 1
        except Exception as exc:
            logger.error(
                f"Self-improve sweep: could not re-enqueue iteration {item['id']}: {exc}"
            )
    if stuck:
        logger.info(f"Self-improve sweep: re-enqueued {attempted} stuck round(s)")
    return attempted
