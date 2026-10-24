"""arq job for the Agent Doctor with non-blocking admission (R2/D19 + R10/D27).

Doctor jobs share the main worker (``max_jobs=10``). A Redis counting
semaphore caps them at ``DOCTOR_GLOBAL_CONCURRENCY`` globally and
``DOCTOR_PER_ORG_CONCURRENCY`` per org. Acquisition is a non-blocking
try-acquire: a job that gets no permit raises ``arq.Retry(defer=...)`` and
returns its worker slot immediately, so waiting Doctor jobs never starve
call-critical jobs. The report stays ``queued`` meanwhile.

The semaphore is a fixed set of slot keys taken with ``SET NX EX`` (atomic
per key); a permit expires with the job timeout, so a crashed worker cannot
hold it forever. Release deletes only keys still holding this job's token.

Register in ``WorkerSettings.functions`` as::

    func(
        analyze_workflow_calls,
        name=FunctionNames.ANALYZE_WORKFLOW_CALLS,
        timeout=DOCTOR_JOB_TIMEOUT_SECONDS,
        max_tries=DOCTOR_ADMISSION_MAX_TRIES + 1,
    )
"""

from __future__ import annotations

import uuid
from typing import Any, Awaitable, Callable, Optional

from arq import Retry
from loguru import logger

from api.db import db_client
from api.services.workflow.doctor.analysis import run_analysis
from api.services.workflow.doctor.contracts import (
    DOCTOR_ADMISSION_MAX_TRIES,
    DOCTOR_ADMISSION_RETRY_DEFER_SECONDS,
    DOCTOR_GLOBAL_CONCURRENCY,
    DOCTOR_JOB_TIMEOUT_SECONDS,
    DOCTOR_PER_ORG_CONCURRENCY,
    ReportStatus,
)

GLOBAL_SLOT_KEY = "doctor:sem:global:{slot}"
ORG_SLOT_KEY = "doctor:sem:org:{org}:{slot}"
#: Retry delay grows 1×, 2×, 4× the base, then stays at 4×.
MAX_DEFER_MULTIPLIER = 4


def _decode(value: Any) -> Optional[str]:
    if isinstance(value, bytes):
        return value.decode()
    return value


async def _take_slot(redis: Any, keys: list[str], token: str) -> Optional[str]:
    for key in keys:
        if await redis.set(key, token, nx=True, ex=DOCTOR_JOB_TIMEOUT_SECONDS):
            return key
    return None


async def _release_slot(redis: Any, key: Optional[str], token: str) -> None:
    if not key:
        return
    try:
        if _decode(await redis.get(key)) == token:
            await redis.delete(key)
    except Exception as exc:  # the TTL frees it anyway
        logger.warning(f"Doctor admission: failed to release {key}: {exc}")


async def try_acquire(redis: Any, organization_id: int, token: str) -> list[str]:
    """Take one org slot and one global slot, or nothing. Never blocks."""
    org_key = await _take_slot(
        redis,
        [
            ORG_SLOT_KEY.format(org=organization_id, slot=i)
            for i in range(DOCTOR_PER_ORG_CONCURRENCY)
        ],
        token,
    )
    if org_key is None:
        return []
    global_key = await _take_slot(
        redis,
        [GLOBAL_SLOT_KEY.format(slot=i) for i in range(DOCTOR_GLOBAL_CONCURRENCY)],
        token,
    )
    if global_key is None:
        await _release_slot(redis, org_key, token)
        return []
    return [org_key, global_key]


async def release(redis: Any, keys: list[str], token: str) -> None:
    for key in keys:
        await _release_slot(redis, key, token)


async def _get_redis(ctx: dict) -> Any:
    redis = ctx.get("redis") if isinstance(ctx, dict) else None
    if redis is None:
        from api.tasks.arq import get_arq_redis

        redis = await get_arq_redis()
    return redis


async def run_with_admission(
    ctx: dict,
    organization_id: int,
    job_label: str,
    work: Callable[[], Awaitable[Any]],
    on_exhausted: Callable[[], Awaitable[Any]],
) -> Any:
    """Run ``work`` under the Doctor semaphore without holding a worker slot.

    No permit: raise ``Retry(defer)`` so arq frees the slot (D27); after
    DOCTOR_ADMISSION_MAX_TRIES call ``on_exhausted`` instead. Shared by the
    analysis and verification jobs so both count against the same caps.
    """
    redis = await _get_redis(ctx)
    job_try = int((ctx or {}).get("job_try") or 1)
    token = f"{job_label}:{uuid.uuid4().hex}"

    keys = await try_acquire(redis, organization_id, token)
    if not keys:
        if job_try >= DOCTOR_ADMISSION_MAX_TRIES:
            logger.warning(f"Doctor {job_label}: no admission after {job_try} tries")
            return await on_exhausted()
        multiplier = min(2 ** (job_try - 1), MAX_DEFER_MULTIPLIER)
        raise Retry(defer=DOCTOR_ADMISSION_RETRY_DEFER_SECONDS * multiplier)

    try:
        return await work()
    finally:
        await release(redis, keys, token)


async def analyze_workflow_calls(
    ctx: dict, report_uuid: str, organization_id: int
) -> Optional[str]:
    """Run one Doctor report once a concurrency permit is free."""

    async def exhausted() -> str:
        await db_client.update_analysis_report(
            report_uuid,
            organization_id,
            status=ReportStatus.failed,
            error="The Doctor was busy for too long. Please retry.",
        )
        return ReportStatus.failed.value

    async def work() -> Optional[str]:
        try:
            return await run_analysis(report_uuid, organization_id)
        except Exception as exc:
            logger.exception(f"Doctor report {report_uuid}: unexpected failure")
            await db_client.update_analysis_report(
                report_uuid,
                organization_id,
                status=ReportStatus.failed,
                error=f"Unexpected error: {str(exc)[:300]}",
            )
            return ReportStatus.failed.value

    return await run_with_admission(
        ctx, organization_id, f"report:{report_uuid}", work, exhausted
    )
