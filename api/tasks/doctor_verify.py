"""ARQ job: verify a proposed Agent Doctor fix by replay (D25/D28/D29/D30).

Registered by the integrator in ``api/tasks/arq.py`` as
``FunctionNames.VERIFY_DOCTOR_FIX``. Enqueued by
``POST /doctor/verifications/{scenario_id}/start`` with
``(scenario_id, organization_id)``.
"""

from loguru import logger

from api.services.workflow.doctor.verify import mark_verification_failed, verify_fix
from api.tasks.doctor import run_with_admission


async def verify_doctor_fix(ctx, scenario_id: int, organization_id: int) -> dict:
    """Run baseline ×10 and candidate ×10 replays and store the verdict.

    ``verify_fix`` records progress and the final ``VerificationResult`` on
    the scenario row itself and turns any replay error into ``failed``, so
    the job never needs an arq retry (a retry would re-bill the replays).
    """

    async def work() -> dict:
        logger.info(
            f"Doctor verification starting for scenario {scenario_id} "
            f"(org {organization_id})"
        )
        result = await verify_fix(
            scenario_id=scenario_id, organization_id=organization_id
        )
        logger.info(
            f"Doctor verification for scenario {scenario_id} finished: "
            f"{result.status.value} (baseline {result.baseline_failures}, "
            f"candidate {result.candidate_failures})"
        )
        return {"scenario_id": scenario_id, "status": result.status.value}

    async def exhausted() -> dict:
        await mark_verification_failed(
            scenario_id,
            organization_id,
            "The Doctor was busy for too long. Please start verification again.",
        )
        return {"scenario_id": scenario_id, "status": "failed"}

    # Same non-blocking admission as analysis (D27): replays are heavy and
    # must never hold worker slots while waiting.
    return await run_with_admission(
        ctx, organization_id, f"verify:{scenario_id}", work, exhausted
    )
