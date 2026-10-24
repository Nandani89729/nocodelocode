"""Agent Doctor report routes (milestone 1).

Register with ``router.include_router(doctor_router)`` in ``api/routes/main.py``.
Paths (under ``/api/v1``):

- ``POST /workflow/{workflow_id}/doctor/reports``  create + enqueue
- ``GET  /workflow/{workflow_id}/doctor/reports``  list (newest first)
- ``GET  /doctor/reports/{report_uuid}``           detail + queue position
- ``POST /doctor/reports/{report_uuid}/resume``    resume a partial/failed report

All lookups are scoped to ``user.selected_organization_id``. The Anthropic
key and org opt-in are enforced server-side in the analysis service
(``doctor.analysis.resolve_ai_access``); the initial result returned here
already carries ``ai_enabled`` / ``ai_disabled_reason`` so the UI can show
the right state while the job is queued.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from api.db import db_client
from api.services.auth.depends import get_user
from api.services.workflow.doctor.analysis import (
    DOCTOR_AI_OPT_IN_KEY,
    opted_in,
    resolve_ai_access,
)
from api.services.workflow.doctor.contracts import (
    CreateReportRequest,
    DoctorSettings,
    ReportResponse,
    ReportResult,
    ReportStatus,
)
from api.services.workflow.doctor.llm import anthropic_key_configured
from api.services.workflow.doctor.timeline import as_obj
from api.tasks.arq import enqueue_job
from api.tasks.function_names import FunctionNames

router = APIRouter(tags=["doctor"])

LIST_LIMIT = 20
RESUMABLE_STATUSES = (ReportStatus.partial.value, ReportStatus.failed.value)


def _status_value(status: Any) -> str:
    return getattr(status, "value", status)


def _to_response(report: Any, queue_position: int | None = None) -> ReportResponse:
    created_at = report.created_at
    return ReportResponse(
        report_uuid=str(report.report_uuid),
        workflow_id=report.workflow_id,
        status=ReportStatus(_status_value(report.status)),
        created_at=created_at.isoformat()
        if hasattr(created_at, "isoformat")
        else str(created_at),
        params=as_obj(report.params) or {},
        result=ReportResult.model_validate(as_obj(report.result) or {}),
        error=report.error or None,
        queue_position=queue_position,
    )


async def _queue_position(report: Any) -> int | None:
    """Approximate: queued/running reports in the org (incl. this one)."""
    if _status_value(report.status) != ReportStatus.queued.value:
        return None
    try:
        return await db_client.count_active_reports(report.organization_id)
    except Exception as exc:
        logger.warning(f"Doctor queue position lookup failed: {exc}")
        return None


async def _enqueue(report_uuid: str, organization_id: int) -> None:
    # A fresh job id per enqueue: arq keeps finished job ids for an hour, so
    # reusing one would silently drop a resume.
    await enqueue_job(
        FunctionNames.ANALYZE_WORKFLOW_CALLS,
        report_uuid,
        organization_id,
        _job_id=f"doctor:{report_uuid}:{uuid.uuid4().hex[:8]}",
    )


async def _enqueue_or_fail(report: Any, organization_id: int) -> None:
    try:
        await _enqueue(str(report.report_uuid), organization_id)
    except Exception as exc:
        logger.exception(f"Doctor: enqueue failed for {report.report_uuid}")
        await db_client.update_analysis_report(
            str(report.report_uuid),
            organization_id,
            status=ReportStatus.failed,
            error="Could not queue the analysis. Please retry.",
        )
        raise HTTPException(
            status_code=503, detail="Could not queue the analysis"
        ) from exc


@router.post(
    "/workflow/{workflow_id}/doctor/reports",
    response_model=ReportResponse,
    summary="Diagnose a workflow from its recent calls",
)
async def create_doctor_report(
    workflow_id: int,
    request: CreateReportRequest,
    user=Depends(get_user),
):
    organization_id = user.selected_organization_id
    workflow = await db_client.get_workflow(
        workflow_id, organization_id=organization_id
    )
    if workflow is None:
        raise HTTPException(status_code=404, detail="Workflow not found")

    report = await db_client.create_analysis_report(
        organization_id=organization_id,
        workflow_id=workflow_id,
        created_by=user.id,
        params=request.model_dump(mode="json"),
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
    await _enqueue_or_fail(report, organization_id)
    return _to_response(report, await _queue_position(report))


@router.get(
    "/workflow/{workflow_id}/doctor/reports",
    response_model=list[ReportResponse],
    summary="List Doctor reports for a workflow",
)
async def list_doctor_reports(workflow_id: int, user=Depends(get_user)):
    organization_id = user.selected_organization_id
    workflow = await db_client.get_workflow(
        workflow_id, organization_id=organization_id
    )
    if workflow is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    reports = await db_client.list_analysis_reports(
        workflow_id, organization_id, limit=LIST_LIMIT
    )
    return [_to_response(r) for r in reports]


@router.get(
    "/doctor/reports/{report_uuid}",
    response_model=ReportResponse,
    summary="Get a Doctor report",
)
async def get_doctor_report(report_uuid: str, user=Depends(get_user)):
    report = await db_client.get_analysis_report(
        report_uuid, user.selected_organization_id
    )
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return _to_response(report, await _queue_position(report))


@router.post(
    "/doctor/reports/{report_uuid}/resume",
    response_model=ReportResponse,
    summary="Resume a partial or failed Doctor report",
)
async def resume_doctor_report(report_uuid: str, user=Depends(get_user)):
    organization_id = user.selected_organization_id
    report = await db_client.get_analysis_report(report_uuid, organization_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    if _status_value(report.status) not in RESUMABLE_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Only partial or failed reports can be resumed "
            f"(this one is {_status_value(report.status)})",
        )
    # Re-check AI access now: the org may have opted in or added a key since.
    result = ReportResult.model_validate(as_obj(report.result) or {})
    result.ai_enabled, result.ai_disabled_reason = await resolve_ai_access(
        organization_id
    )
    report = (
        await db_client.update_analysis_report(
            report_uuid,
            organization_id,
            status=ReportStatus.queued,
            result=result.model_dump(mode="json"),
            error="",
        )
        or report
    )
    await _enqueue_or_fail(report, organization_id)
    return _to_response(report, await _queue_position(report))


@router.get(
    "/doctor/settings",
    response_model=DoctorSettings,
    summary="Doctor AI analysis settings for the current organization",
)
async def get_doctor_settings(user=Depends(get_user)):
    value = await db_client.get_configuration_value(
        user.selected_organization_id, DOCTOR_AI_OPT_IN_KEY, False
    )
    return DoctorSettings(
        ai_opt_in=opted_in(value),
        anthropic_key_configured=anthropic_key_configured(),
    )


@router.put(
    "/doctor/settings",
    response_model=DoctorSettings,
    summary="Opt the organization in or out of AI call analysis",
)
async def update_doctor_settings(settings: DoctorSettings, user=Depends(get_user)):
    await db_client.upsert_configuration(
        user.selected_organization_id,
        DOCTOR_AI_OPT_IN_KEY,
        {"enabled": bool(settings.ai_opt_in)},
    )
    return DoctorSettings(
        ai_opt_in=bool(settings.ai_opt_in),
        anthropic_key_configured=anthropic_key_configured(),
    )
