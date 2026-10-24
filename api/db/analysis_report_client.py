"""DB access for the Agent Doctor: reports, run input, candidates, scenarios.

Every method is organization-scoped (see api/AGENTS.md "Organization Scoping").
"""

from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, func, update
from sqlalchemy.future import select

from api.db.base_client import BaseDBClient
from api.db.filters import (
    apply_workflow_run_filters,
    exclude_internal_runs,
    get_workflow_run_order_clause,
)
from api.db.models import (
    RegressionScenarioModel,
    WorkflowAnalysisReportModel,
    WorkflowDefinitionModel,
    WorkflowModel,
    WorkflowRunModel,
)
from api.services.workflow.doctor.contracts import (
    CANDIDATE_ARCHIVED_STATUS,
    CANDIDATE_DEFINITION_STATUS,
    ReportStatus,
)

ACTIVE_REPORT_STATUSES = (ReportStatus.queued.value, ReportStatus.running.value)

# Columns callers may change through update_regression_scenario.
_SCENARIO_UPDATABLE_FIELDS = frozenset(
    {
        "report_id",
        "cluster_id",
        "source_run_id",
        "turn",
        "expected_behavior",
        "expected_confirmed",
        "verification",
        "candidate_definition_id",
    }
)


def _status_value(status: ReportStatus | str) -> str:
    return (
        status.value if isinstance(status, ReportStatus) else ReportStatus(status).value
    )


class AnalysisReportClient(BaseDBClient):
    async def _require_workflow_in_org(
        self, session, workflow_id: int, organization_id: int
    ) -> None:
        owned = await session.scalar(
            select(WorkflowModel.id).where(
                WorkflowModel.id == workflow_id,
                WorkflowModel.organization_id == organization_id,
            )
        )
        if owned is None:
            raise ValueError(f"Workflow {workflow_id} not found")

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    async def create_analysis_report(
        self,
        *,
        organization_id: int,
        workflow_id: int,
        created_by: Optional[int],
        params: Optional[dict] = None,
    ) -> WorkflowAnalysisReportModel:
        async with self.async_session() as session:
            await self._require_workflow_in_org(session, workflow_id, organization_id)
            report = WorkflowAnalysisReportModel(
                organization_id=organization_id,
                workflow_id=workflow_id,
                created_by=created_by,
                status=ReportStatus.queued.value,
                params=params or {},
                result={},
            )
            session.add(report)
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            await session.refresh(report)
            return report

    async def get_analysis_report(
        self, report_uuid: str, organization_id: int
    ) -> Optional[WorkflowAnalysisReportModel]:
        async with self.async_session() as session:
            return await session.scalar(
                select(WorkflowAnalysisReportModel).where(
                    WorkflowAnalysisReportModel.report_uuid == report_uuid,
                    WorkflowAnalysisReportModel.organization_id == organization_id,
                )
            )

    async def update_analysis_report(
        self,
        report_uuid: str,
        organization_id: int,
        *,
        status: Optional[ReportStatus | str] = None,
        result: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> Optional[WorkflowAnalysisReportModel]:
        """Update a report; returns None when it does not exist in the org.

        ``result`` replaces the stored JSON wholesale (callers own the merge,
        so a stage checkpoint is one atomic write).
        """
        async with self.async_session() as session:
            report = await session.scalar(
                select(WorkflowAnalysisReportModel)
                .where(
                    WorkflowAnalysisReportModel.report_uuid == report_uuid,
                    WorkflowAnalysisReportModel.organization_id == organization_id,
                )
                .with_for_update()
            )
            if report is None:
                return None
            if status is not None:
                report.status = _status_value(status)
            if result is not None:
                report.result = dict(result)
            if error is not None:
                report.error = error
            report.updated_at = datetime.now(UTC)
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            await session.refresh(report)
            return report

    async def list_analysis_reports(
        self, workflow_id: int, organization_id: int, limit: int = 20
    ) -> list[WorkflowAnalysisReportModel]:
        async with self.async_session() as session:
            result = await session.execute(
                select(WorkflowAnalysisReportModel)
                .where(
                    WorkflowAnalysisReportModel.workflow_id == workflow_id,
                    WorkflowAnalysisReportModel.organization_id == organization_id,
                )
                .order_by(
                    WorkflowAnalysisReportModel.created_at.desc(),
                    WorkflowAnalysisReportModel.id.desc(),
                )
                .limit(limit)
            )
            return list(result.scalars().all())

    async def count_active_reports(self, organization_id: Optional[int] = None) -> int:
        """Reports queued or running, for one org or (None) globally."""
        async with self.async_session() as session:
            query = select(func.count(WorkflowAnalysisReportModel.id)).where(
                WorkflowAnalysisReportModel.status.in_(ACTIVE_REPORT_STATUSES)
            )
            if organization_id is not None:
                query = query.where(
                    WorkflowAnalysisReportModel.organization_id == organization_id
                )
            return (await session.scalar(query)) or 0

    # ------------------------------------------------------------------
    # Doctor input
    # ------------------------------------------------------------------

    async def get_runs_with_logs_for_workflow(
        self,
        workflow_id: int,
        organization_id: int,
        *,
        limit: int,
        filters: Optional[List[Dict[str, Any]]] = None,
        run_ids: Optional[List[int]] = None,
    ) -> list[dict]:
        """Completed customer runs of a workflow with their logs, newest first.

        Internal runs (Doctor replays) are excluded, and ``filters`` take the
        same shape as the runs list endpoint. With ``run_ids`` (a Self-improve
        round) exactly those runs are returned, still scoped to the workflow
        and org and still without internal runs; ``limit`` and ``filters``
        are ignored.
        """
        async with self.async_session() as session:
            query = (
                select(
                    WorkflowRunModel.id,
                    WorkflowRunModel.created_at,
                    WorkflowRunModel.definition_id,
                    WorkflowRunModel.mode,
                    WorkflowRunModel.logs,
                    WorkflowRunModel.usage_info,
                    WorkflowRunModel.gathered_context,
                    WorkflowRunModel.initial_context,
                    WorkflowRunModel.annotations,
                )
                .join(WorkflowModel, WorkflowRunModel.workflow_id == WorkflowModel.id)
                .where(
                    WorkflowRunModel.workflow_id == workflow_id,
                    WorkflowModel.organization_id == organization_id,
                    WorkflowRunModel.is_completed.is_(True),
                )
            )
            query = exclude_internal_runs(query)
            if run_ids is not None:
                query = query.where(WorkflowRunModel.id.in_(list(run_ids)))
            else:
                query = apply_workflow_run_filters(query, filters)
            query = query.order_by(
                get_workflow_run_order_clause(), WorkflowRunModel.id.desc()
            )
            if run_ids is None:
                query = query.limit(limit)
            rows = (await session.execute(query)).all()
            return [
                {
                    "id": row.id,
                    "created_at": row.created_at,
                    "definition_id": row.definition_id,
                    "mode": row.mode,
                    "logs": row.logs or {},
                    "usage_info": row.usage_info or {},
                    "gathered_context": row.gathered_context or {},
                    "initial_context": row.initial_context or {},
                    "annotations": row.annotations or {},
                }
                for row in rows
            ]

    # ------------------------------------------------------------------
    # Candidate definitions (D17): hidden from history, drafts, publishing
    # ------------------------------------------------------------------

    async def create_candidate_definition(
        self,
        *,
        workflow_id: int,
        organization_id: int,
        workflow_json: dict,
        workflow_configurations: Optional[dict] = None,
    ) -> WorkflowDefinitionModel:
        async with self.async_session() as session:
            await self._require_workflow_in_org(session, workflow_id, organization_id)
            definition = WorkflowDefinitionModel(
                workflow_id=workflow_id,
                workflow_json=workflow_json,
                workflow_configurations=workflow_configurations or {},
                status=CANDIDATE_DEFINITION_STATUS,
                is_current=False,
                version_number=None,
            )
            session.add(definition)
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            await session.refresh(definition)
            return definition

    def _candidate_query(
        self, definition_id: int, organization_id: int, column=WorkflowDefinitionModel
    ):
        return (
            select(column)
            .join(
                WorkflowModel, WorkflowDefinitionModel.workflow_id == WorkflowModel.id
            )
            .where(
                WorkflowDefinitionModel.id == definition_id,
                WorkflowDefinitionModel.status == CANDIDATE_DEFINITION_STATUS,
                WorkflowModel.organization_id == organization_id,
            )
        )

    async def get_candidate_definition(
        self, definition_id: int, organization_id: int
    ) -> Optional[WorkflowDefinitionModel]:
        async with self.async_session() as session:
            return await session.scalar(
                self._candidate_query(definition_id, organization_id)
            )

    async def archive_candidate_definition(
        self, definition_id: int, organization_id: int
    ) -> bool:
        """Mark a candidate as archived (saved or discarded) without deleting it.

        Replay runs keep referencing the row (workflow_runs.definition_id has
        no ON DELETE), so archiving is the safe end-of-life for candidates.
        """
        async with self.async_session() as session:
            owned = await session.scalar(
                self._candidate_query(
                    definition_id, organization_id, WorkflowDefinitionModel.id
                )
            )
            if owned is None:
                return False
            await session.execute(
                update(WorkflowDefinitionModel)
                .where(
                    WorkflowDefinitionModel.id == owned,
                    WorkflowDefinitionModel.status == CANDIDATE_DEFINITION_STATUS,
                )
                .values(status=CANDIDATE_ARCHIVED_STATUS)
            )
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            return True

    async def delete_candidate_definition(
        self, definition_id: int, organization_id: int
    ) -> bool:
        """Delete a candidate row; never touches non-candidate definitions.

        Replay runs that point at it keep their row (definition_id is the FK
        on workflow_runs, which has no ON DELETE action), so callers must only
        delete candidates no run references, or accept the FK error.
        """
        async with self.async_session() as session:
            owned = await session.scalar(
                self._candidate_query(
                    definition_id, organization_id, WorkflowDefinitionModel.id
                )
            )
            if owned is None:
                return False
            await session.execute(
                delete(WorkflowDefinitionModel).where(
                    WorkflowDefinitionModel.id == owned,
                    WorkflowDefinitionModel.status == CANDIDATE_DEFINITION_STATUS,
                )
            )
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            return True

    # ------------------------------------------------------------------
    # Regression scenarios
    # ------------------------------------------------------------------

    async def create_regression_scenario(
        self,
        *,
        organization_id: int,
        workflow_id: int,
        report_id: Optional[int] = None,
        cluster_id: Optional[str] = None,
        source_run_id: Optional[int] = None,
        turn: Optional[int] = None,
        expected_behavior: Optional[str] = None,
        expected_confirmed: bool = False,
        verification: Optional[dict] = None,
        candidate_definition_id: Optional[int] = None,
    ) -> RegressionScenarioModel:
        async with self.async_session() as session:
            await self._require_workflow_in_org(session, workflow_id, organization_id)
            if report_id is not None:
                report_ok = await session.scalar(
                    select(WorkflowAnalysisReportModel.id).where(
                        WorkflowAnalysisReportModel.id == report_id,
                        WorkflowAnalysisReportModel.organization_id == organization_id,
                        WorkflowAnalysisReportModel.workflow_id == workflow_id,
                    )
                )
                if report_ok is None:
                    raise ValueError(f"Report {report_id} not found")
            if source_run_id is not None:
                run_ok = await session.scalar(
                    select(WorkflowRunModel.id).where(
                        WorkflowRunModel.id == source_run_id,
                        WorkflowRunModel.workflow_id == workflow_id,
                    )
                )
                if run_ok is None:
                    raise ValueError(f"Run {source_run_id} not found")
            if candidate_definition_id is not None:
                candidate_ok = await session.scalar(
                    self._candidate_query(
                        candidate_definition_id,
                        organization_id,
                        WorkflowDefinitionModel.id,
                    )
                )
                if candidate_ok is None:
                    raise ValueError(
                        f"Candidate definition {candidate_definition_id} not found"
                    )
            scenario = RegressionScenarioModel(
                organization_id=organization_id,
                workflow_id=workflow_id,
                report_id=report_id,
                cluster_id=cluster_id,
                source_run_id=source_run_id,
                turn=turn,
                expected_behavior=expected_behavior,
                expected_confirmed=expected_confirmed,
                verification=verification or {},
                candidate_definition_id=candidate_definition_id,
            )
            session.add(scenario)
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            await session.refresh(scenario)
            return scenario

    async def get_regression_scenario(
        self, scenario_id: int, organization_id: int
    ) -> Optional[RegressionScenarioModel]:
        async with self.async_session() as session:
            return await session.scalar(
                select(RegressionScenarioModel).where(
                    RegressionScenarioModel.id == scenario_id,
                    RegressionScenarioModel.organization_id == organization_id,
                )
            )

    async def list_regression_scenarios(
        self,
        workflow_id: int,
        organization_id: int,
        report_id: Optional[int] = None,
    ) -> list[RegressionScenarioModel]:
        async with self.async_session() as session:
            query = select(RegressionScenarioModel).where(
                RegressionScenarioModel.workflow_id == workflow_id,
                RegressionScenarioModel.organization_id == organization_id,
            )
            if report_id is not None:
                query = query.where(RegressionScenarioModel.report_id == report_id)
            result = await session.execute(
                query.order_by(RegressionScenarioModel.id.asc())
            )
            return list(result.scalars().all())

    async def update_regression_scenario(
        self, scenario_id: int, organization_id: int, **fields: Any
    ) -> Optional[RegressionScenarioModel]:
        """Update scenario columns; returns None when not found in the org.

        Only columns in ``_SCENARIO_UPDATABLE_FIELDS`` are accepted; an unknown
        key raises ValueError rather than being silently dropped.
        """
        unknown = set(fields) - _SCENARIO_UPDATABLE_FIELDS
        if unknown:
            raise ValueError(f"Cannot update scenario fields: {sorted(unknown)}")
        async with self.async_session() as session:
            scenario = await session.scalar(
                select(RegressionScenarioModel)
                .where(
                    RegressionScenarioModel.id == scenario_id,
                    RegressionScenarioModel.organization_id == organization_id,
                )
                .with_for_update()
            )
            if scenario is None:
                return None
            candidate_id = fields.get("candidate_definition_id")
            if candidate_id is not None:
                candidate_ok = await session.scalar(
                    self._candidate_query(
                        candidate_id, organization_id, WorkflowDefinitionModel.id
                    )
                )
                if candidate_ok is None:
                    raise ValueError(f"Candidate definition {candidate_id} not found")
            for key, value in fields.items():
                if key == "verification" and value is not None:
                    value = dict(value)
                setattr(scenario, key, value)
            scenario.updated_at = datetime.now(UTC)
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            await session.refresh(scenario)
            return scenario
