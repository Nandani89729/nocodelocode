"""DB access for Self-improve (milestone 2b): settings, counted runs, rounds.

Counting and round creation are atomic per workflow: both lock the
workflow's ``self_improve`` row (``SELECT ... FOR UPDATE``), so two
completion jobs racing at the Nth call create exactly one round (RI-2).
``self_improve_iterations (workflow_id, number)`` is unique as a backstop.

Every method except ``list_stuck_iterations`` (the cross-org sweep) is
organization-scoped (see api/AGENTS.md "Organization Scoping").
"""

from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Literal, Optional

from sqlalchemy import and_, func, or_, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.future import select

from api.db.base_client import BaseDBClient
from api.db.models import (
    CopilotProposalModel,
    SelfImproveIterationModel,
    SelfImproveModel,
    SelfImproveRunModel,
    WorkflowAnalysisReportModel,
    WorkflowModel,
    WorkflowRunModel,
)
from api.services.self_improve.contracts import (
    ACTIVE_ITERATION_STATUSES,
    ALLOWED_EVERY_N_CALLS,
    FAILURES_BEFORE_PAUSE,
    MAX_WINDOW_MULTIPLIER,
    IterationStatus,
)

#: Round statuses that are reviewed by themselves (D4).
REVIEWED_ITERATION_STATUSES = (
    IterationStatus.no_changes.value,
    IterationStatus.failed.value,
)
#: Proposal statuses that make a round reviewed (D4, O2: never ``applying``).
REVIEWED_PROPOSAL_STATUSES = ("applied", "discarded", "stale")

_ACTIVE_STATUS_VALUES = tuple(s.value for s in ACTIVE_ITERATION_STATUSES)
_ITERATION_STATUS_VALUES = frozenset(s.value for s in IterationStatus)

# Columns callers may change through update_iteration.
_ITERATION_UPDATABLE_FIELDS = frozenset(
    {
        "status",
        "report_id",
        "proposal_id",
        "carryover",
        "summary",
        "error",
        "stage_started_at",
    }
)

RoundOutcome = Literal["changes", "no_changes", "failed"]


def _plain(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def iteration_is_reviewed(status: str, proposal_status: Optional[str]) -> bool:
    """D4: a round is reviewed when it ended with no changes or failed, or
    its proposal was applied, discarded or went stale."""
    return _plain(status) in REVIEWED_ITERATION_STATUSES or (
        _plain(proposal_status) in REVIEWED_PROPOSAL_STATUSES
    )


class SelfImproveClient(BaseDBClient):
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _settings_query(self, workflow_id: int, organization_id: int):
        return select(SelfImproveModel).where(
            SelfImproveModel.workflow_id == workflow_id,
            SelfImproveModel.organization_id == organization_id,
        )

    async def _lock_settings(
        self, session, workflow_id: int, organization_id: int
    ) -> Optional[SelfImproveModel]:
        return await session.scalar(
            self._settings_query(workflow_id, organization_id).with_for_update()
        )

    def _collected_filter(self, workflow_id: int, reviewed_at: Optional[datetime]):
        """Counted runs not yet in a round and inside the current window."""
        conditions = [
            SelfImproveRunModel.workflow_id == workflow_id,
            SelfImproveRunModel.iteration_id.is_(None),
        ]
        if reviewed_at is not None:
            conditions.append(SelfImproveRunModel.counted_at >= reviewed_at)
        return and_(*conditions)

    async def _count_collected(
        self, session, workflow_id: int, reviewed_at: Optional[datetime]
    ) -> int:
        return (
            await session.scalar(
                select(func.count(SelfImproveRunModel.id)).where(
                    self._collected_filter(workflow_id, reviewed_at)
                )
            )
        ) or 0

    def _iteration_query(self):
        """Iterations with the joined report / proposal identifiers."""
        return (
            select(
                SelfImproveIterationModel,
                WorkflowAnalysisReportModel.report_uuid,
                CopilotProposalModel.proposal_uuid,
                CopilotProposalModel.status,
            )
            .outerjoin(
                WorkflowAnalysisReportModel,
                SelfImproveIterationModel.report_id == WorkflowAnalysisReportModel.id,
            )
            .outerjoin(
                CopilotProposalModel,
                SelfImproveIterationModel.proposal_id == CopilotProposalModel.id,
            )
        )

    @staticmethod
    def _annotate(row) -> SelfImproveIterationModel:
        iteration, report_uuid, proposal_uuid, proposal_status = row
        iteration.report_uuid = report_uuid
        iteration.proposal_uuid = proposal_uuid
        iteration.proposal_status = proposal_status
        iteration.reviewed = iteration_is_reviewed(iteration.status, proposal_status)
        return iteration

    async def _load_iteration(
        self, session, iteration_id: int, organization_id: int
    ) -> Optional[SelfImproveIterationModel]:
        row = (
            await session.execute(
                self._iteration_query().where(
                    SelfImproveIterationModel.id == iteration_id,
                    SelfImproveIterationModel.organization_id == organization_id,
                )
            )
        ).first()
        return self._annotate(row) if row is not None else None

    def _unreviewed_round_exists(self, workflow_id: int):
        """EXISTS: a queued/running/ready round of the workflow not yet
        reviewed (D4). ``applying`` is not reviewed (O2)."""
        return (
            select(SelfImproveIterationModel.id)
            .outerjoin(
                CopilotProposalModel,
                SelfImproveIterationModel.proposal_id == CopilotProposalModel.id,
            )
            .where(
                SelfImproveIterationModel.workflow_id == workflow_id,
                SelfImproveIterationModel.status.in_(
                    _ACTIVE_STATUS_VALUES + (IterationStatus.ready.value,)
                ),
                or_(
                    CopilotProposalModel.id.is_(None),
                    CopilotProposalModel.status.not_in(REVIEWED_PROPOSAL_STATUSES),
                ),
            )
            .exists()
        )

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    async def get_self_improve(
        self, workflow_id: int, organization_id: int
    ) -> Optional[SelfImproveModel]:
        """The workflow's Self-improve row, or None when never configured."""
        async with self.async_session() as session:
            return await session.scalar(
                self._settings_query(workflow_id, organization_id)
            )

    async def count_collected_runs(self, workflow_id: int, organization_id: int) -> int:
        """Calls counted in the current window (the "7" of "7/10")."""
        async with self.async_session() as session:
            row = await session.scalar(
                self._settings_query(workflow_id, organization_id)
            )
            if row is None:
                return 0
            return await self._count_collected(session, workflow_id, row.reviewed_at)

    async def upsert_self_improve_settings(
        self,
        *,
        workflow_id: int,
        organization_id: int,
        enabled: bool,
        every_n_calls: int,
        user_id: Optional[int],
    ) -> SelfImproveModel:
        """Create or update the settings.

        Turning it on (off -> on, including the first time) records
        ``enabled_by_user_id`` and starts a fresh counting window
        (``reviewed_at = now``). Calls already counted are kept when only
        ``every_n_calls`` changes (DS-10). Raises ValueError for an N outside
        ``ALLOWED_EVERY_N_CALLS`` or a workflow outside the org.
        """
        if every_n_calls not in ALLOWED_EVERY_N_CALLS:
            raise ValueError(
                f"every_n_calls must be one of {list(ALLOWED_EVERY_N_CALLS)}"
            )
        async with self.async_session() as session:
            owned = await session.scalar(
                select(WorkflowModel.id).where(
                    WorkflowModel.id == workflow_id,
                    WorkflowModel.organization_id == organization_id,
                )
            )
            if owned is None:
                raise ValueError(f"Workflow {workflow_id} not found")
            try:
                await session.execute(
                    pg_insert(SelfImproveModel)
                    .values(
                        workflow_id=workflow_id,
                        organization_id=organization_id,
                        enabled=False,
                        every_n_calls=every_n_calls,
                    )
                    .on_conflict_do_nothing(index_elements=["workflow_id"])
                )
                row = await self._lock_settings(session, workflow_id, organization_id)
                now = datetime.now(UTC)
                if enabled and not row.enabled:
                    row.enabled_by_user_id = user_id
                    row.reviewed_at = now
                row.enabled = enabled
                row.every_n_calls = every_n_calls
                row.updated_at = now
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            return await session.scalar(
                self._settings_query(workflow_id, organization_id)
            )

    async def resume_self_improve(
        self, workflow_id: int, organization_id: int
    ) -> Optional[SelfImproveModel]:
        """Clear a D6 pause; returns None when not configured in the org."""
        return await self._update_settings(
            workflow_id,
            organization_id,
            paused=False,
            paused_reason=None,
            consecutive_failures=0,
        )

    async def mark_reviewed(
        self, *, workflow_id: int, organization_id: int
    ) -> Optional[SelfImproveModel]:
        """The user reviewed the waiting round: the next window starts now
        (D4: N counts calls completed after the review)."""
        return await self._update_settings(
            workflow_id, organization_id, reviewed_at=datetime.now(UTC)
        )

    async def _update_settings(
        self, workflow_id: int, organization_id: int, **values: Any
    ) -> Optional[SelfImproveModel]:
        async with self.async_session() as session:
            row = await self._lock_settings(session, workflow_id, organization_id)
            if row is None:
                return None
            try:
                for key, value in values.items():
                    setattr(row, key, value)
                row.updated_at = datetime.now(UTC)
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            return await session.scalar(
                self._settings_query(workflow_id, organization_id)
            )

    async def mark_round_outcome(
        self,
        *,
        workflow_id: int,
        organization_id: int,
        outcome: RoundOutcome,
        reason: Optional[str] = None,
    ) -> Optional[SelfImproveModel]:
        """Apply the D6 spend guard after a round ends.

        - ``changes``: multiplier back to 1, failure streak reset.
        - ``no_changes``: multiplier doubled (max ``MAX_WINDOW_MULTIPLIER``),
          failure streak reset; the round counts as reviewed, so the next
          window starts now.
        - ``failed``: failure streak +1; at ``FAILURES_BEFORE_PAUSE`` it
          pauses with ``reason``; the round counts as reviewed, so the next
          window starts now.

        Not idempotent: call once per round. Returns None when not configured.
        """
        outcome = _plain(outcome)
        if outcome not in ("changes", "no_changes", "failed"):
            raise ValueError(f"Unknown round outcome: {outcome!r}")
        async with self.async_session() as session:
            row = await self._lock_settings(session, workflow_id, organization_id)
            if row is None:
                return None
            try:
                now = datetime.now(UTC)
                if outcome == "changes":
                    row.window_multiplier = 1
                    row.consecutive_failures = 0
                elif outcome == "no_changes":
                    row.window_multiplier = min(
                        (row.window_multiplier or 1) * 2, MAX_WINDOW_MULTIPLIER
                    )
                    row.consecutive_failures = 0
                    row.reviewed_at = now
                else:
                    row.consecutive_failures = (row.consecutive_failures or 0) + 1
                    if row.consecutive_failures >= FAILURES_BEFORE_PAUSE:
                        row.paused = True
                        row.paused_reason = reason or "Rounds failed repeatedly."
                    row.reviewed_at = now
                row.updated_at = now
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            return await session.scalar(
                self._settings_query(workflow_id, organization_id)
            )

    # ------------------------------------------------------------------
    # Counting (D3, RI-2, RI-3, O7)
    # ------------------------------------------------------------------

    async def count_run_for_self_improve(
        self, *, workflow_id: int, organization_id: int, run_id: int
    ) -> tuple[bool, int]:
        """Count one completed real conversation toward the window.

        Atomic under the settings row lock. Returns ``(counted, collected)``:
        ``counted`` is False when the run was already counted (dedupe on
        ``run_id``), or Self-improve is not configured / off / paused;
        ``collected`` is the number of calls in the current window. Raises
        ValueError when the run does not belong to the workflow.
        """
        # Early returns / errors below skip an explicit rollback: nothing was
        # written, and closing the session ends the transaction (and lock).
        async with self.async_session() as session:
            settings = await self._lock_settings(session, workflow_id, organization_id)
            if settings is None:
                return False, 0
            reviewed_at = settings.reviewed_at
            if not settings.enabled or settings.paused:
                return False, await self._count_collected(
                    session, workflow_id, reviewed_at
                )
            run_ok = await session.scalar(
                select(WorkflowRunModel.id).where(
                    WorkflowRunModel.id == run_id,
                    WorkflowRunModel.workflow_id == workflow_id,
                )
            )
            if run_ok is None:
                raise ValueError(f"Run {run_id} not found")
            try:
                inserted = await session.scalar(
                    pg_insert(SelfImproveRunModel)
                    .values(
                        workflow_id=workflow_id,
                        organization_id=organization_id,
                        run_id=run_id,
                        counted_at=datetime.now(UTC),
                    )
                    .on_conflict_do_nothing(index_elements=["run_id"])
                    .returning(SelfImproveRunModel.id)
                )
                collected = await self._count_collected(
                    session, workflow_id, reviewed_at
                )
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            return inserted is not None, collected

    async def claim_window_for_iteration(
        self, *, workflow_id: int, organization_id: int, force: bool = False
    ) -> Optional[SelfImproveIterationModel]:
        """Create the next round when the window is full, atomically.

        Under the settings row lock: only when enabled, not paused, no
        earlier round is still unreviewed (D4), and at least
        ``every_n_calls * window_multiplier`` calls are collected. The round
        (number = max + 1, status queued) takes the oldest window-size
        collected calls as ``run_ids`` and stamps them with its id. Returns
        None when no round may start.

        ``force=True`` ("Run now") skips only the window-size check: it takes
        up to window-size collected calls and needs at least one.
        """
        # Early returns skip an explicit rollback: nothing was written, and
        # closing the session ends the transaction (and the row lock).
        async with self.async_session() as session:
            settings = await self._lock_settings(session, workflow_id, organization_id)
            if settings is None or not settings.enabled or settings.paused:
                return None
            if await session.scalar(select(self._unreviewed_round_exists(workflow_id))):
                return None
            window = settings.every_n_calls * (settings.window_multiplier or 1)
            run_ids = list(
                (
                    await session.execute(
                        select(SelfImproveRunModel.run_id)
                        .where(
                            self._collected_filter(workflow_id, settings.reviewed_at)
                        )
                        .order_by(
                            SelfImproveRunModel.counted_at.asc(),
                            SelfImproveRunModel.id.asc(),
                        )
                        .limit(window)
                    )
                ).scalars()
            )
            if not run_ids or (len(run_ids) < window and not force):
                return None
            last_number = await session.scalar(
                select(func.max(SelfImproveIterationModel.number)).where(
                    SelfImproveIterationModel.workflow_id == workflow_id
                )
            )
            try:
                now = datetime.now(UTC)
                iteration = SelfImproveIterationModel(
                    workflow_id=workflow_id,
                    organization_id=organization_id,
                    number=(last_number or 0) + 1,
                    status=IterationStatus.queued.value,
                    run_ids=run_ids,
                    carryover=[],
                    stage_started_at=now,
                )
                session.add(iteration)
                await session.flush()
                # Read before commit (commit expires the row).
                iteration_id = iteration.id
                await session.execute(
                    update(SelfImproveRunModel)
                    .where(
                        SelfImproveRunModel.workflow_id == workflow_id,
                        SelfImproveRunModel.run_id.in_(run_ids),
                    )
                    .values(iteration_id=iteration_id)
                    .execution_options(synchronize_session=False)
                )
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            return await self._load_iteration(session, iteration_id, organization_id)

    # ------------------------------------------------------------------
    # Rounds
    # ------------------------------------------------------------------

    async def get_iteration(
        self, iteration_id: int, organization_id: int
    ) -> Optional[SelfImproveIterationModel]:
        """A round with ``report_uuid``, ``proposal_uuid``,
        ``proposal_status`` and ``reviewed`` set."""
        async with self.async_session() as session:
            return await self._load_iteration(session, iteration_id, organization_id)

    async def list_iterations(
        self, workflow_id: int, organization_id: int, limit: int = 20
    ) -> list[SelfImproveIterationModel]:
        """The workflow's rounds, newest first, annotated like get_iteration."""
        async with self.async_session() as session:
            rows = (
                await session.execute(
                    self._iteration_query()
                    .where(
                        SelfImproveIterationModel.workflow_id == workflow_id,
                        SelfImproveIterationModel.organization_id == organization_id,
                    )
                    .order_by(SelfImproveIterationModel.number.desc())
                    .limit(limit)
                )
            ).all()
            return [self._annotate(row) for row in rows]

    async def update_iteration(
        self, iteration_id: int, organization_id: int, **fields: Any
    ) -> Optional[SelfImproveIterationModel]:
        """Update round columns; returns None when not found in the org.

        A ``status`` change also sets ``stage_started_at = now`` unless the
        caller passes it. ``report_id`` / ``proposal_id`` must belong to the
        round's workflow in the org (ValueError otherwise).
        """
        unknown = set(fields) - _ITERATION_UPDATABLE_FIELDS
        if unknown:
            raise ValueError(f"Cannot update iteration fields: {sorted(unknown)}")
        if "status" in fields:
            fields["status"] = _plain(fields["status"])
            if fields["status"] not in _ITERATION_STATUS_VALUES:
                raise ValueError(f"Unknown iteration status: {fields['status']!r}")
        async with self.async_session() as session:
            iteration = await session.scalar(
                select(SelfImproveIterationModel)
                .where(
                    SelfImproveIterationModel.id == iteration_id,
                    SelfImproveIterationModel.organization_id == organization_id,
                )
                .with_for_update()
            )
            if iteration is None:
                return None
            if fields.get("report_id") is not None:
                report_ok = await session.scalar(
                    select(WorkflowAnalysisReportModel.id).where(
                        WorkflowAnalysisReportModel.id == fields["report_id"],
                        WorkflowAnalysisReportModel.organization_id == organization_id,
                        WorkflowAnalysisReportModel.workflow_id
                        == iteration.workflow_id,
                    )
                )
                if report_ok is None:
                    raise ValueError(f"Report {fields['report_id']} not found")
            if fields.get("proposal_id") is not None:
                proposal_ok = await session.scalar(
                    select(CopilotProposalModel.id).where(
                        CopilotProposalModel.id == fields["proposal_id"],
                        CopilotProposalModel.organization_id == organization_id,
                        CopilotProposalModel.workflow_id == iteration.workflow_id,
                    )
                )
                if proposal_ok is None:
                    raise ValueError(f"Proposal {fields['proposal_id']} not found")
            try:
                now = datetime.now(UTC)
                if (
                    "status" in fields
                    and fields["status"] != iteration.status
                    and "stage_started_at" not in fields
                ):
                    iteration.stage_started_at = now
                for key, value in fields.items():
                    if key == "carryover" and value is not None:
                        value = [
                            v.model_dump(mode="json") if hasattr(v, "model_dump") else v
                            for v in value
                        ]
                    setattr(iteration, key, value)
                iteration.updated_at = now
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            return await self._load_iteration(session, iteration_id, organization_id)

    async def list_stuck_iterations(
        self, older_than_seconds: int, max_age_seconds: Optional[int] = None
    ) -> list[dict]:
        """Rounds in queued/diagnosing/proposing whose stage began more than
        ``older_than_seconds`` ago, across all orgs (O5 sweep). Each item:
        ``{id, organization_id, workflow_id, number, status, expired}``;
        ``expired`` = created more than ``max_age_seconds`` ago (the sweep
        fails those instead of retrying forever)."""
        now = datetime.now(UTC)
        cutoff = now - timedelta(seconds=older_than_seconds)
        expired_before = (
            now - timedelta(seconds=max_age_seconds)
            if max_age_seconds is not None
            else None
        )
        async with self.async_session() as session:
            rows = (
                await session.execute(
                    select(
                        SelfImproveIterationModel.id,
                        SelfImproveIterationModel.organization_id,
                        SelfImproveIterationModel.workflow_id,
                        SelfImproveIterationModel.number,
                        SelfImproveIterationModel.status,
                        SelfImproveIterationModel.created_at,
                    )
                    .where(
                        SelfImproveIterationModel.status.in_(_ACTIVE_STATUS_VALUES),
                        func.coalesce(
                            SelfImproveIterationModel.stage_started_at,
                            SelfImproveIterationModel.updated_at,
                        )
                        < cutoff,
                    )
                    .order_by(SelfImproveIterationModel.id.asc())
                )
            ).all()
            return [
                {
                    "id": row.id,
                    "organization_id": row.organization_id,
                    "workflow_id": row.workflow_id,
                    "number": row.number,
                    "status": row.status,
                    "expired": expired_before is not None
                    and row.created_at is not None
                    and row.created_at < expired_before,
                }
                for row in rows
            ]
