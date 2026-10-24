"""DB access for the workflow brief / North Star (M2b R15/D2, O8).

The brief is versioned: ``version`` is the optimistic lock and every save
appends the previous version to ``history``. Copilot-proposed edits are
``workflow_brief_changes`` rows the user confirms or dismisses.

Every method is organization-scoped (see api/AGENTS.md "Organization Scoping").
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional

from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.future import select

from api.db.base_client import BaseDBClient
from api.db.models import WorkflowBriefChangeModel, WorkflowBriefModel, WorkflowModel

#: Earlier versions kept in ``workflow_briefs.history`` (oldest dropped).
BRIEF_HISTORY_LIMIT = 50

#: copilot.contracts.BriefChangeStatus values
BRIEF_CHANGE_PENDING = "pending"
_BRIEF_CHANGE_RESOLVED_STATUSES = frozenset({"confirmed", "dismissed", "stale"})


class BriefVersionConflict(Exception):
    """The brief changed since the caller read it (expected_version mismatch)."""

    def __init__(self, expected_version: int, current_version: int):
        self.expected_version = expected_version
        self.current_version = current_version
        super().__init__(
            f"Brief version conflict: expected {expected_version}, "
            f"current {current_version}"
        )


def _plain(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _brief_dict(brief: Any) -> dict:
    """A WorkflowBrief model or a plain dict, stored as JSON."""
    if hasattr(brief, "model_dump"):
        return brief.model_dump(mode="json")
    return dict(brief or {})


class WorkflowBriefClient(BaseDBClient):
    async def _require_brief_workflow_in_org(
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
    # Brief
    # ------------------------------------------------------------------

    def _brief_query(self, workflow_id: int, organization_id: int):
        return select(WorkflowBriefModel).where(
            WorkflowBriefModel.workflow_id == workflow_id,
            WorkflowBriefModel.organization_id == organization_id,
        )

    async def get_workflow_brief(
        self, workflow_id: int, organization_id: int
    ) -> Optional[WorkflowBriefModel]:
        """The workflow's brief row, or None when it was never saved."""
        async with self.async_session() as session:
            return await session.scalar(self._brief_query(workflow_id, organization_id))

    async def save_workflow_brief(
        self,
        *,
        workflow_id: int,
        organization_id: int,
        brief: Any,
        expected_version: int,
        updated_by: str,
    ) -> WorkflowBriefModel:
        """Save a new brief version atomically.

        Locks the row, checks ``expected_version`` against the stored version
        (0 = never saved, which creates the row), appends the previous
        version to ``history`` and bumps ``version``. Raises
        ``BriefVersionConflict`` on mismatch, ValueError when the workflow is
        not in the org.
        """
        new_brief = _brief_dict(brief)
        now = datetime.now(UTC)
        async with self.async_session() as session:
            await self._require_brief_workflow_in_org(
                session, workflow_id, organization_id
            )
            # Conflicts are raised without an explicit rollback: nothing was
            # written, and closing the session releases the row lock.
            row = await session.scalar(
                self._brief_query(workflow_id, organization_id).with_for_update()
            )
            if row is None and expected_version != 0:
                raise BriefVersionConflict(expected_version, 0)
            if row is not None and row.version != expected_version:
                raise BriefVersionConflict(expected_version, row.version)
            if row is None:
                # ON CONFLICT: a concurrent first save won; report a conflict
                # instead of overwriting it.
                created_id = await session.scalar(
                    pg_insert(WorkflowBriefModel)
                    .values(
                        workflow_id=workflow_id,
                        organization_id=organization_id,
                        brief=new_brief,
                        version=1,
                        history=[],
                        updated_by=_plain(updated_by),
                        updated_at=now,
                        created_at=now,
                    )
                    .on_conflict_do_nothing(index_elements=["workflow_id"])
                    .returning(WorkflowBriefModel.id)
                )
                if created_id is None:
                    raise BriefVersionConflict(expected_version, 1)
            try:
                if row is not None:
                    history = list(row.history or [])
                    history.append(
                        {
                            "version": row.version,
                            "brief": dict(row.brief or {}),
                            "updated_at": (
                                row.updated_at.isoformat() if row.updated_at else None
                            ),
                            "updated_by": row.updated_by,
                        }
                    )
                    row.history = history[-BRIEF_HISTORY_LIMIT:]
                    row.brief = new_brief
                    row.version = row.version + 1
                    row.updated_by = _plain(updated_by)
                    row.updated_at = now
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            return await session.scalar(self._brief_query(workflow_id, organization_id))

    # ------------------------------------------------------------------
    # Brief changes (copilot proposals awaiting Confirm / Dismiss)
    # ------------------------------------------------------------------

    def _change_query(self, change_uuid: str, organization_id: int):
        return select(WorkflowBriefChangeModel).where(
            WorkflowBriefChangeModel.change_uuid == change_uuid,
            WorkflowBriefChangeModel.organization_id == organization_id,
        )

    async def create_brief_change(
        self,
        *,
        workflow_id: int,
        organization_id: int,
        base_version: int,
        proposed: Any,
        reason: Optional[str] = None,
    ) -> WorkflowBriefChangeModel:
        """Record a pending brief change. Older pending changes of the same
        workflow are dismissed, so at most one is pending at a time."""
        async with self.async_session() as session:
            await self._require_brief_workflow_in_org(
                session, workflow_id, organization_id
            )
            now = datetime.now(UTC)
            await session.execute(
                update(WorkflowBriefChangeModel)
                .where(
                    WorkflowBriefChangeModel.workflow_id == workflow_id,
                    WorkflowBriefChangeModel.organization_id == organization_id,
                    WorkflowBriefChangeModel.status == BRIEF_CHANGE_PENDING,
                )
                .values(status="dismissed", resolved_at=now)
            )
            row = WorkflowBriefChangeModel(
                workflow_id=workflow_id,
                organization_id=organization_id,
                base_version=base_version,
                proposed=_brief_dict(proposed),
                reason=reason or "",
                status=BRIEF_CHANGE_PENDING,
            )
            session.add(row)
            try:
                await session.flush()
                # Read before commit (commit expires the row).
                change_uuid = row.change_uuid
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            return await session.scalar(
                self._change_query(change_uuid, organization_id)
            )

    async def get_brief_change(
        self, change_uuid: str, organization_id: int
    ) -> Optional[WorkflowBriefChangeModel]:
        async with self.async_session() as session:
            return await session.scalar(
                self._change_query(change_uuid, organization_id)
            )

    async def get_pending_brief_change(
        self, workflow_id: int, organization_id: int
    ) -> Optional[WorkflowBriefChangeModel]:
        """The newest pending change of the workflow, if any."""
        async with self.async_session() as session:
            return await session.scalar(
                select(WorkflowBriefChangeModel)
                .where(
                    WorkflowBriefChangeModel.workflow_id == workflow_id,
                    WorkflowBriefChangeModel.organization_id == organization_id,
                    WorkflowBriefChangeModel.status == BRIEF_CHANGE_PENDING,
                )
                .order_by(
                    WorkflowBriefChangeModel.created_at.desc(),
                    WorkflowBriefChangeModel.id.desc(),
                )
                .limit(1)
            )

    async def resolve_brief_change(
        self, change_uuid: str, organization_id: int, status: Any
    ) -> Optional[WorkflowBriefChangeModel]:
        """Move a pending change to confirmed / dismissed / stale.

        One conditional UPDATE (pending only), so of two concurrent resolves
        one wins; returns None when the change is missing, in another org or
        already resolved. Does not touch the brief: to confirm, call
        ``save_workflow_brief(expected_version=change.base_version, ...)``
        first and resolve as ``stale`` on ``BriefVersionConflict``.
        """
        status = _plain(status)
        if status not in _BRIEF_CHANGE_RESOLVED_STATUSES:
            raise ValueError(f"Cannot resolve a brief change as {status!r}")
        async with self.async_session() as session:
            result = await session.execute(
                update(WorkflowBriefChangeModel)
                .where(
                    WorkflowBriefChangeModel.change_uuid == change_uuid,
                    WorkflowBriefChangeModel.organization_id == organization_id,
                    WorkflowBriefChangeModel.status == BRIEF_CHANGE_PENDING,
                )
                .values(status=status, resolved_at=datetime.now(UTC))
            )
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            if result.rowcount != 1:
                return None
            return await session.scalar(
                self._change_query(change_uuid, organization_id)
            )
