"""DB access for the Agent Copilot: chat sessions and change proposals.

Every method is organization-scoped (see api/AGENTS.md "Organization Scoping").
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.future import select

from api.db.base_client import BaseDBClient
from api.db.models import (
    CopilotProposalModel,
    CopilotSessionModel,
    SelfImproveIterationModel,
    WorkflowDefinitionModel,
    WorkflowModel,
)

#: copilot.contracts.SessionKind values
SESSION_KIND_CHAT = "chat"
SESSION_KIND_SELF_IMPROVE = "self_improve"
_SESSION_KINDS = frozenset({SESSION_KIND_CHAT, SESSION_KIND_SELF_IMPROVE})

# Columns callers may change through update_copilot_session / _proposal; an
# unknown key raises instead of being silently dropped.
_SESSION_UPDATABLE_FIELDS = frozenset({"title", "transcript", "plan", "workflow_id"})
_PROPOSAL_UPDATABLE_FIELDS = frozenset(
    {
        "status",
        "applied_at",
        "workflow_id",
        "summary",
        "base_hash",
        "candidate_json",
        "code",
        "changes",
        "workflow_name",
        "iteration_id",
        "accepted_change_ids",
        "applied_definition_id",
        "hypotheses",
        "settings_candidate",
        "settings_base_hash",
        "new_tools",
    }
)


def _plain(value: Any) -> Any:
    """Enums (e.g. ProposalStatus) are stored by value."""
    return value.value if isinstance(value, Enum) else value


def _jsonable_hypotheses(hypotheses: Optional[dict]) -> dict:
    """change_id -> Hypothesis; accepts pydantic models or plain dicts."""
    out: dict = {}
    for change_id, hypothesis in (hypotheses or {}).items():
        if hasattr(hypothesis, "model_dump"):
            hypothesis = hypothesis.model_dump(mode="json")
        out[str(change_id)] = dict(hypothesis)
    return out


class CopilotClient(BaseDBClient):
    async def _require_copilot_workflow_in_org(
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

    async def _require_iteration_in_org(
        self, session, iteration_id: int, organization_id: int
    ) -> None:
        owned = await session.scalar(
            select(SelfImproveIterationModel.id).where(
                SelfImproveIterationModel.id == iteration_id,
                SelfImproveIterationModel.organization_id == organization_id,
            )
        )
        if owned is None:
            raise ValueError(f"Self-improve iteration {iteration_id} not found")

    async def _require_definition_in_org(
        self, session, definition_id: int, organization_id: int
    ) -> None:
        owned = await session.scalar(
            select(WorkflowDefinitionModel.id)
            .join(
                WorkflowModel, WorkflowDefinitionModel.workflow_id == WorkflowModel.id
            )
            .where(
                WorkflowDefinitionModel.id == definition_id,
                WorkflowModel.organization_id == organization_id,
            )
        )
        if owned is None:
            raise ValueError(f"Workflow definition {definition_id} not found")

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    async def create_copilot_session(
        self,
        *,
        organization_id: int,
        user_id: int,
        workflow_id: Optional[int],
        kind: str = SESSION_KIND_CHAT,
    ) -> CopilotSessionModel:
        """Create a session. ``kind="self_improve"`` needs a workflow and is
        unique per workflow; prefer ``get_or_create_self_improve_session``."""
        kind = _plain(kind)
        if kind not in _SESSION_KINDS:
            raise ValueError(f"Unknown copilot session kind: {kind}")
        if kind == SESSION_KIND_SELF_IMPROVE and workflow_id is None:
            raise ValueError("A self-improve session needs a workflow")
        async with self.async_session() as session:
            if workflow_id is not None:
                await self._require_copilot_workflow_in_org(
                    session, workflow_id, organization_id
                )
            row = CopilotSessionModel(
                organization_id=organization_id,
                user_id=user_id,
                workflow_id=workflow_id,
                kind=kind,
                transcript=[],
            )
            session.add(row)
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            await session.refresh(row)
            return row

    def _self_improve_session_query(self, workflow_id: int, organization_id: int):
        return select(CopilotSessionModel).where(
            CopilotSessionModel.workflow_id == workflow_id,
            CopilotSessionModel.organization_id == organization_id,
            CopilotSessionModel.kind == SESSION_KIND_SELF_IMPROVE,
        )

    async def get_or_create_self_improve_session(
        self, *, organization_id: int, workflow_id: int, user_id: int
    ) -> CopilotSessionModel:
        """The workflow's single self-improve session (RI-1), created on first
        use. ``user_id`` is only used when creating it."""
        existing = await self._get_self_improve_session(workflow_id, organization_id)
        if existing is not None:
            return existing
        try:
            return await self.create_copilot_session(
                organization_id=organization_id,
                user_id=user_id,
                workflow_id=workflow_id,
                kind=SESSION_KIND_SELF_IMPROVE,
            )
        except IntegrityError:
            # A concurrent caller created it first (partial unique index).
            existing = await self._get_self_improve_session(
                workflow_id, organization_id
            )
            if existing is None:
                raise
            return existing

    async def _get_self_improve_session(
        self, workflow_id: int, organization_id: int
    ) -> Optional[CopilotSessionModel]:
        async with self.async_session() as session:
            return await session.scalar(
                self._self_improve_session_query(workflow_id, organization_id)
            )

    async def get_copilot_session(
        self, session_uuid: str, organization_id: int
    ) -> Optional[CopilotSessionModel]:
        async with self.async_session() as session:
            return await session.scalar(
                select(CopilotSessionModel).where(
                    CopilotSessionModel.session_uuid == session_uuid,
                    CopilotSessionModel.organization_id == organization_id,
                )
            )

    async def list_copilot_sessions(
        self,
        organization_id: int,
        *,
        workflow_id: Optional[int] = None,
        user_id: Optional[int] = None,
        limit: int = 20,
        kind: Optional[str] = SESSION_KIND_CHAT,
    ) -> list[CopilotSessionModel]:
        """Sessions in the org, most recently active first.

        Only chat sessions by default, so the chat resume list never returns
        the self-improve session (RI-1); ``kind=None`` lists every kind.
        """
        async with self.async_session() as session:
            query = select(CopilotSessionModel).where(
                CopilotSessionModel.organization_id == organization_id
            )
            if kind is not None:
                query = query.where(CopilotSessionModel.kind == _plain(kind))
            if workflow_id is not None:
                query = query.where(CopilotSessionModel.workflow_id == workflow_id)
            if user_id is not None:
                query = query.where(CopilotSessionModel.user_id == user_id)
            result = await session.execute(
                query.order_by(
                    CopilotSessionModel.updated_at.desc(),
                    CopilotSessionModel.id.desc(),
                ).limit(limit)
            )
            return list(result.scalars().all())

    async def update_copilot_session(
        self, session_uuid: str, organization_id: int, **fields: Any
    ) -> Optional[CopilotSessionModel]:
        """Update session columns; returns None when not found in the org.

        ``transcript`` / ``plan`` replace the stored JSON wholesale (callers
        own the merge, so each checkpoint is one atomic write).
        """
        unknown = set(fields) - _SESSION_UPDATABLE_FIELDS
        if unknown:
            raise ValueError(f"Cannot update copilot session fields: {sorted(unknown)}")
        async with self.async_session() as session:
            row = await session.scalar(
                select(CopilotSessionModel)
                .where(
                    CopilotSessionModel.session_uuid == session_uuid,
                    CopilotSessionModel.organization_id == organization_id,
                )
                .with_for_update()
            )
            if row is None:
                return None
            if fields.get("workflow_id") is not None:
                await self._require_copilot_workflow_in_org(
                    session, fields["workflow_id"], organization_id
                )
            for key, value in fields.items():
                if key == "transcript" and value is not None:
                    value = list(value)
                setattr(row, key, value)
            row.updated_at = datetime.now(UTC)
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            await session.refresh(row)
            return row

    # ------------------------------------------------------------------
    # Proposals
    # ------------------------------------------------------------------

    def _proposal_query(self, proposal_uuid: str, organization_id: int):
        return select(CopilotProposalModel).where(
            CopilotProposalModel.proposal_uuid == proposal_uuid,
            CopilotProposalModel.organization_id == organization_id,
        )

    async def create_copilot_proposal(
        self,
        *,
        session_id: int,
        organization_id: int,
        workflow_id: Optional[int],
        kind: Any,
        summary: str,
        base_hash: Optional[str],
        candidate_json: dict,
        code: str,
        changes: list,
        workflow_name: Optional[str] = None,
        iteration_id: Optional[int] = None,
        hypotheses: Optional[dict] = None,
        settings_candidate: Optional[dict] = None,
        settings_base_hash: Optional[str] = None,
        new_tools: Optional[list] = None,
    ) -> CopilotProposalModel:
        """Create a proposal. With ``iteration_id`` (a Self-improve round) a
        second proposal for the same round raises ValueError (O5).

        ``settings_candidate`` / ``settings_base_hash`` / ``new_tools`` (M2c)
        default to a graph-only proposal."""
        async with self.async_session() as session:
            session_ok = await session.scalar(
                select(CopilotSessionModel.id).where(
                    CopilotSessionModel.id == session_id,
                    CopilotSessionModel.organization_id == organization_id,
                )
            )
            if session_ok is None:
                raise ValueError(f"Copilot session {session_id} not found")
            if workflow_id is not None:
                await self._require_copilot_workflow_in_org(
                    session, workflow_id, organization_id
                )
            if iteration_id is not None:
                await self._require_iteration_in_org(
                    session, iteration_id, organization_id
                )
                already = await session.scalar(
                    select(CopilotProposalModel.id).where(
                        CopilotProposalModel.iteration_id == iteration_id
                    )
                )
                if already is not None:
                    raise ValueError(
                        f"Self-improve iteration {iteration_id} already has a proposal"
                    )
            row = CopilotProposalModel(
                session_id=session_id,
                organization_id=organization_id,
                workflow_id=workflow_id,
                kind=_plain(kind),
                summary=summary or "",
                base_hash=base_hash,
                candidate_json=dict(candidate_json or {}),
                code=code or "",
                changes=list(changes or []),
                workflow_name=workflow_name,
                iteration_id=iteration_id,
                hypotheses=_jsonable_hypotheses(hypotheses),
                accepted_change_ids=[],
                settings_candidate=(
                    dict(settings_candidate) if settings_candidate is not None else None
                ),
                settings_base_hash=settings_base_hash,
                new_tools=list(new_tools or []),
            )
            session.add(row)
            try:
                await session.flush()
                # Read before commit: commit expires the row, and touching an
                # expired attribute would trigger a sync lazy load.
                proposal_uuid = row.proposal_uuid
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                if iteration_id is not None:
                    # Lost a race with a concurrent proposal for this round.
                    raise ValueError(
                        f"Self-improve iteration {iteration_id} already has a proposal"
                    ) from exc
                raise
            except Exception:
                await session.rollback()
                raise
            # Re-select so the joined ``session`` relationship is loaded.
            return await session.scalar(
                self._proposal_query(proposal_uuid, organization_id)
            )

    async def get_copilot_proposal(
        self, proposal_uuid: str, organization_id: int
    ) -> Optional[CopilotProposalModel]:
        async with self.async_session() as session:
            return await session.scalar(
                self._proposal_query(proposal_uuid, organization_id)
            )

    async def get_proposal_for_iteration(
        self, iteration_id: int, organization_id: int
    ) -> Optional[CopilotProposalModel]:
        """The proposal a Self-improve round made, if any (at most one)."""
        async with self.async_session() as session:
            return await session.scalar(
                select(CopilotProposalModel).where(
                    CopilotProposalModel.iteration_id == iteration_id,
                    CopilotProposalModel.organization_id == organization_id,
                )
            )

    async def list_copilot_proposals(
        self, session_id: int, organization_id: int
    ) -> list[CopilotProposalModel]:
        """Proposals of one session, oldest first (chat order)."""
        async with self.async_session() as session:
            result = await session.execute(
                select(CopilotProposalModel)
                .where(
                    CopilotProposalModel.session_id == session_id,
                    CopilotProposalModel.organization_id == organization_id,
                )
                .order_by(
                    CopilotProposalModel.created_at.asc(),
                    CopilotProposalModel.id.asc(),
                )
            )
            return list(result.scalars().unique().all())

    async def claim_copilot_proposal(
        self,
        proposal_uuid: str,
        organization_id: int,
        *,
        from_status: str,
        to_status: str,
    ) -> bool:
        """Atomically move a proposal from one status to another.

        One conditional UPDATE, so of two concurrent applies only one wins;
        returns False when the proposal is missing, in another org, or no
        longer in ``from_status``.
        """
        async with self.async_session() as session:
            result = await session.execute(
                update(CopilotProposalModel)
                .where(
                    CopilotProposalModel.proposal_uuid == proposal_uuid,
                    CopilotProposalModel.organization_id == organization_id,
                    CopilotProposalModel.status == from_status,
                )
                .values(status=to_status)
            )
            await session.commit()
            return result.rowcount == 1

    async def update_copilot_proposal(
        self, proposal_uuid: str, organization_id: int, **fields: Any
    ) -> Optional[CopilotProposalModel]:
        """Update proposal columns; returns None when not found in the org."""
        unknown = set(fields) - _PROPOSAL_UPDATABLE_FIELDS
        if unknown:
            raise ValueError(
                f"Cannot update copilot proposal fields: {sorted(unknown)}"
            )
        async with self.async_session() as session:
            # Lock only the proposal row: FOR UPDATE cannot apply to the
            # nullable side of the joined session relationship.
            row = await session.scalar(
                self._proposal_query(proposal_uuid, organization_id).with_for_update(
                    of=CopilotProposalModel
                )
            )
            if row is None:
                return None
            if fields.get("workflow_id") is not None:
                await self._require_copilot_workflow_in_org(
                    session, fields["workflow_id"], organization_id
                )
            if fields.get("iteration_id") is not None:
                await self._require_iteration_in_org(
                    session, fields["iteration_id"], organization_id
                )
            if fields.get("applied_definition_id") is not None:
                await self._require_definition_in_org(
                    session, fields["applied_definition_id"], organization_id
                )
            for key, value in fields.items():
                if key == "accepted_change_ids" and value is not None:
                    value = [str(_plain(v)) for v in value]
                elif key == "hypotheses" and value is not None:
                    value = _jsonable_hypotheses(value)
                setattr(row, key, _plain(value))
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            return await session.scalar(
                self._proposal_query(proposal_uuid, organization_id)
            )
