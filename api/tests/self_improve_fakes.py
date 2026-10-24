"""In-memory stand-in for the db_client methods Self-improve uses.

Mirrors the semantics of ``api/db/self_improve_client.py`` closely enough
for orchestration tests (window size, D4 review gate, D6 outcomes); the real
SQL is covered by ``test_self_improve_client.py``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, Optional

from api.services.self_improve.contracts import (
    FAILURES_BEFORE_PAUSE,
    MAX_WINDOW_MULTIPLIER,
)

ORG = 1
OTHER_ORG = 2
WF = 9
USER_ID = 3

ACTIVE = ("queued", "diagnosing", "proposing")
REVIEWED_PROPOSAL = ("applied", "discarded", "stale")


def _v(value: Any) -> Any:
    return getattr(value, "value", value)


class FakeSelfImproveDB:
    def __init__(self):
        self.now = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)
        self.workflows = {WF: ORG}
        self.settings: dict[int, SimpleNamespace] = {}
        self.counted: list[SimpleNamespace] = []  # self_improve_runs
        self.iterations: dict[int, SimpleNamespace] = {}
        self.reports: dict[str, SimpleNamespace] = {}
        self.proposals: dict[str, SimpleNamespace] = {}
        self.sessions: dict[int, SimpleNamespace] = {}
        self.members = {(USER_ID, ORG)}
        self.users = {
            USER_ID: SimpleNamespace(id=USER_ID, selected_organization_id=ORG)
        }
        self.runs: dict[int, dict] = {}  # run rows (get_runs_with_logs shape)
        self.run_models: dict[int, Any] = {}  # get_workflow_run_by_id
        self.versions: list[SimpleNamespace] = []
        self.brief = None
        self.outcomes: list[tuple] = []
        self.iteration_updates: list[dict] = []
        self.reports_created = 0
        self.raise_on: set[str] = set()

    def _maybe_raise(self, name: str) -> None:
        if name in self.raise_on:
            raise RuntimeError(f"{name} exploded")

    # ------------------------------------------------------------ workflows
    async def get_workflow(self, workflow_id, organization_id=None):
        if self.workflows.get(workflow_id) == organization_id:
            return SimpleNamespace(id=workflow_id, organization_id=organization_id)
        return None

    async def get_workflow_run_by_id(self, run_id):
        return self.run_models.get(run_id)

    async def get_workflow_versions(self, workflow_id, limit=None, offset=0):
        return list(self.versions)

    async def get_workflow_brief(self, workflow_id, organization_id):
        return self.brief

    # ------------------------------------------------------------ settings
    def configure(self, *, enabled=True, every_n_calls=10, **extra):
        self.settings[WF] = SimpleNamespace(
            workflow_id=WF,
            organization_id=ORG,
            enabled=enabled,
            every_n_calls=every_n_calls,
            enabled_by_user_id=USER_ID,
            window_multiplier=1,
            consecutive_failures=0,
            paused=False,
            paused_reason=None,
            reviewed_at=self.now - timedelta(days=1),
        )
        for key, value in extra.items():
            setattr(self.settings[WF], key, value)
        return self.settings[WF]

    async def get_self_improve(self, workflow_id, organization_id):
        row = self.settings.get(workflow_id)
        return (
            row if row is not None and row.organization_id == organization_id else None
        )

    def _collected(self, workflow_id):
        row = self.settings[workflow_id]
        return [
            c
            for c in self.counted
            if c.workflow_id == workflow_id
            and c.iteration_id is None
            and (row.reviewed_at is None or c.counted_at >= row.reviewed_at)
        ]

    async def count_collected_runs(self, workflow_id, organization_id):
        if await self.get_self_improve(workflow_id, organization_id) is None:
            return 0
        return len(self._collected(workflow_id))

    async def upsert_self_improve_settings(
        self, *, workflow_id, organization_id, enabled, every_n_calls, user_id
    ):
        if self.workflows.get(workflow_id) != organization_id:
            raise ValueError("Workflow not found")
        row = self.settings.get(workflow_id) or self.configure(
            enabled=False, every_n_calls=every_n_calls
        )
        if enabled and not row.enabled:
            row.enabled_by_user_id = user_id
            row.reviewed_at = self.now
        row.enabled = enabled
        row.every_n_calls = every_n_calls
        return row

    async def resume_self_improve(self, workflow_id, organization_id):
        row = await self.get_self_improve(workflow_id, organization_id)
        if row is None:
            return None
        row.paused, row.paused_reason, row.consecutive_failures = False, None, 0
        return row

    async def mark_round_outcome(
        self, *, workflow_id, organization_id, outcome, reason=None
    ):
        self.outcomes.append((workflow_id, outcome, reason))
        row = self.settings[workflow_id]
        if outcome == "changes":
            row.window_multiplier, row.consecutive_failures = 1, 0
        elif outcome == "no_changes":
            row.window_multiplier = min(
                row.window_multiplier * 2, MAX_WINDOW_MULTIPLIER
            )
            row.consecutive_failures = 0
            row.reviewed_at = self.now
        else:
            row.consecutive_failures += 1
            if row.consecutive_failures >= FAILURES_BEFORE_PAUSE:
                row.paused, row.paused_reason = True, reason
            row.reviewed_at = self.now
        return row

    # ------------------------------------------------------------ counting
    def add_counted(self, workflow_id, run_id, *, iteration_id=None):
        self.counted.append(
            SimpleNamespace(
                workflow_id=workflow_id,
                run_id=run_id,
                counted_at=self.now,
                iteration_id=iteration_id,
            )
        )

    async def count_run_for_self_improve(self, *, workflow_id, organization_id, run_id):
        row = await self.get_self_improve(workflow_id, organization_id)
        if row is None:
            return False, 0
        if not row.enabled or row.paused:
            return False, len(self._collected(workflow_id))
        if any(c.run_id == run_id for c in self.counted):
            return False, len(self._collected(workflow_id))
        self.add_counted(workflow_id, run_id)
        return True, len(self._collected(workflow_id))

    def _annotate(self, it):
        report = next((r for r in self.reports.values() if r.id == it.report_id), None)
        proposal = next(
            (p for p in self.proposals.values() if p.id == it.proposal_id), None
        )
        it.report_uuid = report.report_uuid if report else None
        it.proposal_uuid = proposal.proposal_uuid if proposal else None
        it.proposal_status = proposal.status if proposal else None
        it.reviewed = it.status in ("no_changes", "failed") or (
            it.proposal_status in REVIEWED_PROPOSAL
        )
        return it

    def add_iteration(self, *, number, status="queued", run_ids=(), **extra):
        iteration_id = len(self.iterations) + 100
        it = SimpleNamespace(
            id=iteration_id,
            workflow_id=WF,
            organization_id=ORG,
            number=number,
            status=status,
            run_ids=list(run_ids),
            report_id=None,
            proposal_id=None,
            carryover=[],
            summary=None,
            error=None,
            created_at=self.now,
            updated_at=self.now,
            stage_started_at=self.now,
        )
        for key, value in extra.items():
            setattr(it, key, value)
        self.iterations[iteration_id] = it
        return self._annotate(it)

    async def claim_window_for_iteration(
        self, *, workflow_id, organization_id, force=False
    ):
        row = await self.get_self_improve(workflow_id, organization_id)
        if row is None or not row.enabled or row.paused:
            return None
        for it in self.iterations.values():
            self._annotate(it)
            if it.workflow_id == workflow_id and (
                it.status in ACTIVE or (it.status == "ready" and not it.reviewed)
            ):
                return None
        window = row.every_n_calls * row.window_multiplier
        collected = self._collected(workflow_id)[:window]
        if not collected or (len(collected) < window and not force):
            return None
        number = max((i.number for i in self.iterations.values()), default=0) + 1
        it = self.add_iteration(number=number, run_ids=[c.run_id for c in collected])
        for c in collected:
            c.iteration_id = it.id
        return it

    # ------------------------------------------------------------ rounds
    async def get_iteration(self, iteration_id, organization_id):
        it = self.iterations.get(iteration_id)
        if it is None or it.organization_id != organization_id:
            return None
        return self._annotate(SimpleNamespace(**vars(it)))

    async def list_iterations(self, workflow_id, organization_id, limit=20):
        rows = [
            self._annotate(SimpleNamespace(**vars(it)))
            for it in self.iterations.values()
            if it.workflow_id == workflow_id and it.organization_id == organization_id
        ]
        return sorted(rows, key=lambda r: -r.number)[:limit]

    async def update_iteration(self, iteration_id, organization_id, **fields):
        self._maybe_raise("update_iteration")
        it = self.iterations.get(iteration_id)
        if it is None or it.organization_id != organization_id:
            return None
        if "status" in fields:
            fields["status"] = _v(fields["status"])
        if "carryover" in fields and fields["carryover"] is not None:
            fields["carryover"] = [
                c.model_dump(mode="json") if hasattr(c, "model_dump") else c
                for c in fields["carryover"]
            ]
        self.iteration_updates.append({"id": iteration_id, **fields})
        for key, value in fields.items():
            setattr(it, key, value)
        return await self.get_iteration(iteration_id, organization_id)

    async def list_stuck_iterations(self, older_than_seconds, max_age_seconds=None):
        cutoff = datetime.now(UTC) - timedelta(seconds=older_than_seconds)
        return [
            {
                "id": it.id,
                "organization_id": it.organization_id,
                "workflow_id": it.workflow_id,
                "number": it.number,
                "status": it.status,
                "expired": max_age_seconds is not None
                and getattr(it, "created_at", None) is not None
                and it.created_at < self.now - timedelta(seconds=max_age_seconds),
            }
            for it in self.iterations.values()
            if it.status in ACTIVE and it.stage_started_at < cutoff
        ]

    # ------------------------------------------------------------ reports
    async def create_analysis_report(
        self, *, organization_id, workflow_id, created_by, params
    ):
        self.reports_created += 1
        report = SimpleNamespace(
            id=len(self.reports) + 500,
            report_uuid=f"rep-{len(self.reports) + 1}",
            organization_id=organization_id,
            workflow_id=workflow_id,
            created_by=created_by,
            status="queued",
            params=params,
            result={},
            error=None,
        )
        self.reports[report.report_uuid] = report
        return report

    async def get_analysis_report(self, report_uuid, organization_id):
        report = self.reports.get(report_uuid)
        if report is None or report.organization_id != organization_id:
            return None
        return report

    async def update_analysis_report(
        self, report_uuid, organization_id, *, status=None, result=None, error=None
    ):
        report = await self.get_analysis_report(report_uuid, organization_id)
        if report is None:
            return None
        if status is not None:
            report.status = _v(status)
        if result is not None:
            report.result = result
        if error is not None:
            report.error = error
        return report

    async def get_runs_with_logs_for_workflow(
        self, workflow_id, organization_id, *, limit, filters=None, run_ids=None
    ):
        ids = run_ids if run_ids is not None else list(self.runs)[:limit]
        return [dict(self.runs[rid]) for rid in ids if rid in self.runs]

    # ------------------------------------------------------------ copilot
    def add_proposal(self, *, iteration_id=None, status="pending", **extra):
        proposal = SimpleNamespace(
            id=len(self.proposals) + 700,
            proposal_uuid=f"prop-{len(self.proposals) + 1}",
            organization_id=ORG,
            workflow_id=WF,
            iteration_id=iteration_id,
            status=status,
            summary="Fix the welcome step",
            changes=[],
            accepted_change_ids=[],
            applied_definition_id=None,
            hypotheses={},
        )
        for key, value in extra.items():
            setattr(proposal, key, value)
        self.proposals[proposal.proposal_uuid] = proposal
        if iteration_id is not None and iteration_id in self.iterations:
            self.iterations[iteration_id].proposal_id = proposal.id
        return proposal

    async def get_proposal_for_iteration(self, iteration_id, organization_id):
        for p in self.proposals.values():
            if p.iteration_id == iteration_id and p.organization_id == organization_id:
                return p
        return None

    async def get_copilot_proposal(self, proposal_uuid, organization_id):
        p = self.proposals.get(proposal_uuid)
        return p if p is not None and p.organization_id == organization_id else None

    async def get_or_create_self_improve_session(
        self, *, organization_id, workflow_id, user_id
    ):
        if workflow_id not in self.sessions:
            self.sessions[workflow_id] = SimpleNamespace(
                id=1, session_uuid=str(uuid.uuid4()), workflow_id=workflow_id
            )
        return self.sessions[workflow_id]

    # ------------------------------------------------------------ users
    async def is_user_member_of_organization(self, user_id, organization_id):
        return (user_id, organization_id) in self.members

    async def get_user_by_id(self, user_id):
        return self.users.get(user_id)


def ai(enabled: bool = True, reason: Optional[str] = None):
    """A resolve_ai_access replacement."""

    async def resolve(organization_id):
        if enabled:
            return True, None
        return False, reason or "not_opted_in"

    return resolve
