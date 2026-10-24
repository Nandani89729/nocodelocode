"""Regression contract for the internal-run filter (Agent Doctor R1/R6, D18/D23).

`exclude_internal_runs()` hides Doctor replay runs (mode='doctor_replay') from
every customer-facing run list, run count and CSV/daily report. The contract:

* For every real mode, each enumerated query returns exactly the same rows, in
  the same order, with the same counts as before a replay run existed.
* Replay runs never appear in those queries.
* Usage aggregation does NOT hide replay usage; it reports it on its own
  `doctor_replay` line (D21).
* Single-run lookups by id still see replay runs (internal code needs them).

Two layers:
1. SQL-shape tests (no database): every enumerated query carries the filter,
   single-run lookups do not.
2. Seeded database tests (Postgres test_db): seed one run per existing mode,
   snapshot each query, add replay runs, assert the output is unchanged.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.dialects import postgresql

from api.db.db_client import DBClient
from api.db.filters import exclude_internal_runs
from api.db.models import CampaignModel, WorkflowRunModel
from api.enums import WorkflowRunMode
from api.services.workflow.doctor.contracts import (
    DOCTOR_REPLAY_RUN_MODE,
    INTERNAL_RUN_MODES,
)

REAL_MODES = [mode.value for mode in WorkflowRunMode]


# ---------------------------------------------------------------------------
# Layer 1: SQL shape (no database)
# ---------------------------------------------------------------------------


class _FakeResult:
    def scalars(self):
        return self

    def all(self):
        return []

    def first(self):
        return None

    def scalar(self):
        return 0

    def scalar_one_or_none(self):
        return object()  # "campaign exists" for the ownership pre-checks

    def scalar_one(self):
        return object()

    def one(self):
        class _Row:
            total_seconds = 0
            call_count = 0

        return _Row()

    def __iter__(self):
        return iter([])


class _FakeSession:
    def __init__(self, statements):
        self._statements = statements

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, statement, *args, **kwargs):
        self._statements.append(statement)
        return _FakeResult()

    async def scalar(self, statement, *args, **kwargs):
        self._statements.append(statement)
        return 0


def _client_recording(statements) -> DBClient:
    client = DBClient.__new__(DBClient)
    client.async_session = lambda: _FakeSession(statements)
    return client


def _excludes_internal(statement) -> bool:
    compiled = statement.compile(
        dialect=postgresql.dialect(), compile_kwargs={"render_postcompile": True}
    )
    sql = str(compiled).replace("\n", " ")
    values = set()
    for value in compiled.params.values():
        if isinstance(value, (list, tuple)):
            values.update(value)
        else:
            values.add(value)
    return "workflow_runs.mode NOT IN" in sql and set(INTERNAL_RUN_MODES) <= values


START = datetime(2026, 1, 1, tzinfo=UTC)
END = datetime(2026, 12, 31, tzinfo=UTC)

# name -> coroutine factory; each must issue at least one run query and every
# workflow_runs list/count/report query it issues must carry the filter.
ENUMERATED_QUERIES = {
    "workflow_runs_by_workflow_id": lambda c: c.get_workflow_runs_by_workflow_id(
        1, organization_id=1
    ),
    "workflow_runs_for_superadmin": lambda c: c.get_workflow_runs_for_superadmin(),
    "campaign_runs": lambda c: c.get_campaign_runs(1, 1),
    "campaign_runs_paginated": lambda c: c.get_campaign_runs_paginated(1, 1),
    "completed_runs_for_report_campaign": lambda c: c.get_completed_runs_for_report(
        campaign_id=1
    ),
    "completed_runs_for_report_workflow": lambda c: c.get_completed_runs_for_report(
        workflow_id=1
    ),
    "usage_runs_for_report": lambda c: c.get_usage_runs_for_report(1),
    "usage_history": lambda c: c.get_usage_history(1),
    "daily_report": lambda c: c.get_workflow_runs_for_daily_report(1, START, END),
    "workflow_run_count": lambda c: c.get_workflow_run_count(1),
    "workflow_run_counts": lambda c: c.get_workflow_run_counts([1, 2]),
    "doctor_input": lambda c: c.get_runs_with_logs_for_workflow(1, 1, limit=10),
}


def _run_statements(statements):
    """Statements that read workflow_runs (skip ownership pre-checks)."""
    out = []
    for statement in statements:
        froms = {getattr(t, "name", None) for t in statement.get_final_froms()}
        sql = str(statement.compile(dialect=postgresql.dialect()))
        if "workflow_runs" in froms or "FROM workflow_runs" in sql:
            out.append(statement)
    return out


@pytest.mark.parametrize("name", sorted(ENUMERATED_QUERIES))
async def test_enumerated_query_excludes_internal_runs(name):
    statements: list = []
    await ENUMERATED_QUERIES[name](_client_recording(statements))

    run_statements = _run_statements(statements)
    assert run_statements, f"{name} issued no workflow_runs query"
    for statement in run_statements:
        assert _excludes_internal(statement), f"{name} lacks the filter:\n{statement}"


async def test_daily_usage_breakdown_reports_replays_on_their_own_line():
    statements: list = []
    result = await _client_recording(statements).get_daily_usage_breakdown(
        1, START, END, 0.01
    )

    run_statements = _run_statements(statements)
    assert len(run_statements) == 2
    per_day, replay = run_statements
    assert _excludes_internal(per_day)
    replay_sql = str(
        replay.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )
    assert f"workflow_runs.mode = '{DOCTOR_REPLAY_RUN_MODE}'" in replay_sql
    assert set(result["doctor_replay"]) == {
        "minutes",
        "cost_usd",
        "dograh_tokens",
        "call_count",
    }


@pytest.mark.parametrize(
    "lookup",
    [
        lambda c: c.get_workflow_run(1, organization_id=1),
        lambda c: c.get_workflow_run_by_id(1),
        lambda c: c.get_workflow_run_with_context(1),
    ],
    ids=["get_workflow_run", "get_workflow_run_by_id", "with_context"],
)
async def test_single_run_lookups_still_see_internal_runs(lookup):
    statements: list = []
    await lookup(_client_recording(statements))
    assert statements
    assert not any(_excludes_internal(s) for s in statements)


def test_filter_never_drops_real_modes():
    """Every real mode survives; mode is NOT NULL so NOT IN is null-safe."""
    assert WorkflowRunModel.__table__.c.mode.nullable is False
    assert not set(REAL_MODES) & set(INTERNAL_RUN_MODES)
    from sqlalchemy import select

    sql = str(
        exclude_internal_runs(select(WorkflowRunModel.id)).compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )
    assert "workflow_runs.mode NOT IN ('doctor_replay')" in sql


# ---------------------------------------------------------------------------
# Layer 2: seeded database (requires Postgres test_db, see api/.env.test)
# ---------------------------------------------------------------------------


async def _seed_org(db_session, async_session, slug: str):
    user, _ = await db_session.get_or_create_user_by_provider_id(f"{slug}_user")
    org, _ = await db_session.get_or_create_organization_by_provider_id(
        f"{slug}_org", user.id
    )
    await db_session.update_user_selected_organization(user.id, org.id)
    workflow = await db_session.create_workflow(
        name=f"{slug} workflow",
        workflow_definition={},
        user_id=user.id,
        organization_id=org.id,
    )
    campaign = CampaignModel(
        name=f"{slug} campaign",
        organization_id=org.id,
        workflow_id=workflow.id,
        created_by=user.id,
        source_type="csv",
        source_id="seed.csv",
    )
    async_session.add(campaign)
    await async_session.flush()
    return user, org, workflow, campaign


def _run(workflow, campaign, mode: str, created_at: datetime, seconds: float):
    return WorkflowRunModel(
        name=f"run-{mode}",
        workflow_id=workflow.id,
        definition_id=workflow.released_definition_id,
        mode=mode,
        call_type="outbound",
        state="completed",
        is_completed=True,
        campaign_id=campaign.id,
        created_at=created_at,
        usage_info={"call_duration_seconds": seconds},
        cost_info={"total_cost_usd": 0.01},
        initial_context={"phone_number": "+15550100"},
        gathered_context={"mapped_call_disposition": "XFER"},
        logs={"realtime_feedback_events": []},
        annotations={},
    )


def _ids(rows):
    out = []
    for row in rows:
        if isinstance(row, dict):
            out.append(row["id"])
        elif isinstance(row, tuple):
            out.append(row[0])
        elif isinstance(row, int):
            out.append(row)
        else:
            out.append(row.id)
    return out


def _seeded_queries(org, workflow, campaign, start, end):
    wf_filter = [
        {"attribute": "workflowId", "type": "number", "value": {"value": workflow.id}}
    ]

    async def by_workflow(c):
        rows, total = await c.get_workflow_runs_by_workflow_id(
            workflow.id, organization_id=org.id, limit=100
        )
        return [r.model_dump() for r in rows], total

    async def superadmin(c):
        rows, total = await c.get_workflow_runs_for_superadmin(
            limit=100, filters=wf_filter
        )
        return rows, total

    async def campaign_runs(c):
        rows = await c.get_campaign_runs(campaign.id, org.id)
        return [r.id for r in rows], len(rows)

    async def campaign_paginated(c):
        rows, total = await c.get_campaign_runs_paginated(
            campaign.id, org.id, limit=100
        )
        return [r.model_dump() for r in rows], total

    async def csv_campaign(c):
        rows = await c.get_completed_runs_for_report(campaign_id=campaign.id)
        return [tuple(r) for r in rows], len(rows)

    async def csv_workflow(c):
        rows = await c.get_completed_runs_for_report(workflow_id=workflow.id)
        return [tuple(r) for r in rows], len(rows)

    async def usage_csv(c):
        rows = await c.get_usage_runs_for_report(org.id)
        return [tuple(r) for r in rows], len(rows)

    async def usage_history(c):
        rows, total, tokens, seconds = await c.get_usage_history(org.id, limit=100)
        return rows, (total, tokens, seconds)

    async def daily_report(c):
        rows = await c.get_workflow_runs_for_daily_report(org.id, start, end)
        # The original query has no ORDER BY: compare as a sorted set.
        return sorted(rows, key=lambda r: r["id"]), len(rows)

    async def run_count(c):
        return [], await c.get_workflow_run_count(workflow.id)

    async def run_counts(c):
        return [], await c.get_workflow_run_counts([workflow.id])

    async def doctor_input(c):
        rows = await c.get_runs_with_logs_for_workflow(workflow.id, org.id, limit=100)
        return rows, len(rows)

    return {
        "workflow_runs_by_workflow_id": by_workflow,
        "workflow_runs_for_superadmin": superadmin,
        "campaign_runs": campaign_runs,
        "campaign_runs_paginated": campaign_paginated,
        "completed_runs_for_report_campaign": csv_campaign,
        "completed_runs_for_report_workflow": csv_workflow,
        "usage_runs_for_report": usage_csv,
        "usage_history": usage_history,
        "daily_report": daily_report,
        "workflow_run_count": run_count,
        "workflow_run_counts": run_counts,
        "doctor_input": doctor_input,
    }


@pytest.mark.parametrize("name", sorted(ENUMERATED_QUERIES))
async def test_seeded_query_output_unchanged_by_replay_runs(
    db_session, async_session, name
):
    _, org, workflow, campaign = await _seed_org(
        db_session, async_session, f"internal_runs_{name}"
    )
    now = datetime.now(UTC).replace(microsecond=0)
    real_runs = [
        _run(workflow, campaign, mode, now - timedelta(minutes=i + 1), 10 + i)
        for i, mode in enumerate(REAL_MODES)
    ]
    async_session.add_all(real_runs)
    await async_session.flush()
    newest_first = [r.id for r in sorted(real_runs, key=lambda r: r.created_at)][::-1]

    query = _seeded_queries(org, workflow, campaign, now - timedelta(days=1), now)[name]
    before_rows, before_count = await query(db_session)

    # Replay runs interleaved with real ones: newest, middle and oldest.
    replays = [
        _run(workflow, campaign, DOCTOR_REPLAY_RUN_MODE, at, 99)
        for at in (
            now - timedelta(seconds=10),
            now - timedelta(minutes=3, seconds=30),
            now - timedelta(hours=2),
        )
    ]
    async_session.add_all(replays)
    await async_session.flush()
    replay_ids = {r.id for r in replays}

    after_rows, after_count = await query(db_session)

    assert after_rows == before_rows
    assert after_count == before_count
    if before_rows:
        ids = _ids(before_rows)
        assert not set(ids) & replay_ids
        if name == "daily_report":
            assert ids == sorted(newest_first)
        else:
            assert ids == newest_first
    if name == "workflow_run_count":
        assert after_count == len(REAL_MODES)
    if name == "workflow_run_counts":
        assert after_count == {workflow.id: len(REAL_MODES)}


async def test_seeded_daily_usage_shows_replay_line(db_session, async_session):
    _, org, workflow, campaign = await _seed_org(
        db_session, async_session, "internal_runs_usage"
    )
    now = datetime.now(UTC).replace(microsecond=0)
    async_session.add_all(
        [
            _run(workflow, campaign, mode, now - timedelta(minutes=1), 60)
            for mode in REAL_MODES
        ]
    )
    await async_session.flush()
    start, end, price = now - timedelta(days=1), now + timedelta(minutes=1), 0.01

    before = await db_session.get_daily_usage_breakdown(org.id, start, end, price)
    async_session.add(
        _run(
            workflow, campaign, DOCTOR_REPLAY_RUN_MODE, now - timedelta(minutes=1), 120
        )
    )
    await async_session.flush()
    after = await db_session.get_daily_usage_breakdown(org.id, start, end, price)

    assert after["breakdown"] == before["breakdown"]
    assert before["doctor_replay"]["call_count"] == 0
    assert after["doctor_replay"] == {
        "minutes": 2.0,
        "cost_usd": 1.2,
        "dograh_tokens": 120.0,
        "call_count": 1,
    }
    assert after["total_minutes"] == pytest.approx(before["total_minutes"] + 2.0)
