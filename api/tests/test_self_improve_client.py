"""SelfImproveClient (milestone 2b) against Postgres test_db.

Covers the migration objects, settings, run counting with dedupe, the
round claim (window size, D4 review gate, D6 backoff / pause / resume),
round updates, the stuck-round sweep, org scoping, and two races on real
separate sessions (RI-2).
"""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from api.db.models import WorkflowRunModel
from api.services.self_improve.contracts import (
    FAILURES_BEFORE_PAUSE,
    MAX_WINDOW_MULTIPLIER,
    IterationStatus,
)


async def _org_with_workflow(db_session, slug: str):
    user, _ = await db_session.get_or_create_user_by_provider_id(f"{slug}_user")
    org, _ = await db_session.get_or_create_organization_by_provider_id(
        f"{slug}_org", user.id
    )
    await db_session.update_user_selected_organization(user.id, org.id)
    workflow = await db_session.create_workflow(
        name=f"{slug} workflow",
        workflow_definition={"nodes": [], "edges": []},
        user_id=user.id,
        organization_id=org.id,
    )
    return user, org, workflow


async def _runs(async_session, workflow, count: int, mode: str = "twilio"):
    rows = [
        WorkflowRunModel(
            name=f"run-{i}", workflow_id=workflow.id, mode=mode, is_completed=True
        )
        for i in range(count)
    ]
    async_session.add_all(rows)
    await async_session.flush()
    return [row.id for row in rows]


async def _enabled(db_session, org, workflow, user, n: int = 5):
    return await db_session.upsert_self_improve_settings(
        workflow_id=workflow.id,
        organization_id=org.id,
        enabled=True,
        every_n_calls=n,
        user_id=user.id,
    )


async def _count_all(db_session, org, workflow, run_ids):
    results = []
    for run_id in run_ids:
        results.append(
            await db_session.count_run_for_self_improve(
                workflow_id=workflow.id, organization_id=org.id, run_id=run_id
            )
        )
    return results


async def _claim(db_session, org, workflow, **kwargs):
    return await db_session.claim_window_for_iteration(
        workflow_id=workflow.id, organization_id=org.id, **kwargs
    )


async def _proposal(db_session, user, org, workflow, iteration):
    session = await db_session.get_or_create_self_improve_session(
        organization_id=org.id, workflow_id=workflow.id, user_id=user.id
    )
    return await db_session.create_copilot_proposal(
        session_id=session.id,
        organization_id=org.id,
        workflow_id=workflow.id,
        kind="edit",
        summary="Fix pricing",
        base_hash="h",
        candidate_json={},
        code="",
        changes=[],
        iteration_id=iteration.id,
    )


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


async def test_migration_objects_exist(async_session):
    tables = set(
        (
            await async_session.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public'"
                )
            )
        ).scalars()
    )
    assert {
        "workflow_briefs",
        "workflow_brief_changes",
        "self_improve",
        "self_improve_iterations",
        "self_improve_runs",
    } <= tables

    async def columns(table):
        return {
            row.column_name: row
            for row in (
                await async_session.execute(
                    text(
                        "SELECT column_name, is_nullable, column_default "
                        "FROM information_schema.columns WHERE table_name = :t"
                    ),
                    {"t": table},
                )
            ).all()
        }

    session_cols = await columns("copilot_sessions")
    assert session_cols["kind"].is_nullable == "NO"
    assert "chat" in session_cols["kind"].column_default
    proposal_cols = await columns("copilot_proposals")
    assert {
        "iteration_id",
        "accepted_change_ids",
        "applied_definition_id",
        "hypotheses",
    } <= set(proposal_cols)

    constraints = set(
        (
            await async_session.execute(
                text("SELECT conname FROM pg_constraint WHERE conname = ANY(:names)"),
                {
                    "names": [
                        "fk_copilot_proposals_iteration_id",
                        "fk_self_improve_iterations_proposal_id",
                        "fk_copilot_proposals_applied_definition_id",
                        "uq_self_improve_iterations_workflow_number",
                    ]
                },
            )
        ).scalars()
    )
    assert len(constraints) == 4
    indexes = set(
        (
            await async_session.execute(
                text("SELECT indexname FROM pg_indexes WHERE indexname = ANY(:names)"),
                {
                    "names": [
                        "uq_copilot_sessions_self_improve_workflow",
                        "uq_copilot_proposals_iteration_id",
                        "ix_self_improve_runs_workflow_iteration",
                    ]
                },
            )
        ).scalars()
    )
    assert len(indexes) == 3


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


async def test_settings_upsert_and_org_scoping(db_session):
    user, org, workflow = await _org_with_workflow(db_session, "si_settings")
    _, other_org, foreign = await _org_with_workflow(db_session, "si_settings_x")

    assert await db_session.get_self_improve(workflow.id, org.id) is None
    with pytest.raises(ValueError):
        await _enabled(db_session, org, workflow, user, n=7)
    with pytest.raises(ValueError):
        await _enabled(db_session, org, foreign, user)

    off = await db_session.upsert_self_improve_settings(
        workflow_id=workflow.id,
        organization_id=org.id,
        enabled=False,
        every_n_calls=10,
        user_id=user.id,
    )
    assert off.enabled is False and off.reviewed_at is None
    assert off.window_multiplier == 1 and off.consecutive_failures == 0
    assert off.paused is False and off.enabled_by_user_id is None

    on = await _enabled(db_session, org, workflow, user, n=5)
    assert on.enabled is True and on.every_n_calls == 5
    assert on.enabled_by_user_id == user.id and on.reviewed_at is not None
    first_window = on.reviewed_at

    # Changing N keeps the window (DS-10).
    changed = await _enabled(db_session, org, workflow, user, n=20)
    assert changed.every_n_calls == 20 and changed.reviewed_at == first_window

    # Off then on starts a fresh window.
    await db_session.upsert_self_improve_settings(
        workflow_id=workflow.id,
        organization_id=org.id,
        enabled=False,
        every_n_calls=20,
        user_id=user.id,
    )
    again = await _enabled(db_session, org, workflow, user, n=20)
    assert again.reviewed_at > first_window

    assert await db_session.get_self_improve(workflow.id, other_org.id) is None
    assert await db_session.resume_self_improve(workflow.id, other_org.id) is None
    assert (
        await db_session.mark_reviewed(
            workflow_id=workflow.id, organization_id=other_org.id
        )
        is None
    )
    assert (
        await db_session.mark_round_outcome(
            workflow_id=workflow.id, organization_id=other_org.id, outcome="failed"
        )
        is None
    )
    assert await db_session.count_collected_runs(workflow.id, other_org.id) == 0


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


async def test_count_run_dedupe_and_gating(db_session, async_session):
    user, org, workflow = await _org_with_workflow(db_session, "si_count")
    _, other_org, foreign = await _org_with_workflow(db_session, "si_count_x")
    run_a, run_b = await _runs(async_session, workflow, 2)
    (foreign_run,) = await _runs(async_session, foreign, 1)

    # Not configured / off: nothing counted.
    assert await _count_all(db_session, org, workflow, [run_a]) == [(False, 0)]
    await db_session.upsert_self_improve_settings(
        workflow_id=workflow.id,
        organization_id=org.id,
        enabled=False,
        every_n_calls=5,
        user_id=user.id,
    )
    assert await _count_all(db_session, org, workflow, [run_a]) == [(False, 0)]

    await _enabled(db_session, org, workflow, user)
    # The same run twice is counted once.
    assert await _count_all(db_session, org, workflow, [run_a, run_a, run_b]) == [
        (True, 1),
        (False, 1),
        (True, 2),
    ]
    assert await db_session.count_collected_runs(workflow.id, org.id) == 2

    with pytest.raises(ValueError):
        await _count_all(db_session, org, workflow, [foreign_run])
    # Other org: not configured there.
    assert await db_session.count_run_for_self_improve(
        workflow_id=workflow.id, organization_id=other_org.id, run_id=run_b
    ) == (False, 0)


async def test_window_starts_at_review(db_session, async_session):
    user, org, workflow = await _org_with_workflow(db_session, "si_window")
    await _enabled(db_session, org, workflow, user, n=5)
    early = await _runs(async_session, workflow, 3)
    await _count_all(db_session, org, workflow, early)
    assert await db_session.count_collected_runs(workflow.id, org.id) == 3

    await db_session.mark_reviewed(workflow_id=workflow.id, organization_id=org.id)
    assert await db_session.count_collected_runs(workflow.id, org.id) == 0

    late = await _runs(async_session, workflow, 5)
    results = await _count_all(db_session, org, workflow, late)
    assert results[-1] == (True, 5)
    iteration = await _claim(db_session, org, workflow)
    assert iteration.run_ids == late


# ---------------------------------------------------------------------------
# Claim + D4 review gate
# ---------------------------------------------------------------------------


async def test_claim_takes_exactly_the_oldest_window(db_session, async_session):
    user, org, workflow = await _org_with_workflow(db_session, "si_claim")
    _, other_org, _ = await _org_with_workflow(db_session, "si_claim_x")
    await _enabled(db_session, org, workflow, user, n=5)
    run_ids = await _runs(async_session, workflow, 7)

    await _count_all(db_session, org, workflow, run_ids[:4])
    assert await _claim(db_session, org, workflow) is None
    await _count_all(db_session, org, workflow, run_ids[4:])
    assert (
        await db_session.claim_window_for_iteration(
            workflow_id=workflow.id, organization_id=other_org.id
        )
        is None
    )

    iteration = await _claim(db_session, org, workflow)
    assert iteration.number == 1
    assert iteration.status == IterationStatus.queued.value
    assert iteration.run_ids == run_ids[:5]
    assert iteration.stage_started_at is not None
    assert iteration.reviewed is False and iteration.proposal_uuid is None
    assert await db_session.count_collected_runs(workflow.id, org.id) == 2

    stamped = (
        await async_session.execute(
            text(
                "SELECT run_id FROM self_improve_runs WHERE iteration_id = :i "
                "ORDER BY run_id"
            ),
            {"i": iteration.id},
        )
    ).scalars()
    assert list(stamped) == sorted(run_ids[:5])


async def test_force_claim_takes_what_is_collected(db_session, async_session):
    user, org, workflow = await _org_with_workflow(db_session, "si_force")
    await _enabled(db_session, org, workflow, user, n=10)
    assert await _claim(db_session, org, workflow, force=True) is None
    run_ids = await _runs(async_session, workflow, 3)
    await _count_all(db_session, org, workflow, run_ids)
    assert await _claim(db_session, org, workflow) is None
    iteration = await _claim(db_session, org, workflow, force=True)
    assert iteration.run_ids == run_ids


@pytest.mark.parametrize(
    "review", ["applied", "discarded", "stale", "no_changes", "failed"]
)
async def test_claim_waits_for_review(db_session, async_session, review):
    user, org, workflow = await _org_with_workflow(db_session, f"si_gate_{review}")
    await _enabled(db_session, org, workflow, user, n=5)
    first = await _runs(async_session, workflow, 5)
    await _count_all(db_session, org, workflow, first)
    round_one = await _claim(db_session, org, workflow)
    second = await _runs(async_session, workflow, 5)
    await _count_all(db_session, org, workflow, second)

    # Running rounds block.
    for status in ("queued", "diagnosing", "proposing"):
        await db_session.update_iteration(round_one.id, org.id, status=status)
        assert await _claim(db_session, org, workflow) is None

    if review in ("no_changes", "failed"):
        updated = await db_session.update_iteration(round_one.id, org.id, status=review)
    else:
        proposal = await _proposal(db_session, user, org, workflow, round_one)
        updated = await db_session.update_iteration(
            round_one.id,
            org.id,
            status=IterationStatus.ready,
            proposal_id=proposal.id,
        )
        assert updated.proposal_uuid == proposal.proposal_uuid
        assert updated.reviewed is False
        # A pending or applying proposal still blocks (O2).
        for status in ("pending", "applying"):
            await db_session.update_copilot_proposal(
                proposal.proposal_uuid, org.id, status=status
            )
            assert await _claim(db_session, org, workflow) is None
        await db_session.update_copilot_proposal(
            proposal.proposal_uuid, org.id, status=review
        )
        updated = await db_session.get_iteration(round_one.id, org.id)

    assert updated.reviewed is True
    round_two = await _claim(db_session, org, workflow)
    assert round_two is not None
    assert round_two.number == 2 and round_two.run_ids == second


# ---------------------------------------------------------------------------
# D6 spend guard
# ---------------------------------------------------------------------------


async def test_round_outcome_backoff_pause_and_resume(db_session, async_session):
    user, org, workflow = await _org_with_workflow(db_session, "si_d6")
    await _enabled(db_session, org, workflow, user, n=5)

    async def outcome(value, reason=None):
        return await db_session.mark_round_outcome(
            workflow_id=workflow.id,
            organization_id=org.id,
            outcome=value,
            reason=reason,
        )

    with pytest.raises(ValueError):
        await outcome("maybe")

    multipliers = [(await outcome("no_changes")).window_multiplier for _ in range(3)]
    assert multipliers == [2, 4, MAX_WINDOW_MULTIPLIER]

    # The backed-off window needs N x multiplier calls.
    run_ids = await _runs(async_session, workflow, 20)
    await _count_all(db_session, org, workflow, run_ids[:19])
    assert await _claim(db_session, org, workflow) is None
    await _count_all(db_session, org, workflow, run_ids[19:])
    iteration = await _claim(db_session, org, workflow)
    assert len(iteration.run_ids) == 20
    await db_session.update_iteration(iteration.id, org.id, status="no_changes")

    reset = await outcome("changes")
    assert reset.window_multiplier == 1 and reset.consecutive_failures == 0

    first_fail = await outcome("failed", "Doctor timed out")
    assert first_fail.consecutive_failures == 1 and first_fail.paused is False
    assert FAILURES_BEFORE_PAUSE == 2
    paused = await outcome("failed", "Doctor timed out")
    assert paused.consecutive_failures == 2
    assert paused.paused is True and paused.paused_reason == "Doctor timed out"

    # Paused: nothing counted, no round.
    more = await _runs(async_session, workflow, 5)
    counted = await _count_all(db_session, org, workflow, more)
    assert all(result[0] is False for result in counted)
    assert await _claim(db_session, org, workflow, force=True) is None

    resumed = await db_session.resume_self_improve(workflow.id, org.id)
    assert resumed.paused is False and resumed.consecutive_failures == 0
    assert resumed.paused_reason is None
    counted = await _count_all(db_session, org, workflow, more)
    assert counted[-1] == (True, 5)
    assert await _claim(db_session, org, workflow) is not None

    # A failure after a success streak resets does not pause at once.
    await outcome("changes")
    assert (await outcome("failed")).paused is False


# ---------------------------------------------------------------------------
# Rounds
# ---------------------------------------------------------------------------


async def test_iteration_updates_and_org_scoping(db_session, async_session):
    user, org, workflow = await _org_with_workflow(db_session, "si_iter")
    other_user, other_org, foreign = await _org_with_workflow(db_session, "si_iter_x")
    await _enabled(db_session, org, workflow, user, n=5)
    for _ in range(2):
        await _count_all(
            db_session, org, workflow, await _runs(async_session, workflow, 5)
        )
        iteration = await _claim(db_session, org, workflow)
        await db_session.update_iteration(iteration.id, org.id, status="failed")

    listed = await db_session.list_iterations(workflow.id, org.id)
    assert [i.number for i in listed] == [2, 1]
    assert all(i.reviewed for i in listed)
    assert [
        i.number for i in await db_session.list_iterations(workflow.id, org.id, limit=1)
    ] == [2]
    assert await db_session.list_iterations(workflow.id, other_org.id) == []
    assert await db_session.get_iteration(iteration.id, other_org.id) is None
    assert (
        await db_session.update_iteration(iteration.id, other_org.id, summary="x")
        is None
    )

    with pytest.raises(ValueError):
        await db_session.update_iteration(iteration.id, org.id, number=9)
    with pytest.raises(ValueError):
        await db_session.update_iteration(iteration.id, org.id, status="bogus")

    report = await db_session.create_analysis_report(
        organization_id=org.id, workflow_id=workflow.id, created_by=user.id
    )
    foreign_report = await db_session.create_analysis_report(
        organization_id=other_org.id, workflow_id=foreign.id, created_by=other_user.id
    )
    with pytest.raises(ValueError):
        await db_session.update_iteration(
            iteration.id, org.id, report_id=foreign_report.id
        )

    before = (await db_session.get_iteration(iteration.id, org.id)).stage_started_at
    carry = [{"proposal_uuid": "p", "change_id": "c", "title": "t", "outcome": "fixed"}]
    updated = await db_session.update_iteration(
        iteration.id,
        org.id,
        status=IterationStatus.diagnosing,
        report_id=report.id,
        carryover=carry,
        summary="Analysing",
        error=None,
    )
    assert updated.status == "diagnosing" and updated.report_uuid == report.report_uuid
    assert updated.carryover == carry and updated.summary == "Analysing"
    assert updated.stage_started_at > before
    assert updated.reviewed is False

    # Same status again keeps the stage start.
    same = await db_session.update_iteration(iteration.id, org.id, status="diagnosing")
    assert same.stage_started_at == updated.stage_started_at


async def test_list_stuck_iterations(db_session, async_session):
    user, org, workflow = await _org_with_workflow(db_session, "si_stuck")
    await _enabled(db_session, org, workflow, user, n=5)
    await _count_all(db_session, org, workflow, await _runs(async_session, workflow, 5))
    iteration = await _claim(db_session, org, workflow)

    def mine(rows):
        return [r for r in rows if r["id"] == iteration.id]

    assert mine(await db_session.list_stuck_iterations(1800)) == []
    old = datetime.now(UTC) - timedelta(hours=1)
    await db_session.update_iteration(
        iteration.id, org.id, status="diagnosing", stage_started_at=old
    )
    stuck = mine(await db_session.list_stuck_iterations(1800))
    assert stuck == [
        {
            "id": iteration.id,
            "organization_id": org.id,
            "workflow_id": workflow.id,
            "number": 1,
            "status": "diagnosing",
            "expired": False,
        }
    ]
    # Created just now: not expired at a 1h max age, expired at a 0s max age.
    [fresh] = mine(await db_session.list_stuck_iterations(1800, max_age_seconds=3600))
    assert fresh["expired"] is False
    [old_round] = mine(await db_session.list_stuck_iterations(1800, max_age_seconds=0))
    assert old_round["expired"] is True
    await db_session.update_iteration(
        iteration.id, org.id, status="ready", stage_started_at=old
    )
    assert mine(await db_session.list_stuck_iterations(1800)) == []


# ---------------------------------------------------------------------------
# Real separate sessions (prod session settings: expire_on_commit=True)
# ---------------------------------------------------------------------------


@pytest.fixture
async def real_client(test_engine):
    """A DBClient on its own connections, committing for real.

    Data is created under a unique slug and deleted afterwards.
    """
    from api.db.db_client import DBClient

    client = DBClient()
    await client.engine.dispose()
    client.engine = test_engine
    client.async_session = async_sessionmaker(bind=test_engine)
    created: list[int] = []
    yield client, created
    async with test_engine.begin() as conn:
        for workflow_id in created:
            params = {"w": workflow_id}
            await conn.execute(
                text(
                    "UPDATE workflows SET released_definition_id = NULL WHERE id = :w"
                ),
                params,
            )
            for statement in (
                "DELETE FROM copilot_sessions WHERE workflow_id = :w",
                "DELETE FROM self_improve_runs WHERE workflow_id = :w",
                "DELETE FROM self_improve_iterations WHERE workflow_id = :w",
                "DELETE FROM self_improve WHERE workflow_id = :w",
                "DELETE FROM workflow_brief_changes WHERE workflow_id = :w",
                "DELETE FROM workflow_briefs WHERE workflow_id = :w",
                "DELETE FROM workflow_analysis_reports WHERE workflow_id = :w",
                "DELETE FROM workflow_runs WHERE workflow_id = :w",
                "DELETE FROM workflow_definitions WHERE workflow_id = :w",
                "DELETE FROM workflows WHERE id = :w",
            ):
                await conn.execute(text(statement), params)


async def _real_setup(client, created, slug: str, runs: int):
    user, org, workflow = await _org_with_workflow(client, slug)
    created.append(workflow.id)
    async with client.async_session() as session:
        rows = [
            WorkflowRunModel(
                name=f"run-{i}",
                workflow_id=workflow.id,
                mode="twilio",
                is_completed=True,
            )
            for i in range(runs)
        ]
        session.add_all(rows)
        await session.flush()
        run_ids = [row.id for row in rows]
        await session.commit()
    await _enabled(client, org, workflow, user, n=5)
    return user, org, workflow, run_ids


async def test_concurrent_completions_create_one_round(real_client):
    client, created = real_client
    user, org, workflow, run_ids = await _real_setup(client, created, "si_race", 6)

    await _count_all(client, org, workflow, run_ids[:3])

    async def completion(run_id):
        counted = await client.count_run_for_self_improve(
            workflow_id=workflow.id, organization_id=org.id, run_id=run_id
        )
        iteration = await client.claim_window_for_iteration(
            workflow_id=workflow.id, organization_id=org.id
        )
        return counted, iteration

    # Runs 4 and 5 complete at once (the window boundary), plus a duplicate
    # completion job for run 5.
    results = await asyncio.gather(
        completion(run_ids[3]), completion(run_ids[4]), completion(run_ids[4])
    )
    counted_flags = [counted for (counted, _), _ in results]
    assert sorted(counted_flags) == [False, True, True]
    rounds = [iteration for _, iteration in results if iteration is not None]
    assert len(rounds) == 1
    # The two racing runs may be counted in either order (oldest-counted first).
    assert rounds[0].number == 1
    assert rounds[0].run_ids[:3] == run_ids[:3]
    assert sorted(rounds[0].run_ids) == sorted(run_ids[:5])

    listed = await client.list_iterations(workflow.id, org.id)
    assert [i.number for i in listed] == [1]
    assert await client.count_collected_runs(workflow.id, org.id) == 0

    # Two concurrent claims on an already-full window: still one round.
    await client.update_iteration(rounds[0].id, org.id, status="no_changes")
    extra = await _real_runs(client, workflow, 4)
    await _count_all(client, org, workflow, [run_ids[5], *extra])
    claims = await asyncio.gather(
        *(
            client.claim_window_for_iteration(
                workflow_id=workflow.id, organization_id=org.id
            )
            for _ in range(3)
        )
    )
    made = [c for c in claims if c is not None]
    assert len(made) == 1 and made[0].number == 2


async def _real_runs(client, workflow, count):
    async with client.async_session() as session:
        rows = [
            WorkflowRunModel(
                name=f"run-x{i}",
                workflow_id=workflow.id,
                mode="twilio",
                is_completed=True,
            )
            for i in range(count)
        ]
        session.add_all(rows)
        await session.flush()
        run_ids = [row.id for row in rows]
        await session.commit()
    return run_ids


async def test_write_paths_on_expiring_sessions(real_client):
    """Every write method returns usable rows when commit expires them."""
    client, created = real_client
    user, org, workflow, run_ids = await _real_setup(client, created, "si_real", 5)

    await _count_all(client, org, workflow, run_ids)
    iteration = await client.claim_window_for_iteration(
        workflow_id=workflow.id, organization_id=org.id
    )
    assert iteration.number == 1 and iteration.reviewed is False

    session = await client.get_or_create_self_improve_session(
        organization_id=org.id, workflow_id=workflow.id, user_id=user.id
    )
    again = await client.get_or_create_self_improve_session(
        organization_id=org.id, workflow_id=workflow.id, user_id=user.id
    )
    assert again.session_uuid == session.session_uuid and session.kind == "self_improve"

    proposal = await client.create_copilot_proposal(
        session_id=session.id,
        organization_id=org.id,
        workflow_id=workflow.id,
        kind="edit",
        summary="s",
        base_hash=None,
        candidate_json={},
        code="",
        changes=[],
        iteration_id=iteration.id,
        hypotheses={"n:1": {"title": "Pricing in welcome", "fix_target": "Welcome"}},
    )
    assert proposal.iteration_id == iteration.id
    assert proposal.hypotheses["n:1"]["title"] == "Pricing in welcome"
    found = await client.get_proposal_for_iteration(iteration.id, org.id)
    assert found.proposal_uuid == proposal.proposal_uuid

    updated = await client.update_iteration(
        iteration.id, org.id, status="ready", proposal_id=proposal.id
    )
    assert updated.status == "ready" and updated.proposal_uuid == proposal.proposal_uuid

    applied = await client.update_copilot_proposal(
        proposal.proposal_uuid,
        org.id,
        status="applied",
        accepted_change_ids=["n:1"],
        applied_definition_id=workflow.released_definition_id,
    )
    assert applied.accepted_change_ids == ["n:1"]
    assert (await client.get_iteration(iteration.id, org.id)).reviewed is True

    settings = await client.mark_round_outcome(
        workflow_id=workflow.id, organization_id=org.id, outcome="changes"
    )
    assert settings.window_multiplier == 1
    assert (
        await client.mark_reviewed(workflow_id=workflow.id, organization_id=org.id)
    ).reviewed_at is not None
    assert (await client.resume_self_improve(workflow.id, org.id)).paused is False

    brief = await client.save_workflow_brief(
        workflow_id=workflow.id,
        organization_id=org.id,
        brief={"goal": "Book demos"},
        expected_version=0,
        updated_by="user",
    )
    assert brief.version == 1
    brief = await client.save_workflow_brief(
        workflow_id=workflow.id,
        organization_id=org.id,
        brief={"goal": "Book more demos"},
        expected_version=1,
        updated_by="copilot",
    )
    assert brief.version == 2 and brief.history[0]["version"] == 1
    change = await client.create_brief_change(
        workflow_id=workflow.id,
        organization_id=org.id,
        base_version=2,
        proposed={"goal": "x"},
        reason="r",
    )
    resolved = await client.resolve_brief_change(
        change.change_uuid, org.id, "dismissed"
    )
    assert resolved.status == "dismissed" and resolved.resolved_at is not None

    async with client.async_session() as s:
        counted = (
            await s.execute(
                text("SELECT count(*) FROM self_improve_runs WHERE iteration_id = :i"),
                {"i": iteration.id},
            )
        ).scalar()
        assert counted == 5
