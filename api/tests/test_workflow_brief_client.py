"""WorkflowBriefClient (milestone 2b R15/O8) against Postgres test_db.

Covers versioned saves with the optimistic lock and history, pending brief
changes (create / supersede / resolve once), and org scoping.
"""

import pytest

from api.db.workflow_brief_client import BriefVersionConflict
from api.services.copilot.contracts import BriefChangeStatus, WorkflowBrief


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


async def test_brief_save_versions_and_history(db_session):
    _, org, workflow = await _org_with_workflow(db_session, "brief_save")
    _, other_org, foreign = await _org_with_workflow(db_session, "brief_save_x")

    assert await db_session.get_workflow_brief(workflow.id, org.id) is None

    async def save(brief, expected, by="user", workflow_id=workflow.id, org_id=org.id):
        return await db_session.save_workflow_brief(
            workflow_id=workflow_id,
            organization_id=org_id,
            brief=brief,
            expected_version=expected,
            updated_by=by,
        )

    # No brief yet: only expected_version 0 may create it.
    with pytest.raises(BriefVersionConflict) as conflict:
        await save({"goal": "x"}, 3)
    assert conflict.value.current_version == 0

    first = await save(WorkflowBrief(goal="Book demos", must_do=["Ask email"]), 0)
    assert first.version == 1 and first.history == []
    assert first.brief["goal"] == "Book demos"
    assert first.brief["must_do"] == ["Ask email"]
    assert first.updated_by == "user"

    with pytest.raises(BriefVersionConflict) as conflict:
        await save({"goal": "stale"}, 0, by="copilot")
    assert conflict.value.expected_version == 0
    assert conflict.value.current_version == 1

    second = await save({"goal": "Book more demos"}, 1, by="copilot")
    assert second.version == 2 and second.updated_by == "copilot"
    assert second.brief == {"goal": "Book more demos"}
    assert [h["version"] for h in second.history] == [1]
    assert second.history[0]["brief"]["goal"] == "Book demos"
    assert second.history[0]["updated_by"] == "user"
    assert second.history[0]["updated_at"]

    third = await save({"goal": "v3"}, 2)
    assert [h["version"] for h in third.history] == [1, 2]

    fetched = await db_session.get_workflow_brief(workflow.id, org.id)
    assert fetched.version == 3 and fetched.brief == {"goal": "v3"}

    # Org scoping.
    assert await db_session.get_workflow_brief(workflow.id, other_org.id) is None
    with pytest.raises(ValueError):
        await save({"goal": "x"}, 3, workflow_id=workflow.id, org_id=other_org.id)
    with pytest.raises(ValueError):
        await save({"goal": "x"}, 0, workflow_id=foreign.id, org_id=org.id)
    assert (await db_session.get_workflow_brief(workflow.id, org.id)).version == 3


async def test_brief_change_lifecycle(db_session):
    _, org, workflow = await _org_with_workflow(db_session, "brief_change")
    _, other_org, foreign = await _org_with_workflow(db_session, "brief_change_x")

    assert await db_session.get_pending_brief_change(workflow.id, org.id) is None
    with pytest.raises(ValueError):
        await db_session.create_brief_change(
            workflow_id=foreign.id,
            organization_id=org.id,
            base_version=0,
            proposed={},
            reason="r",
        )

    first = await db_session.create_brief_change(
        workflow_id=workflow.id,
        organization_id=org.id,
        base_version=0,
        proposed=WorkflowBrief(goal="Qualify leads"),
        reason="You said the goal is qualifying leads",
    )
    assert len(first.change_uuid) == 36
    assert first.status == "pending" and first.resolved_at is None
    assert first.proposed["goal"] == "Qualify leads"
    assert first.base_version == 0

    pending = await db_session.get_pending_brief_change(workflow.id, org.id)
    assert pending.change_uuid == first.change_uuid

    # A newer proposal supersedes the older pending one.
    second = await db_session.create_brief_change(
        workflow_id=workflow.id,
        organization_id=org.id,
        base_version=0,
        proposed={"goal": "Qualify and book"},
        reason=None,
    )
    assert second.reason == ""
    superseded = await db_session.get_brief_change(first.change_uuid, org.id)
    assert superseded.status == "dismissed" and superseded.resolved_at is not None
    pending = await db_session.get_pending_brief_change(workflow.id, org.id)
    assert pending.change_uuid == second.change_uuid

    # Org scoping.
    assert await db_session.get_brief_change(second.change_uuid, other_org.id) is None
    assert await db_session.get_pending_brief_change(workflow.id, other_org.id) is None
    assert (
        await db_session.resolve_brief_change(
            second.change_uuid, other_org.id, "confirmed"
        )
        is None
    )

    with pytest.raises(ValueError):
        await db_session.resolve_brief_change(second.change_uuid, org.id, "pending")

    confirmed = await db_session.resolve_brief_change(
        second.change_uuid, org.id, BriefChangeStatus.confirmed
    )
    assert confirmed.status == "confirmed" and confirmed.resolved_at is not None
    # Resolves once.
    assert (
        await db_session.resolve_brief_change(second.change_uuid, org.id, "stale")
        is None
    )
    assert await db_session.get_pending_brief_change(workflow.id, org.id) is None
