"""CopilotClient against Postgres test_db: org scoping and round trips."""

from datetime import UTC, datetime

import pytest

from api.services.copilot.contracts import ProposalKind, ProposalStatus


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


async def test_session_lifecycle_is_org_scoped(db_session):
    user, org, workflow = await _org_with_workflow(db_session, "copilot_session")
    _, other_org, _ = await _org_with_workflow(db_session, "copilot_session_other")

    session = await db_session.create_copilot_session(
        organization_id=org.id, user_id=user.id, workflow_id=workflow.id
    )
    assert len(session.session_uuid) == 36
    assert session.transcript == [] and session.plan is None

    build_new = await db_session.create_copilot_session(
        organization_id=org.id, user_id=user.id, workflow_id=None
    )
    assert build_new.workflow_id is None

    assert (
        await db_session.get_copilot_session(session.session_uuid, other_org.id) is None
    )
    assert (
        await db_session.update_copilot_session(
            session.session_uuid, other_org.id, title="x"
        )
        is None
    )

    transcript = [{"role": "user", "content": "hi"}]
    updated = await db_session.update_copilot_session(
        session.session_uuid,
        org.id,
        title="Build a lead qualifier",
        transcript=transcript,
        plan={"summary": "s", "steps": []},
    )
    assert updated.title == "Build a lead qualifier"
    assert updated.transcript == transcript
    assert updated.plan == {"summary": "s", "steps": []}
    assert updated.updated_at >= session.updated_at

    listed = await db_session.list_copilot_sessions(org.id, workflow_id=workflow.id)
    assert [s.session_uuid for s in listed] == [session.session_uuid]
    all_mine = await db_session.list_copilot_sessions(org.id, user_id=user.id)
    # Most recently updated first.
    assert [s.session_uuid for s in all_mine] == [
        session.session_uuid,
        build_new.session_uuid,
    ]
    assert await db_session.list_copilot_sessions(other_org.id) == []

    with pytest.raises(ValueError):
        await db_session.update_copilot_session(
            session.session_uuid, org.id, user_id=user.id
        )


async def test_create_session_rejects_foreign_workflow(db_session):
    user, org, _ = await _org_with_workflow(db_session, "copilot_foreign_a")
    _, _, foreign = await _org_with_workflow(db_session, "copilot_foreign_b")
    with pytest.raises(ValueError):
        await db_session.create_copilot_session(
            organization_id=org.id, user_id=user.id, workflow_id=foreign.id
        )
    session = await db_session.create_copilot_session(
        organization_id=org.id, user_id=user.id, workflow_id=None
    )
    with pytest.raises(ValueError):
        await db_session.update_copilot_session(
            session.session_uuid, org.id, workflow_id=foreign.id
        )


async def test_proposal_round_trip(db_session):
    user, org, workflow = await _org_with_workflow(db_session, "copilot_proposal")
    _, other_org, foreign = await _org_with_workflow(
        db_session, "copilot_proposal_other"
    )
    session = await db_session.create_copilot_session(
        organization_id=org.id, user_id=user.id, workflow_id=workflow.id
    )

    changes = [{"change_id": "n:1", "target": "node", "op": "add"}]
    proposal = await db_session.create_copilot_proposal(
        session_id=session.id,
        organization_id=org.id,
        workflow_id=workflow.id,
        kind=ProposalKind.edit,
        summary="Add a node",
        base_hash="abc",
        candidate_json={"nodes": [], "edges": []},
        code="const wf = new Workflow({ name: 'x' });",
        changes=changes,
    )
    assert len(proposal.proposal_uuid) == 36
    assert proposal.kind == "edit" and proposal.status == "pending"
    assert proposal.changes == changes and proposal.applied_at is None
    # The joined session is usable after the DB session closed.
    assert proposal.session.session_uuid == session.session_uuid

    assert (
        await db_session.get_copilot_proposal(proposal.proposal_uuid, other_org.id)
        is None
    )
    fetched = await db_session.get_copilot_proposal(proposal.proposal_uuid, org.id)
    assert fetched.session.session_uuid == session.session_uuid

    second = await db_session.create_copilot_proposal(
        session_id=session.id,
        organization_id=org.id,
        workflow_id=None,
        kind="create",
        summary="New",
        base_hash=None,
        candidate_json={},
        code="",
        changes=[],
        workflow_name="Fresh",
    )
    listed = await db_session.list_copilot_proposals(session.id, org.id)
    assert [p.proposal_uuid for p in listed] == [
        proposal.proposal_uuid,
        second.proposal_uuid,
    ]
    assert await db_session.list_copilot_proposals(session.id, other_org.id) == []

    applied_at = datetime.now(UTC)
    updated = await db_session.update_copilot_proposal(
        proposal.proposal_uuid,
        org.id,
        status=ProposalStatus.applied,
        applied_at=applied_at,
    )
    assert updated.status == "applied" and updated.applied_at is not None
    assert updated.session.session_uuid == session.session_uuid
    assert (
        await db_session.update_copilot_proposal(
            proposal.proposal_uuid, other_org.id, status="discarded"
        )
        is None
    )
    with pytest.raises(ValueError):
        await db_session.update_copilot_proposal(
            proposal.proposal_uuid, org.id, session_id=1
        )
    with pytest.raises(ValueError):
        await db_session.update_copilot_proposal(
            proposal.proposal_uuid, org.id, workflow_id=foreign.id
        )


async def test_create_proposal_validates_session_and_workflow(db_session):
    user, org, workflow = await _org_with_workflow(db_session, "copilot_prop_val")
    other_user, other_org, foreign = await _org_with_workflow(
        db_session, "copilot_prop_val_other"
    )
    foreign_session = await db_session.create_copilot_session(
        organization_id=other_org.id, user_id=other_user.id, workflow_id=foreign.id
    )
    own_session = await db_session.create_copilot_session(
        organization_id=org.id, user_id=user.id, workflow_id=workflow.id
    )
    common = dict(
        kind="edit",
        summary="s",
        base_hash=None,
        candidate_json={},
        code="",
        changes=[],
    )
    with pytest.raises(ValueError):
        await db_session.create_copilot_proposal(
            session_id=foreign_session.id,
            organization_id=org.id,
            workflow_id=workflow.id,
            **common,
        )
    with pytest.raises(ValueError):
        await db_session.create_copilot_proposal(
            session_id=own_session.id,
            organization_id=org.id,
            workflow_id=foreign.id,
            **common,
        )


async def test_session_kind_keeps_self_improve_out_of_chat_list(db_session):
    user, org, workflow = await _org_with_workflow(db_session, "copilot_kind")
    _, other_org, _ = await _org_with_workflow(db_session, "copilot_kind_x")

    chat = await db_session.create_copilot_session(
        organization_id=org.id, user_id=user.id, workflow_id=workflow.id
    )
    assert chat.kind == "chat"
    self_improve = await db_session.get_or_create_self_improve_session(
        organization_id=org.id, workflow_id=workflow.id, user_id=user.id
    )
    assert self_improve.kind == "self_improve"
    again = await db_session.get_or_create_self_improve_session(
        organization_id=org.id, workflow_id=workflow.id, user_id=user.id
    )
    assert again.session_uuid == self_improve.session_uuid

    # The self-improve session is the most recently created, yet the chat
    # resume list never returns it (RI-1).
    listed = await db_session.list_copilot_sessions(org.id, workflow_id=workflow.id)
    assert [s.session_uuid for s in listed] == [chat.session_uuid]
    every_kind = await db_session.list_copilot_sessions(
        org.id, workflow_id=workflow.id, kind=None
    )
    assert {s.session_uuid for s in every_kind} == {
        chat.session_uuid,
        self_improve.session_uuid,
    }
    only_si = await db_session.list_copilot_sessions(
        org.id, workflow_id=workflow.id, kind="self_improve"
    )
    assert [s.session_uuid for s in only_si] == [self_improve.session_uuid]

    with pytest.raises(ValueError):
        await db_session.create_copilot_session(
            organization_id=org.id,
            user_id=user.id,
            workflow_id=None,
            kind="self_improve",
        )
    with pytest.raises(ValueError):
        await db_session.create_copilot_session(
            organization_id=org.id, user_id=user.id, workflow_id=None, kind="other"
        )
    with pytest.raises(ValueError):
        await db_session.get_or_create_self_improve_session(
            organization_id=other_org.id, workflow_id=workflow.id, user_id=user.id
        )


async def test_proposal_self_improve_fields(db_session, async_session):
    from api.db.models import SelfImproveIterationModel
    from api.services.copilot.contracts import Hypothesis

    user, org, workflow = await _org_with_workflow(db_session, "copilot_si_prop")
    _, other_org, foreign = await _org_with_workflow(db_session, "copilot_si_prop_x")
    iteration = SelfImproveIterationModel(
        workflow_id=workflow.id, organization_id=org.id, number=1, status="proposing"
    )
    foreign_iteration = SelfImproveIterationModel(
        workflow_id=foreign.id, organization_id=other_org.id, number=1, status="ready"
    )
    async_session.add_all([iteration, foreign_iteration])
    await async_session.flush()

    session = await db_session.get_or_create_self_improve_session(
        organization_id=org.id, workflow_id=workflow.id, user_id=user.id
    )
    common = dict(
        session_id=session.id,
        organization_id=org.id,
        workflow_id=workflow.id,
        kind="edit",
        summary="Move pricing",
        base_hash="h",
        candidate_json={},
        code="",
        changes=[{"change_id": "n:1"}],
    )
    with pytest.raises(ValueError):
        await db_session.create_copilot_proposal(
            **common, iteration_id=foreign_iteration.id
        )

    proposal = await db_session.create_copilot_proposal(
        **common,
        iteration_id=iteration.id,
        hypotheses={
            "n:1": Hypothesis(
                cluster_id="C1", title="Welcome answers pricing", fix_target="Welcome"
            )
        },
    )
    assert proposal.iteration_id == iteration.id
    assert proposal.accepted_change_ids == []
    assert proposal.applied_definition_id is None
    assert proposal.hypotheses["n:1"]["cluster_id"] == "C1"
    assert proposal.hypotheses["n:1"]["fix_target"] == "Welcome"

    # A round proposes once (O5).
    with pytest.raises(ValueError):
        await db_session.create_copilot_proposal(**common, iteration_id=iteration.id)

    found = await db_session.get_proposal_for_iteration(iteration.id, org.id)
    assert found.proposal_uuid == proposal.proposal_uuid
    assert (
        await db_session.get_proposal_for_iteration(iteration.id, other_org.id) is None
    )

    # A plain chat proposal keeps the defaults.
    chat_session = await db_session.create_copilot_session(
        organization_id=org.id, user_id=user.id, workflow_id=workflow.id
    )
    plain = await db_session.create_copilot_proposal(
        **{**common, "session_id": chat_session.id}
    )
    assert plain.iteration_id is None and plain.hypotheses == {}

    assert await db_session.claim_copilot_proposal(
        proposal.proposal_uuid, org.id, from_status="pending", to_status="applying"
    )
    with pytest.raises(ValueError):
        await db_session.update_copilot_proposal(
            proposal.proposal_uuid,
            org.id,
            applied_definition_id=foreign.released_definition_id,
        )
    with pytest.raises(ValueError):
        await db_session.update_copilot_proposal(
            proposal.proposal_uuid, org.id, iteration_id=foreign_iteration.id
        )
    applied = await db_session.update_copilot_proposal(
        proposal.proposal_uuid,
        org.id,
        status=ProposalStatus.applied,
        accepted_change_ids=["n:1"],
        applied_definition_id=workflow.released_definition_id,
        hypotheses={"n:1": {"title": "Welcome answers pricing"}},
    )
    assert applied.status == "applied"
    assert applied.accepted_change_ids == ["n:1"]
    assert applied.applied_definition_id == workflow.released_definition_id
    assert applied.hypotheses == {"n:1": {"title": "Welcome answers pricing"}}
