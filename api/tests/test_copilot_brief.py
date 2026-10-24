"""Workflow brief service + routes (M2b R15/D2, O8, DS-6, DS-9), RI-1 listing,
and GET /copilot/proposals/{uuid}, against Postgres test_db.

The draft-from-chats LLM call is faked; everything else is real DB.
"""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api.routes import copilot as routes
from api.services.copilot import brief as brief_service
from api.services.copilot.contracts import (
    BriefChangeStatus,
    UpdateBriefRequest,
    WorkflowBrief,
)
from api.services.workflow.doctor import llm as doctor_llm


async def _org_with_workflow(db, slug: str):
    user, _ = await db.get_or_create_user_by_provider_id(f"{slug}_user")
    org, _ = await db.get_or_create_organization_by_provider_id(f"{slug}_org", user.id)
    await db.update_user_selected_organization(user.id, org.id)
    workflow = await db.create_workflow(
        name=f"{slug} workflow",
        workflow_definition={"nodes": [], "edges": []},
        user_id=user.id,
        organization_id=org.id,
    )
    acting = SimpleNamespace(id=user.id, selected_organization_id=org.id)
    return acting, org, workflow


@pytest.fixture
def ai_on(monkeypatch):
    async def allowed(organization_id):
        return None

    monkeypatch.setattr(routes, "_require_ai", allowed)


BRIEF = WorkflowBrief(
    goal="Book demos",
    must_do=["Collect the caller's email"],
    must_never=["Quote prices"],
)


# ─── service ─────────────────────────────────────────────────────────────


async def test_get_brief_defaults_and_save_with_version_check(db_session):
    _, org, workflow = await _org_with_workflow(db_session, "bs_save")

    empty = await brief_service.get_brief(workflow.id, org.id)
    assert empty.version == 0 and empty.brief == WorkflowBrief()
    assert empty.pending_change is None and empty.updated_by is None

    saved = await brief_service.save_brief(
        workflow_id=workflow.id,
        organization_id=org.id,
        brief=WorkflowBrief(goal="  Book demos ", must_do=["a", " a ", "", "b"]),
        expected_version=0,
    )
    assert saved.version == 1 and saved.updated_by == "user"
    # Normalised: trimmed, empty and duplicate items dropped.
    assert saved.brief.goal == "Book demos" and saved.brief.must_do == ["a", "b"]

    with pytest.raises(brief_service.BriefVersionConflict):
        await brief_service.save_brief(
            workflow_id=workflow.id,
            organization_id=org.id,
            brief=BRIEF,
            expected_version=0,
        )
    assert (await brief_service.get_brief(workflow.id, org.id)).version == 1


async def test_propose_confirm_and_dismiss(db_session):
    _, org, workflow = await _org_with_workflow(db_session, "bs_confirm")

    change = await brief_service.propose_brief_change(
        workflow_id=workflow.id,
        organization_id=org.id,
        proposed=BRIEF,
        reason="User: never quote prices",
    )
    assert change.status == BriefChangeStatus.pending and change.base_version == 0
    # Proposing never mutates the brief.
    current = await brief_service.get_brief(workflow.id, org.id)
    assert current.version == 0 and current.brief == WorkflowBrief()
    assert current.pending_change.change_uuid == change.change_uuid

    confirmed = await brief_service.confirm_brief_change(
        workflow_id=workflow.id, organization_id=org.id, change_uuid=change.change_uuid
    )
    assert confirmed.version == 1 and confirmed.brief == BRIEF
    assert confirmed.updated_by == "copilot" and confirmed.pending_change is None
    with pytest.raises(brief_service.BriefChangeNotPending):
        await brief_service.confirm_brief_change(
            workflow_id=workflow.id,
            organization_id=org.id,
            change_uuid=change.change_uuid,
        )

    second = await brief_service.propose_brief_change(
        workflow_id=workflow.id,
        organization_id=org.id,
        proposed=WorkflowBrief(goal="Other"),
    )
    assert second.base_version == 1
    dismissed = await brief_service.dismiss_brief_change(
        workflow_id=workflow.id, organization_id=org.id, change_uuid=second.change_uuid
    )
    assert dismissed.version == 1 and dismissed.brief == BRIEF
    assert dismissed.pending_change is None
    stored = await db_session.get_brief_change(second.change_uuid, org.id)
    assert stored.status == "dismissed"
    # Dismissing again is a no-op.
    await brief_service.dismiss_brief_change(
        workflow_id=workflow.id, organization_id=org.id, change_uuid=second.change_uuid
    )


async def test_confirm_after_a_direct_edit_goes_stale(db_session):
    _, org, workflow = await _org_with_workflow(db_session, "bs_stale")
    change = await brief_service.propose_brief_change(
        workflow_id=workflow.id, organization_id=org.id, proposed=BRIEF
    )
    await brief_service.save_brief(
        workflow_id=workflow.id,
        organization_id=org.id,
        brief=WorkflowBrief(goal="Edited directly"),
        expected_version=0,
    )
    with pytest.raises(brief_service.BriefChangeStale) as stale:
        await brief_service.confirm_brief_change(
            workflow_id=workflow.id,
            organization_id=org.id,
            change_uuid=change.change_uuid,
        )
    assert stale.value.brief.brief.goal == "Edited directly"
    stored = await db_session.get_brief_change(change.change_uuid, org.id)
    assert stored.status == "stale"
    assert (await brief_service.get_brief(workflow.id, org.id)).version == 1


async def test_org_isolation(db_session):
    _, org, workflow = await _org_with_workflow(db_session, "bs_iso")
    _, other_org, other_workflow = await _org_with_workflow(db_session, "bs_iso_x")
    change = await brief_service.propose_brief_change(
        workflow_id=workflow.id, organization_id=org.id, proposed=BRIEF
    )

    with pytest.raises(brief_service.BriefNotFound):
        await brief_service.get_brief(workflow.id, other_org.id)
    with pytest.raises(brief_service.BriefNotFound):
        await brief_service.save_brief(
            workflow_id=workflow.id,
            organization_id=other_org.id,
            brief=BRIEF,
            expected_version=0,
        )
    with pytest.raises(brief_service.BriefNotFound):
        await brief_service.confirm_brief_change(
            workflow_id=workflow.id,
            organization_id=other_org.id,
            change_uuid=change.change_uuid,
        )
    # A change of another workflow is not found through this workflow.
    with pytest.raises(brief_service.BriefChangeNotFound):
        await brief_service.confirm_brief_change(
            workflow_id=other_workflow.id,
            organization_id=other_org.id,
            change_uuid=change.change_uuid,
        )
    assert (await brief_service.get_brief(workflow.id, org.id)).version == 0


def test_brief_prompt_text():
    assert "no brief yet" in brief_service.brief_prompt_text(None)
    assert "no brief yet" in brief_service.brief_prompt_text(WorkflowBrief())
    text = brief_service.brief_prompt_text(
        WorkflowBrief(goal="Book demos", must_never=["Quote prices"], notes="n")
    )
    assert "Goal: Book demos" in text
    assert "Must never:\n- Quote prices" in text
    assert "Accepted tradeoffs" in text and "Notes:\nn" in text


# ─── draft from chats (DS-6) ─────────────────────────────────────────────


async def _chat(db, user, org, workflow, texts, kind="chat"):
    session = await db.create_copilot_session(
        organization_id=org.id, user_id=user.id, workflow_id=workflow.id, kind=kind
    )
    transcript = []
    for i, text in enumerate(texts):
        transcript.append(
            {"role": "user" if i % 2 == 0 else "assistant", "content": text}
        )
    await db.update_copilot_session(
        str(session.session_uuid), org.id, transcript=transcript, title="Chat"
    )
    return session


async def test_draft_from_chats_creates_a_pending_change_only(db_session, monkeypatch):
    user, org, workflow = await _org_with_workflow(db_session, "bs_draft")
    await _chat(db_session, user, org, workflow, ["Never quote prices.", "Noted."])
    await _chat(
        db_session,
        user,
        org,
        workflow,
        ["SELF-IMPROVE SECRET ROUND TEXT"],
        kind="self_improve",
    )
    seen = {}

    async def fake_structured(*, system, user_content, output_model, **kwargs):
        seen["user_content"] = user_content
        assert output_model is WorkflowBrief
        return WorkflowBrief(goal="Book demos", must_never=["Quote prices"]), {}

    monkeypatch.setattr(doctor_llm, "claude_structured", fake_structured)

    change = await brief_service.draft_brief_from_chats(
        workflow_id=workflow.id, organization_id=org.id
    )
    assert change.status == BriefChangeStatus.pending
    assert change.proposed.must_never == ["Quote prices"]
    assert change.reason == "Drafted from 1 Copilot chat"
    assert "Never quote prices." in seen["user_content"]
    # Only chat sessions feed the draft (RI-1).
    assert "SELF-IMPROVE SECRET" not in seen["user_content"]
    current = await brief_service.get_brief(workflow.id, org.id)
    assert current.version == 0 and current.brief == WorkflowBrief()
    assert current.pending_change.change_uuid == change.change_uuid


async def test_draft_without_chats_or_news(db_session, monkeypatch):
    user, org, workflow = await _org_with_workflow(db_session, "bs_draft_none")

    async def fake_structured(**kwargs):
        return WorkflowBrief(), {}

    monkeypatch.setattr(doctor_llm, "claude_structured", fake_structured)
    with pytest.raises(brief_service.BriefDraftUnavailable) as none:
        await brief_service.draft_brief_from_chats(
            workflow_id=workflow.id, organization_id=org.id
        )
    assert none.value.reason == "no_chat_history"

    await _chat(db_session, user, org, workflow, ["hello"])
    with pytest.raises(brief_service.BriefDraftUnavailable) as same:
        await brief_service.draft_brief_from_chats(
            workflow_id=workflow.id, organization_id=org.id
        )
    assert same.value.reason == "no_new_information"
    assert await db_session.get_pending_brief_change(workflow.id, org.id) is None


# ─── routes ──────────────────────────────────────────────────────────────


async def test_brief_routes_conflict_confirm_dismiss(db_session, ai_on):
    user, org, workflow = await _org_with_workflow(db_session, "br_routes")

    got = await routes.get_workflow_brief(workflow.id, user)
    assert got.version == 0

    saved = await routes.update_workflow_brief(
        workflow.id, UpdateBriefRequest(brief=BRIEF, expected_version=0), user
    )
    assert saved.version == 1
    with pytest.raises(HTTPException) as conflict:
        await routes.update_workflow_brief(
            workflow.id, UpdateBriefRequest(brief=BRIEF, expected_version=0), user
        )
    assert conflict.value.status_code == 409
    assert conflict.value.detail == "brief_version_conflict"

    change = await brief_service.propose_brief_change(
        workflow_id=workflow.id,
        organization_id=org.id,
        proposed=WorkflowBrief(goal="v2"),
    )
    assert (await routes.get_workflow_brief(workflow.id, user)).pending_change
    confirmed = await routes.confirm_workflow_brief_change(
        workflow.id, change.change_uuid, user
    )
    assert confirmed.version == 2 and confirmed.brief.goal == "v2"

    stale = await brief_service.propose_brief_change(
        workflow_id=workflow.id,
        organization_id=org.id,
        proposed=WorkflowBrief(goal="v3"),
    )
    await routes.update_workflow_brief(
        workflow.id,
        UpdateBriefRequest(brief=WorkflowBrief(goal="direct"), expected_version=2),
        user,
    )
    with pytest.raises(HTTPException) as gone:
        await routes.confirm_workflow_brief_change(workflow.id, stale.change_uuid, user)
    assert gone.value.status_code == 409
    assert gone.value.detail == "brief_change_stale"

    dismissable = await brief_service.propose_brief_change(
        workflow_id=workflow.id,
        organization_id=org.id,
        proposed=WorkflowBrief(goal="v4"),
    )
    dismissed = await routes.dismiss_workflow_brief_change(
        workflow.id, dismissable.change_uuid, user
    )
    assert dismissed.brief.goal == "direct" and dismissed.pending_change is None

    with pytest.raises(HTTPException) as missing:
        await routes.dismiss_workflow_brief_change(workflow.id, "nope", user)
    assert missing.value.status_code == 404


async def test_brief_routes_are_org_scoped(db_session, ai_on):
    user, org, workflow = await _org_with_workflow(db_session, "br_iso")
    other_user, _, _ = await _org_with_workflow(db_session, "br_iso_x")
    change = await brief_service.propose_brief_change(
        workflow_id=workflow.id, organization_id=org.id, proposed=BRIEF
    )
    attempts = [
        routes.get_workflow_brief(workflow.id, other_user),
        routes.update_workflow_brief(
            workflow.id, UpdateBriefRequest(brief=BRIEF, expected_version=0), other_user
        ),
        routes.confirm_workflow_brief_change(
            workflow.id, change.change_uuid, other_user
        ),
        routes.dismiss_workflow_brief_change(
            workflow.id, change.change_uuid, other_user
        ),
        routes.draft_workflow_brief(workflow.id, other_user),
    ]
    for attempt in attempts:
        with pytest.raises(HTTPException) as info:
            await attempt
        assert info.value.status_code == 404
    current = await brief_service.get_brief(workflow.id, org.id)
    assert current.version == 0 and current.pending_change is not None


async def test_brief_routes_are_gated(monkeypatch):
    async def not_opted_in(organization_id):
        return False, "not_opted_in"

    monkeypatch.setattr(routes, "resolve_ai_access", not_opted_in)
    user = SimpleNamespace(id=1, selected_organization_id=1)
    attempts = [
        routes.get_workflow_brief(1, user),
        routes.update_workflow_brief(
            1, UpdateBriefRequest(brief=BRIEF, expected_version=0), user
        ),
        routes.confirm_workflow_brief_change(1, "c", user),
        routes.dismiss_workflow_brief_change(1, "c", user),
        routes.draft_workflow_brief(1, user),
        routes.get_copilot_proposal("p", user),
    ]
    for attempt in attempts:
        with pytest.raises(HTTPException) as info:
            await attempt
        assert info.value.status_code == 403


async def test_draft_route(db_session, ai_on, monkeypatch):
    user, org, workflow = await _org_with_workflow(db_session, "br_draft")
    with pytest.raises(HTTPException) as empty:
        await routes.draft_workflow_brief(workflow.id, user)
    assert empty.value.status_code == 422 and empty.value.detail == "no_chat_history"

    await _chat(db_session, user, org, workflow, ["Always ask for the email."])

    async def fake_structured(**kwargs):
        return WorkflowBrief(must_do=["Ask for the email"]), {}

    monkeypatch.setattr(doctor_llm, "claude_structured", fake_structured)
    response = await routes.draft_workflow_brief(workflow.id, user)
    assert response.brief_change.status == BriefChangeStatus.pending
    assert (await brief_service.get_brief(workflow.id, org.id)).version == 0

    async def broken(**kwargs):
        raise doctor_llm.DoctorLLMError("boom")

    monkeypatch.setattr(doctor_llm, "claude_structured", broken)
    with pytest.raises(HTTPException) as failed:
        await routes.draft_workflow_brief(workflow.id, user)
    assert failed.value.status_code == 502


# ─── RI-1 + GET proposal ─────────────────────────────────────────────────


async def test_chat_listing_never_returns_self_improve_sessions(db_session, ai_on):
    user, org, workflow = await _org_with_workflow(db_session, "ri1_list")
    chat = await db_session.create_copilot_session(
        organization_id=org.id, user_id=user.id, workflow_id=workflow.id
    )
    rounds = await db_session.get_or_create_self_improve_session(
        organization_id=org.id, workflow_id=workflow.id, user_id=user.id
    )
    listed = await routes.list_copilot_sessions(workflow.id, user)
    uuids = [s.session_uuid for s in listed]
    assert str(chat.session_uuid) in uuids
    assert str(rounds.session_uuid) not in uuids

    # A self-improve session cannot be driven from chat.
    from api.services.copilot.contracts import SendMessageRequest

    with pytest.raises(HTTPException) as info:
        await routes.send_copilot_message(
            str(rounds.session_uuid), SendMessageRequest(text="hi"), user
        )
    assert info.value.status_code == 404


async def test_get_proposal_endpoint_is_org_scoped(db_session, ai_on):
    user, org, workflow = await _org_with_workflow(db_session, "gp_get")
    other_user, _, _ = await _org_with_workflow(db_session, "gp_get_x")
    session = await db_session.create_copilot_session(
        organization_id=org.id, user_id=user.id, workflow_id=workflow.id
    )
    model = await db_session.create_copilot_proposal(
        session_id=session.id,
        organization_id=org.id,
        workflow_id=workflow.id,
        kind="edit",
        summary="Fix greeting",
        base_hash="h",
        candidate_json={},
        code="",
        changes=[],
        hypotheses={"n:1": {"title": "Greets twice", "cluster_id": "C1"}},
    )
    got = await routes.get_copilot_proposal(str(model.proposal_uuid), user)
    assert got.summary == "Fix greeting" and got.session_uuid == str(
        session.session_uuid
    )
    assert got.hypotheses["n:1"].cluster_id == "C1"
    for who, uuid in ((other_user, str(model.proposal_uuid)), (user, "nope")):
        with pytest.raises(HTTPException) as info:
            await routes.get_copilot_proposal(uuid, who)
        assert info.value.status_code == 404
