"""Copilot routes: gating, org scoping, history shaping, SSE framing.

Handlers are called directly where possible; the SSE test goes through a
minimal FastAPI app with the auth dependency overridden. DB, agent loop and
proposals module are faked.
"""

import sys
import types
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException

import api.services.copilot as copilot_pkg
from api.routes import copilot as routes
from api.services.auth.depends import get_user
from api.services.copilot.contracts import (
    ApplyProposalRequest,
    ApplyProposalResponse,
    CopilotEvent,
    CopilotEventType,
    CreateSessionRequest,
    Proposal,
    ProposalKind,
    SendMessageRequest,
)
from api.services.workflow.doctor import analysis

USER = SimpleNamespace(id=3, selected_organization_id=1)
NOW = datetime(2026, 10, 1, tzinfo=UTC)


def _session(**kw):
    base = dict(
        id=11,
        session_uuid="s-1",
        organization_id=1,
        user_id=3,
        workflow_id=9,
        title="Fix greeting",
        transcript=[],
        plan=None,
        created_at=NOW,
        updated_at=NOW,
    )
    base.update(kw)
    return SimpleNamespace(**base)


class FakeDB:
    def __init__(self):
        self.opted_in = True
        self.session = _session()
        self.created = []
        self.list_calls = []

    async def get_configuration_value(self, organization_id, key, default=None):
        return self.opted_in

    async def get_workflow(self, workflow_id, organization_id=None):
        if workflow_id == 9 and organization_id == 1:
            return SimpleNamespace(id=9, name="Lead")
        return None

    async def create_copilot_session(self, *, organization_id, user_id, workflow_id):
        self.created.append((organization_id, user_id, workflow_id))
        return _session(workflow_id=workflow_id, title=None)

    async def get_copilot_session(self, session_uuid, organization_id):
        if session_uuid == "s-1" and organization_id == 1:
            return self.session
        return None

    async def list_copilot_sessions(self, organization_id, **kwargs):
        self.list_calls.append((organization_id, kwargs))
        return [self.session]

    async def list_copilot_proposals(self, session_id, organization_id):
        assert (session_id, organization_id) == (11, 1)
        return [SimpleNamespace(proposal_uuid="p-1")]


def _proposal(uuid="p-1"):
    return Proposal(
        proposal_uuid=uuid,
        session_uuid="s-1",
        kind=ProposalKind.edit,
        workflow_id=9,
        summary="Edit greeting",
    )


@pytest.fixture
def env(monkeypatch):
    db = FakeDB()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.delenv("DOCTOR_LLM_PROVIDER", raising=False)
    monkeypatch.setattr(routes, "db_client", db)
    monkeypatch.setattr(analysis, "db_client", db)

    calls = {}
    module = types.ModuleType("api.services.copilot.proposals")

    def proposal_from_model(model, *, session_uuid=None):
        assert session_uuid == "s-1"
        return _proposal(model.proposal_uuid)

    module.proposal_from_model = proposal_from_model

    async def apply_proposal(*, proposal_uuid, organization_id, user, request):
        calls["apply"] = (proposal_uuid, organization_id, user, request)
        return ApplyProposalResponse(saved=True, workflow_id=9, draft_hash="h")

    async def discard_proposal(*, proposal_uuid, organization_id):
        calls["discard"] = (proposal_uuid, organization_id)
        return _proposal(proposal_uuid) if proposal_uuid == "p-1" else None

    module.apply_proposal = apply_proposal
    module.discard_proposal = discard_proposal
    monkeypatch.setitem(sys.modules, "api.services.copilot.proposals", module)
    monkeypatch.setattr(copilot_pkg, "proposals", module, raising=False)
    routes.running_turns._started.clear()
    return db, calls


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------


async def test_no_key_is_503(env, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(HTTPException) as info:
        await routes.create_copilot_session(CreateSessionRequest(workflow_id=9), USER)
    assert info.value.status_code == 503


async def test_not_opted_in_is_403_on_every_route(env):
    db, _ = env
    db.opted_in = False
    attempts = [
        routes.create_copilot_session(CreateSessionRequest(), USER),
        routes.get_copilot_session("s-1", USER),
        routes.list_copilot_sessions(9, USER),
        routes.send_copilot_message("s-1", SendMessageRequest(text="hi"), USER),
        routes.apply_copilot_proposal("p-1", ApplyProposalRequest(), USER),
        routes.discard_copilot_proposal("p-1", USER),
    ]
    for attempt in attempts:
        with pytest.raises(HTTPException) as info:
            await attempt
        assert info.value.status_code == 403
        assert info.value.detail == routes.NOT_OPTED_IN_DETAIL


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


async def test_create_session_scopes_workflow(env):
    db, _ = env
    with pytest.raises(HTTPException) as info:
        await routes.create_copilot_session(CreateSessionRequest(workflow_id=77), USER)
    assert info.value.status_code == 404
    assert db.created == []

    resp = await routes.create_copilot_session(
        CreateSessionRequest(workflow_id=9), USER
    )
    assert db.created == [(1, 3, 9)]
    assert resp.workflow_id == 9 and resp.created_at == NOW.isoformat()

    await routes.create_copilot_session(CreateSessionRequest(), USER)
    assert db.created[-1] == (1, 3, None)


async def test_get_session_shapes_history(env):
    db, _ = env
    db.session.transcript = [
        {"role": "user", "content": "Fix it", "created_at": "t0"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "name": "get_workflow_code", "input": {}}],
        },
        {
            "role": "tool",
            "content": "x" * 1000,
            "tool_call_id": "c1",
            "tool_name": "get_workflow_code",
        },
        {
            "role": "tool",
            "content": "bad",
            "tool_call_id": "c2",
            "tool_name": "propose_changes",
            "is_error": True,
        },
        {"role": "assistant", "content": "Here is the plan."},
    ]
    db.session.plan = {"summary": "Fix", "steps": [{"title": "Edit greeting"}]}

    resp = await routes.get_copilot_session("s-1", USER)

    assert [(m.role, m.tool_name) for m in resp.messages] == [
        ("user", None),
        ("tool", "get_workflow_code"),
        ("tool", "propose_changes"),
        ("assistant", None),
    ]
    assert len(resp.messages[1].content) == routes.TOOL_PREVIEW_CHARS
    assert resp.messages[2].is_error is True
    assert [p.proposal_uuid for p in resp.proposals] == ["p-1"]
    assert resp.plan.steps[0].title == "Edit greeting"

    with pytest.raises(HTTPException) as info:
        await routes.get_copilot_session("other", USER)
    assert info.value.status_code == 404


async def test_list_sessions_is_light_and_user_scoped(env):
    db, _ = env
    db.session.transcript = [{"role": "user", "content": "hi"}]
    resp = await routes.list_copilot_sessions(9, USER)
    assert len(resp) == 1 and resp[0].messages == [] and resp[0].proposals == []
    assert db.list_calls == [(1, {"workflow_id": 9, "user_id": 3, "limit": 20})]

    with pytest.raises(HTTPException) as info:
        await routes.list_copilot_sessions(77, USER)
    assert info.value.status_code == 404


# ---------------------------------------------------------------------------
# Messages (SSE)
# ---------------------------------------------------------------------------


def test_sse_frame():
    event = CopilotEvent(type=CopilotEventType.text_delta, data={"text": "hé\nllo"})
    assert (
        routes.sse_frame(event) == 'event: text_delta\ndata: {"text": "hé\\nllo"}\n\n'
    )


async def test_send_message_streams_sse_and_releases_lock(env, monkeypatch):
    seen = {}

    async def fake_run_turn(*, session_uuid, organization_id, user, text):
        seen.update(session=session_uuid, org=organization_id, text=text)
        # Concurrent send while this turn runs -> 409.
        with pytest.raises(HTTPException) as info:
            await routes.send_copilot_message(
                session_uuid, SendMessageRequest(text="again"), USER
            )
        seen["conflict"] = info.value.status_code
        yield CopilotEvent(type=CopilotEventType.text_delta, data={"text": "Hi"})
        yield CopilotEvent(type=CopilotEventType.message_done)

    monkeypatch.setattr(routes.agent, "run_turn", fake_run_turn)

    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_user] = lambda: USER
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post(
            "/copilot/sessions/s-1/messages", json={"text": "Hello"}
        )
        missing = await client.post(
            "/copilot/sessions/nope/messages", json={"text": "Hello"}
        )
        empty = await client.post("/copilot/sessions/s-1/messages", json={"text": ""})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["x-accel-buffering"] == "no"
    assert resp.text == (
        'event: text_delta\ndata: {"text": "Hi"}\n\nevent: message_done\ndata: {}\n\n'
    )
    assert seen == {"session": "s-1", "org": 1, "text": "Hello", "conflict": 409}
    assert routes.running_turns._started == {}
    assert missing.status_code == 404
    assert empty.status_code == 422


def test_running_turns_expire():
    turns = routes._RunningTurns()
    assert turns.acquire("a") and not turns.acquire("a")
    turns._started["a"] -= turns.TTL_SECONDS + 1
    assert turns.acquire("a")
    turns.release("a")
    assert turns.acquire("a")


# ---------------------------------------------------------------------------
# Proposals
# ---------------------------------------------------------------------------


async def test_apply_and_discard_delegate(env):
    _, calls = env
    request = ApplyProposalRequest(
        accepted_change_ids=["n:1"], expected_draft_version=3
    )
    resp = await routes.apply_copilot_proposal("p-1", request, USER)
    assert resp.saved is True
    assert calls["apply"] == ("p-1", 1, USER, request)

    discarded = await routes.discard_copilot_proposal("p-1", USER)
    assert discarded.proposal_uuid == "p-1" and calls["discard"] == ("p-1", 1)
    with pytest.raises(HTTPException) as info:
        await routes.discard_copilot_proposal("p-x", USER)
    assert info.value.status_code == 404
