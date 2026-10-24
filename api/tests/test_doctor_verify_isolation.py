"""O1 replay isolation: a doctor_replay run sends nothing outside Dograh.

Drives the real text-chat lifecycle (initialize -> execute greeting -> append
user message -> execute with a custom HTTP tool call -> complete) against the
test DB with an in-process mock LLM, while every non-loopback socket/DNS
lookup and every aiohttp/httpx request is blocked and recorded. A twin test
runs the same workflow as a normal text chat to prove that path still does
the pre-call fetch, the real tool call and the post-call work.
"""

import json
import socket
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

import aiohttp
import httpx
import pytest

from api.db.models import OrganizationModel, UserModel, organization_users_association
from api.enums import OrganizationConfigurationKey, WorkflowRunMode
from api.schemas.ai_model_configuration import EffectiveAIModelConfiguration
from api.services.configuration.ai_model_configuration import (
    convert_legacy_ai_model_configuration_to_v2,
)
from api.services.pipecat import pre_call_fetch as pre_call_fetch_module
from api.services.pipecat.worker_runner import wait_for_pipeline_worker_started
from api.services.workflow import text_chat_session_service as session_service
from api.services.workflow.doctor.contracts import DOCTOR_REPLAY_RUN_MODE
from api.services.workflow.run_creation import prepare_workflow_run_inputs
from api.services.workflow.text_chat_runner import default_text_chat_checkpoint
from api.services.workflow.text_chat_session_service import (
    append_text_chat_user_message,
    complete_text_chat_session,
    default_text_chat_session_data,
    execute_pending_text_chat_turn,
    initialize_text_chat_session,
)
from api.tests.integrations._run_pipeline_helpers import USER_CONFIGURATION
from pipecat.tests import MockLLMService

_LOOPBACK = {"localhost", "127.0.0.1", "::1", "", None}
RECORDED_RESULT = {"status": "success", "data": {"order": "shipped"}}


class _NetworkGuard:
    """Blocks and records every request that would leave the machine."""

    def __init__(self) -> None:
        self.attempts: list[str] = []

    def install(self, stack: ExitStack) -> None:
        real_getaddrinfo = socket.getaddrinfo
        real_create_connection = socket.create_connection
        real_connect = socket.socket.connect

        def host_of(address):
            return address[0] if isinstance(address, tuple) else address

        def getaddrinfo(host, *args, **kwargs):
            if isinstance(host, bytes):
                host = host.decode()
            if host not in _LOOPBACK:
                self.attempts.append(f"dns:{host}")
                raise OSError(f"network blocked in replay test: {host}")
            return real_getaddrinfo(host, *args, **kwargs)

        def create_connection(address, *args, **kwargs):
            if host_of(address) not in _LOOPBACK:
                self.attempts.append(f"tcp:{address}")
                raise OSError(f"network blocked in replay test: {address}")
            return real_create_connection(address, *args, **kwargs)

        def connect(sock, address):
            if (
                sock.family in (socket.AF_INET, socket.AF_INET6)
                and host_of(address) not in _LOOPBACK
            ):
                self.attempts.append(f"connect:{address}")
                raise OSError(f"network blocked in replay test: {address}")
            return real_connect(sock, address)

        async def aiohttp_request(_session, method, url, *args, **kwargs):
            self.attempts.append(f"aiohttp:{method} {url}")
            raise aiohttp.ClientConnectionError("network blocked in replay test")

        async def httpx_send(_client, request, *args, **kwargs):
            self.attempts.append(f"httpx:{request.method} {request.url}")
            raise httpx.ConnectError("network blocked in replay test")

        def httpx_send_sync(_client, request, *args, **kwargs):
            self.attempts.append(f"httpx:{request.method} {request.url}")
            raise httpx.ConnectError("network blocked in replay test")

        stack.enter_context(patch.object(socket, "getaddrinfo", getaddrinfo))
        stack.enter_context(
            patch.object(socket, "create_connection", create_connection)
        )
        stack.enter_context(patch.object(socket.socket, "connect", connect))
        stack.enter_context(
            patch.object(aiohttp.ClientSession, "_request", aiohttp_request)
        )
        stack.enter_context(patch.object(httpx.AsyncClient, "send", httpx_send))
        stack.enter_context(patch.object(httpx.Client, "send", httpx_send_sync))


async def _setup(db_session, async_session, suffix: str):
    org = OrganizationModel(provider_id=f"doctor-replay-org-{suffix}")
    async_session.add(org)
    await async_session.flush()
    user = UserModel(
        provider_id=f"doctor-replay-user-{suffix}", selected_organization_id=org.id
    )
    async_session.add(user)
    await async_session.flush()
    await async_session.execute(
        organization_users_association.insert().values(
            user_id=user.id, organization_id=org.id
        )
    )
    await db_session.upsert_configuration(
        org.id,
        OrganizationConfigurationKey.MODEL_CONFIGURATION_V2.value,
        convert_legacy_ai_model_configuration_to_v2(
            EffectiveAIModelConfiguration.model_validate(USER_CONFIGURATION)
        ).model_dump(mode="json", exclude_none=True),
    )
    tool = await db_session.create_tool(
        organization_id=org.id,
        user_id=user.id,
        name="Lookup Order",
        description="Look up an order",
        definition={
            "schema_version": 1,
            "type": "http_api",
            "config": {
                "method": "GET",
                "url": "https://orders.customer-api.example/orders",
                "parameters": [
                    {
                        "name": "order_id",
                        "type": "string",
                        "description": "Order id",
                        "required": True,
                    }
                ],
            },
        },
    )
    workflow = await db_session.create_workflow(
        name=f"Doctor replay workflow {suffix}",
        workflow_definition={
            "nodes": [
                {
                    "id": "start",
                    "type": "startCall",
                    "position": {"x": 0, "y": 0},
                    "data": {
                        "name": "Start",
                        "prompt": "Help {{customer_name}} with orders.",
                        "is_start": True,
                        "allow_interrupt": False,
                        "add_global_prompt": False,
                        "greeting_type": "text",
                        "greeting": "Hi {{customer_name}}.",
                        "pre_call_fetch_mode": "always",
                        "pre_call_fetch_url": "https://crm.customer-api.example/lookup",
                        "tool_uuids": [tool.tool_uuid],
                    },
                },
                {
                    "id": "end",
                    "type": "endCall",
                    "position": {"x": 0, "y": 200},
                    "data": {
                        "name": "End",
                        "prompt": "Say goodbye.",
                        "is_end": True,
                        "allow_interrupt": False,
                        "add_global_prompt": False,
                    },
                },
            ],
            "edges": [
                {
                    "id": "start-end",
                    "source": "start",
                    "target": "end",
                    "data": {"label": "End Call", "condition": "When done."},
                }
            ],
        },
        user_id=user.id,
        organization_id=org.id,
    )
    return user, workflow


def _llms():
    return [
        MockLLMService(mock_steps=[], chunk_delay=0.001),
        MockLLMService(
            mock_steps=[
                MockLLMService.create_function_call_chunks(
                    "lookup_order", {"order_id": "7"}, tool_call_id="call_lookup"
                ),
                MockLLMService.create_text_chunks("Your order shipped."),
            ],
            chunk_delay=0.001,
        ),
    ]


async def _run_lifecycle(db_session, workflow, user, mode: str):
    # Reload with relationships eager-loaded (as the routes do).
    workflow = await db_session.get_workflow(
        workflow.id, organization_id=user.selected_organization_id
    )
    run_inputs = await prepare_workflow_run_inputs(
        db_session,
        workflow,
        initial_context={"customer_name": "Ana"},
        use_draft=True,
        include_template_context=True,
    )
    run = await db_session.create_workflow_run(
        name=f"run-{mode}",
        workflow_id=workflow.id,
        mode=mode,
        user_id=user.id,
        initial_context=run_inputs.initial_context,
        organization_id=user.selected_organization_id,
        definition_id=run_inputs.definition_id,
        use_draft=run_inputs.use_draft,
    )
    if mode == DOCTOR_REPLAY_RUN_MODE:
        await db_session.update_workflow_run(
            run.id,
            annotations={
                "doctor_replay": {
                    "tool_results": {"lookup_order": [json.dumps(RECORDED_RESULT)]}
                }
            },
        )
    text_session = await db_session.ensure_workflow_run_text_session(
        run.id,
        session_data=default_text_chat_session_data(),
        checkpoint=default_text_chat_checkpoint(),
    )
    text_session = await initialize_text_chat_session(
        run_id=run.id, text_session=text_session
    )
    text_session = await execute_pending_text_chat_turn(
        workflow_id=workflow.id, run_id=run.id, text_session=text_session
    )
    text_session = await append_text_chat_user_message(
        run_id=run.id,
        text_session=text_session,
        user_text="Where is order 7?",
        expected_revision=text_session.revision,
    )
    text_session = await execute_pending_text_chat_turn(
        workflow_id=workflow.id, run_id=run.id, text_session=text_session
    )
    text_session = await complete_text_chat_session(
        run_id=run.id, text_session=text_session, expected_revision=None
    )
    return run, text_session


async def _patient_worker_start(task, timeout=5.0, run_task=None):
    # The first pipeline built in a test process pays cold imports; on a slow
    # box that can exceed the runner's 5s start window. Test-only allowance.
    await wait_for_pipeline_worker_started(task, timeout=30.0, run_task=run_task)


def _common_patches(stack: ExitStack):
    stack.enter_context(
        patch(
            "api.services.workflow.text_chat_runner.wait_for_pipeline_worker_started",
            new=_patient_worker_start,
        )
    )
    stack.enter_context(
        patch(
            "api.services.workflow.text_chat_runner.create_llm_service",
            side_effect=_llms(),
        )
    )
    stack.enter_context(
        patch(
            "api.services.managed_model_services.ensure_mps_correlation_id",
            new=AsyncMock(return_value=None),
        )
    )
    stack.enter_context(
        patch(
            "api.services.workflow.text_chat_runner.db_client.has_active_recordings",
            new=AsyncMock(return_value=False),
        )
    )
    enqueue = stack.enter_context(
        patch.object(session_service, "_enqueue_text_chat_completion", AsyncMock())
    )
    upload = stack.enter_context(
        patch.object(session_service, "_upload_text_chat_transcript", AsyncMock())
    )
    return enqueue, upload


def _tool_results(text_session):
    return [
        event["payload"]["result"]
        for turn in text_session.session_data["turns"]
        for event in turn.get("events") or []
        if event["type"] == "tool_call_result"
    ]


@pytest.mark.asyncio
async def test_doctor_replay_lifecycle_makes_zero_outbound_requests(
    db_session, async_session
):
    user, workflow = await _setup(db_session, async_session, "replay")
    guard = _NetworkGuard()
    with ExitStack() as stack:
        enqueue, upload = _common_patches(stack)
        # Real implementations behind spies: only the guard would stop them.
        pre_call_fetch = stack.enter_context(
            patch(
                "api.services.workflow.text_chat_runner.execute_pre_call_fetch",
                new=AsyncMock(wraps=pre_call_fetch_module.execute_pre_call_fetch),
            )
        )
        from api.services.workflow.tools import custom_tool

        http_tool = stack.enter_context(
            patch(
                "api.services.workflow.pipecat_engine_custom_tools.execute_http_tool",
                new=AsyncMock(wraps=custom_tool.execute_http_tool),
            )
        )
        guard.install(stack)

        run, text_session = await _run_lifecycle(
            db_session, workflow, user, DOCTOR_REPLAY_RUN_MODE
        )

    assert guard.attempts == []
    pre_call_fetch.assert_not_awaited()
    http_tool.assert_not_awaited()
    enqueue.assert_not_awaited()
    upload.assert_not_awaited()

    # The replay still behaved like the agent: greeting, tool call, reply.
    turns = text_session.session_data["turns"]
    assert turns[0]["assistant_message"]["text"] == "Hi Ana."
    assert "Your order shipped." in turns[1]["assistant_message"]["text"]
    # The dry-run tool returned the source call's recorded result.
    assert _tool_results(text_session) == [RECORDED_RESULT]
    stored = await db_session.get_workflow_run_by_id(run.id)
    assert stored.mode == DOCTOR_REPLAY_RUN_MODE
    assert stored.is_completed is True


@pytest.mark.asyncio
async def test_normal_text_chat_still_fetches_calls_tools_and_runs_post_call(
    db_session, async_session
):
    user, workflow = await _setup(db_session, async_session, "normal")
    tool_result = {"status": "success", "status_code": 200, "data": {"order": "x"}}
    with ExitStack() as stack:
        enqueue, upload = _common_patches(stack)
        pre_call_fetch = stack.enter_context(
            patch(
                "api.services.workflow.text_chat_runner.execute_pre_call_fetch",
                new=AsyncMock(return_value={"customer_name": "Fetched"}),
            )
        )
        http_tool = stack.enter_context(
            patch(
                "api.services.workflow.pipecat_engine_custom_tools.execute_http_tool",
                new=AsyncMock(return_value=tool_result),
            )
        )

        _run, text_session = await _run_lifecycle(
            db_session, workflow, user, WorkflowRunMode.TEXTCHAT.value
        )

    pre_call_fetch.assert_awaited_once()
    assert (
        pre_call_fetch.await_args.kwargs["url"]
        == "https://crm.customer-api.example/lookup"
    )
    http_tool.assert_awaited_once()
    assert _tool_results(text_session) == [tool_result]
    assert text_session.session_data["turns"][0]["assistant_message"]["text"] == (
        "Hi Fetched."
    )
    upload.assert_awaited_once()
    enqueue.assert_awaited_once()
