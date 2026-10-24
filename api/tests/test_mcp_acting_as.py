"""`acting_as`: in-process MCP tool calls as an already-authenticated user.

The header/API-key path is covered by `test_mcp_auth.py` (unchanged); these
tests cover the override and that it cannot leak across tasks.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from api.mcp_server.auth import acting_as, authenticate_mcp_request


def _user(user_id: int, org_id: int) -> MagicMock:
    user = MagicMock()
    user.id = user_id
    user.selected_organization_id = org_id
    return user


@pytest.mark.asyncio
async def test_acting_user_is_returned_without_reading_headers():
    user = _user(1, 10)
    with (
        patch("api.mcp_server.auth.get_http_headers") as get_headers,
        patch("api.mcp_server.auth._handle_api_key_auth", AsyncMock()) as handle,
    ):
        with acting_as(user):
            assert await authenticate_mcp_request() is user
    get_headers.assert_not_called()
    handle.assert_not_awaited()


@pytest.mark.asyncio
async def test_header_flow_resumes_after_block_exits():
    with acting_as(_user(1, 10)):
        pass
    with patch("api.mcp_server.auth.get_http_headers", return_value={}):
        with pytest.raises(HTTPException) as exc_info:
            await authenticate_mcp_request()
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_override_is_reset_when_the_block_raises():
    with pytest.raises(RuntimeError):
        with acting_as(_user(1, 10)):
            raise RuntimeError("boom")
    with patch("api.mcp_server.auth.get_http_headers", return_value={}):
        with pytest.raises(HTTPException):
            await authenticate_mcp_request()


@pytest.mark.asyncio
async def test_nested_blocks_restore_the_outer_user():
    outer, inner = _user(1, 10), _user(2, 20)
    with acting_as(outer):
        with acting_as(inner):
            assert await authenticate_mcp_request() is inner
        assert await authenticate_mcp_request() is outer


@pytest.mark.asyncio
async def test_override_does_not_leak_into_concurrent_tasks():
    """A task started elsewhere (another request) never sees the override,
    even while the acting block is open."""
    entered = asyncio.Event()
    release = asyncio.Event()
    other_result: dict = {}

    async def copilot_request():
        with acting_as(_user(1, 10)):
            entered.set()
            await release.wait()
            return await authenticate_mcp_request()

    async def other_request():
        await entered.wait()
        try:
            other_result["user"] = await authenticate_mcp_request()
        except HTTPException as e:
            other_result["status"] = e.status_code
        finally:
            release.set()

    with patch("api.mcp_server.auth.get_http_headers", return_value={}):
        copilot_task = asyncio.create_task(copilot_request())
        other_task = asyncio.create_task(other_request())
        acted, _ = await asyncio.gather(copilot_task, other_task)

    assert acted.id == 1
    assert other_result == {"status": 401}


@pytest.mark.asyncio
async def test_acting_as_rejects_none():
    with pytest.raises(ValueError):
        with acting_as(None):  # type: ignore[arg-type]
            pass


@pytest.mark.asyncio
async def test_mcp_tool_function_runs_as_acting_user():
    """A real MCP tool body scopes its query to the acting user's org."""
    from api.db import db_client
    from api.mcp_server.tools.workflows import list_workflows

    with patch.object(
        db_client, "get_all_workflows_for_listing", AsyncMock(return_value=[])
    ) as listing:
        with acting_as(_user(3, 77)):
            assert await list_workflows() == []
    listing.assert_awaited_once_with(organization_id=77, status="active")
