from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from fastapi import HTTPException
from fastmcp.server.dependencies import get_http_headers
from opentelemetry import trace

from api.db.models import UserModel
from api.services.auth.depends import _handle_api_key_auth

# In-process override: server-side code (e.g. the Agent Copilot) runs MCP
# tool functions as a user it has already authenticated, without an HTTP
# request carrying an API key. A ContextVar keeps the override local to the
# current task: asyncio tasks copy the context at creation, so a value set
# inside one request/task never becomes visible to another, and `acting_as`
# resets it on exit (also on error).
_acting_user: ContextVar[UserModel | None] = ContextVar(
    "dograh_mcp_acting_user", default=None
)


@contextmanager
def acting_as(user: UserModel) -> Iterator[UserModel]:
    """Run MCP tool functions in-process as `user`.

    Only for server-side callers that already authenticated `user` and
    resolved its organization; `authenticate_mcp_request` returns this user
    instead of reading API-key headers while the block runs.
    """
    if user is None:
        raise ValueError("acting_as requires a user")
    token = _acting_user.set(user)
    try:
        yield user
    finally:
        _acting_user.reset(token)


async def authenticate_mcp_request() -> UserModel:
    """Resolve the authenticated Dograh user for an MCP tool invocation.

    Accepts either `X-API-Key: <key>` or `Authorization: Bearer <key>`,
    reusing the API-key flow from `api.services.auth.depends`.

    Tags the currently-active OTel span with the resolved organization
    and user identifiers. `_OrgRoutingExporter` reads `dograh.org_id`
    at export time to dispatch the span to the right Langfuse project;
    the `langfuse.user.id` / `langfuse.session.id` attributes make the
    span filterable in the Langfuse UI.

    Inside an `acting_as(user)` block the acting user is returned instead
    and no headers are read.
    """
    acting = _acting_user.get()
    if acting is not None:
        _tag_span(acting)
        return acting

    # FastMCP strips Authorization by default unless explicitly included.
    # Preserve it here so Bearer API keys work for MCP tool invocations.
    headers = get_http_headers(include={"authorization"})
    api_key = headers.get("x-api-key")
    if not api_key:
        auth = headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            api_key = auth.split(" ", 1)[1].strip()
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key — send X-API-Key or Authorization: Bearer <key>",
        )
    user = await _handle_api_key_auth(api_key)
    _tag_span(user)
    return user


def _tag_span(user: UserModel) -> None:
    span = trace.get_current_span()
    if span.is_recording():
        org_id = user.selected_organization_id
        # Intentionally NOT `dograh.org_id` — that attribute triggers the
        # per-org Langfuse routing for pipeline spans, and MCP traffic
        # should land in the default (developer-facing) project only.
        # Exposed under `mcp.org_id` for Langfuse UI filtering without
        # affecting the router.
        span.set_attribute("mcp.org_id", str(org_id))
        span.set_attribute("mcp.user_id", str(user.id))
        span.set_attribute("langfuse.user.id", str(user.id))
