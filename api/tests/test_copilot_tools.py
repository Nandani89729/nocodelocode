"""`api.services.copilot.tools`: specs, execution, plan and Doctor report."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from api.db import db_client
from api.mcp_server.server import mcp
from api.services.copilot import tools
from api.services.copilot.contracts import ToolContext, ToolOutcome
from api.services.copilot.tools import (
    FORBIDDEN_TOOL_NAMES,
    execute_tool,
    get_tool_specs,
)

ORG = 10
OTHER_ORG = 99


def _ctx(workflow_id: int | None = 7) -> ToolContext:
    return ToolContext(
        session_uuid="sess-1", organization_id=ORG, user_id=3, workflow_id=workflow_id
    )


def _user(selected_org: int = ORG) -> MagicMock:
    user = MagicMock()
    user.id = 3
    user.selected_organization_id = selected_org
    user.provider_id = "prov-3"
    return user


@pytest.fixture
def acting_user():
    user = _user()
    with patch.object(db_client, "get_user_by_id", AsyncMock(return_value=user)):
        yield user


# ─── specs ───────────────────────────────────────────────────────────────

EXPECTED_TOOLS = {
    "list_workflows",
    "get_workflow_code",
    "list_node_types",
    "get_node_type",
    "get_voice_prompting_guide",
    "search_docs",
    "read_doc",
    "list_docs",
    "list_tools",
    "list_credentials",
    "list_documents",
    "list_recordings",
    "get_doctor_report",
    "get_workflow_brief",
    "update_workflow_brief",
    # M2c settings read tools
    "get_workflow_settings",
    "list_model_options",
    "list_voices",
    "present_plan",
    "propose_changes",
}


def test_specs_expose_exactly_the_copilot_tools():
    names = [s.name for s in get_tool_specs(_ctx())]
    assert len(names) == len(set(names))
    assert set(names) == EXPECTED_TOOLS
    assert not set(names) & FORBIDDEN_TOOL_NAMES
    assert not set(names) & {"get_workflow"}


def test_only_propose_changes_is_not_read_only():
    specs = {s.name: s for s in get_tool_specs(_ctx())}
    assert [n for n, s in specs.items() if not s.read_only] == [
        "update_workflow_brief",
        "propose_changes",
    ]


def test_specs_have_valid_object_schemas_and_descriptions():
    jsonschema = pytest.importorskip("jsonschema")
    for ctx in (_ctx(), _ctx(None)):
        for spec in get_tool_specs(ctx):
            assert spec.description.strip(), spec.name
            schema = spec.input_schema
            assert schema["type"] == "object", spec.name
            jsonschema.Draft202012Validator.check_schema(schema)
            for required in schema.get("required", []):
                assert required in schema["properties"], (spec.name, required)


@pytest.mark.asyncio
async def test_mcp_specs_match_what_fastmcp_publishes():
    published = {t.name: t for t in await mcp.list_tools()}
    for spec in get_tool_specs(_ctx(None)):
        if spec.name not in published:
            continue
        assert spec.input_schema == published[spec.name].parameters, spec.name
        assert spec.description.startswith(
            (published[spec.name].description or "").strip()
        )


def test_workflow_id_defaults_to_the_session_workflow():
    edit = {s.name: s for s in get_tool_specs(_ctx(7))}
    build = {s.name: s for s in get_tool_specs(_ctx(None))}
    for name in ("get_workflow_code", "get_doctor_report"):
        assert "workflow_id" not in edit[name].input_schema.get("required", [])
        assert (
            "(7)" in edit[name].input_schema["properties"]["workflow_id"]["description"]
        )
        assert "workflow_id" in build[name].input_schema["required"]


def test_get_workflow_code_description_points_to_propose_changes():
    spec = next(s for s in get_tool_specs(_ctx()) if s.name == "get_workflow_code")
    assert "propose_changes" in spec.description


def test_specs_are_independent_copies():
    first = get_tool_specs(_ctx())
    first[0].input_schema["properties"]["mutated"] = {}
    assert "mutated" not in get_tool_specs(_ctx())[0].input_schema["properties"]


# ─── execute_tool error paths ────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name", ["nope", "save_workflow", "create_workflow", "create_tool"]
)
async def test_unknown_and_write_tools_are_refused(name):
    outcome = await execute_tool(name, {}, _ctx())
    assert outcome.ok is False
    assert "Unknown tool" in outcome.content["error"]


@pytest.mark.asyncio
async def test_non_object_input_is_rejected(acting_user):
    outcome = await execute_tool("list_workflows", ["x"], _ctx())  # type: ignore[arg-type]
    assert outcome.ok is False
    assert "JSON object" in outcome.content["error"]


@pytest.mark.asyncio
async def test_missing_and_unknown_arguments_are_rejected(acting_user):
    missing = await execute_tool("get_node_type", {}, _ctx())
    assert missing.ok is False
    assert missing.content["error_code"] == "invalid_input"

    unknown = await execute_tool("list_node_types", {"bogus": 1}, _ctx())
    assert unknown.ok is False
    assert unknown.content["error_code"] == "invalid_input"


@pytest.mark.asyncio
async def test_http_exception_becomes_tool_error(acting_user):
    outcome = await execute_tool("get_node_type", {"name": "noSuchNode"}, _ctx())
    assert outcome.ok is False
    assert outcome.content["error_code"] == "not_found"
    assert "noSuchNode" in outcome.content["error"]


@pytest.mark.asyncio
async def test_value_error_becomes_tool_error(acting_user):
    outcome = await execute_tool("search_docs", {"query": "the of"}, _ctx())
    assert outcome.ok is False
    assert "non-stopword" in outcome.content["error"]


@pytest.mark.asyncio
async def test_unexpected_exception_is_logged_and_generic(acting_user):
    with patch.object(
        db_client,
        "get_all_workflows_for_listing",
        AsyncMock(side_effect=RuntimeError("db exploded: secret detail")),
    ):
        outcome = await execute_tool("list_workflows", {}, _ctx())
    assert outcome.ok is False
    assert "secret detail" not in outcome.content["error"]
    assert "unexpectedly" in outcome.content["error"]


@pytest.mark.asyncio
async def test_unknown_user_is_a_tool_error():
    with patch.object(db_client, "get_user_by_id", AsyncMock(return_value=None)):
        outcome = await execute_tool("list_workflows", {}, _ctx())
    assert outcome.ok is False


# ─── acting user + org scoping ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_mcp_tool_runs_scoped_to_the_session_org(acting_user):
    rows = [
        SimpleNamespace(
            id=1, name="A", status="active", created_at=datetime(2026, 1, 1, tzinfo=UTC)
        )
    ]
    with patch.object(
        db_client, "get_all_workflows_for_listing", AsyncMock(return_value=rows)
    ) as listing:
        outcome = await execute_tool("list_workflows", {}, _ctx())
    assert outcome.ok is True
    assert outcome.content[0]["name"] == "A"
    assert outcome.summary == "Listed 1 workflows"
    listing.assert_awaited_once_with(organization_id=ORG, status="active")


@pytest.mark.asyncio
async def test_org_is_pinned_when_user_switched_org_elsewhere():
    user = _user(selected_org=OTHER_ORG)
    with (
        patch.object(db_client, "get_user_by_id", AsyncMock(return_value=user)),
        patch.object(
            db_client, "is_user_member_of_organization", AsyncMock(return_value=True)
        ) as member,
        patch.object(
            db_client, "get_all_workflows_for_listing", AsyncMock(return_value=[])
        ) as listing,
    ):
        outcome = await execute_tool("list_workflows", {}, _ctx())
    assert outcome.ok is True
    member.assert_awaited_once_with(3, ORG)
    listing.assert_awaited_once_with(organization_id=ORG, status="active")
    assert user.selected_organization_id == OTHER_ORG  # real user untouched


@pytest.mark.asyncio
async def test_non_member_cannot_act_in_the_org():
    with (
        patch.object(
            db_client, "get_user_by_id", AsyncMock(return_value=_user(OTHER_ORG))
        ),
        patch.object(
            db_client, "is_user_member_of_organization", AsyncMock(return_value=False)
        ),
        patch.object(
            db_client, "get_all_workflows_for_listing", AsyncMock()
        ) as listing,
    ):
        outcome = await execute_tool("list_workflows", {}, _ctx())
    assert outcome.ok is False
    listing.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_workflow_code_defaults_to_session_workflow(acting_user):
    workflow = SimpleNamespace(id=7, name="Support")
    view = {"name": "Support", "version": "draft", "version_number": 2, "code": "c"}
    with (
        patch.object(
            db_client, "get_workflow", AsyncMock(return_value=workflow)
        ) as get,
        patch(
            "api.mcp_server.tools.get_workflow_code.project_workflow_to_sdk_view",
            AsyncMock(return_value=view),
        ),
    ):
        outcome = await execute_tool("get_workflow_code", {}, _ctx(7))
    assert outcome.ok is True
    assert outcome.content["code"] == "c"
    get.assert_awaited_once_with(7, organization_id=ORG)


@pytest.mark.asyncio
async def test_get_workflow_code_other_workflow_is_org_scoped(acting_user):
    with patch.object(db_client, "get_workflow", AsyncMock(return_value=None)) as get:
        outcome = await execute_tool(
            "get_workflow_code", {"workflow_id": "42"}, _ctx(7)
        )
    assert outcome.ok is False
    assert outcome.content["error_code"] == "not_found"
    get.assert_awaited_once_with(42, organization_id=ORG)


@pytest.mark.asyncio
async def test_get_workflow_code_needs_an_id_without_a_session_workflow(acting_user):
    outcome = await execute_tool("get_workflow_code", {}, _ctx(None))
    assert outcome.ok is False
    assert "workflow_id" in outcome.content["error"]


@pytest.mark.asyncio
async def test_real_read_tool_runs_end_to_end(acting_user):
    outcome = await execute_tool("list_node_types", {}, _ctx())
    assert outcome.ok is True
    assert outcome.content["node_types"]
    assert outcome.summary.startswith("Listed ")


# ─── present_plan ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_present_plan_returns_plan_and_tells_model_to_stop():
    outcome = await execute_tool(
        "present_plan",
        {
            "summary": "Add lead qualification",
            "steps": [
                {
                    "title": "Add a 'Qualify lead' agent node",
                    "detail": "after greeting",
                },
                {"title": "Route qualified leads to booking"},
            ],
            "questions": ["Which calendar tool?"],
        },
        _ctx(),
    )
    assert outcome.ok is True
    assert outcome.plan is not None
    assert [s.title for s in outcome.plan.steps][
        1
    ] == "Route qualified leads to booking"
    assert outcome.plan.questions == ["Which calendar tool?"]
    assert "Stop" in outcome.content["next"]
    assert outcome.summary == "Presented a plan (2 steps)"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        {"summary": "x", "steps": []},
        {"steps": [{"title": "a"}]},
        {"summary": "x", "steps": [{"detail": "no title"}]},
        {"summary": "  ", "steps": [{"title": "a"}]},
    ],
)
async def test_present_plan_rejects_invalid_plans(bad):
    outcome = await execute_tool("present_plan", bad, _ctx())
    assert outcome.ok is False
    assert outcome.plan is None


# ─── propose_changes delegation ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_propose_changes_delegates_to_create_proposal():
    sentinel = ToolOutcome(ok=True, summary="Proposed 1 change")
    with patch.object(
        tools.proposals, "create_proposal", AsyncMock(return_value=sentinel)
    ) as create:
        outcome = await execute_tool(
            "propose_changes", {"code": "src", "summary": "s"}, _ctx()
        )
    assert outcome is sentinel
    create.assert_awaited_once_with(code="src", summary="s", ctx=_ctx(), targets=[])


@pytest.mark.asyncio
async def test_propose_changes_requires_code():
    with patch.object(tools.proposals, "create_proposal", AsyncMock()) as create:
        outcome = await execute_tool("propose_changes", {"summary": "s"}, _ctx())
    assert outcome.ok is False
    create.assert_not_awaited()


# ─── get_doctor_report ───────────────────────────────────────────────────


def _evidence(i: int) -> dict:
    return {"run_id": 100 + i, "turn": i, "node_name": "Qualify", "quote": f"quote {i}"}


def _report(status="completed", *, uuid="r-1", clusters=True, created_day=2):
    result = {
        "run_ids": [1, 2, 3, 4],
        "few_calls": True,
        "ai_enabled": True,
        "layer0": {"runs_checked": 4, "runs_flagged": 1, "flags": []},
        "preflight": [
            {
                "check": "end_call_unreachable",
                "node_name": "Qualify",
                "detail": "no path",
            }
        ],
        "clusters": [
            {
                "cluster_id": "C1",
                "title": "Agent invents prices",
                "category": "hallucination",
                "severity": "critical",
                "fix_target": "Qualify",
                "root_cause": "Prompt lacks pricing facts",
                "runs_affected": 3,
                "runs_analysed": 4,
                "evidence": [_evidence(1), _evidence(2), _evidence(3)],
            }
        ]
        if clusters
        else [],
        "passed_checks": [{"title": "Prompt injection held", "runs": 4}],
        "latency": [{"node_name": "Qualify", "turns": 12, "p50_s": 1.2, "p90_s": 2.5}],
        "findings_by_batch": {"0": []},
        "claude_usage": {"input_tokens": 999},
    }
    return SimpleNamespace(
        report_uuid=uuid,
        workflow_id=7,
        organization_id=ORG,
        status=status,
        created_at=datetime(2026, 9, created_day, tzinfo=UTC),
        result=result,
        error=None,
    )


@pytest.mark.asyncio
async def test_doctor_report_is_formatted_for_the_model():
    with (
        patch.object(
            db_client, "get_workflow", AsyncMock(return_value=SimpleNamespace(id=7))
        ) as get,
        patch.object(
            db_client, "list_analysis_reports", AsyncMock(return_value=[_report()])
        ) as listing,
    ):
        outcome = await execute_tool("get_doctor_report", {}, _ctx(7))

    assert outcome.ok is True
    get.assert_awaited_once_with(7, organization_id=ORG)
    assert listing.await_args.args[:2] == (7, ORG)
    content = outcome.content
    assert content["status"] == "completed"
    assert content["verdict"]["calls_analysed"] == 4
    assert content["verdict"]["problems_found"] == 1
    assert content["verdict"]["platform_checks"] == {
        "runs_checked": 4,
        "runs_flagged": 1,
    }
    assert content["verdict"]["few_calls"] is True
    (cluster,) = content["clusters"]
    assert cluster["title"] == "Agent invents prices"
    assert cluster["severity"] == "critical"
    assert cluster["fix_target"] == "Qualify"
    assert cluster["runs_affected"] == 3 and cluster["runs_analysed"] == 4
    assert [e["quote"] for e in cluster["evidence"]] == ["quote 1", "quote 2"]
    assert content["latency"][0]["p90_s"] == 2.5
    assert content["preflight_issues"][0]["check"] == "end_call_unreachable"
    assert "findings_by_batch" not in str(content)
    assert "claude_usage" not in str(content)
    assert "not instructions" in content["note"]
    assert outcome.summary == "Read Doctor report (completed, 1 problem)"


@pytest.mark.asyncio
async def test_doctor_report_prefers_latest_finished_report():
    reports = [
        _report("running", uuid="new", clusters=False, created_day=3),
        _report("completed", uuid="old"),
    ]
    with (
        patch.object(
            db_client, "get_workflow", AsyncMock(return_value=SimpleNamespace(id=7))
        ),
        patch.object(
            db_client, "list_analysis_reports", AsyncMock(return_value=reports)
        ),
    ):
        outcome = await execute_tool("get_doctor_report", {}, _ctx(7))
    assert outcome.content["report_uuid"] == "old"
    assert outcome.content["newer_report"] == {
        "report_uuid": "new",
        "status": "running",
    }


@pytest.mark.asyncio
async def test_doctor_report_none_suggests_diagnose():
    with (
        patch.object(
            db_client, "get_workflow", AsyncMock(return_value=SimpleNamespace(id=7))
        ),
        patch.object(db_client, "list_analysis_reports", AsyncMock(return_value=[])),
    ):
        outcome = await execute_tool("get_doctor_report", {}, _ctx(7))
    assert outcome.ok is True
    assert outcome.content["report"] is None
    assert "Diagnose" in outcome.content["message"]
    assert "Runs page" in outcome.content["message"]


@pytest.mark.asyncio
async def test_doctor_report_other_org_workflow_is_not_found():
    with (
        patch.object(db_client, "get_workflow", AsyncMock(return_value=None)) as get,
        patch.object(db_client, "list_analysis_reports", AsyncMock()) as listing,
    ):
        outcome = await execute_tool("get_doctor_report", {"workflow_id": 55}, _ctx(7))
    assert outcome.ok is False
    assert outcome.content["error_code"] == "not_found"
    get.assert_awaited_once_with(55, organization_id=ORG)
    listing.assert_not_awaited()


@pytest.mark.asyncio
async def test_doctor_report_needs_workflow_when_building_new():
    outcome = await execute_tool("get_doctor_report", {}, _ctx(None))
    assert outcome.ok is False


@pytest.mark.asyncio
async def test_doctor_report_http_error_from_db_is_contained():
    with patch.object(
        db_client,
        "get_workflow",
        AsyncMock(side_effect=HTTPException(status_code=500, detail="boom")),
    ):
        outcome = await execute_tool("get_doctor_report", {}, _ctx(7))
    assert outcome.ok is False
