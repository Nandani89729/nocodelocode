"""Doctor orchestrator: AI gating, checkpoint/resume, partial status."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from api.services.workflow.doctor import analysis, layer1, rules
from api.services.workflow.doctor.contracts import (
    Evidence,
    Finding,
    ReportResult,
    StageName,
)
from api.services.workflow.doctor.layer1 import (
    ClusterDraft,
    ClusteringOutput,
    FindingsOutput,
)

WORKFLOW_JSON = {
    "nodes": [
        {
            "id": "1",
            "type": "startCall",
            "position": {"x": 0, "y": 0},
            "data": {
                "name": "Welcome",
                "prompt": "You are Sarah from Acme. Ask how you can help.",
                "greeting_type": "text",
                "greeting": "Hi {{first_name}}, this is Sarah from Acme.",
            },
        },
        {
            "id": "2",
            "type": "endCall",
            "position": {"x": 0, "y": 0},
            "data": {"name": "End", "prompt": "Say goodbye."},
        },
    ],
    "edges": [
        {
            "id": "e1",
            "source": "1",
            "target": "2",
            "data": {"label": "done", "condition": "caller is done"},
        }
    ],
}


def make_run(run_id, generic=False):
    greeting = (
        "Hello! How can I help you today?"
        if generic
        else "Hi Ann, this is Sarah from Acme."
    )
    events = [
        {
            "type": "rtf-bot-text",
            "turn": 0,
            "node_name": "Welcome",
            "payload": {"text": greeting},
        },
        {
            "type": "rtf-user-transcription",
            "turn": 1,
            "node_name": "Welcome",
            "payload": {"text": "call me on 555 123 4567", "final": True},
        },
        {
            "type": "rtf-bot-text",
            "turn": 1,
            "node_name": "Welcome",
            "payload": {"text": "Sure."},
        },
        {
            "type": "rtf-latency-breakdown",
            "turn": 1,
            "node_name": "Welcome",
            "payload": {"total_secs": 1.0 + run_id / 100, "ttfb": []},
        },
    ]
    return {
        "id": run_id,
        "created_at": datetime(2026, 9, 30, tzinfo=UTC),
        "definition_id": 5,
        "mode": "twilio",
        "logs": {"realtime_feedback_events": events},
        "usage_info": {},
        "gathered_context": {},
        "initial_context": {"first_name": "Ann"},
        "annotations": {},
    }


class FakeDB:
    def __init__(self, n_runs=12, opted_in=True):
        self.runs = [make_run(100 + i, generic=(i == 0)) for i in range(n_runs)]
        self.opted_in = opted_in
        self.report = SimpleNamespace(
            report_uuid="r-1",
            organization_id=1,
            workflow_id=9,
            status="queued",
            params={"limit": 50, "filters": None},
            result={},
            error=None,
            created_at=datetime.now(UTC),
        )
        self.checkpoints: list[dict] = []
        self.statuses: list[str] = []

    async def get_analysis_report(self, report_uuid, organization_id):
        assert organization_id == 1
        return self.report if report_uuid == "r-1" else None

    async def update_analysis_report(
        self, report_uuid, organization_id, *, status=None, result=None, error=None
    ):
        if status is not None:
            self.report.status = getattr(status, "value", status)
            self.statuses.append(self.report.status)
        if result is not None:
            self.report.result = result
            self.checkpoints.append(result)
        if error is not None:
            self.report.error = error
        return self.report

    async def get_workflow(self, workflow_id, organization_id=None):
        assert organization_id == 1
        return SimpleNamespace(
            id=workflow_id,
            released_definition=SimpleNamespace(id=5, workflow_json=WORKFLOW_JSON),
            current_definition=None,
            workflow_definition=WORKFLOW_JSON,
        )

    async def get_workflow_versions(self, workflow_id):
        return [SimpleNamespace(id=5, workflow_json=WORKFLOW_JSON)]

    async def get_runs_with_logs_for_workflow(
        self, workflow_id, organization_id, *, limit, filters=None
    ):
        return [dict(r) for r in self.runs[:limit]]

    async def get_configuration_value(self, organization_id, key, default=None):
        assert key == analysis.DOCTOR_AI_OPT_IN_KEY
        return self.opted_in

    async def get_tools_for_organization(self, organization_id):
        return []


class FakeClaude:
    """Dispatches on output model; can fail on chosen Layer 1 batch calls."""

    def __init__(self, fail_batch_calls=()):
        self.fail_batch_calls = set(fail_batch_calls)
        self.batch_calls = 0
        self.batch_run_ids: list[list[int]] = []
        self.cluster_calls = 0
        self.rule_calls = 0

    async def __call__(self, *, system, user_content, output_model, **kwargs):
        if output_model is FindingsOutput:
            self.batch_calls += 1
            if self.batch_calls in self.fail_batch_calls:
                raise RuntimeError("529 overloaded")
            ids = [
                int(part.split('"')[1])
                for part in user_content.split("<call_transcript run_id=")[1:]
            ]
            self.batch_run_ids.append(ids)
            assert "555 123 4567" not in user_content  # redacted
            return (
                FindingsOutput(
                    findings=[
                        Finding(
                            category="workflow_error",
                            severity="major",
                            fix_target="Welcome",
                            root_cause="Never moves on",
                            evidence=[Evidence(run_id=i, turn=1, quote="Sure.")],
                        )
                        for i in ids[:2]
                    ]
                ),
                {"input_tokens": 1},
            )
        if output_model is ClusteringOutput:
            self.cluster_calls += 1
            n = user_content.count('"finding_id"')
            return (
                ClusteringOutput(
                    clusters=[
                        ClusterDraft(
                            title="Never moves on",
                            category="workflow_error",
                            severity="major",
                            fix_target="Welcome",
                            root_cause="Missing transition cue",
                            finding_ids=list(range(n)),
                        )
                    ]
                ),
                {},
            )
        self.rule_calls += 1
        return rules.RulesOutput(
            rules=[rules.RuleDraft(node_name="Welcome", text="Ask how you can help.")]
        ), {}


@pytest.fixture
def fake_db(monkeypatch):
    db = FakeDB()
    monkeypatch.setattr(analysis, "db_client", db)
    monkeypatch.setattr("api.db.db_client", db)
    return db


@pytest.fixture
def fake_claude(monkeypatch):
    claude = FakeClaude()
    monkeypatch.setattr(layer1, "claude_structured", claude)
    monkeypatch.setattr(rules, "claude_structured", claude)
    return claude


def stage(result, name):
    return result.stages[StageName(name)].status


async def test_no_key_runs_layer0_and_layer2_only(fake_db, fake_claude, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    status = await analysis.run_analysis("r-1", 1)
    assert status == "completed"
    result = ReportResult.model_validate(fake_db.report.result)
    assert result.ai_enabled is False and result.ai_disabled_reason == "no_key"
    assert result.layer0 is not None and result.layer0.runs_flagged == 1
    assert result.latency and result.latency[0].node_name == "Welcome"
    assert result.few_calls is False and len(result.run_ids) == 12
    for s in ("layer1", "clusters", "rules"):
        assert stage(result, s) == "skipped"
    for s in ("preflight", "layer0", "layer2"):
        assert stage(result, s) == "done"
    assert fake_claude.batch_calls == 0 and fake_claude.rule_calls == 0


async def test_not_opted_in(fake_db, fake_claude, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    fake_db.opted_in = False
    await analysis.run_analysis("r-1", 1)
    result = ReportResult.model_validate(fake_db.report.result)
    assert result.ai_disabled_reason == "not_opted_in"
    assert fake_claude.batch_calls == 0


async def test_batch_failure_gives_partial_then_resume_skips_done_work(
    fake_db, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    claude = FakeClaude(fail_batch_calls={2})
    monkeypatch.setattr(layer1, "claude_structured", claude)
    monkeypatch.setattr(rules, "claude_structured", claude)

    status = await analysis.run_analysis("r-1", 1)
    assert status == "partial"
    assert fake_db.report.status == "partial"
    assert "problems" in fake_db.report.error and "batch 2/2" in fake_db.report.error
    result = ReportResult.model_validate(fake_db.report.result)
    l1 = result.stages[StageName.layer1]
    assert l1.status == "failed" and l1.completed_batches == [0]
    assert l1.total_batches == 2
    assert stage(result, "clusters") == "pending"
    assert stage(result, "layer2") == "done" and stage(result, "rules") == "done"
    assert result.layer0 is not None and result.latency
    # A checkpoint was written right after batch 0.
    assert any(
        cp["stages"]["layer1"]["completed_batches"] == [0] for cp in fake_db.checkpoints
    )

    # Resume: count re-executed stages.
    calls = {"preflight": 0, "layer0": 0}

    async def counting_preflight(*a, **k):
        calls["preflight"] += 1
        return []

    def counting_layer0(*a, **k):
        calls["layer0"] += 1
        raise AssertionError("layer0 must not rerun")

    monkeypatch.setattr(analysis, "run_preflight", counting_preflight)
    monkeypatch.setattr(analysis, "run_layer0", counting_layer0)
    claude.fail_batch_calls = set()
    first_batch_ids = claude.batch_run_ids[0]
    claude.batch_run_ids.clear()
    fake_db.report.status = "queued"
    # New runs arriving meanwhile must not change the analysed set.
    fake_db.runs.insert(0, make_run(999))

    status = await analysis.run_analysis("r-1", 1)
    assert status == "completed"
    result = ReportResult.model_validate(fake_db.report.result)
    assert calls == {"preflight": 0, "layer0": 0}
    assert len(claude.batch_run_ids) == 1  # only batch index 1 re-sent
    assert set(claude.batch_run_ids[0]).isdisjoint(first_batch_ids)
    assert 999 not in result.run_ids
    assert result.stages[StageName.layer1].completed_batches == [0, 1]
    assert claude.cluster_calls == 1 and claude.rule_calls == 1  # rules not redone
    (cluster,) = result.clusters
    assert cluster.runs_affected == 4 and cluster.runs_analysed == 12
    assert not cluster.watch_only
    assert fake_db.report.error == ""


async def test_completed_report_is_not_rerun(fake_db, fake_claude):
    fake_db.report.status = "completed"
    assert await analysis.run_analysis("r-1", 1) == "completed"
    assert fake_db.checkpoints == []


async def test_missing_workflow_fails_report(fake_db, fake_claude, monkeypatch):
    async def none(*a, **k):
        return None

    monkeypatch.setattr(fake_db, "get_workflow", none)
    assert await analysis.run_analysis("r-1", 1) == "failed"
    assert fake_db.report.status == "failed"
