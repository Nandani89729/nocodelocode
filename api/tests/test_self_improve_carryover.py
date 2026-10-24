"""Next-calls check for earlier accepted changes (O1, O3, DS-4)."""

import copy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from api.services.copilot.contracts import ChangeItem, ChangeOp, ChangeTarget, FieldDiff
from api.services.self_improve import carryover
from api.services.self_improve.contracts import FixOutcome
from api.services.workflow.doctor.contracts import (
    Cluster,
    Evidence,
    Finding,
    ReportResult,
    StageName,
    StageState,
)
from api.tests.self_improve_fakes import ORG, WF, FakeSelfImproveDB

PUBLISHED = datetime(2026, 10, 2, 10, 0, tzinfo=UTC)


def workflow_json(prompt):
    return {
        "nodes": [
            {
                "id": "1",
                "type": "startCall",
                "position": {"x": 0, "y": 0},
                "data": {"name": "Welcome", "prompt": prompt},
            },
            {
                "id": "2",
                "type": "endCall",
                "position": {"x": 5, "y": 5},
                "data": {"name": "End", "prompt": "Say goodbye."},
            },
        ],
        "edges": [
            {
                "id": "1-2",
                "source": "1",
                "target": "2",
                "data": {"label": "done", "condition": "caller is done"},
            }
        ],
    }


OLD, NEW = "Quote prices from memory.", "Say a specialist will send prices."

CHANGE = ChangeItem(
    change_id="n:1",
    target=ChangeTarget.node,
    op=ChangeOp.update,
    ref_id="1",
    label='Update node "Welcome" (prompt)',
    before={"id": "1", "type": "startCall", "data": {"name": "Welcome", "prompt": OLD}},
    after={"id": "1", "type": "startCall", "data": {"name": "Welcome", "prompt": NEW}},
    field_diffs=[FieldDiff(field="prompt", before=OLD, after=NEW)],
)
HYPOTHESIS = {
    "cluster_id": "C1",
    "title": "Welcome step answers pricing",
    "category": "knowledge_gap",
    "fix_target": "Welcome",
}


def definition(def_id, prompt, published_at):
    return SimpleNamespace(
        id=def_id, workflow_json=workflow_json(prompt), published_at=published_at
    )


def run_row(run_id, def_id, *, minutes_after=30, nodes=("Welcome",)):
    events = [
        {
            "type": "rtf-bot-text",
            "turn": i,
            "node_name": node,
            "payload": {"text": "hello"},
        }
        for i, node in enumerate(nodes)
    ]
    return {
        "id": run_id,
        "created_at": PUBLISHED + timedelta(minutes=minutes_after),
        "definition_id": def_id,
        "logs": {"realtime_feedback_events": events},
    }


def report(clusters=(), findings=(), clustered=True, run_ids=None):
    result = ReportResult(
        run_ids=list(run_ids or range(1, 11)),
        clusters=list(clusters),
        findings_by_batch={0: list(findings)},
    )
    status = "done" if clustered else "failed"
    result.stages[StageName.layer1] = StageState(status="done")
    result.stages[StageName.clusters] = StageState(status=status)
    return result


def pricing_cluster(*run_ids):
    return Cluster(
        cluster_id="C1",
        title="Agent answers pricing questions itself",
        category="knowledge_gap",
        severity="major",
        fix_target="Welcome",
        root_cause="Prompt has prices",
        runs_affected=len(run_ids),
        runs_analysed=10,
        evidence=[Evidence(run_id=r, quote="It costs $50") for r in run_ids[:2]],
    )


def finding(run_id):
    return Finding(
        category="knowledge_gap",
        severity="major",
        fix_target="Welcome",
        root_cause="Quoted a price",
        evidence=[Evidence(run_id=run_id, quote="$50")],
    )


def proposal(applied_definition_id=20, accepted=("n:1",)):
    return carryover.AppliedProposal(
        proposal_uuid="prop-1",
        changes=[CHANGE],
        accepted_change_ids=list(accepted),
        hypotheses={"n:1": HYPOTHESIS},
        applied_definition_id=applied_definition_id,
    )


def check(definitions, rows, result, prop=None):
    defs = {d.id: carryover.definition_facts(d) for d in definitions}
    runs = [carryover.run_facts(r) for r in rows]
    return carryover.compute_carryover(
        proposals=[prop or proposal()], runs=runs, definitions=defs, result=result
    )


BASE_DEFS = [definition(10, OLD, PUBLISHED - timedelta(days=3))]


def test_unpublished_change_is_waiting_for_publish():
    defs = BASE_DEFS + [definition(20, NEW, None)]  # applied draft, never published
    rows = [run_row(i, 10) for i in range(1, 6)]
    (line,) = check(defs, rows, report())
    assert line.outcome == FixOutcome.waiting_for_publish
    assert line.title == "Welcome step answers pricing" and line.eligible_calls == 0


def test_reverted_change_does_not_count_calls():
    # Published, then reverted: the revert copies the old content into a
    # newer, published definition (O1). Its calls are not "after the fix".
    defs = BASE_DEFS + [
        definition(20, NEW, PUBLISHED),
        definition(30, OLD, PUBLISHED + timedelta(minutes=10)),
    ]
    rows = [run_row(i, 30, minutes_after=20) for i in range(1, 5)] + [run_row(5, 20)]
    (line,) = check(defs, rows, report())
    assert line.outcome == FixOutcome.insufficient
    assert line.eligible_calls == 1


def test_fewer_than_three_eligible_calls_is_insufficient():
    defs = BASE_DEFS + [definition(20, NEW, PUBLISHED)]
    rows = [
        run_row(1, 20),
        run_row(2, 20),
        run_row(3, 20, minutes_after=-5),  # created before the publish
        run_row(4, 20, nodes=("End",)),  # never reached the Welcome node
        run_row(5, 10),  # old published definition, no fix
    ]
    (line,) = check(defs, rows, report())
    assert line.outcome == FixOutcome.insufficient
    assert line.eligible_calls == 2 and line.published_at == PUBLISHED.isoformat()


def test_no_matching_problem_in_enough_calls_is_fixed():
    defs = BASE_DEFS + [definition(20, NEW, PUBLISHED)]
    rows = [run_row(i, 20) for i in range(1, 4)]
    unrelated = pricing_cluster(7).model_copy(
        update={
            "fix_target": "End",
            "title": "Goodbye is abrupt",
            "category": "voice_style",
        }
    )
    result = report([unrelated])
    (line,) = check(defs, rows, result)
    assert line.outcome == FixOutcome.fixed
    assert (line.eligible_calls, line.affected_calls) == (3, 0)
    assert result.clusters[0].since_fix is None


def test_problem_still_in_calls_is_still_happening_and_fills_since_fix():
    defs = BASE_DEFS + [definition(20, NEW, PUBLISHED)]
    rows = [run_row(i, 20) for i in range(1, 5)] + [run_row(9, 10)]
    # Cluster evidence shows only 2 quotes; run 3 comes from the raw finding,
    # run 9 is affected but ran on the old definition.
    cluster = pricing_cluster(2, 9)
    result = report([cluster], findings=[finding(3), finding(9)])
    (line,) = check(defs, rows, result)
    assert line.outcome == FixOutcome.still_happening
    assert (line.eligible_calls, line.affected_calls) == (4, 2)
    assert result.clusters[0].since_fix["outcome"] == "still_happening"
    assert result.clusters[0].since_fix["change_id"] == "n:1"


def test_cluster_matched_by_title_when_target_differs():
    defs = BASE_DEFS + [definition(20, NEW, PUBLISHED)]
    rows = [run_row(i, 20) for i in range(1, 4)]
    cluster = pricing_cluster(1).model_copy(
        update={"fix_target": "global", "title": "Welcome step answers pricing!"}
    )
    (line,) = check(defs, rows, report([cluster]))
    assert line.outcome == FixOutcome.still_happening and line.affected_calls == 1


def test_partial_clustering_is_insufficient():
    defs = BASE_DEFS + [definition(20, NEW, PUBLISHED)]
    rows = [run_row(i, 20) for i in range(1, 6)]
    (line,) = check(defs, rows, report(clustered=False))
    assert line.outcome == FixOutcome.insufficient and line.eligible_calls == 5


def test_definitions_older_than_the_applied_draft_are_ignored():
    # v10 happens to hold the new text already (an old version); only the
    # applied draft (id 20) and later count.
    defs = [definition(10, NEW, PUBLISHED), definition(20, NEW, None)]
    rows = [run_row(i, 10) for i in range(1, 5)]
    (line,) = check(defs, rows, report())
    assert line.outcome == FixOutcome.waiting_for_publish


def test_later_unrelated_edits_keep_the_change():
    wf = workflow_json(NEW)
    wf["nodes"][0]["data"]["greeting"] = "Hi there"
    wf["nodes"][0]["position"] = {"x": 99, "y": 1}
    assert carryover.definition_contains_change(wf, CHANGE)
    assert not carryover.definition_contains_change(workflow_json(OLD), CHANGE)


def test_edge_and_removal_changes():
    wf = workflow_json(NEW)
    add_edge = ChangeItem(
        change_id="e:2-1",
        target=ChangeTarget.edge,
        op=ChangeOp.add,
        ref_id="2-1",
        label="Add edge",
        after={"id": "2-1", "source": "2", "target": "1", "data": {"label": "back"}},
    )
    assert not carryover.definition_contains_change(wf, add_edge)
    wf2 = copy.deepcopy(wf)
    wf2["edges"].append(
        {"id": "x", "source": "2", "target": "1", "data": {"label": "back"}}
    )
    assert carryover.definition_contains_change(wf2, add_edge)

    remove_node = ChangeItem(
        change_id="n:2",
        target=ChangeTarget.node,
        op=ChangeOp.remove,
        ref_id="2",
        label="Remove",
        before={"id": "2", "type": "endCall", "data": {"name": "End"}},
    )
    assert not carryover.definition_contains_change(wf, remove_node)
    wf3 = copy.deepcopy(wf)
    wf3["nodes"] = wf3["nodes"][:1]
    assert carryover.definition_contains_change(wf3, remove_node)


def test_prompt_wide_targets_expose_every_call():
    run = carryover.run_facts(run_row(1, 20, nodes=("End",)))
    wf = workflow_json(NEW)
    assert carryover.run_reached_target(run, "kb", wf)
    assert carryover.run_reached_target(run, "config:llm", wf)
    assert not carryover.run_reached_target(run, "Welcome", wf)
    assert not carryover.run_reached_target(run, "edge:Welcome->End", wf)


def test_ds4_lines():
    line = carryover.FixCarryover(
        proposal_uuid="p",
        change_id="n:1",
        title="Welcome step answers pricing",
        outcome=FixOutcome.fixed,
        eligible_calls=9,
        published_at=PUBLISHED.isoformat(),
    )
    assert carryover.carryover_line(line) == (
        "✓ Welcome step answers pricing: 0 of 9 calls since the fix (published 2 Oct)"
    )
    line.outcome, line.affected_calls, line.eligible_calls = (
        FixOutcome.still_happening,
        3,
        7,
    )
    assert "still in 3 of 7 calls" in carryover.carryover_line(line)
    line.outcome, line.eligible_calls = FixOutcome.insufficient, 2
    assert "(2 reached that step; need 3)" in carryover.carryover_line(line)
    line.outcome = FixOutcome.waiting_for_publish
    assert carryover.carryover_line(line).endswith(
        "waiting for you to publish the draft"
    )


@pytest.fixture
def fake_db(monkeypatch):
    db = FakeSelfImproveDB()
    db.configure()
    monkeypatch.setattr(carryover, "db_client", db)
    return db


async def test_build_carryover_reads_applied_proposals_of_earlier_rounds(fake_db):
    db = fake_db
    applied_round = db.add_iteration(number=1, status="ready")
    db.add_proposal(
        iteration_id=applied_round.id,
        status="applied",
        changes=[CHANGE.model_dump(mode="json")],
        accepted_change_ids=["n:1"],
        hypotheses={"n:1": HYPOTHESIS},
        applied_definition_id=20,
    )
    discarded_round = db.add_iteration(number=2, status="ready")
    db.add_proposal(
        iteration_id=discarded_round.id,
        status="discarded",
        changes=[CHANGE.model_dump(mode="json")],
        accepted_change_ids=["n:1"],
    )
    db.versions = BASE_DEFS + [definition(20, NEW, PUBLISHED)]
    for run_id in range(1, 5):
        db.runs[run_id] = run_row(run_id, 20)
    result = report(run_ids=[1, 2, 3, 4])

    lines = await carryover.build_carryover(
        workflow_id=WF,
        organization_id=ORG,
        number=3,
        run_ids=[1, 2, 3, 4],
        result=result,
    )
    assert [(c.proposal_uuid, c.outcome) for c in lines] == [
        ("prop-1", FixOutcome.fixed)
    ]


async def test_build_carryover_without_applied_proposals_loads_nothing(
    fake_db, monkeypatch
):
    async def boom(*args, **kwargs):
        raise AssertionError("runs must not be loaded")

    monkeypatch.setattr(fake_db, "get_runs_with_logs_for_workflow", boom)
    assert (
        await carryover.build_carryover(
            workflow_id=WF, organization_id=ORG, number=1, run_ids=[1], result=report()
        )
        == []
    )


def test_cluster_run_ids_are_exact_when_present():
    """Clusters now carry every affected run; evidence keeps only 2-3."""
    from api.services.workflow.doctor.contracts import Cluster, Evidence, ReportResult

    cluster = Cluster(
        cluster_id="C1",
        title="Answers pricing in Welcome",
        category="workflow_error",
        severity="critical",
        fix_target="Welcome and Source",
        root_cause="r",
        runs_affected=5,
        runs_analysed=10,
        evidence=[Evidence(run_id=1, quote="q"), Evidence(run_id=2, quote="q")],
        run_ids=[1, 2, 3, 4, 5],
    )
    assert carryover.cluster_run_ids(cluster, ReportResult()) == {1, 2, 3, 4, 5}
