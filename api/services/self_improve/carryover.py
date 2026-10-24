"""Next-calls check: how earlier accepted changes are doing (O1, O3, DS-4).

For every change the user accepted in an applied proposal of an earlier
round, this round's calls are sorted like this::

    window run ── pinned definition published, run created after publish,
       │          and the definition still contains the change?  (O1)
       │ no ─► not "after the fix" (draft test calls, reverted fixes, ...)
       ▼ yes
    reached the change's target node?  ─ no ─► not exposed to the fix
       ▼ yes
    eligible ── in the new report's matching cluster? ─ yes ─► affected

    no published definition contains the change ─► waiting_for_publish
    eligible < MIN_CALLS_FOR_EVIDENCE, or clustering unfinished ─► insufficient
    affected == 0 ─► fixed                       else ─► still_happening

The matching cluster (same fix_target + category, else a similar title)
gets the result in ``Cluster.since_fix``.

Everything except ``build_carryover`` is pure, so it is tested without a DB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Any, Optional

from loguru import logger

from api.db import db_client
from api.services.configuration.masking import mask_workflow_definition
from api.services.copilot.contracts import ChangeItem, ChangeOp, ChangeTarget
from api.services.self_improve.contracts import (
    MIN_CALLS_FOR_EVIDENCE,
    FixCarryover,
    FixOutcome,
)
from api.services.workflow.doctor.contracts import Cluster, ReportResult, StageName
from api.services.workflow.doctor.timeline import as_obj, build_timeline, run_events

#: Two cluster titles this similar (0..1) describe the same problem.
TITLE_SIMILARITY_MIN = 0.6

#: How many earlier rounds are searched for applied proposals.
EARLIER_ROUNDS_LIMIT = 50


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass
class RunFacts:
    run_id: int
    created_at: Optional[datetime]
    definition_id: Optional[int]
    #: normalized names of the nodes the call was in at any point
    nodes: set[str] = field(default_factory=set)


@dataclass
class DefinitionFacts:
    id: int
    published_at: Optional[datetime]
    #: workflow JSON with secrets masked (proposal changes are masked too)
    workflow_json: dict = field(default_factory=dict)


@dataclass
class AppliedProposal:
    proposal_uuid: str
    changes: list[ChangeItem]
    accepted_change_ids: list[str]
    hypotheses: dict[str, dict]
    applied_definition_id: Optional[int] = None


def _norm(value: Any) -> str:
    return " ".join(str(value or "").split()).lower()


def _aware(value: Any) -> Optional[datetime]:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def run_facts(row: dict) -> RunFacts:
    """Compact one row of ``get_runs_with_logs_for_workflow``."""
    nodes: set[str] = set()
    for entry in build_timeline(run_events(row)):
        for key in ("node", "to", "from"):
            if key == "node" or entry.get("kind") == "transition":
                name = _norm(entry.get(key))
                if name:
                    nodes.add(name)
    return RunFacts(
        run_id=row.get("id"),
        created_at=_aware(row.get("created_at")),
        definition_id=row.get("definition_id"),
        nodes=nodes,
    )


def definition_facts(definition: Any) -> DefinitionFacts:
    wf = as_obj(getattr(definition, "workflow_json", None)) or {}
    return DefinitionFacts(
        id=getattr(definition, "id"),
        published_at=_aware(getattr(definition, "published_at", None)),
        workflow_json=(mask_workflow_definition(wf) or {})
        if isinstance(wf, dict)
        else {},
    )


def applied_proposal(proposal: Any) -> Optional[AppliedProposal]:
    changes = []
    for raw in as_obj(getattr(proposal, "changes", None)) or []:
        try:
            changes.append(ChangeItem.model_validate(raw))
        except Exception:
            continue
    accepted = [
        str(c) for c in as_obj(getattr(proposal, "accepted_change_ids", None)) or []
    ]
    if not changes or not accepted:
        return None
    hypotheses = as_obj(getattr(proposal, "hypotheses", None)) or {}
    return AppliedProposal(
        proposal_uuid=str(getattr(proposal, "proposal_uuid")),
        changes=changes,
        accepted_change_ids=accepted,
        hypotheses=hypotheses if isinstance(hypotheses, dict) else {},
        applied_definition_id=getattr(proposal, "applied_definition_id", None),
    )


# ---------------------------------------------------------------------------
# Does a definition still contain an accepted change? (O1)
# ---------------------------------------------------------------------------


def _items(wf: dict, key: str) -> list[dict]:
    return [i for i in (wf or {}).get(key) or [] if isinstance(i, dict)]


def _data(item: dict) -> dict:
    raw = item.get("data")
    return (
        {k: v for k, v in raw.items() if v is not None} if isinstance(raw, dict) else {}
    )


def _find_node(wf: dict, change: ChangeItem) -> Optional[dict]:
    nodes = _items(wf, "nodes")
    for node in nodes:
        if str(node.get("id")) == str(change.ref_id):
            return node
    # Fall back to the node name (ids can be renumbered by later edits).
    name = _norm(((change.after or {}).get("data") or {}).get("name"))
    if name:
        matches = [n for n in nodes if _norm(_data(n).get("name")) == name]
        if len(matches) == 1:
            return matches[0]
    return None


def _edge_key(edge: dict) -> tuple:
    return (
        str(edge.get("source")),
        str(edge.get("target")),
        _norm(_data(edge).get("label")),
    )


def _find_edge(wf: dict, ref_id: str, view: Optional[dict]) -> Optional[dict]:
    edges = _items(wf, "edges")
    for edge in edges:
        if str(edge.get("id")) == str(ref_id):
            return edge
    if view:
        key = _edge_key(view)
        for edge in edges:
            if _edge_key(edge) == key:
                return edge
    return None


def _fields_match(item: dict, change: ChangeItem, extra: tuple[str, ...]) -> bool:
    data = _data(item)
    if change.field_diffs:
        for diff in change.field_diffs:
            actual = (
                item.get(diff.field) if diff.field in extra else data.get(diff.field)
            )
            if actual != diff.after:
                return False
        return True
    # No field list (an added node/edge): every field the change set must
    # still hold; fields edited later that the change never set do not matter.
    after = change.after or {}
    if any(item.get(k) != after.get(k) for k in extra if k in after):
        return False
    wanted = {k: v for k, v in (after.get("data") or {}).items() if v is not None}
    return all(data.get(k) == v for k, v in wanted.items())


def definition_contains_change(wf: dict, change: ChangeItem) -> bool:
    """Compare the change's ``after`` content with the (masked) definition."""
    if change.target == ChangeTarget.node:
        if change.op == ChangeOp.remove:
            return not any(
                str(n.get("id")) == str(change.ref_id) for n in _items(wf, "nodes")
            )
        node = _find_node(wf, change)
        return node is not None and _fields_match(node, change, ("type",))
    if change.target == ChangeTarget.edge:
        if change.op == ChangeOp.remove:
            return _find_edge(wf, change.ref_id, change.before) is None
        edge = _find_edge(wf, change.ref_id, change.after)
        return edge is not None and _fields_match(edge, change, ("source", "target"))
    # Workflow-level settings (the name) are not in the definition JSON;
    # publishing a definition made after the apply is the best evidence.
    return True


# ---------------------------------------------------------------------------
# Targets and clusters
# ---------------------------------------------------------------------------


def resolve_target(hypothesis: dict, change: ChangeItem) -> Optional[str]:
    target = hypothesis.get("fix_target")
    if target:
        return str(target)
    if change.target == ChangeTarget.node:
        view = change.after or change.before or {}
        name = (view.get("data") or {}).get("name")
        return str(name) if name else None
    return None


def run_reached_target(run: RunFacts, target: Optional[str], wf: dict) -> bool:
    """Was the call exposed to the fix? Node targets need the call to reach
    the node (an edge target its source node); prompt-wide targets (global,
    kb, config, platform, unknown names) expose every call."""
    if not target:
        return True
    if target.startswith("edge:"):
        source = _norm(target[len("edge:") :].split("->")[0])
        return not source or source in run.nodes
    name = _norm(target)
    node_names = {_norm(_data(n).get("name")) for n in _items(wf, "nodes")}
    if name in node_names:
        return name in run.nodes
    return True


def match_cluster(clusters: list[Cluster], hypothesis: dict) -> Optional[Cluster]:
    target = _norm(hypothesis.get("fix_target"))
    category = hypothesis.get("category")
    if target:
        for cluster in clusters:
            if _norm(cluster.fix_target) == target and (
                not category or cluster.category == category
            ):
                return cluster
    title = _norm(hypothesis.get("title"))
    if not title:
        return None
    best, best_score = None, TITLE_SIMILARITY_MIN
    for cluster in clusters:
        score = SequenceMatcher(None, title, _norm(cluster.title)).ratio()
        if score >= best_score:
            best, best_score = cluster, score
    return best


def cluster_run_ids(cluster: Cluster, result: ReportResult) -> set[int]:
    """Runs showing the cluster's problem. ``Cluster.evidence`` keeps only
    2-3 quotes, so the raw Layer 1 findings with the same target and
    category are added back. Reports made after Cluster.run_ids existed
    carry the exact set."""
    if cluster.run_ids:
        return set(cluster.run_ids)
    ids = {e.run_id for e in cluster.evidence}
    target = _norm(cluster.fix_target)
    for findings in result.findings_by_batch.values():
        for finding in findings:
            if _norm(finding.fix_target) == target and (
                finding.category == cluster.category
            ):
                ids.update(e.run_id for e in finding.evidence)
    return ids


def clustering_complete(result: ReportResult) -> bool:
    stages = result.stages
    return all(
        getattr(stages.get(stage), "status", None) == "done"
        for stage in (StageName.layer1, StageName.clusters)
    )


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def evaluate_change(
    *,
    proposal: AppliedProposal,
    change: ChangeItem,
    runs: list[RunFacts],
    definitions: dict[int, DefinitionFacts],
    result: ReportResult,
) -> tuple[FixCarryover, Optional[Cluster]]:
    hypothesis = proposal.hypotheses.get(change.change_id) or {}
    if not isinstance(hypothesis, dict):
        hypothesis = as_obj(hypothesis) or {}
    title = str(hypothesis.get("title") or change.label)
    target = resolve_target(hypothesis, change)

    floor = proposal.applied_definition_id
    containing = {
        d.id: d
        for d in definitions.values()
        if (floor is None or d.id >= floor)
        and d.published_at is not None
        and definition_contains_change(d.workflow_json, change)
    }
    base = {
        "proposal_uuid": proposal.proposal_uuid,
        "change_id": change.change_id,
        "title": title,
    }
    if not containing:
        return FixCarryover(**base, outcome=FixOutcome.waiting_for_publish), None
    published_at = min(d.published_at for d in containing.values())

    eligible: list[RunFacts] = []
    for run in runs:
        definition = containing.get(run.definition_id)
        if definition is None or run.created_at is None:
            continue
        if run.created_at <= definition.published_at:
            continue
        if run_reached_target(run, target, definition.workflow_json):
            eligible.append(run)

    cluster = match_cluster(result.clusters, {**hypothesis, "fix_target": target})
    affected = 0
    if cluster is not None:
        shown = cluster_run_ids(cluster, result)
        affected = sum(1 for run in eligible if run.run_id in shown)

    if not clustering_complete(result) or len(eligible) < MIN_CALLS_FOR_EVIDENCE:
        outcome = FixOutcome.insufficient
    elif affected == 0:
        outcome = FixOutcome.fixed
    else:
        outcome = FixOutcome.still_happening
    return (
        FixCarryover(
            **base,
            outcome=outcome,
            eligible_calls=len(eligible),
            affected_calls=affected,
            published_at=_iso(published_at),
        ),
        cluster,
    )


def compute_carryover(
    *,
    proposals: list[AppliedProposal],
    runs: list[RunFacts],
    definitions: dict[int, DefinitionFacts],
    result: ReportResult,
) -> list[FixCarryover]:
    """One line per accepted change; fills ``since_fix`` on matched clusters
    of ``result`` in place (first match wins)."""
    lines: list[FixCarryover] = []
    for proposal in proposals:
        by_id = {c.change_id: c for c in proposal.changes}
        for change_id in proposal.accepted_change_ids:
            change = by_id.get(change_id)
            if change is None:
                continue
            line, cluster = evaluate_change(
                proposal=proposal,
                change=change,
                runs=runs,
                definitions=definitions,
                result=result,
            )
            lines.append(line)
            if cluster is not None and cluster.since_fix is None:
                cluster.since_fix = line.model_dump(mode="json")
    return lines


def _status(value: Any) -> Any:
    return getattr(value, "value", value)


async def load_applied_proposals(
    *, workflow_id: int, organization_id: int, before_number: int
) -> list[AppliedProposal]:
    """Applied proposals of the workflow's rounds numbered below this one,
    oldest first."""
    rounds = await db_client.list_iterations(
        workflow_id, organization_id, limit=EARLIER_ROUNDS_LIMIT
    )
    proposals: list[AppliedProposal] = []
    for it in sorted(rounds, key=lambda r: r.number):
        if it.number >= before_number or not it.proposal_uuid:
            continue
        if _status(it.proposal_status) != "applied":
            continue
        row = await db_client.get_copilot_proposal(it.proposal_uuid, organization_id)
        parsed = applied_proposal(row) if row is not None else None
        if parsed is not None:
            proposals.append(parsed)
    return proposals


async def build_carryover(
    *,
    workflow_id: int,
    organization_id: int,
    number: int,
    run_ids: list[int],
    result: ReportResult,
) -> list[FixCarryover]:
    """Load what the check needs and run it. Mutates ``result`` clusters."""
    proposals = await load_applied_proposals(
        workflow_id=workflow_id, organization_id=organization_id, before_number=number
    )
    if not proposals:
        return []
    analysed = set(result.run_ids or run_ids)
    rows = await db_client.get_runs_with_logs_for_workflow(
        workflow_id,
        organization_id,
        limit=max(len(run_ids), 1),
        run_ids=list(run_ids),
    )
    runs = [run_facts(row) for row in rows if row.get("id") in analysed]
    del rows
    definitions: dict[int, DefinitionFacts] = {}
    for version in await db_client.get_workflow_versions(workflow_id):
        try:
            facts = definition_facts(version)
        except Exception as exc:  # one malformed version must not sink the round
            logger.warning(f"Self-improve: skipping definition: {exc}")
            continue
        definitions[facts.id] = facts
    return compute_carryover(
        proposals=proposals, runs=runs, definitions=definitions, result=result
    )


def carryover_line(item: FixCarryover) -> str:
    """DS-4 wording (used in the round's copilot message)."""
    published = ""
    if item.published_at:
        when = _aware(item.published_at)
        if when:
            published = f" (published {when.day} {when.strftime('%b')})"
    if item.outcome == FixOutcome.fixed:
        return (
            f"✓ {item.title}: 0 of {item.eligible_calls} calls since the fix{published}"
        )
    if item.outcome == FixOutcome.still_happening:
        return (
            f"✗ {item.title}: still in {item.affected_calls} of "
            f"{item.eligible_calls} calls since the fix{published}"
        )
    if item.outcome == FixOutcome.insufficient:
        return (
            f"– {item.title}: not enough calls yet ({item.eligible_calls} reached "
            f"that step; need {MIN_CALLS_FOR_EVIDENCE})"
        )
    return f"– {item.title}: waiting for you to publish the draft"
