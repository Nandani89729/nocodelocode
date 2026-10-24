"""Shared contracts for the Agent Doctor (milestone 1).

Every Doctor module, the DB client, the arq job, the routes and the UI code
against this file. Change it deliberately: it is the seam between parallel
work streams.

Flow (plan: ~/.claude/plans/i-want-this-claude-luminous-ember.md):

    POST /workflow/{id}/doctor/reports ──► WorkflowAnalysisReport(status=queued)
        │ arq analyze_workflow_calls (shared worker, non-blocking admission)
        ▼
    preflight ─► layer0 ─► layer1 batches ─► clusters ─► layer2 ─► rules
        (each stage checkpointed into report.result; error ⇒ status=partial)
        ▼
    POST .../clusters/{cluster_id}/propose ─► candidate WorkflowDefinition
    POST /doctor/verifications/{id}/start   ─► replay ×10 baseline + candidate
    POST /doctor/verifications/{id}/save    ─► sdk_authoring save (hash match)
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Constants (D17, D18, D21, D19/D27, D25, design-doc caps)
# ---------------------------------------------------------------------------

#: WorkflowDefinition.status for Doctor fix candidates. Hidden from version
#: history, draft lookups and publishing (those filter explicit statuses).
CANDIDATE_DEFINITION_STATUS = "candidate"

#: Status after a candidate is saved or discarded. Rows are kept (replay runs
#: reference them via workflow_runs.definition_id, which has no ON DELETE);
#: hidden everywhere because history/draft queries filter explicit statuses.
CANDIDATE_ARCHIVED_STATUS = "candidate_archived"

#: workflow_runs.mode for Doctor replay runs. Excluded from run lists, CSV
#: reports and Doctor input via ``exclude_internal_runs()``; their usage is
#: reported on its own line (D21).
DOCTOR_REPLAY_RUN_MODE = "doctor_replay"

#: All run modes that are internal and never customer-visible as calls.
INTERNAL_RUN_MODES: tuple[str, ...] = (DOCTOR_REPLAY_RUN_MODE,)

#: usage_context tags for billing/metering.
USAGE_CONTEXT_DOCTOR = "doctor"  # Claude calls (Layer 1, clustering, judge)
USAGE_CONTEXT_DOCTOR_REPLAY = "doctor_replay"  # workflow-LLM replay turns

#: Claude model for Doctor analysis and grading.
DOCTOR_MODEL = "claude-opus-5-5"

#: Caps (design doc "Limits").
MAX_CALLS_PER_ANALYSIS = 100
DEFAULT_CALLS_PER_ANALYSIS = 50
LAYER1_BATCH_SIZE = 10
MAX_SCENARIOS_PER_VERIFICATION = 30
REGENERATIONS_PER_SIDE = 10  # ×10 single-turn verification
REGENERATION_CONCURRENCY = 5  # D25
FEW_CALLS_THRESHOLD = 10  # below this the report shows a "patterns may be noise" caveat

#: Keep rule (teardown A, D28 kept 0/10): baseline must fail ≥ this many of 10,
#: candidate must fail 0 of 10.
BASELINE_MIN_FAILURES = 3

#: Admission control (D19 + D27): non-blocking semaphore, Retry(defer) otherwise.
DOCTOR_GLOBAL_CONCURRENCY = 2
DOCTOR_PER_ORG_CONCURRENCY = 1
DOCTOR_JOB_TIMEOUT_SECONDS = 1800
DOCTOR_ADMISSION_RETRY_DEFER_SECONDS = 30
DOCTOR_ADMISSION_MAX_TRIES = 20

#: Retention (design doc + D30).
REPORT_RETENTION_DAYS = 90

#: Layer 0 thresholds (design-doc check table).
L0_GREETING_SIMILARITY_MIN = 0.6
L0_PROMPT_TOKEN_RATIO_MIN = 0.30


# ---------------------------------------------------------------------------
# Report lifecycle
# ---------------------------------------------------------------------------


class ReportStatus(str, Enum):
    queued = "queued"
    running = "running"
    partial = "partial"  # D20: stage(s) failed after retries; resumable
    completed = "completed"
    failed = "failed"


class StageName(str, Enum):
    preflight = "preflight"
    layer0 = "layer0"
    layer1 = "layer1"  # sub-checkpointed per batch
    clusters = "clusters"
    layer2 = "layer2"
    rules = "rules"


STAGE_ORDER: tuple[StageName, ...] = (
    StageName.preflight,
    StageName.layer0,
    StageName.layer1,
    StageName.clusters,
    StageName.layer2,
    StageName.rules,
)


class StageState(BaseModel):
    status: Literal["pending", "running", "done", "failed", "skipped"] = "pending"
    error: Optional[str] = None
    #: For layer1: indices of completed batches (resume skips them).
    completed_batches: list[int] = Field(default_factory=list)
    total_batches: Optional[int] = None


# ---------------------------------------------------------------------------
# Report content
# ---------------------------------------------------------------------------

Severity = Literal["critical", "major", "minor"]
Category = Literal[
    "platform",
    "hallucination",
    "knowledge_gap",
    "workflow_error",
    "stt_mishearing",
    "turn_taking",
    "voice_style",
]


class Evidence(BaseModel):
    run_id: int
    turn: Optional[int] = None
    node_name: Optional[str] = None
    quote: str


class PreflightIssue(BaseModel):
    check: Literal[
        "end_call_unreachable",
        "tool_not_attached",
        "tool_name_collision",
        "graph_invalid",
    ]
    node_name: Optional[str] = None
    detail: str


class Layer0Flag(BaseModel):
    run_id: int
    check: Literal["greeting_mismatch", "system_prompt_missing", "pipeline_errors"]
    detail: str


class Layer0Summary(BaseModel):
    runs_checked: int
    runs_flagged: int
    flags: list[Layer0Flag] = Field(default_factory=list)
    #: per-check runs lacking required fields (legacy runs) → "insufficient data"
    insufficient_data: dict[str, int] = Field(default_factory=dict)


class Finding(BaseModel):
    """One Layer 1 finding from one run (Claude structured output)."""

    category: Category
    severity: Severity
    fix_target: str  # node name / "edge:A->B" / "kb" / "config:<key>" / "platform"
    root_cause: str
    evidence: list[Evidence] = Field(min_length=1)


class Cluster(BaseModel):
    cluster_id: str  # stable within a report, e.g. "C1"
    title: str  # plain-English problem title (design 1A)
    category: Category
    severity: Severity
    fix_target: str
    root_cause: str
    runs_affected: int
    runs_analysed: int
    evidence: list[Evidence] = Field(default_factory=list)  # 2-3 shown
    #: every analysed run showing this problem (evidence keeps only 2-3)
    run_ids: list[int] = Field(default_factory=list)
    #: True when only one run shows it ("watch", not a fix).
    watch_only: bool = False
    #: Hypothesis carry-over: rate since the last published fix, if any.
    since_fix: Optional[dict] = None


class PassedCheck(BaseModel):
    title: str  # e.g. "Prompt injection held"
    runs: int


class LatencyRow(BaseModel):
    node_name: str
    turns: int
    p50_s: float
    p90_s: float
    llm_ttfb_p50_s: Optional[float] = None
    stt_p50_s: Optional[float] = None
    tts_ttfb_p50_s: Optional[float] = None


class Rule(BaseModel):
    """Atomic rule extracted from a node/global prompt (teardown B)."""

    rule_id: str
    node_name: str  # "global" for the global prompt
    text: str
    #: regression guard, not a correctness oracle (R14 context)
    kind: Literal["regression_guard"] = "regression_guard"


class ReportResult(BaseModel):
    """Stored in workflow_analysis_reports.result (JSON)."""

    stages: dict[StageName, StageState] = Field(
        default_factory=lambda: {s: StageState() for s in STAGE_ORDER}
    )
    run_ids: list[int] = Field(default_factory=list)
    definition_ids: list[int] = Field(default_factory=list)
    few_calls: bool = False
    preflight: list[PreflightIssue] = Field(default_factory=list)
    layer0: Optional[Layer0Summary] = None
    #: raw findings per completed layer1 batch index (checkpoint)
    findings_by_batch: dict[int, list[Finding]] = Field(default_factory=dict)
    clusters: list[Cluster] = Field(default_factory=list)
    passed_checks: list[PassedCheck] = Field(default_factory=list)
    latency: list[LatencyRow] = Field(default_factory=list)
    rules: list[Rule] = Field(default_factory=list)
    #: True when org opted in and a key exists (Layer 1 ran or will run).
    ai_enabled: bool = False
    ai_disabled_reason: Optional[Literal["no_key", "not_opted_in"]] = None
    #: Cumulative Claude token usage for this report (metered as
    #: USAGE_CONTEXT_DOCTOR); summed across resumes.
    claude_usage: dict[str, int] = Field(default_factory=dict)


class CreateReportRequest(BaseModel):
    limit: int = Field(DEFAULT_CALLS_PER_ANALYSIS, ge=1, le=MAX_CALLS_PER_ANALYSIS)
    #: Same filter shape as the runs list endpoint (api/db/filters.py), D12.
    filters: Optional[list[dict]] = None
    #: Analyse exactly these runs (a Self-improve round, M2b); ``limit`` and
    #: ``filters`` are then ignored. Still scoped to the workflow and org.
    run_ids: Optional[list[int]] = Field(
        None, min_length=1, max_length=MAX_CALLS_PER_ANALYSIS
    )


class ReportResponse(BaseModel):
    report_uuid: str
    workflow_id: int
    status: ReportStatus
    created_at: str
    params: dict
    result: ReportResult
    error: Optional[str] = None
    queue_position: Optional[int] = None


# ---------------------------------------------------------------------------
# Verification (D29 checkpoint, D25 concurrency, D28 keep rule, D30 hash)
# ---------------------------------------------------------------------------


class VerificationStatus(str, Enum):
    proposing = "proposing"
    awaiting_start = "awaiting_start"  # cost estimate shown
    verifying = "verifying"
    kept = "kept"
    not_kept = "not_kept"
    not_reproducible = "not_reproducible"
    not_text_verifiable = "not_text_verifiable"
    failed = "failed"


class SampleResult(BaseModel):
    text: str
    tool_calls: list[str] = Field(default_factory=list)
    passed: bool
    reason: Optional[str] = None


class ScenarioOutcome(BaseModel):
    """One recorded call's verdict when a fix is checked on several calls."""

    scenario_id: int
    source_run_id: Optional[int] = None
    turn: Optional[int] = None
    status: VerificationStatus
    baseline_failures: Optional[int] = None
    candidate_failures: Optional[int] = None


class VerificationResult(BaseModel):
    """Stored on the regression scenario / verification record.

    On the primary scenario, samples and failure counts are that scenario's
    own; ``status`` is the verdict across all of them and
    ``scenario_outcomes`` lists each call's result.
    """

    status: VerificationStatus
    cluster_id: str
    scenario_ids: list[int] = Field(default_factory=list)
    candidate_definition_id: Optional[int] = None
    #: sha256 of canonical post-sanitize candidate workflow JSON (D30).
    candidate_hash: Optional[str] = None
    baseline_failures: Optional[int] = None
    candidate_failures: Optional[int] = None
    samples_per_side: int = REGENERATIONS_PER_SIDE
    baseline_samples: list[SampleResult] = Field(default_factory=list)
    candidate_samples: list[SampleResult] = Field(default_factory=list)
    diff_summary: Optional[str] = None
    cost_estimate_usd: Optional[float] = None
    error: Optional[str] = None
    scenario_outcomes: list[ScenarioOutcome] = Field(default_factory=list)


class RegenerateRequest(BaseModel):
    """Tester 'Regenerate ×10 at this turn' (D13)."""

    run_id: int
    turn: int
    n: int = Field(REGENERATIONS_PER_SIDE, ge=1, le=REGENERATIONS_PER_SIDE)


class RegenerateGroup(BaseModel):
    text: str  # representative reply
    count: int
    tool_calls: list[str] = Field(default_factory=list)
    flagged: bool = False
    flag_reason: Optional[str] = None
    members: list[str] = Field(default_factory=list)


class RegenerateResponse(BaseModel):
    n: int
    groups: list[RegenerateGroup]  # sorted by count desc


class DoctorSettings(BaseModel):
    """Per-org Doctor settings (GET/PUT /doctor/settings)."""

    ai_opt_in: bool
    anthropic_key_configured: bool = False
