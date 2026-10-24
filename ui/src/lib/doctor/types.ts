/**
 * Agent Doctor (milestone 1) API types.
 *
 * Hand-mirrored from api/services/workflow/doctor/contracts.py until the
 * generated client (src/client, hey-api) is regenerated with the Doctor
 * routes. Keep field names and literals identical to the Python contract.
 */

export const MAX_CALLS_PER_ANALYSIS = 100;
export const DEFAULT_CALLS_PER_ANALYSIS = 50;
export const LAYER1_BATCH_SIZE = 10;
export const REGENERATIONS_PER_SIDE = 10;
export const FEW_CALLS_THRESHOLD = 10;
export const BASELINE_MIN_FAILURES = 3;

// ---------------------------------------------------------------------------
// Report lifecycle
// ---------------------------------------------------------------------------

export type ReportStatus = "queued" | "running" | "partial" | "completed" | "failed";

export type StageName = "preflight" | "layer0" | "layer1" | "clusters" | "layer2" | "rules";

export const STAGE_ORDER: readonly StageName[] = [
    "preflight",
    "layer0",
    "layer1",
    "clusters",
    "layer2",
    "rules",
];

export interface StageState {
    status: "pending" | "running" | "done" | "failed" | "skipped";
    error?: string | null;
    /** For layer1: indices of completed batches (resume skips them). */
    completed_batches: number[];
    total_batches?: number | null;
}

// ---------------------------------------------------------------------------
// Report content
// ---------------------------------------------------------------------------

export type Severity = "critical" | "major" | "minor";

export type Category =
    | "platform"
    | "hallucination"
    | "knowledge_gap"
    | "workflow_error"
    | "stt_mishearing"
    | "turn_taking"
    | "voice_style";

export interface Evidence {
    run_id: number;
    turn?: number | null;
    node_name?: string | null;
    quote: string;
}

export interface PreflightIssue {
    check: "end_call_unreachable" | "tool_not_attached" | "tool_name_collision" | "graph_invalid";
    node_name?: string | null;
    detail: string;
}

export interface Layer0Flag {
    run_id: number;
    check: "greeting_mismatch" | "system_prompt_missing" | "pipeline_errors";
    detail: string;
}

export interface Layer0Summary {
    runs_checked: number;
    runs_flagged: number;
    flags: Layer0Flag[];
    /** per-check runs lacking required fields (legacy runs) → "insufficient data" */
    insufficient_data: Record<string, number>;
}

export interface Finding {
    category: Category;
    severity: Severity;
    fix_target: string;
    root_cause: string;
    evidence: Evidence[];
}

export interface Cluster {
    cluster_id: string;
    title: string;
    category: Category;
    severity: Severity;
    fix_target: string;
    root_cause: string;
    runs_affected: number;
    runs_analysed: number;
    evidence: Evidence[];
    watch_only: boolean;
    /** Hypothesis carry-over: rate since the last published fix, if any. */
    since_fix?: Record<string, unknown> | null;
}

export interface PassedCheck {
    title: string;
    runs: number;
}

export interface LatencyRow {
    node_name: string;
    turns: number;
    p50_s: number;
    p90_s: number;
    llm_ttfb_p50_s?: number | null;
    stt_p50_s?: number | null;
    tts_ttfb_p50_s?: number | null;
}

export interface Rule {
    rule_id: string;
    node_name: string;
    text: string;
    kind: "regression_guard";
}

export interface ReportResult {
    stages: Partial<Record<StageName, StageState>>;
    run_ids: number[];
    definition_ids: number[];
    few_calls: boolean;
    preflight: PreflightIssue[];
    layer0?: Layer0Summary | null;
    /** raw findings per completed layer1 batch index (JSON keys are strings) */
    findings_by_batch: Record<string, Finding[]>;
    clusters: Cluster[];
    passed_checks: PassedCheck[];
    latency: LatencyRow[];
    rules: Rule[];
    ai_enabled: boolean;
    ai_disabled_reason?: "no_key" | "not_opted_in" | null;
}

/** Same filter shape as the runs list endpoint (api/db/filters.py), D12. */
export interface DoctorRunFilter {
    attribute: string;
    type: string;
    value: unknown;
}

export interface CreateReportRequest {
    limit: number;
    filters?: DoctorRunFilter[] | null;
}

export interface ReportResponse {
    report_uuid: string;
    workflow_id: number;
    status: ReportStatus;
    created_at: string;
    params: Record<string, unknown>;
    result: ReportResult;
    error?: string | null;
    queue_position?: number | null;
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

export type VerificationStatus =
    | "proposing"
    | "awaiting_start"
    | "verifying"
    | "kept"
    | "not_kept"
    | "not_reproducible"
    | "not_text_verifiable"
    | "failed";

export interface SampleResult {
    text: string;
    tool_calls: string[];
    passed: boolean;
    reason?: string | null;
}

/** One recorded call's verdict when a fix is checked on several calls. */
export interface ScenarioOutcome {
    scenario_id: number;
    source_run_id?: number | null;
    turn?: number | null;
    status: VerificationStatus;
    baseline_failures?: number | null;
    candidate_failures?: number | null;
}

export interface VerificationResult {
    status: VerificationStatus;
    cluster_id: string;
    scenario_ids: number[];
    candidate_definition_id?: number | null;
    candidate_hash?: string | null;
    baseline_failures?: number | null;
    candidate_failures?: number | null;
    samples_per_side: number;
    baseline_samples: SampleResult[];
    candidate_samples: SampleResult[];
    diff_summary?: string | null;
    cost_estimate_usd?: number | null;
    error?: string | null;
    /** Per-call results when more than one call was replayed; counts and
     * samples above are the primary call's own. */
    scenario_outcomes?: ScenarioOutcome[];
}

/**
 * Response of the propose / verification endpoints. The contract returns a
 * VerificationResult; the route may additionally echo the scenario id the
 * verification endpoints are keyed by.
 */
export type VerificationResponse = VerificationResult & { scenario_id?: number | null };

export interface SaveVerificationRequest {
    expected_draft_version: number | null;
}

export interface RegenerateRequest {
    run_id: number;
    turn: number;
    n: number;
}

export interface RegenerateGroup {
    text: string;
    count: number;
    tool_calls: string[];
    flagged: boolean;
    flag_reason?: string | null;
    members: string[];
}

export interface RegenerateResponse {
    n: number;
    /** sorted by count desc */
    groups: RegenerateGroup[];
}
