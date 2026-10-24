/**
 * Self-improve (milestone 2b) API types.
 *
 * Hand-mirrored from api/services/self_improve/contracts.py until the
 * generated client picks up the routes. Keep field names and literals
 * identical to the Python contract.
 */

/** Allowed values for "every N calls". */
export const ALLOWED_EVERY_N_CALLS = [5, 10, 20] as const;
export const DEFAULT_EVERY_N_CALLS = 10;

/** Backoff after a no_changes round: N, 2N, 4N (D6). */
export const MAX_WINDOW_MULTIPLIER = 4;

/** Consecutive failed rounds that pause Self-improve (D6). */
export const FAILURES_BEFORE_PAUSE = 2;

/** Next-calls check: fewer eligible calls than this = insufficient evidence (O3). */
export const MIN_CALLS_FOR_EVIDENCE = 3;

/** What the status line shows (DS-3). */
export type SelfImproveState =
    | "off"
    | "ai_disabled"
    | "no_key"
    | "collecting"
    | "waiting"
    | "running"
    | "paused";

export interface SelfImproveSettings {
    enabled: boolean;
    every_n_calls: number;
}

export type IterationStatus = "queued" | "diagnosing" | "proposing" | "ready" | "no_changes" | "failed";

/** Round statuses during which the job is still running. */
export const ACTIVE_ITERATION_STATUSES: readonly IterationStatus[] = ["queued", "diagnosing", "proposing"];

/** Next-calls check result for one earlier accepted change (DS-4). */
export type FixOutcome = "fixed" | "still_happening" | "insufficient" | "waiting_for_publish";

/** One line under a round: how an earlier accepted change is doing. */
export interface FixCarryover {
    proposal_uuid: string;
    change_id: string;
    /** the hypothesis title, e.g. "Welcome step answers pricing" */
    title: string;
    outcome: FixOutcome;
    /** calls since publish that reached the target node */
    eligible_calls: number;
    /** of those, how many still showed the problem */
    affected_calls: number;
    published_at?: string | null;
}

export interface IterationResponse {
    iteration_id: number;
    number: number;
    status: IterationStatus;
    run_ids: number[];
    report_uuid?: string | null;
    proposal_uuid?: string | null;
    /** reviewed = proposal applied/discarded/stale, or no_changes/failed */
    reviewed: boolean;
    carryover: FixCarryover[];
    summary?: string | null;
    error?: string | null;
    created_at: string;
    updated_at?: string | null;
    /**
     * NOT in the contract yet (UI assumption): problems the round is writing
     * fixes for, for "Writing fixes for 2 problems…". Without it the line
     * reads "Writing fixes…".
     */
    problem_count?: number | null;
}

/** GET /workflow/{id}/self-improve */
export interface SelfImproveStatus {
    workflow_id: number;
    settings: SelfImproveSettings;
    state: SelfImproveState;
    /** calls counted in the current window */
    collected: number;
    /** the window size (N x multiplier) */
    window_size: number;
    window_multiplier: number;
    paused_reason?: string | null;
    /** rounds waiting for review (0 or 1 under D4), for DS-2 counts */
    to_review: number;
    /** the round to show first (DS-1): waiting/running round, else latest */
    current_iteration?: IterationResponse | null;
    /** false when the workflow has no brief yet (DS-6 prompt on turn-on) */
    has_brief: boolean;
}

export interface UpdateSelfImproveRequest {
    enabled: boolean;
    every_n_calls: number;
}

export interface IterationListResponse {
    iterations: IterationResponse[];
}

export type RunNowReason = "waiting_for_review" | "running" | "no_calls" | "ai_disabled" | "no_key" | "off" | "paused";

export interface RunNowResponse {
    started: boolean;
    iteration?: IterationResponse | null;
    reason?: RunNowReason | null;
}
