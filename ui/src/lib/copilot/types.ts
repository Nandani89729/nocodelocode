/**
 * Agent Copilot (milestone 2) API types.
 *
 * Hand-mirrored from api/services/copilot/contracts.py until the generated
 * client (src/client, hey-api) is regenerated with the copilot routes. Keep
 * field names and literals identical to the Python contract.
 */

/** SendMessageRequest.text max_length. */
export const MAX_MESSAGE_CHARS = 20_000;

// ---------------------------------------------------------------------------
// Plans and proposals
// ---------------------------------------------------------------------------

export interface PlanStep {
    title: string;
    detail?: string | null;
}

export interface Plan {
    summary: string;
    steps: PlanStep[];
    /** open questions for the user before building, if any */
    questions: string[];
}

export type ChangeTarget = "node" | "edge" | "workflow" | "tool";

export type ChangeOp = "add" | "update" | "remove";

export interface FieldDiff {
    /** dotted path inside node/edge data, e.g. "prompt", "tool_uuids" */
    field: string;
    before?: unknown;
    after?: unknown;
}

export interface ChangeItem {
    /** stable within a proposal, e.g. "n:3", "e:2-4:end_call" */
    change_id: string;
    target: ChangeTarget;
    op: ChangeOp;
    /** node id / edge id / "workflow" */
    ref_id: string;
    /** human label, e.g. 'Update node "Welcome and Source" (prompt)' */
    label: string;
    before?: Record<string, unknown> | null;
    after?: Record<string, unknown> | null;
    field_diffs: FieldDiff[];
    /** change_ids that must also be accepted */
    depends_on: string[];
}

export type ProposalKind = "edit" | "create";

/** "applying" = an apply is in progress (atomic claim); not "reviewed" (M2b O2). */
export type ProposalStatus = "pending" | "applying" | "applied" | "discarded" | "stale";

/** What an accepted change is expected to fix (M2b next-calls check). */
export interface Hypothesis {
    cluster_id?: string | null;
    title: string;
    category?: string | null;
    /** node name (or "edge:A->B", "kb", ...) the problem was attributed to */
    fix_target?: string | null;
}

export interface Proposal {
    proposal_uuid: string;
    session_uuid: string;
    kind: ProposalKind;
    /** null for kind=create until applied */
    workflow_id?: number | null;
    summary: string;
    base_hash?: string | null;
    /** Set when the proposal changes settings (configurations / template variables). */
    settings_base_hash?: string | null;
    changes: ChangeItem[];
    status: ProposalStatus;
    /** for create: the workflow name from `new Workflow({ name })` */
    workflow_name?: string | null;
    created_at?: string | null;
    /** set when the proposal came from a Self-improve round (M2b) */
    iteration_id?: number | null;
    /** after apply: which change_ids were applied and the resulting definition */
    accepted_change_ids?: string[];
    applied_definition_id?: number | null;
    /** per change_id: the Doctor cluster it targets */
    hypotheses?: Record<string, Hypothesis>;
}

// ---------------------------------------------------------------------------
// Workflow brief / North Star (M2b R15/D2)
// ---------------------------------------------------------------------------

export interface WorkflowBrief {
    goal: string;
    must_do: string[];
    must_never: string[];
    accepted_tradeoffs: string[];
    notes: string;
}

export type BriefChangeStatus = "pending" | "confirmed" | "dismissed" | "stale";

/** A copilot-proposed brief update; never applied without the user (O8). */
export interface BriefChange {
    change_uuid: string;
    workflow_id: number;
    base_version: number;
    proposed: WorkflowBrief;
    reason: string;
    status: BriefChangeStatus;
    created_at?: string | null;
}

export interface BriefResponse {
    workflow_id: number;
    brief: WorkflowBrief;
    /** increments on every confirmed change; the optimistic lock */
    version: number;
    updated_at?: string | null;
    updated_by?: "user" | "copilot" | null;
    /** a copilot-proposed change waiting for the user, if any */
    pending_change?: BriefChange | null;
}

export interface UpdateBriefRequest {
    brief: WorkflowBrief;
    expected_version: number;
}

/** Session kinds: the chat panel only ever lists/resumes "chat" (backend filters, RI-1). */
export type SessionKind = "chat" | "self_improve";

// ---------------------------------------------------------------------------
// Streaming events (SSE: `event: <type>` + `data: <json>`)
// ---------------------------------------------------------------------------

export type CopilotEventType =
    | "text_delta"
    | "tool_call"
    | "tool_result"
    | "plan"
    | "proposal"
    | "brief_change"
    | "message_done"
    | "error";

export const COPILOT_EVENT_TYPES: readonly CopilotEventType[] = [
    "text_delta",
    "tool_call",
    "tool_result",
    "plan",
    "proposal",
    "brief_change",
    "message_done",
    "error",
];

export type CopilotEvent =
    | { type: "text_delta"; data: { text: string } }
    | { type: "tool_call"; data: { id: string; name: string; summary?: string } }
    | { type: "tool_result"; data: { id: string; name: string; ok: boolean; summary?: string } }
    | { type: "plan"; data: { plan: Plan } }
    | { type: "proposal"; data: { proposal: Proposal } }
    | { type: "brief_change"; data: { brief_change: BriefChange } }
    | { type: "message_done"; data: Record<string, never> }
    | { type: "error"; data: { message: string } };

// ---------------------------------------------------------------------------
// API shapes
// ---------------------------------------------------------------------------

export interface CreateSessionRequest {
    /** null = "Build with Copilot" from /workflow/create (proposals are kind=create) */
    workflow_id?: number | null;
}

export interface SendMessageRequest {
    text: string;
}

/** What the UI renders for history (tool messages collapsed to chips). */
export interface DisplayMessage {
    role: "user" | "assistant" | "tool";
    content: string;
    tool_name?: string | null;
    is_error: boolean;
    created_at?: string | null;
}

export interface SessionResponse {
    session_uuid: string;
    workflow_id?: number | null;
    title?: string | null;
    messages: DisplayMessage[];
    proposals: Proposal[];
    /** last plan presented in this session, if any */
    plan?: Plan | null;
    created_at: string;
    updated_at: string;
}

export interface ApplyProposalRequest {
    /** change_ids to apply (edit). Ignored for create (all-or-nothing). */
    accepted_change_ids: string[];
    /** optimistic lock: the working copy's version_number (int) or hash (str) */
    expected_draft_version: number | string | null;
}

export type ApplyErrorCode =
    | "draft_conflict"
    | "stale"
    | "missing_dependency"
    | "validation_error"
    | "graph_validation"
    | "not_found"
    | "already_applied";

export interface ApplyProposalResponse {
    saved: boolean;
    workflow_id?: number | null;
    /** hash of the new working copy after the save */
    draft_hash?: string | null;
    error_code?: ApplyErrorCode | string | null;
    error?: string | null;
    proposal?: Proposal | null;
}
