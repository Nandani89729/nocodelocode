/**
 * Thin typed client for the Agent Copilot routes.
 *
 * Same approach as lib/doctor/api.ts: requests go through the generated
 * hey-api `client` singleton so they get its base URL and the
 * `Authorization: Bearer` interceptor (setupAuthInterceptor). The message
 * stream is a POST, so instead of EventSource it uses `client.request` with
 * `parseAs: "stream"` (still through the interceptor) and parses the body
 * with ./sse. Replace the REST calls with generated functions once
 * `npm run generate-client` picks up the copilot routes.
 */
import { client } from "@/client/client.gen";
import { detailFromError } from "@/lib/apiError";

import { readCopilotEvents } from "./sse";
import type {
    ApplyErrorCode,
    ApplyProposalRequest,
    ApplyProposalResponse,
    BriefChange,
    BriefResponse,
    CopilotEvent,
    Proposal,
    SessionResponse,
    UpdateBriefRequest,
} from "./types";

export class CopilotApiError extends Error {
    readonly status: number | undefined;
    readonly body: unknown;

    constructor(message: string, status: number | undefined, body: unknown) {
        super(message);
        this.name = "CopilotApiError";
        this.status = status;
        this.body = body;
    }
}

/**
 * Why the copilot can't answer at all: 503 = no AI key on the server,
 * 403 = the organisation hasn't opted in to AI (same switch as the Doctor).
 */
export type CopilotGate = "no_key" | "not_opted_in";

export function gateOf(err: unknown): CopilotGate | null {
    if (!(err instanceof CopilotApiError)) return null;
    if (err.status === 503) return "no_key";
    if (err.status === 403) return "not_opted_in";
    return null;
}

/** 409 on send: a turn is already running in this session. */
export const isBusyError = (err: unknown) => err instanceof CopilotApiError && err.status === 409;

export const isAbortError = (err: unknown) =>
    err instanceof DOMException ? err.name === "AbortError" : (err as { name?: string } | null)?.name === "AbortError";

type Method = "GET" | "POST" | "PUT";

async function call<T>(method: Method, url: string, body?: unknown, fallback = "Request failed"): Promise<T> {
    const result = await client.request<{ 200: T }, unknown, false>({
        method,
        url,
        ...(body !== undefined ? { body, headers: { "Content-Type": "application/json" } } : {}),
    });
    if (result.error !== undefined) {
        throw new CopilotApiError(detailFromError(result.error, fallback), result.response?.status, result.error);
    }
    return result.data as T;
}

const seg = (value: string | number) => encodeURIComponent(String(value));

export function createCopilotSession(workflowId: number | null) {
    return call<SessionResponse>(
        "POST",
        "/api/v1/copilot/sessions",
        { workflow_id: workflowId },
        "Could not start a copilot chat",
    );
}

export function getCopilotSession(sessionUuid: string) {
    return call<SessionResponse>("GET", `/api/v1/copilot/sessions/${seg(sessionUuid)}`, undefined, "Could not load the chat");
}

/** Latest first; `messages` may be empty in the list. */
export function listCopilotSessions(workflowId: number) {
    return call<SessionResponse[]>(
        "GET",
        `/api/v1/workflow/${seg(workflowId)}/copilot/sessions`,
        undefined,
        "Could not load copilot chats",
    );
}

/** One proposal by uuid (Self-improve rounds link to theirs by proposal_uuid). */
export function getProposal(proposalUuid: string) {
    return call<Proposal>("GET", `/api/v1/copilot/proposals/${seg(proposalUuid)}`, undefined, "Could not load the proposal");
}

export function discardProposal(proposalUuid: string) {
    return call<Proposal>(
        "POST",
        `/api/v1/copilot/proposals/${seg(proposalUuid)}/discard`,
        {},
        "Could not discard the proposal",
    );
}

const STATUS_TO_APPLY_CODE: Record<number, ApplyErrorCode> = {
    404: "not_found",
    409: "draft_conflict",
    422: "validation_error",
};

/**
 * Apply accepted changes. Apply failures come back as ApplyProposalResponse
 * with `saved: false` + `error_code`, whether the backend sends them with a
 * 200 or with an error status (body = the response, or `{detail: {...}}`).
 * Only failures that carry no apply shape at all (network, 500) throw.
 */
export async function applyProposal(proposalUuid: string, request: ApplyProposalRequest): Promise<ApplyProposalResponse> {
    const result = await client.request<{ 200: ApplyProposalResponse }, unknown, false>({
        method: "POST",
        url: `/api/v1/copilot/proposals/${seg(proposalUuid)}/apply`,
        body: request,
        headers: { "Content-Type": "application/json" },
    });
    if (result.error === undefined) return result.data as ApplyProposalResponse;

    const status = result.response?.status;
    const body = result.error as Record<string, unknown> | null;
    const shaped = (candidate: unknown): ApplyProposalResponse | null =>
        candidate && typeof candidate === "object" && "error_code" in candidate
            ? { saved: false, ...(candidate as Partial<ApplyProposalResponse>) }
            : null;
    const fromBody = shaped(body) ?? shaped(body?.detail);
    if (fromBody) return fromBody;
    const code = status !== undefined ? STATUS_TO_APPLY_CODE[status] : undefined;
    if (code) return { saved: false, error_code: code, error: detailFromError(body, "") || null };
    throw new CopilotApiError(detailFromError(body, "Could not apply the changes"), status, body);
}

/**
 * Send a user message and stream the assistant turn. Resolves when the stream
 * ends (normally after `message_done`). Throws CopilotApiError for HTTP
 * errors (409 busy, 503 no key, 403 not opted in) and the fetch AbortError
 * when `signal` aborts.
 */
export async function streamCopilotMessage(
    sessionUuid: string,
    text: string,
    onEvent: (event: CopilotEvent) => void,
    signal?: AbortSignal,
): Promise<void> {
    const result = await client.request<{ 200: ReadableStream<Uint8Array> }, unknown, false>({
        method: "POST",
        url: `/api/v1/copilot/sessions/${seg(sessionUuid)}/messages`,
        body: { text },
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        parseAs: "stream",
        signal,
    });
    if (result.error !== undefined) {
        if (signal?.aborted || isAbortError(result.error)) {
            throw new DOMException("The copilot reply was stopped", "AbortError");
        }
        throw new CopilotApiError(
            detailFromError(result.error, "The copilot could not reply"),
            result.response?.status,
            result.error,
        );
    }
    const stream = result.data as ReadableStream<Uint8Array> | null | undefined;
    if (!stream) return;
    for await (const event of readCopilotEvents(stream)) {
        onEvent(event);
    }
}

// ---------------------------------------------------------------------------
// Workflow brief (M2b R15/D2)
// ---------------------------------------------------------------------------

/** 409 detail sent when the brief's version moved on (optimistic lock). */
export const BRIEF_VERSION_CONFLICT = "brief_version_conflict";

/** A brief save/confirm was refused because someone changed the brief first. */
export const isBriefConflict = (err: unknown) => err instanceof CopilotApiError && err.status === 409;

export function getBrief(workflowId: number) {
    return call<BriefResponse>("GET", `/api/v1/workflow/${seg(workflowId)}/brief`, undefined, "Could not load the brief");
}

/** Throws CopilotApiError 409 (detail "brief_version_conflict") on a version mismatch. */
export function updateBrief(workflowId: number, request: UpdateBriefRequest) {
    return call<BriefResponse>("PUT", `/api/v1/workflow/${seg(workflowId)}/brief`, request, "Could not save the brief");
}

export function confirmBriefChange(workflowId: number, changeUuid: string) {
    return call<BriefResponse>(
        "POST",
        `/api/v1/workflow/${seg(workflowId)}/brief/changes/${seg(changeUuid)}/confirm`,
        {},
        "Could not update the brief",
    );
}

export function dismissBriefChange(workflowId: number, changeUuid: string) {
    return call<BriefResponse>(
        "POST",
        `/api/v1/workflow/${seg(workflowId)}/brief/changes/${seg(changeUuid)}/dismiss`,
        {},
        "Could not dismiss the brief change",
    );
}

/** DS-6 "Draft from my chats": the copilot drafts a brief as a pending change. */
export function draftBrief(workflowId: number) {
    return call<{ brief_change: BriefChange }>(
        "POST",
        `/api/v1/workflow/${seg(workflowId)}/brief/draft`,
        {},
        "Could not draft a brief from your chats",
    );
}
