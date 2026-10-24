/**
 * Thin typed client for the Self-improve routes (milestone 2b).
 *
 * Same approach as lib/copilot/api.ts: requests go through the generated
 * hey-api `client` singleton (base URL + Bearer interceptor). Errors throw
 * CopilotApiError so the panel handles both features' failures one way.
 * Replace with generated functions once `npm run generate-client` picks up
 * the self-improve routes.
 */
import { client } from "@/client/client.gen";
import { detailFromError } from "@/lib/apiError";
import { CopilotApiError } from "@/lib/copilot/api";

import type {
    IterationListResponse,
    RunNowResponse,
    SelfImproveStatus,
    UpdateSelfImproveRequest,
} from "./types";

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

const base = (workflowId: number) => `/api/v1/workflow/${encodeURIComponent(String(workflowId))}/self-improve`;

export function getSelfImproveStatus(workflowId: number) {
    return call<SelfImproveStatus>("GET", base(workflowId), undefined, "Could not load Self-improve");
}

export function updateSelfImprove(workflowId: number, request: UpdateSelfImproveRequest) {
    return call<SelfImproveStatus>("PUT", base(workflowId), request, "Could not save Self-improve settings");
}

export function runSelfImproveNow(workflowId: number) {
    return call<RunNowResponse>("POST", `${base(workflowId)}/run`, {}, "Could not start a round");
}

export function resumeSelfImprove(workflowId: number) {
    return call<SelfImproveStatus>("POST", `${base(workflowId)}/resume`, {}, "Could not resume Self-improve");
}

/** Newest first. */
export function listSelfImproveIterations(workflowId: number) {
    return call<IterationListResponse>("GET", `${base(workflowId)}/iterations`, undefined, "Could not load rounds");
}
