/**
 * Thin typed client for the Agent Doctor routes.
 *
 * Uses the generated hey-api `client` singleton, so requests get the same
 * base URL (lib/apiClient.ts createClientConfig, upgraded by AppConfigProvider)
 * and the same `Authorization: Bearer` interceptor (setupAuthInterceptor in
 * OrgConfigContext) as every generated SDK call. Replace with the generated
 * functions once `npm run generate-client` picks up the Doctor routes.
 */
import { client } from "@/client/client.gen";
import { detailFromError } from "@/lib/apiError";

import type {
    CreateReportRequest,
    RegenerateRequest,
    RegenerateResponse,
    ReportResponse,
    SaveVerificationRequest,
    VerificationResponse,
} from "./types";

export class DoctorApiError extends Error {
    readonly status: number | undefined;
    readonly body: unknown;

    constructor(message: string, status: number | undefined, body: unknown) {
        super(message);
        this.name = "DoctorApiError";
        this.status = status;
        this.body = body;
    }
}

type Method = "GET" | "POST" | "PUT";

async function call<T>(method: Method, url: string, body?: unknown, fallback = "Request failed"): Promise<T> {
    const result = await client.request<{ 200: T }, unknown, false>({
        method,
        url,
        ...(body !== undefined
            ? { body, headers: { "Content-Type": "application/json" } }
            : {}),
    });
    if (result.error !== undefined) {
        throw new DoctorApiError(
            detailFromError(result.error, fallback),
            result.response?.status,
            result.error,
        );
    }
    return result.data as T;
}

const seg = (value: string | number) => encodeURIComponent(String(value));

export function createDoctorReport(workflowId: number, request: CreateReportRequest) {
    return call<ReportResponse>(
        "POST",
        `/api/v1/workflow/${seg(workflowId)}/doctor/reports`,
        request,
        "Could not start the analysis",
    );
}

export function listDoctorReports(workflowId: number) {
    return call<ReportResponse[]>("GET", `/api/v1/workflow/${seg(workflowId)}/doctor/reports`);
}

export function getDoctorReport(reportUuid: string) {
    return call<ReportResponse>("GET", `/api/v1/doctor/reports/${seg(reportUuid)}`, undefined, "Could not load the report");
}

export function resumeDoctorReport(reportUuid: string) {
    return call<ReportResponse>("POST", `/api/v1/doctor/reports/${seg(reportUuid)}/resume`, {}, "Could not resume the analysis");
}

export function proposeFix(reportUuid: string, clusterId: string) {
    return call<VerificationResponse>(
        "POST",
        `/api/v1/doctor/reports/${seg(reportUuid)}/clusters/${seg(clusterId)}/propose`,
        {},
        "Could not write a fix",
    );
}

export function startVerification(scenarioId: number) {
    return call<VerificationResponse>("POST", `/api/v1/doctor/verifications/${seg(scenarioId)}/start`, {}, "Could not start verification");
}

export function getVerification(scenarioId: number) {
    return call<VerificationResponse>("GET", `/api/v1/doctor/verifications/${seg(scenarioId)}`);
}

export function saveVerification(scenarioId: number, request: SaveVerificationRequest) {
    return call<unknown>("POST", `/api/v1/doctor/verifications/${seg(scenarioId)}/save`, request, "Could not save the draft");
}

export function discardVerification(scenarioId: number) {
    return call<unknown>("POST", `/api/v1/doctor/verifications/${seg(scenarioId)}/discard`, {});
}

export function regenerateTurn(workflowId: number, request: RegenerateRequest) {
    return call<RegenerateResponse>(
        "POST",
        `/api/v1/workflow/${seg(workflowId)}/text-chat/regenerate`,
        request,
        "Could not regenerate this turn",
    );
}

/** The scenario id the verification endpoints are keyed by. */
export function scenarioIdOf(v: VerificationResponse | null | undefined): number | null {
    if (!v) return null;
    if (typeof v.scenario_id === "number") return v.scenario_id;
    return v.scenario_ids?.[0] ?? null;
}

export interface DoctorSettings {
    ai_opt_in: boolean;
    anthropic_key_configured?: boolean;
}

export function getDoctorSettings() {
    return call<DoctorSettings>("GET", "/api/v1/doctor/settings");
}

export function setDoctorAiOptIn(aiOptIn: boolean) {
    return call<DoctorSettings>(
        "PUT",
        "/api/v1/doctor/settings",
        { ai_opt_in: aiOptIn },
        "Could not update AI analysis settings",
    );
}
