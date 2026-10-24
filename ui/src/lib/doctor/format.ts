/**
 * Pure helpers for the Doctor report and verify screens: ranking, labels and
 * the exact user-facing wording from the design spec (4A: counts, never
 * the word "Verified").
 */
import type {
    Category,
    Cluster,
    Evidence,
    LatencyRow,
    Layer0Summary,
    ReportResponse,
    Severity,
    StageName,
    StageState,
    VerificationResult,
} from "./types";
import { STAGE_ORDER } from "./types";

export const SEVERITY_LABEL: Record<Severity, string> = {
    critical: "Critical",
    major: "Major",
    minor: "Minor",
};

const SEVERITY_WEIGHT: Record<Severity, number> = { critical: 3, major: 2, minor: 1 };

export const CATEGORY_LABEL: Record<Category, string> = {
    platform: "Platform",
    hallucination: "Hallucination",
    knowledge_gap: "Knowledge gap",
    workflow_error: "Workflow error",
    stt_mishearing: "Speech-to-text mishearing",
    turn_taking: "Turn-taking",
    voice_style: "Voice and style",
};

export const STAGE_LABEL: Record<StageName, string> = {
    preflight: "Setup checks",
    layer0: "Platform",
    layer1: "Problems",
    clusters: "Grouping",
    layer2: "Latency",
    rules: "Rules",
};

const LAYER0_CHECK_LABEL: Record<string, string> = {
    greeting_mismatch: "greeting didn't match",
    system_prompt_missing: "system prompt didn't load",
    pipeline_errors: "pipeline errors",
};

export const layer0CheckLabel = (check: string): string => LAYER0_CHECK_LABEL[check] ?? check.replace(/_/g, " ");

/** Rank problems by severity × frequency, highest first (design 1A). */
export function rankClusters(clusters: Cluster[]): Cluster[] {
    const score = (c: Cluster) =>
        SEVERITY_WEIGHT[c.severity] * (c.runs_analysed > 0 ? c.runs_affected / c.runs_analysed : 0);
    return [...clusters].sort((a, b) => {
        const diff = score(b) - score(a);
        if (diff !== 0) return diff;
        const sev = SEVERITY_WEIGHT[b.severity] - SEVERITY_WEIGHT[a.severity];
        if (sev !== 0) return sev;
        return b.runs_affected - a.runs_affected;
    });
}

export const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

export const frequencyLabel = (c: Cluster) => `${c.runs_affected} of ${c.runs_analysed} calls`;

/** Descriptive evidence link name (8A): "Call #16, turn 5". */
export const evidenceLinkName = (e: Evidence) =>
    e.turn !== null && e.turn !== undefined ? `Call #${e.run_id}, turn ${e.turn}` : `Call #${e.run_id}`;

/**
 * The run page cannot scroll to a turn yet, so evidence links open the run
 * and the turn lives in the link name.
 */
export const evidenceHref = (workflowId: number, e: Evidence) => `/workflow/${workflowId}/run/${e.run_id}`;

export const formatSeconds = (s: number | null | undefined) =>
    s === null || s === undefined || Number.isNaN(s) ? "–" : `${s.toFixed(1)}s`;

/** Turn-weighted median of the per-step p50 reply latency. */
export function medianReplySeconds(rows: LatencyRow[]): number | null {
    const usable = rows.filter((r) => r.turns > 0 && Number.isFinite(r.p50_s));
    if (usable.length === 0) return null;
    const sorted = [...usable].sort((a, b) => a.p50_s - b.p50_s);
    const total = sorted.reduce((sum, r) => sum + r.turns, 0);
    let acc = 0;
    for (const r of sorted) {
        acc += r.turns;
        if (acc >= total / 2) return r.p50_s;
    }
    return sorted[sorted.length - 1].p50_s;
}

export const topLatency = (rows: LatencyRow[], n = 3) => [...rows].sort((a, b) => b.p90_s - a.p90_s).slice(0, n);

export interface PlatformFailure {
    check: string;
    runs: number;
    runsChecked: number;
    runIds: number[];
}

/**
 * Platform failures that must be fixed first (2A "Platform failure").
 * Picks the Layer 0 check that flagged the most distinct calls.
 */
export function platformFailure(layer0: Layer0Summary | null | undefined): PlatformFailure | null {
    if (!layer0 || layer0.runs_flagged === 0 || layer0.flags.length === 0) return null;
    const byCheck = new Map<string, Set<number>>();
    for (const f of layer0.flags) {
        const set = byCheck.get(f.check) ?? new Set<number>();
        set.add(f.run_id);
        byCheck.set(f.check, set);
    }
    let best: PlatformFailure | null = null;
    byCheck.forEach((runs, check) => {
        if (!best || runs.size > best.runs) {
            best = { check, runs: runs.size, runsChecked: layer0.runs_checked, runIds: [...runs].sort((a, b) => a - b) };
        }
    });
    return best;
}

export const callsAnalysed = (report: ReportResponse) =>
    report.result.run_ids.length || report.result.layer0?.runs_checked || 0;

export const isActive = (status: ReportResponse["status"]) => status === "queued" || status === "running";

export function stageEntries(report: ReportResponse): Array<{ name: StageName; state: StageState }> {
    return STAGE_ORDER.map((name) => ({
        name,
        state: report.result.stages?.[name] ?? { status: "pending", completed_batches: [] },
    }));
}

/** "Problems 4/10 batches" style progress for a stage. */
export function stageProgressText(name: StageName, state: StageState): string {
    const label = STAGE_LABEL[name];
    if (name === "layer1" && state.total_batches) {
        return `${label} ${state.completed_batches.length}/${state.total_batches} batches`;
    }
    return label;
}

/** First failed stage, for "Analysis stopped at problems batch 6/10 (provider error)". */
export function partialStopText(report: ReportResponse): string {
    const failed = stageEntries(report).find((s) => s.state.status === "failed");
    if (!failed) return "Analysis stopped before it finished.";
    const reason = failed.state.error ? ` (${failed.state.error})` : "";
    if (failed.name === "layer1" && failed.state.total_batches) {
        const at = failed.state.completed_batches.length + 1;
        return `Analysis stopped at problems batch ${at}/${failed.state.total_batches}${reason}`;
    }
    return `Analysis stopped at ${STAGE_LABEL[failed.name].toLowerCase()}${reason}`;
}

export function formatElapsed(fromIso: string, now: number): string {
    const start = Date.parse(fromIso);
    if (Number.isNaN(start)) return "";
    const secs = Math.max(0, Math.floor((now - start) / 1000));
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return m > 0 ? `${m}m ${s.toString().padStart(2, "0")}s` : `${s}s`;
}

export function formatReportTime(iso: string): string {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

export const formatUsd = (v: number | null | undefined) =>
    v === null || v === undefined ? null : v < 0.01 ? "under $0.01" : `about $${v.toFixed(2)}`;

// ---------------------------------------------------------------------------
// Verification wording (4A)
// ---------------------------------------------------------------------------

export function verifyProgress(v: VerificationResult) {
    const total = v.samples_per_side || 10;
    return {
        total,
        baselineDone: Math.min(v.baseline_samples.length, total),
        candidateDone: Math.min(v.candidate_samples.length, total),
    };
}

const failures = (explicit: number | null | undefined, samples: { passed: boolean }[]) =>
    explicit ?? samples.filter((s) => !s.passed).length;

export function replayCountsText(v: VerificationResult): string {
    const total = v.samples_per_side || 10;
    const base = failures(v.baseline_failures, v.baseline_samples);
    const cand = failures(v.candidate_failures, v.candidate_samples);
    return `Replays: old version failed ${base}/${total} → new version ${cand}/${total}`;
}

export function notKeptText(v: VerificationResult): string {
    const total = v.samples_per_side || 10;
    const outcomes = v.scenario_outcomes ?? [];
    if (outcomes.length > 1) {
        const failing = outcomes.filter((o) => o.status === "not_kept").length;
        return `New version still failed on ${failing} of ${outcomes.length} calls`;
    }
    return `New version still failed ${failures(v.candidate_failures, v.candidate_samples)}/${total}`;
}

/** One line per replayed call, e.g. "Call #16, turn 3: old failed 10/10 → new 4/10". */
export function scenarioOutcomeLines(v: VerificationResult): string[] {
    const total = v.samples_per_side || 10;
    const outcomes = v.scenario_outcomes ?? [];
    if (outcomes.length < 2) return [];
    return outcomes.map((o) => {
        const where = `Call #${o.source_run_id ?? "?"}${o.turn != null ? `, turn ${o.turn}` : ""}`;
        if (o.status === "not_text_verifiable") return `${where}: can't be checked by a text replay`;
        const base = `old failed ${o.baseline_failures ?? "?"}/${total}`;
        if (o.candidate_failures == null) return `${where}: ${base}, couldn't reproduce`;
        return `${where}: ${base} → new ${o.candidate_failures}/${total}`;
    });
}

export function notReproducibleText(v: VerificationResult): string {
    const total = v.samples_per_side || 10;
    return `Old version failed ${failures(v.baseline_failures, v.baseline_samples)}/${total}: couldn't reproduce this problem in a replay. Check on a real call.`;
}
