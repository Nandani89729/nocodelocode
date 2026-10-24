import { describe, expect, it } from "vitest";

import {
    evidenceLinkName,
    medianReplySeconds,
    notKeptText,
    notReproducibleText,
    partialStopText,
    platformFailure,
    rankClusters,
    replayCountsText,
    scenarioOutcomeLines,
} from "./format";
import type { Cluster, ReportResponse, VerificationResult } from "./types";

const cluster = (over: Partial<Cluster>): Cluster => ({
    cluster_id: "C",
    title: "t",
    category: "workflow_error",
    severity: "minor",
    fix_target: "node",
    root_cause: "r",
    runs_affected: 1,
    runs_analysed: 10,
    evidence: [],
    watch_only: false,
    ...over,
});

const verification = (over: Partial<VerificationResult>): VerificationResult => ({
    status: "kept",
    cluster_id: "C1",
    scenario_ids: [1],
    samples_per_side: 10,
    baseline_samples: [],
    candidate_samples: [],
    ...over,
});

describe("rankClusters", () => {
    it("ranks by severity × frequency", () => {
        const ranked = rankClusters([
            cluster({ cluster_id: "minor-all", severity: "minor", runs_affected: 10 }),
            cluster({ cluster_id: "critical-few", severity: "critical", runs_affected: 2 }),
            cluster({ cluster_id: "major-half", severity: "major", runs_affected: 5 }),
        ]);
        expect(ranked.map((c) => c.cluster_id)).toEqual(["major-half", "minor-all", "critical-few"]);
    });
});

describe("wording (4A)", () => {
    it("shows counts, never 'Verified'", () => {
        const text = replayCountsText(verification({ baseline_failures: 7, candidate_failures: 0 }));
        expect(text).toBe("Replays: old version failed 7/10 → new version 0/10");
        expect(text.toLowerCase()).not.toContain("verified");
    });

    it("falls back to counting failed samples", () => {
        const text = notReproducibleText(
            verification({ status: "not_reproducible", baseline_samples: [{ text: "a", tool_calls: [], passed: true }] }),
        );
        expect(text).toMatch(/^Old version failed 0\/10: couldn't reproduce/);
    });

    it("names evidence links descriptively", () => {
        expect(evidenceLinkName({ run_id: 16, turn: 5, quote: "q" })).toBe("Call #16, turn 5");
        expect(evidenceLinkName({ run_id: 16, quote: "q" })).toBe("Call #16");
    });
});

describe("report helpers", () => {
    it("computes the turn-weighted median reply", () => {
        expect(
            medianReplySeconds([
                { node_name: "a", turns: 1, p50_s: 9, p90_s: 9 },
                { node_name: "b", turns: 10, p50_s: 2, p90_s: 3 },
            ]),
        ).toBe(2);
        expect(medianReplySeconds([])).toBeNull();
    });

    it("finds the dominant platform failure", () => {
        const f = platformFailure({
            runs_checked: 20,
            runs_flagged: 2,
            insufficient_data: {},
            flags: [
                { run_id: 3, check: "system_prompt_missing", detail: "" },
                { run_id: 4, check: "system_prompt_missing", detail: "" },
                { run_id: 4, check: "pipeline_errors", detail: "" },
            ],
        });
        expect(f).toMatchObject({ check: "system_prompt_missing", runs: 2, runsChecked: 20, runIds: [3, 4] });
    });

    it("describes where a partial report stopped", () => {
        const report = {
            result: {
                stages: {
                    layer1: { status: "failed", error: "provider error", completed_batches: [0, 1, 2, 3, 4], total_batches: 10 },
                },
            },
        } as unknown as ReportResponse;
        expect(partialStopText(report)).toBe("Analysis stopped at problems batch 6/10 (provider error)");
    });
});

describe("multi-call verification text", () => {
    const base = {
        status: "not_kept" as const,
        cluster_id: "C3",
        scenario_ids: [1, 2],
        samples_per_side: 10,
        baseline_samples: [],
        candidate_samples: [],
        candidate_failures: 4,
        scenario_outcomes: [
            { scenario_id: 1, source_run_id: 16, turn: 3, status: "not_kept" as const, baseline_failures: 10, candidate_failures: 4 },
            { scenario_id: 2, source_run_id: 12, turn: 3, status: "kept" as const, baseline_failures: 9, candidate_failures: 0 },
        ],
    };

    it("summarises across calls and lists each call", () => {
        expect(notKeptText(base)).toBe("New version still failed on 1 of 2 calls");
        expect(scenarioOutcomeLines(base)).toEqual([
            "Call #16, turn 3: old failed 10/10 → new 4/10",
            "Call #12, turn 3: old failed 9/10 → new 0/10",
        ]);
    });

    it("keeps the single-call wording", () => {
        const single = { ...base, scenario_outcomes: [] };
        expect(notKeptText(single)).toBe("New version still failed 4/10");
        expect(scenarioOutcomeLines(single)).toEqual([]);
    });
});
