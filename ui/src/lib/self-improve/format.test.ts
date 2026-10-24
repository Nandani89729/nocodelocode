import { describe, expect, it } from "vitest";

import type { Proposal } from "@/lib/copilot/types";

import {
    appliedRoundText,
    carryoverLine,
    collectingText,
    copilotButtonLabel,
    diagnosingText,
    failedText,
    isProposalPublished,
    isRoundDone,
    MOBILE_APPLY_TEXT,
    nChangeText,
    NO_BRIEF_TEXT,
    NO_KEY_TEXT,
    noChangesText,
    noRoundsText,
    offText,
    pausedText,
    proposingText,
    queuedText,
    readyAnnouncement,
    roundPhase,
    roundRowText,
    roundTitle,
    runNowText,
    selfImproveTabLabel,
    shortDate,
    statusLineText,
    waitingText,
} from "./format";
import type { FixCarryover, IterationResponse, SelfImproveStatus } from "./types";

// Midday UTC so the day is the same in every test timezone.
const OCT2 = "2026-10-02T12:00:00Z";

const iteration = (over: Partial<IterationResponse> = {}): IterationResponse => ({
    iteration_id: 1,
    number: 3,
    status: "ready",
    run_ids: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    reviewed: false,
    carryover: [],
    created_at: OCT2,
    ...over,
});

const status = (over: Partial<SelfImproveStatus> = {}): SelfImproveStatus => ({
    workflow_id: 7,
    settings: { enabled: true, every_n_calls: 10 },
    state: "collecting",
    collected: 7,
    window_size: 10,
    window_multiplier: 1,
    to_review: 0,
    has_brief: true,
    ...over,
});

const proposal = (over: Partial<Proposal> = {}): Proposal => ({
    proposal_uuid: "p1",
    session_uuid: "s1",
    kind: "edit",
    summary: "",
    status: "pending",
    changes: Array.from({ length: 5 }, (_, i) => ({
        change_id: `n:${i}`,
        target: "node" as const,
        op: "update" as const,
        ref_id: String(i),
        label: `Change ${i}`,
        field_diffs: [],
        depends_on: [],
    })),
    ...over,
});

const carry = (over: Partial<FixCarryover>): FixCarryover => ({
    proposal_uuid: "p1",
    change_id: "n:1",
    title: "Welcome step answers pricing",
    outcome: "fixed",
    eligible_calls: 9,
    affected_calls: 0,
    published_at: OCT2,
    ...over,
});

describe("DS-2 ready signal", () => {
    it("labels the Copilot button and the tab with the count only while a round waits", () => {
        expect(copilotButtonLabel(1)).toBe("Copilot · 1 to review");
        expect(copilotButtonLabel(0)).toBe("Copilot");
        expect(selfImproveTabLabel(1)).toBe("Self-improve (1)");
        expect(selfImproveTabLabel(0)).toBe("Self-improve");
        expect(readyAnnouncement(1)).toBe("1 round waiting for review");
        expect(readyAnnouncement(0)).toBe("");
    });
});

describe("DS-3 status line", () => {
    it("Off", () => {
        expect(offText(10)).toBe("Self-improve is off. Turn it on to review every 10 real calls automatically.");
        expect(statusLineText(status({ state: "off", settings: { enabled: false, every_n_calls: 10 } }))).toBe(
            "Self-improve is off. Turn it on to review every 10 real calls automatically.",
        );
    });

    it("No key", () => {
        expect(NO_KEY_TEXT).toBe("This server has no AI key.");
        expect(statusLineText(status({ state: "no_key" }))).toBe("This server has no AI key.");
    });

    it("AI not enabled has no sentence (AiDisabledNotice shows instead)", () => {
        expect(statusLineText(status({ state: "ai_disabled" }))).toBeNull();
    });

    it("Collecting", () => {
        expect(collectingText(10, 1, 7, 10)).toBe("On · every 10 calls · 7/10 collected");
        expect(statusLineText(status())).toBe("On · every 10 calls · 7/10 collected");
    });

    it("Collecting, backed off", () => {
        expect(statusLineText(status({ window_multiplier: 2, window_size: 20, collected: 3 }))).toBe(
            "On · every 20 calls (backed off) · 3/20 collected",
        );
    });

    it("Waiting", () => {
        expect(waitingText(3)).toBe("Waiting for your review. Next round starts after you apply or discard Round 3.");
        expect(statusLineText(status({ state: "waiting", to_review: 1, current_iteration: iteration() }))).toBe(
            "Waiting for your review. Next round starts after you apply or discard Round 3.",
        );
    });

    it("Paused", () => {
        expect(pausedText("the AI provider timed out")).toBe("Paused after 2 failed rounds: the AI provider timed out.");
        expect(pausedText("the AI provider timed out.")).toBe("Paused after 2 failed rounds: the AI provider timed out.");
        expect(pausedText(null)).toBe("Paused after 2 failed rounds.");
        expect(statusLineText(status({ state: "paused", paused_reason: "no credits" }))).toBe(
            "Paused after 2 failed rounds: no credits.",
        );
    });

    it("Running", () => {
        expect(statusLineText(status({ state: "running" }))).toBe("On · every 10 calls");
    });
});

describe("DS-3 rounds", () => {
    it("queued", () => {
        expect(queuedText(4)).toBe("Round 4 · starting");
    });

    it("diagnosing", () => {
        expect(diagnosingText(10)).toBe("Analysing 10 calls…");
        expect(diagnosingText(1)).toBe("Analysing 1 call…");
    });

    it("proposing", () => {
        expect(proposingText(2)).toBe("Writing fixes for 2 problems…");
        expect(proposingText(1)).toBe("Writing fixes for 1 problem…");
        expect(proposingText(undefined)).toBe("Writing fixes…");
    });

    it("ready header", () => {
        expect(roundTitle(3, "waiting for review")).toBe("Round 3 · waiting for review");
    });

    it("no_changes", () => {
        expect(noChangesText(10, 20)).toBe("No changes needed in 10 calls. Next check after 20 calls.");
    });

    it("failed", () => {
        expect(failedText("the analysis timed out", 10)).toBe(
            "Round failed: the analysis timed out. Next round runs after 10 more calls.",
        );
        expect(failedText(null, 10)).toBe(
            "Round failed: something went wrong on our side. Next round runs after 10 more calls.",
        );
    });

    it("applied", () => {
        expect(appliedRoundText(3, 5)).toBe("Applied 3 of 5 changes · draft");
        expect(appliedRoundText(1, 1, true)).toBe("Applied 1 of 1 change · published");
    });

    it("On, no rounds yet", () => {
        expect(noRoundsText(10)).toBe(
            "No rounds yet. The first one runs after 10 real calls (phone or web, where the caller speaks). Text-chat tests don't count.",
        );
    });

    it("derives the phase from the proposal for ready rounds", () => {
        expect(roundPhase(iteration({ status: "diagnosing" }))).toBe("diagnosing");
        expect(roundPhase(iteration(), proposal())).toBe("ready");
        expect(roundPhase(iteration(), proposal({ status: "applying" }))).toBe("applying");
        expect(roundPhase(iteration(), proposal({ status: "applied" }))).toBe("applied");
        expect(roundPhase(iteration())).toBe("ready");
        expect(isRoundDone(iteration(), proposal({ status: "discarded" }))).toBe(true);
        expect(isRoundDone(iteration(), proposal({ status: "applying" }))).toBe(false);
        expect(isRoundDone(iteration({ status: "no_changes" }))).toBe(true);
    });

    it("earlier rounds are one line with the time", () => {
        const applied = proposal({ status: "applied", accepted_change_ids: ["n:0", "n:1", "n:2"] });
        expect(roundRowText(iteration({ number: 2 }), applied)).toBe("Round 2 · Applied 3 of 5 changes · draft · 2 Oct");
        expect(roundRowText(iteration({ number: 2 }), proposal({ status: "applied" }))).toBe(
            "Round 2 · Applied · draft · 2 Oct",
        );
        expect(roundRowText(iteration({ number: 2 }), applied, true)).toBe(
            "Round 2 · Applied 3 of 5 changes · published · 2 Oct",
        );
        expect(roundRowText(iteration({ number: 2 }), proposal({ status: "discarded" }))).toBe(
            "Round 2 · Discarded · 2 Oct",
        );
        expect(roundRowText(iteration({ number: 2 }), proposal({ status: "stale" }))).toBe(
            "Round 2 · Out of date: the workflow changed before review · 2 Oct",
        );
        expect(roundRowText(iteration({ number: 1, status: "no_changes" }), null)).toBe(
            "Round 1 · No changes needed in 10 calls · 2 Oct",
        );
        expect(roundRowText(iteration({ number: 1, status: "failed", error: "timed out." }), null)).toBe(
            "Round 1 · Failed: timed out · 2 Oct",
        );
    });

    it("knows a proposal was published from a later round's carry-over", () => {
        const later = iteration({ carryover: [carry({})] });
        expect(isProposalPublished("p1", [later])).toBe(true);
        expect(isProposalPublished("p1", [iteration({ carryover: [carry({ outcome: "waiting_for_publish", published_at: null })] })])).toBe(false);
        expect(isProposalPublished("p2", [later])).toBe(false);
    });
});

describe("DS-4 since earlier fixes", () => {
    it("fixed", () => {
        expect(carryoverLine(carry({}))).toBe("✓ Welcome step answers pricing: 0 of 9 calls since the fix (published 2 Oct)");
    });

    it("still happening", () => {
        expect(
            carryoverLine(
                carry({ title: "Agrees a competitor is better", outcome: "still_happening", eligible_calls: 7, affected_calls: 3 }),
            ),
        ).toBe("✗ Agrees a competitor is better: still in 3 of 7 calls since the fix (published 2 Oct)");
    });

    it("not enough calls", () => {
        expect(
            carryoverLine(carry({ title: "Collect follow-up email", outcome: "insufficient", eligible_calls: 2 })),
        ).toBe("– Collect follow-up email: not enough calls yet (2 reached that step; need 3)");
    });

    it("waiting for publish", () => {
        expect(
            carryoverLine(carry({ title: "Goodbye wording", outcome: "waiting_for_publish", published_at: null })),
        ).toBe("– Goodbye wording: waiting for you to publish the draft");
    });

    it("never says Verified", () => {
        const all = (["fixed", "still_happening", "insufficient", "waiting_for_publish"] as const).map((outcome) =>
            carryoverLine(carry({ outcome })),
        );
        expect(all.join(" ").toLowerCase()).not.toContain("verified");
    });

    it("leaves out the date when unknown", () => {
        expect(carryoverLine(carry({ published_at: null }))).toBe("✓ Welcome step answers pricing: 0 of 9 calls since the fix");
        expect(shortDate("nope")).toBe("");
    });
});

describe("DS-10 changing N", () => {
    it("a smaller N starts a round now", () => {
        expect(nChangeText(7, 10, 5)).toBe("7 calls already counted. With N=5 a round starts now.");
    });

    it("a larger N keeps collecting", () => {
        expect(nChangeText(7, 10, 20)).toBe("7 calls already counted. With N=20 a round starts after 13 more calls.");
        expect(nChangeText(1, 10, 5)).toBe("1 call already counted. With N=5 a round starts after 4 more calls.");
    });

    it("applies the backoff multiplier", () => {
        expect(nChangeText(7, 10, 5, 2)).toBe("7 calls already counted. With N=5 a round starts after 3 more calls.");
    });

    it("says nothing when N is unchanged or nothing is counted", () => {
        expect(nChangeText(7, 10, 10)).toBeNull();
        expect(nChangeText(0, 10, 5)).toBeNull();
    });
});

describe("other copy", () => {
    it("DS-6 / DS-8 strings", () => {
        expect(NO_BRIEF_TEXT).toBe("Rounds follow your brief. You don't have one yet.");
        expect(MOBILE_APPLY_TEXT).toBe("Open on desktop to apply");
    });

    it("Run now results", () => {
        expect(runNowText({ started: true, iteration: iteration({ number: 4 }) })).toBe("Round 4 started.");
        expect(runNowText({ started: false, reason: "waiting_for_review" })).toBe("Review the waiting round first.");
        expect(runNowText({ started: false, reason: "no_calls" })).toBe("No new real calls to analyse yet.");
    });
});
