/**
 * Every user-facing string of the Self-improve tab (design spec DS-2, DS-3,
 * DS-4, DS-6, DS-8, DS-10), as pure functions so each one is unit-tested.
 * Plain words, tabular numbers, glyphs inline with text, never "Verified".
 */
import type { Proposal } from "@/lib/copilot/types";

import {
    FAILURES_BEFORE_PAUSE,
    type FixCarryover,
    type IterationResponse,
    MIN_CALLS_FOR_EVIDENCE,
    type RunNowResponse,
    type SelfImproveStatus,
} from "./types";

export const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/** Glyphs allowed by 5A, inline with text. */
export const FIXED_GLYPH = "✓";
export const STILL_GLYPH = "✗";
export const OPEN_GLYPH = "–";

/** "2 Oct" (DS-4 "published 2 Oct"); "" for a missing or bad date. */
export function shortDate(iso: string | null | undefined): string {
    const t = iso ? Date.parse(iso) : NaN;
    if (Number.isNaN(t)) return "";
    return new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short" }).format(new Date(t));
}

// ---------------------------------------------------------------------------
// DS-2 ready signal
// ---------------------------------------------------------------------------

/** Editor header button: "Copilot · 1 to review" while a round waits. */
export const copilotButtonLabel = (toReview: number) => (toReview > 0 ? `Copilot · ${toReview} to review` : "Copilot");

/** Panel tab: "Self-improve (1)" while a round waits. */
export const selfImproveTabLabel = (toReview: number) => (toReview > 0 ? `Self-improve (${toReview})` : "Self-improve");

/** Screen-reader announcement when the ready count changes. */
export const readyAnnouncement = (toReview: number) =>
    toReview > 0 ? `${plural(toReview, "round")} waiting for review` : "";

// ---------------------------------------------------------------------------
// DS-3 status line
// ---------------------------------------------------------------------------

export const offText = (everyN: number) =>
    `Self-improve is off. Turn it on to review every ${everyN} real calls automatically.`;

export const NO_KEY_TEXT = "This server has no AI key.";
export const NO_KEY_DETAIL = "Ask your administrator to set ANTHROPIC_API_KEY on the server.";
export const ENABLE_AI_LABEL = "Enable AI analysis";

/** "every 10 calls", or "every 20 calls (backed off)" after empty rounds (D6). */
export const everyText = (everyN: number, multiplier: number) =>
    multiplier > 1 ? `every ${everyN * multiplier} calls (backed off)` : `every ${everyN} calls`;

/** "On · every 10 calls · 7/10 collected". */
export const collectingText = (everyN: number, multiplier: number, collected: number, windowSize: number) =>
    `On · ${everyText(everyN, multiplier)} · ${collected}/${windowSize} collected`;

/** A round is queued/diagnosing/proposing; the round itself says how far it got. */
export const runningText = (everyN: number, multiplier: number) => `On · ${everyText(everyN, multiplier)}`;

export const waitingText = (roundNumber: number) =>
    `Waiting for your review. Next round starts after you apply or discard Round ${roundNumber}.`;

const stripEnd = (s: string) => s.trim().replace(/[.\s]+$/, "");

/** "Paused after 2 failed rounds: <reason>." */
export function pausedText(reason: string | null | undefined, failures = FAILURES_BEFORE_PAUSE): string {
    const why = reason ? stripEnd(reason) : "";
    return why ? `Paused after ${failures} failed rounds: ${why}.` : `Paused after ${failures} failed rounds.`;
}

/** The status line's sentence for each state; null for the AI-gated states (they show AiDisabledNotice). */
export function statusLineText(status: SelfImproveStatus): string | null {
    const { settings, window_multiplier: mult } = status;
    switch (status.state) {
        case "off":
            return offText(settings.every_n_calls);
        case "collecting":
            return collectingText(settings.every_n_calls, mult, status.collected, status.window_size);
        case "running":
            return runningText(settings.every_n_calls, mult);
        case "waiting":
            return waitingText(status.current_iteration?.number ?? 0);
        case "paused":
            return pausedText(status.paused_reason);
        case "no_key":
            return NO_KEY_TEXT;
        case "ai_disabled":
            return null;
    }
}

/** Tab, Self-improve on and no rounds yet. */
export const noRoundsText = (windowSize: number) =>
    `No rounds yet. The first one runs after ${windowSize} real calls (phone or web, where the caller speaks). Text-chat tests don't count.`;

// ---------------------------------------------------------------------------
// DS-10 changing N mid-window
// ---------------------------------------------------------------------------

/**
 * Shown in the gear popover before Save. Counted calls are kept and the new
 * N applies at once: "7 calls already counted. With N=5 a round starts now."
 * Null when nothing would change (same N, or nothing counted yet).
 */
export function nChangeText(collected: number, currentN: number, newN: number, multiplier = 1): string | null {
    if (newN === currentN || collected <= 0) return null;
    const window = newN * Math.max(1, multiplier);
    const counted = `${plural(collected, "call")} already counted.`;
    if (collected >= window) return `${counted} With N=${newN} a round starts now.`;
    return `${counted} With N=${newN} a round starts after ${plural(window - collected, "more call", "more calls")}.`;
}

export function runNowText(result: RunNowResponse): string {
    if (result.started) return result.iteration ? `Round ${result.iteration.number} started.` : "Round started.";
    switch (result.reason) {
        case "waiting_for_review":
            return "Review the waiting round first.";
        case "running":
            return "A round is already running.";
        case "no_calls":
            return "No new real calls to analyse yet.";
        case "ai_disabled":
            return "Turn on AI analysis first.";
        case "no_key":
            return NO_KEY_TEXT;
        case "off":
            return "Turn Self-improve on first.";
        case "paused":
            return "Self-improve is paused. Resume it first.";
        default:
            return "Could not start a round.";
    }
}

// ---------------------------------------------------------------------------
// DS-3 rounds
// ---------------------------------------------------------------------------

/** What a round looks like to the user: its status, refined by its proposal's status. */
export type RoundPhase =
    | "queued"
    | "diagnosing"
    | "proposing"
    | "ready"
    | "applying"
    | "applied"
    | "discarded"
    | "stale"
    | "no_changes"
    | "failed";

export function roundPhase(iteration: IterationResponse, proposal?: Proposal | null): RoundPhase {
    if (iteration.status !== "ready") return iteration.status;
    switch (proposal?.status) {
        case "applying":
            return "applying";
        case "applied":
            return "applied";
        case "discarded":
            return "discarded";
        case "stale":
            return "stale";
        default:
            return "ready";
    }
}

/** The round has reached a final, reviewed state (D4). */
export const isRoundDone = (iteration: IterationResponse, proposal?: Proposal | null) => {
    const phase = roundPhase(iteration, proposal);
    return phase === "applied" || phase === "discarded" || phase === "stale" || phase === "no_changes" || phase === "failed";
};

export const roundTitle = (number: number, label: string) => `Round ${number} · ${label}`;

/** Round, queued: "Round 4 · starting". */
export const queuedText = (number: number) => roundTitle(number, "starting");

/** Round, diagnosing: "Analysing 10 calls…". */
export const diagnosingText = (calls: number) => `Analysing ${plural(calls, "call")}…`;

/** Round, proposing: "Writing fixes for 2 problems…". */
export const proposingText = (problems: number | null | undefined) =>
    problems && problems > 0 ? `Writing fixes for ${plural(problems, "problem")}…` : "Writing fixes…";

/** Round, no_changes: "No changes needed in 10 calls. Next check after 20 calls." */
export const noChangesText = (calls: number, nextWindow: number) =>
    `No changes needed in ${plural(calls, "call")}. Next check after ${plural(nextWindow, "call")}.`;

/** Round, failed: "Round failed: <plain reason>. Next round runs after 10 more calls." */
export function failedText(reason: string | null | undefined, nextWindow: number): string {
    const why = reason ? stripEnd(reason) : "something went wrong on our side";
    return `Round failed: ${why}. Next round runs after ${plural(nextWindow, "more call", "more calls")}.`;
}

/** Round, applied: "Applied 3 of 5 changes · draft". */
export function appliedRoundText(accepted: number, total: number, published = false): string {
    const where = published ? "published" : "draft";
    if (total <= 0) return `Applied · ${where}`;
    return `Applied ${accepted} of ${plural(total, "change")} · ${where}`;
}

export const DISCARDED_ROUND_TEXT = "Discarded";
export const STALE_ROUND_TEXT = "Out of date: the workflow changed before review";

/** Expanded round header suffix (DS-1 "Round 3 · waiting for review"). */
export const ROUND_PHASE_LABEL: Record<RoundPhase, string> = {
    queued: "starting",
    diagnosing: "analysing calls",
    proposing: "writing fixes",
    ready: "waiting for review",
    applying: "applying changes",
    applied: "applied",
    discarded: "discarded",
    stale: "out of date",
    no_changes: "no changes needed",
    failed: "failed",
};

/** A proposal counts as published once a later round's carry-over saw it published. */
export function isProposalPublished(proposalUuid: string, iterations: readonly IterationResponse[]): boolean {
    return iterations.some((it) =>
        it.carryover.some(
            (c) => c.proposal_uuid === proposalUuid && c.outcome !== "waiting_for_publish" && !!c.published_at,
        ),
    );
}

/**
 * Collapsed row for an earlier round (DS-3 "one line + time"):
 * "Round 2 · Applied 3 of 5 changes · draft · 2 Oct".
 */
export function roundRowText(
    iteration: IterationResponse,
    proposal: Proposal | null | undefined,
    published = false,
): string {
    const phase = roundPhase(iteration, proposal);
    let line: string;
    switch (phase) {
        case "applied": {
            // Without accepted_change_ids (older rows, or not refetched yet) the count is unknown.
            const accepted = proposal?.accepted_change_ids?.length ?? 0;
            line = appliedRoundText(accepted, accepted > 0 ? (proposal?.changes.length ?? 0) : 0, published);
            break;
        }
        case "discarded":
            line = DISCARDED_ROUND_TEXT;
            break;
        case "stale":
            line = STALE_ROUND_TEXT;
            break;
        case "no_changes":
            line = `No changes needed in ${plural(iteration.run_ids.length, "call")}`;
            break;
        case "failed":
            line = iteration.error ? `Failed: ${stripEnd(iteration.error)}` : "Failed";
            break;
        default:
            line = ROUND_PHASE_LABEL[phase];
    }
    const when = shortDate(iteration.updated_at ?? iteration.created_at);
    return when ? `Round ${iteration.number} · ${line} · ${when}` : `Round ${iteration.number} · ${line}`;
}

// ---------------------------------------------------------------------------
// DS-4 "Since earlier fixes"
// ---------------------------------------------------------------------------

export const CARRYOVER_HEADING = "Since earlier fixes";

export interface CarryoverParts {
    glyph: string;
    text: string;
}

/**
 * One line per accepted change; the glyph is shown inline before the text:
 * ✓ Welcome step answers pricing: 0 of 9 calls since the fix (published 2 Oct)
 * ✗ Agrees a competitor is better: still in 3 of 7 calls since the fix (published 2 Oct)
 * – Collect follow-up email: not enough calls yet (2 reached that step; need 3)
 * – Goodbye wording: waiting for you to publish the draft
 */
export function carryoverParts(c: FixCarryover): CarryoverParts {
    const date = shortDate(c.published_at);
    const published = date ? ` (published ${date})` : "";
    switch (c.outcome) {
        case "fixed":
            return {
                glyph: FIXED_GLYPH,
                text: `${c.title}: ${c.affected_calls} of ${plural(c.eligible_calls, "call")} since the fix${published}`,
            };
        case "still_happening":
            return {
                glyph: STILL_GLYPH,
                text: `${c.title}: still in ${c.affected_calls} of ${plural(c.eligible_calls, "call")} since the fix${published}`,
            };
        case "insufficient":
            return {
                glyph: OPEN_GLYPH,
                text: `${c.title}: not enough calls yet (${c.eligible_calls} reached that step; need ${MIN_CALLS_FOR_EVIDENCE})`,
            };
        case "waiting_for_publish":
            return { glyph: OPEN_GLYPH, text: `${c.title}: waiting for you to publish the draft` };
    }
}

export const carryoverLine = (c: FixCarryover) => {
    const { glyph, text } = carryoverParts(c);
    return `${glyph} ${text}`;
};

// ---------------------------------------------------------------------------
// DS-6 no brief, DS-8 mobile, links
// ---------------------------------------------------------------------------

export const NO_BRIEF_TEXT = "Rounds follow your brief. You don't have one yet.";
export const DRAFT_BRIEF_LABEL = "Draft from my chats";
export const WRITE_BRIEF_LABEL = "Write it myself";
export const SKIP_BRIEF_LABEL = "Skip";

export const MOBILE_APPLY_TEXT = "Open on desktop to apply";
export const REVIEW_CHANGES_LABEL = "Review changes";
export const DISCARD_ROUND_LABEL = "Discard round";

export const reportHref = (workflowId: number, reportUuid: string) =>
    `/workflow/${workflowId}/doctor/${encodeURIComponent(reportUuid)}`;
