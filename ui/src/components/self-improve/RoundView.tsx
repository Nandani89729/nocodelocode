"use client";

import { useEffect, useId, useRef, useState } from "react";

import { ProposalCard } from "@/components/copilot/ProposalCard";
import { CROSS } from "@/components/doctor/ReportStates";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { discardProposal } from "@/lib/copilot/api";
import type { ApplyProposalResponse, Proposal } from "@/lib/copilot/types";
import {
    diagnosingText,
    DISCARD_ROUND_LABEL,
    failedText,
    MOBILE_APPLY_TEXT,
    noChangesText,
    pausedText,
    proposingText,
    queuedText,
    REVIEW_CHANGES_LABEL,
    ROUND_PHASE_LABEL,
    roundPhase,
    roundTitle,
} from "@/lib/self-improve/format";
import type { IterationResponse, SelfImproveStatus } from "@/lib/self-improve/types";

import { CarryoverLines, ChangeTitles, ReportLink } from "./RoundParts";

export interface RoundViewProps {
    workflowId: number;
    iteration: IterationResponse;
    status: SelfImproveStatus;
    /** undefined = not loaded yet */
    proposal: Proposal | null | undefined;
    proposalError?: string | null;
    onRetryProposal?: () => void;
    onProposalChange: (proposal: Proposal) => void;
    /** Below 768px (DS-8): read the round and discard it; apply on desktop. */
    mobile?: boolean;
    draftVersion?: number | string | null;
    hasUnsavedChanges?: boolean;
    onApplied?: (proposal: Proposal, result: ApplyProposalResponse) => void | Promise<void>;
    onReloadCanvas?: () => void;
    /** Move focus to this round's heading (after apply/discard). */
    focusRequested?: boolean;
}

function SkeletonLines() {
    return (
        <div className="space-y-2" aria-hidden>
            <Skeleton className="h-4 w-2/3" />
            <Skeleton className="h-4 w-1/2" />
        </div>
    );
}

function MobileReview({
    proposal,
    onProposalChange,
}: {
    proposal: Proposal;
    onProposalChange: (proposal: Proposal) => void;
}) {
    const [discarding, setDiscarding] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const noteId = useId();
    const pending = proposal.status === "pending";

    const discard = async () => {
        setDiscarding(true);
        setError(null);
        try {
            onProposalChange(await discardProposal(proposal.proposal_uuid));
        } catch (e) {
            setError(e instanceof Error ? e.message : "Could not discard the round");
        } finally {
            setDiscarding(false);
        }
    };

    return (
        <div className="space-y-3">
            {proposal.summary ? <p className="whitespace-pre-wrap break-words text-sm">{proposal.summary}</p> : null}
            <ChangeTitles proposal={proposal} />
            <div className="flex flex-wrap items-center gap-2">
                <Button variant="outline" className="min-h-11" disabled aria-describedby={noteId}>
                    {REVIEW_CHANGES_LABEL}
                </Button>
                <span id={noteId} className="text-sm text-muted-foreground">
                    {MOBILE_APPLY_TEXT}
                </span>
            </div>
            <Button
                variant="outline"
                className="min-h-11"
                onClick={discard}
                disabled={!pending || discarding}
            >
                {discarding ? "Discarding…" : DISCARD_ROUND_LABEL}
            </Button>
            {error ? (
                <p role="alert" className="text-sm text-destructive">
                    <span aria-hidden>{CROSS} </span>
                    {error}
                </p>
            ) : null}
        </div>
    );
}

/**
 * The round that needs the user, expanded (DS-1, 2nd): its state line from
 * the DS-3 table, the "Since earlier fixes" lines, the reused ProposalCard
 * and a link to the round's Doctor report.
 */
export function RoundView({
    workflowId,
    iteration,
    status,
    proposal,
    proposalError,
    onRetryProposal,
    onProposalChange,
    mobile = false,
    draftVersion,
    hasUnsavedChanges,
    onApplied,
    onReloadCanvas,
    focusRequested = false,
}: RoundViewProps) {
    const headingRef = useRef<HTMLHeadingElement>(null);
    const headingId = useId();
    const phase = roundPhase(iteration, proposal);
    const calls = iteration.run_ids.length;
    const remaining = status.window_size - status.collected;

    useEffect(() => {
        if (focusRequested) headingRef.current?.focus();
    }, [focusRequested]);

    const title = phase === "queued" ? queuedText(iteration.number) : roundTitle(iteration.number, ROUND_PHASE_LABEL[phase]);

    let line: string | null = null;
    switch (phase) {
        case "diagnosing":
            line = diagnosingText(calls);
            break;
        case "proposing":
            line = proposingText(iteration.problem_count);
            break;
        case "no_changes":
            line = noChangesText(calls, status.window_size);
            break;
        case "failed":
            line =
                status.state === "paused"
                    ? pausedText(status.paused_reason)
                    : failedText(iteration.error, remaining > 0 ? remaining : status.window_size);
            break;
        default:
            line = null;
    }

    const showProposal = phase === "ready" || phase === "applying";

    return (
        <section aria-labelledby={headingId} className="space-y-3 border-b px-4 py-3">
            <div aria-live="polite" className="space-y-1">
                <h3
                    id={headingId}
                    ref={headingRef}
                    tabIndex={-1}
                    className="text-sm font-medium tabular-nums outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                    {title}
                </h3>
                {line ? (
                    <p className={phase === "failed" ? "text-sm text-destructive tabular-nums" : "text-sm tabular-nums"}>
                        {phase === "failed" ? <span aria-hidden>{CROSS} </span> : null}
                        {line}
                    </p>
                ) : null}
            </div>

            {phase === "queued" || phase === "proposing" ? <SkeletonLines /> : null}

            {phase !== "queued" ? <CarryoverLines carryover={iteration.carryover} /> : null}

            {showProposal ? (
                proposal ? (
                    mobile ? (
                        <MobileReview proposal={proposal} onProposalChange={onProposalChange} />
                    ) : (
                        <ProposalCard
                            proposal={proposal}
                            draftVersion={draftVersion}
                            hasUnsavedChanges={hasUnsavedChanges}
                            onProposalChange={onProposalChange}
                            onApplied={onApplied}
                            onReloadCanvas={onReloadCanvas}
                        />
                    )
                ) : proposalError ? (
                    <div className="flex flex-wrap items-center gap-2 text-sm">
                        <p className="text-destructive">
                            <span aria-hidden>{CROSS} </span>
                            {proposalError}
                        </p>
                        {onRetryProposal ? (
                            <Button variant="outline" size="sm" className="min-h-11 md:min-h-8" onClick={onRetryProposal}>
                                Retry
                            </Button>
                        ) : null}
                    </div>
                ) : (
                    <div className="space-y-2" aria-hidden>
                        <Skeleton className="h-4 w-1/2" />
                        <Skeleton className="h-20 w-full" />
                    </div>
                )
            ) : null}

            {phase !== "queued" ? (
                <ReportLink workflowId={workflowId} reportUuid={iteration.report_uuid} live={phase === "diagnosing"} />
            ) : null}
        </section>
    );
}
