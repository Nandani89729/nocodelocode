"use client";

import { type RefObject, useEffect, useRef, useState } from "react";

import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import type { Proposal } from "@/lib/copilot/types";
import { roundRowText } from "@/lib/self-improve/format";
import type { IterationResponse } from "@/lib/self-improve/types";

import { CarryoverLines, ChangeTitles, ReportLink } from "./RoundParts";

export interface RoundRowProps {
    workflowId: number;
    iteration: IterationResponse;
    proposal: Proposal | null | undefined;
    published?: boolean;
    /** Take focus once (after this round was applied/discarded), then report it. */
    focusRequested?: boolean;
    onFocused?: () => void;
}

/** An earlier round as a plain collapsible row, not a card (DS-7). */
export function RoundRow({ workflowId, iteration, proposal, published = false, focusRequested, onFocused }: RoundRowProps) {
    const [open, setOpen] = useState(false);
    const triggerRef = useRef<HTMLElement | null>(null);

    useEffect(() => {
        if (!focusRequested) return;
        triggerRef.current?.focus();
        onFocused?.();
    }, [focusRequested, onFocused]);

    const hasDetail =
        iteration.carryover.length > 0 || !!iteration.summary || !!iteration.report_uuid || (proposal?.changes.length ?? 0) > 0;
    const text = roundRowText(iteration, proposal, published);

    if (!hasDetail) {
        return (
            <li>
                <p
                    ref={triggerRef as RefObject<HTMLParagraphElement | null>}
                    tabIndex={-1}
                    className="flex min-h-11 items-center py-2 text-sm tabular-nums outline-none focus-visible:ring-2 focus-visible:ring-ring md:min-h-9"
                >
                    {text}
                </p>
            </li>
        );
    }

    return (
        <li>
            <Collapsible open={open} onOpenChange={setOpen}>
                <CollapsibleTrigger asChild>
                    <button
                        ref={triggerRef as RefObject<HTMLButtonElement | null>}
                        type="button"
                        className="flex min-h-11 w-full items-center gap-2 py-2 text-left text-sm tabular-nums outline-none hover:bg-muted/40 focus-visible:ring-2 focus-visible:ring-ring md:min-h-9"
                    >
                        <span className="min-w-0 flex-1 break-words">{text}</span>
                        <span aria-hidden className="shrink-0 text-xs text-muted-foreground">
                            {open ? "Hide" : "Show"}
                        </span>
                    </button>
                </CollapsibleTrigger>
                <CollapsibleContent className="space-y-3 pb-3">
                    {iteration.summary ? (
                        <p className="whitespace-pre-wrap break-words text-sm text-muted-foreground">{iteration.summary}</p>
                    ) : null}
                    <CarryoverLines carryover={iteration.carryover} />
                    {proposal && proposal.changes.length > 0 ? <ChangeTitles proposal={proposal} /> : null}
                    <ReportLink workflowId={workflowId} reportUuid={iteration.report_uuid} />
                </CollapsibleContent>
            </Collapsible>
        </li>
    );
}
