"use client";

import Link from "next/link";

import { CHECK } from "@/components/doctor/ReportStates";
import { OP_LABEL } from "@/lib/copilot/format";
import type { Proposal } from "@/lib/copilot/types";
import { CARRYOVER_HEADING, carryoverParts, OPEN_GLYPH, reportHref } from "@/lib/self-improve/format";
import type { FixCarryover } from "@/lib/self-improve/types";

/** DS-4: one line per earlier accepted change, glyph inline, tabular numbers. */
export function CarryoverLines({ carryover }: { carryover: readonly FixCarryover[] }) {
    if (carryover.length === 0) return null;
    return (
        <div className="space-y-1">
            <h4 className="text-xs font-medium text-muted-foreground">{CARRYOVER_HEADING}</h4>
            <ul className="space-y-1 text-sm tabular-nums">
                {carryover.map((c) => {
                    const { glyph, text } = carryoverParts(c);
                    return (
                        <li key={`${c.proposal_uuid}:${c.change_id}`} className="break-words">
                            <span aria-hidden>{glyph} </span>
                            {text}
                        </li>
                    );
                })}
            </ul>
        </div>
    );
}

/**
 * Read-only change titles (mobile review, earlier rounds). For an applied
 * proposal each line says whether that change was applied.
 */
export function ChangeTitles({ proposal }: { proposal: Proposal }) {
    const applied = proposal.status === "applied" ? new Set(proposal.accepted_change_ids ?? []) : null;
    return (
        <ul className="space-y-1 text-sm">
            {proposal.changes.map((change) => {
                const isApplied = applied?.has(change.change_id);
                return (
                    <li key={change.change_id} className="break-words">
                        {applied ? <span aria-hidden>{isApplied ? CHECK : OPEN_GLYPH} </span> : null}
                        <span className="text-muted-foreground">{OP_LABEL[change.op]}: </span>
                        {change.label}
                        {applied ? <span className="sr-only">{isApplied ? " (applied)" : " (not applied)"}</span> : null}
                    </li>
                );
            })}
        </ul>
    );
}

export function ReportLink({
    workflowId,
    reportUuid,
    live = false,
}: {
    workflowId: number;
    reportUuid: string | null | undefined;
    live?: boolean;
}) {
    if (!reportUuid) return null;
    return (
        <Link
            href={reportHref(workflowId, reportUuid)}
            className="inline-flex min-h-11 items-center text-sm underline underline-offset-2 md:min-h-0"
        >
            {live ? "Watch the live Doctor report" : "Open the Doctor report"}
        </Link>
    );
}
