"use client";

import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Skeleton } from "@/components/ui/skeleton";
import { regenerateTurn } from "@/lib/doctor/api";
import type { RegenerateGroup, RegenerateResponse } from "@/lib/doctor/types";
import { REGENERATIONS_PER_SIDE } from "@/lib/doctor/types";

import { WARN } from "./ReportStates";

interface RegeneratePopoverProps {
    workflowId: number;
    runId: number;
    /** Zero-based index of the turn in the text-chat session. */
    turn: number;
    disabled?: boolean;
}

function GroupRow({ group, n }: { group: RegenerateGroup; n: number }) {
    const [open, setOpen] = useState(false);
    const others = group.members.filter((m) => m !== group.text);
    return (
        <li className="space-y-1 border-b pb-2 last:border-0 last:pb-0">
            <Collapsible open={open} onOpenChange={setOpen}>
                <CollapsibleTrigger asChild>
                    <button
                        type="button"
                        aria-expanded={open}
                        className="flex w-full items-start gap-2 rounded-sm text-left text-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                    >
                        <span className="shrink-0 font-medium tabular-nums">
                            {group.count}/{n}
                        </span>
                        <span className={open ? "whitespace-pre-wrap break-words" : "line-clamp-2 break-words"}>
                            {group.text}
                        </span>
                    </button>
                </CollapsibleTrigger>
                {group.tool_calls.length > 0 ? (
                    <div className="mt-1 flex flex-wrap gap-1 pl-10">
                        {group.tool_calls.map((t, i) => (
                            <span key={`${t}-${i}`} className="rounded border px-1.5 py-0.5 font-mono text-[11px]">
                                {t}
                            </span>
                        ))}
                    </div>
                ) : null}
                {group.flagged ? (
                    <p className="mt-1 pl-10 text-xs text-amber-700 dark:text-amber-400">
                        <span aria-hidden>{WARN} </span>
                        {group.flag_reason || "Flagged by the rubric"}
                    </p>
                ) : null}
                <CollapsibleContent className="mt-1 space-y-1 pl-10">
                    {others.map((m, i) => (
                        <p key={i} className="whitespace-pre-wrap break-words text-xs text-muted-foreground">
                            {m}
                        </p>
                    ))}
                </CollapsibleContent>
            </Collapsible>
        </li>
    );
}

/**
 * "Regenerate ×10 at this turn" (D13): replies grouped by frequency, sorted
 * descending, each group expandable to its raw text.
 */
export function RegeneratePopover({ workflowId, runId, turn, disabled }: RegeneratePopoverProps) {
    const [open, setOpen] = useState(false);
    const [result, setResult] = useState<RegenerateResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const run = async () => {
        setLoading(true);
        setError(null);
        try {
            setResult(await regenerateTurn(workflowId, { run_id: runId, turn, n: REGENERATIONS_PER_SIDE }));
        } catch (err) {
            setError(err instanceof Error ? err.message : "Could not regenerate this turn");
        } finally {
            setLoading(false);
        }
    };

    const onOpenChange = (next: boolean) => {
        setOpen(next);
        if (next && !result && !loading) void run();
    };

    const groups = result ? [...result.groups].sort((a, b) => b.count - a.count) : [];

    return (
        <Popover open={open} onOpenChange={onOpenChange}>
            <PopoverTrigger asChild>
                <button
                    type="button"
                    disabled={disabled}
                    aria-label="Regenerate this reply 10 times"
                    title="Regenerate ×10 at this turn"
                    className="inline-flex h-6 items-center justify-center rounded px-1.5 text-xs text-muted-foreground hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:opacity-50"
                >
                    ×{REGENERATIONS_PER_SIDE}
                </button>
            </PopoverTrigger>
            <PopoverContent align="start" className="w-96 space-y-3">
                <div className="flex items-center justify-between gap-2">
                    <h3 className="text-sm font-semibold">Regenerated ×{REGENERATIONS_PER_SIDE} at this turn</h3>
                    {result ? (
                        <Button variant="ghost" size="sm" onClick={() => void run()} disabled={loading}>
                            Run again
                        </Button>
                    ) : null}
                </div>
                <div aria-live="polite">
                    {loading ? (
                        <div className="space-y-2" aria-busy="true">
                            <p className="text-xs text-muted-foreground">Regenerating {REGENERATIONS_PER_SIDE} replies…</p>
                            <Skeleton className="h-4 w-full" />
                            <Skeleton className="h-4 w-5/6" />
                            <Skeleton className="h-4 w-2/3" />
                        </div>
                    ) : error ? (
                        <p className="text-sm text-destructive">{error}</p>
                    ) : result ? (
                        <ul className="max-h-80 space-y-2 overflow-y-auto">
                            {groups.map((g, i) => (
                                <GroupRow key={i} group={g} n={result.n} />
                            ))}
                        </ul>
                    ) : null}
                </div>
            </PopoverContent>
        </Popover>
    );
}
