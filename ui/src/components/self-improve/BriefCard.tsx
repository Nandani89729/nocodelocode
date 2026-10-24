"use client";

import { useId, useState } from "react";

import { BriefChangeCard } from "@/components/copilot/BriefChangeCard";
import { BriefEditor } from "@/components/copilot/BriefEditor";
import { CROSS } from "@/components/doctor/ReportStates";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Skeleton } from "@/components/ui/skeleton";
import { BRIEF_FIELD_LABEL, BRIEF_FIELDS, briefGoalLine, coerceBrief, isBriefEmpty, isListField } from "@/lib/copilot/brief";
import type { BriefResponse } from "@/lib/copilot/types";

export interface BriefCardProps {
    workflowId: number;
    brief: BriefResponse | null;
    loading: boolean;
    error?: string | null;
    onRetry?: () => void;
    onBriefChange: (brief: BriefResponse) => void;
    editing: boolean;
    onEditingChange: (editing: boolean) => void;
}

/**
 * The brief, last in the tab and collapsed to its goal (DS-1, 4th). Opens
 * into the field editor (DS-9); a copilot-proposed change waits above it as
 * a diff card with Confirm / Dismiss.
 */
export function BriefCard({
    workflowId,
    brief,
    loading,
    error,
    onRetry,
    onBriefChange,
    editing,
    onEditingChange,
}: BriefCardProps) {
    const [open, setOpen] = useState(false);
    const headingId = useId();
    const pending = brief?.pending_change;
    const showChange = !!brief && !!pending && (pending.status === "pending" || pending.status === "stale");
    const empty = isBriefEmpty(brief?.brief);
    const fields = coerceBrief(brief?.brief);

    let body: React.ReactNode;
    if (loading && !brief) {
        body = <Skeleton className="h-5 w-2/3" aria-hidden />;
    } else if (error && !brief) {
        body = (
            <div className="flex flex-wrap items-center gap-2 text-sm">
                <p className="text-destructive">
                    <span aria-hidden>{CROSS} </span>
                    {error}
                </p>
                {onRetry ? (
                    <Button variant="outline" size="sm" className="min-h-11 md:min-h-8" onClick={onRetry}>
                        Retry
                    </Button>
                ) : null}
            </div>
        );
    } else if (editing) {
        body = (
            <BriefEditor
                workflowId={workflowId}
                brief={brief}
                onSaved={(saved) => {
                    onBriefChange(saved);
                    onEditingChange(false);
                }}
                onCancel={() => onEditingChange(false)}
            />
        );
    } else {
        body = (
            <Collapsible open={open && !empty} onOpenChange={setOpen}>
                <div className="flex items-center gap-2">
                    {empty ? (
                        <p className="min-w-0 flex-1 text-sm">
                            <span className="font-medium">Brief: </span>
                            <span className="text-muted-foreground">none yet</span>
                        </p>
                    ) : (
                        <CollapsibleTrigger asChild>
                            <button
                                type="button"
                                className="flex min-h-11 min-w-0 flex-1 items-center gap-2 text-left text-sm outline-none hover:bg-muted/40 focus-visible:ring-2 focus-visible:ring-ring md:min-h-8"
                            >
                                <span className="min-w-0 flex-1 truncate">
                                    <span className="font-medium">Brief: </span>
                                    {briefGoalLine(brief?.brief) || "no goal yet"}
                                </span>
                                <span aria-hidden className="shrink-0 text-xs text-muted-foreground">
                                    {open ? "Hide" : "Show"}
                                </span>
                            </button>
                        </CollapsibleTrigger>
                    )}
                    <Button
                        variant="ghost"
                        size="sm"
                        className="min-h-11 shrink-0 md:min-h-8"
                        onClick={() => onEditingChange(true)}
                    >
                        {empty ? "Write" : "Edit"}
                    </Button>
                </div>
                <CollapsibleContent>
                    <dl className="space-y-2 pt-2 text-sm">
                        {BRIEF_FIELDS.map((field) => {
                            const value = fields[field];
                            const isEmpty = Array.isArray(value) ? value.length === 0 : value.trim() === "";
                            return (
                                <div key={field}>
                                    <dt className="text-xs font-medium text-muted-foreground">{BRIEF_FIELD_LABEL[field]}</dt>
                                    <dd className="break-words">
                                        {isEmpty ? (
                                            <span className="text-muted-foreground">None</span>
                                        ) : isListField(field) ? (
                                            <ul className="list-disc pl-5">
                                                {(value as string[]).map((item, i) => (
                                                    <li key={i}>{item}</li>
                                                ))}
                                            </ul>
                                        ) : (
                                            <span className="whitespace-pre-wrap">{value as string}</span>
                                        )}
                                    </dd>
                                </div>
                            );
                        })}
                    </dl>
                </CollapsibleContent>
            </Collapsible>
        );
    }

    return (
        <section aria-labelledby={headingId} className="space-y-3 px-4 py-3">
            <h3 id={headingId} className="sr-only">
                Brief
            </h3>
            {showChange && brief && pending ? (
                <BriefChangeCard
                    workflowId={workflowId}
                    change={pending}
                    current={brief}
                    onChange={(change) => onBriefChange({ ...brief, pending_change: change })}
                    onResolved={onBriefChange}
                />
            ) : null}
            {body}
        </section>
    );
}
