"use client";

import { useEffect, useMemo, useState } from "react";

import { CHECK, CROSS, WARN } from "@/components/doctor/ReportStates";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { confirmBriefChange, dismissBriefChange, getBrief, isBriefConflict } from "@/lib/copilot/api";
import { BRIEF_CHANGE_STALE_TEXT, briefFieldDiffs } from "@/lib/copilot/brief";
import type { BriefChange, BriefResponse } from "@/lib/copilot/types";

import { FieldDiffView } from "./FieldDiffView";

export interface BriefChangeCardProps {
    workflowId: number;
    change: BriefChange;
    /**
     * The brief as it is now, to diff against. When omitted the card loads it
     * (chat timeline); the Self-improve tab passes the one it already has.
     */
    current?: BriefResponse | null;
    /** The change's status moved (confirmed / dismissed / stale). */
    onChange?: (change: BriefChange) => void;
    /** The server returned the brief after confirm / dismiss. */
    onResolved?: (brief: BriefResponse) => void;
}

type Phase = "idle" | "confirming" | "dismissing";

/**
 * A copilot-proposed brief update (O8, DS-9): per-field diffs against the
 * current brief, Confirm / Dismiss. Never applied without the user; a change
 * made against an older brief version is shown as out of date.
 */
export function BriefChangeCard({ workflowId, change, current: given, onChange, onResolved }: BriefChangeCardProps) {
    const [loaded, setLoaded] = useState<BriefResponse | null>(null);
    const [loadError, setLoadError] = useState<string | null>(null);
    const [phase, setPhase] = useState<Phase>("idle");
    const [error, setError] = useState<string | null>(null);
    const current = given !== undefined ? given : loaded;
    const needsLoad = given === undefined && change.status === "pending";

    useEffect(() => {
        if (!needsLoad) return;
        let cancelled = false;
        getBrief(workflowId)
            .then((b) => {
                if (!cancelled) setLoaded(b);
            })
            .catch((e: unknown) => {
                if (!cancelled) setLoadError(e instanceof Error ? e.message : "Could not load the brief");
            });
        return () => {
            cancelled = true;
        };
    }, [needsLoad, workflowId]);

    const versionMoved = !!current && current.version !== change.base_version;
    const stale = change.status === "stale" || (change.status === "pending" && versionMoved);
    const actionable = change.status === "pending" && phase === "idle";
    const diffs = useMemo(() => briefFieldDiffs(current?.brief ?? null, change.proposed), [current, change.proposed]);
    const waitingForCurrent = change.status === "pending" && !current && !loadError;

    const confirm = async () => {
        setPhase("confirming");
        setError(null);
        try {
            const brief = await confirmBriefChange(workflowId, change.change_uuid);
            onChange?.({ ...change, status: "confirmed" });
            onResolved?.(brief);
        } catch (e) {
            if (isBriefConflict(e)) onChange?.({ ...change, status: "stale" });
            else setError(e instanceof Error ? e.message : "Could not update the brief");
        } finally {
            setPhase("idle");
        }
    };

    const dismiss = async () => {
        setPhase("dismissing");
        setError(null);
        try {
            const brief = await dismissBriefChange(workflowId, change.change_uuid);
            onChange?.({ ...change, status: "dismissed" });
            onResolved?.(brief);
        } catch (e) {
            setError(e instanceof Error ? e.message : "Could not dismiss the brief change");
        } finally {
            setPhase("idle");
        }
    };

    return (
        <section aria-label="Proposed brief change" className="space-y-3 rounded-lg border bg-background p-3">
            <div className="space-y-1">
                <h3 className="text-sm font-medium">Brief change</h3>
                {change.reason ? <p className="whitespace-pre-wrap break-words text-sm">{change.reason}</p> : null}
            </div>

            {waitingForCurrent ? (
                <div className="space-y-2" aria-hidden>
                    <Skeleton className="h-4 w-1/3" />
                    <Skeleton className="h-16 w-full" />
                </div>
            ) : diffs.length > 0 ? (
                <div className="space-y-3">
                    {diffs.map((diff) => (
                        <FieldDiffView key={diff.field} diff={diff} />
                    ))}
                </div>
            ) : (
                <p className="text-sm text-muted-foreground">Same as the current brief.</p>
            )}

            <div aria-live="polite" className="space-y-1 text-sm">
                {change.status === "confirmed" ? (
                    <p>
                        <span aria-hidden>{CHECK} </span>Brief updated.
                    </p>
                ) : null}
                {change.status === "dismissed" ? <p className="text-muted-foreground">Dismissed.</p> : null}
                {stale && change.status !== "confirmed" && change.status !== "dismissed" ? (
                    <p>
                        <span aria-hidden>{WARN} </span>
                        {BRIEF_CHANGE_STALE_TEXT}
                    </p>
                ) : null}
                {loadError ? (
                    <p className="text-destructive">
                        <span aria-hidden>{CROSS} </span>
                        {loadError}
                    </p>
                ) : null}
                {error ? (
                    <p className="text-destructive">
                        <span aria-hidden>{CROSS} </span>
                        {error}
                    </p>
                ) : null}
            </div>

            {change.status === "pending" ? (
                <div className="flex flex-wrap gap-2">
                    {!stale ? (
                        <Button
                            className="min-h-11 md:min-h-9"
                            onClick={confirm}
                            disabled={!actionable || waitingForCurrent}
                        >
                            {phase === "confirming" ? "Confirming…" : "Confirm"}
                        </Button>
                    ) : null}
                    <Button variant="outline" className="min-h-11 md:min-h-9" onClick={dismiss} disabled={!actionable}>
                        {phase === "dismissing" ? "Dismissing…" : "Dismiss"}
                    </Button>
                </div>
            ) : null}
        </section>
    );
}
