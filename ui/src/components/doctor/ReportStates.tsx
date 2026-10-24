"use client";

import { useEffect, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { setDoctorAiOptIn } from "@/lib/doctor/api";
import {
    formatElapsed,
    plural,
    stageEntries,
    stageProgressText,
} from "@/lib/doctor/format";
import type { ReportResponse } from "@/lib/doctor/types";

/** Glyphs allowed by 5A: check, cross and warning, inline with text. */
export const CHECK = "✓";
export const CROSS = "✗";
export const WARN = "⚠";

function useNow(active: boolean) {
    const [now, setNow] = useState(() => Date.now());
    useEffect(() => {
        if (!active) return;
        const id = setInterval(() => setNow(Date.now()), 1000);
        return () => clearInterval(id);
    }, [active]);
    return now;
}

/** Running: "Platform ✓ · Problems 4/10 batches · Latency · Rules" plus elapsed time. */
export function StageChecklist({ report }: { report: ReportResponse }) {
    const now = useNow(report.status === "running");
    const stages = stageEntries(report).filter((s) => s.name !== "preflight" && s.name !== "clusters");
    return (
        <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm tabular-nums">
            {stages.map(({ name, state }, i) => (
                <span key={name} className="inline-flex items-center gap-1">
                    {i > 0 ? <span aria-hidden className="text-muted-foreground">·</span> : null}
                    <span className={state.status === "pending" ? "text-muted-foreground" : undefined}>
                        {stageProgressText(name, state)}
                    </span>
                    {state.status === "done" ? <span aria-label="done">{CHECK}</span> : null}
                    {state.status === "failed" ? <span aria-label="failed">{CROSS}</span> : null}
                    {state.status === "running" ? <span className="sr-only">(in progress)</span> : null}
                </span>
            ))}
            <span aria-hidden className="text-muted-foreground">·</span>
            <span className="text-muted-foreground">{formatElapsed(report.created_at, now)} elapsed</span>
        </p>
    );
}

/** Queued / running problems area: skeleton rows. */
export function ProblemRowsSkeleton({ rows = 3 }: { rows?: number }) {
    return (
        <div className="space-y-3" aria-hidden>
            {Array.from({ length: rows }, (_, i) => (
                <div key={i} className="flex items-center gap-3 border-b py-3">
                    <Skeleton className="h-5 w-16 rounded-full" />
                    <Skeleton className="h-4 flex-1" />
                    <Skeleton className="h-4 w-20" />
                </div>
            ))}
        </div>
    );
}

export interface AiDisabledCopy {
    noKeyTitle: string;
    noKeyDetail: string;
    optInTitle: string;
    optInDetail: string;
    enableLabel: string;
}

const DOCTOR_AI_COPY: AiDisabledCopy = {
    noKeyTitle: "AI diagnosis is not available: this Dograh server has no Anthropic API key.",
    noKeyDetail:
        "Ask your administrator to set ANTHROPIC_API_KEY on the server. Platform checks and latency are shown below.",
    optInTitle: "AI diagnosis is off for your organisation. Platform checks and latency are shown below.",
    optInDetail:
        "When enabled, call transcripts are sent to Anthropic for analysis. Phone numbers, emails and card numbers are masked first; names and addresses spoken in calls may not be.",
    enableLabel: "Enable AI analysis",
};

/**
 * No Anthropic key / org not opted in (2A). Layer 0 + latency still show.
 * The same org opt-in gates the Agent Copilot, which passes its own copy.
 */
export function AiDisabledNotice({
    reason,
    onEnabled,
    copy = DOCTOR_AI_COPY,
}: {
    reason: "no_key" | "not_opted_in";
    /** Called after the org opts in, e.g. to re-run the analysis. */
    onEnabled?: () => void;
    copy?: AiDisabledCopy;
}) {
    const [enabling, setEnabling] = useState(false);
    const [error, setError] = useState<string | null>(null);

    if (reason === "no_key") {
        return (
            <div className="py-4 text-sm" role="status">
                <p>{copy.noKeyTitle}</p>
                <p className="text-muted-foreground">{copy.noKeyDetail}</p>
            </div>
        );
    }

    const enable = async () => {
        setEnabling(true);
        setError(null);
        try {
            await setDoctorAiOptIn(true);
            onEnabled?.();
        } catch (e) {
            setError(e instanceof Error ? e.message : "Could not enable AI analysis");
        } finally {
            setEnabling(false);
        }
    };

    return (
        <div className="flex flex-col gap-3 py-4 sm:flex-row sm:items-start sm:justify-between">
            <div className="space-y-1 text-sm">
                <p>{copy.optInTitle}</p>
                <p className="text-muted-foreground">{copy.optInDetail}</p>
                {error ? <p className="text-destructive">{error}</p> : null}
            </div>
            <Button variant="outline" className="min-h-11 shrink-0 md:min-h-9" onClick={enable} disabled={enabling}>
                {enabling ? "Enabling…" : copy.enableLabel}
            </Button>
        </div>
    );
}

/** No problems (2A). */
export function NoProblemsNotice({ calls }: { calls: number }) {
    return (
        <div className="flex flex-col gap-1 py-4">
            <p className="text-sm">No problems found in {plural(calls, "call")}.</p>
            <p className="text-sm text-muted-foreground">Run again after more calls.</p>
        </div>
    );
}

/** Few calls (<10) caveat chip. */
export function FewCallsChip({ calls }: { calls: number }) {
    return (
        <Badge variant="outline" className="font-normal tabular-nums">
            Only {plural(calls, "call")}: patterns may be noise
        </Badge>
    );
}

/** Failed (2A): error in plain words + reference id + Retry. */
export function FailedNotice({
    report,
    onRetry,
    retrying,
}: {
    report: ReportResponse;
    onRetry: () => void;
    retrying: boolean;
}) {
    return (
        <div className="flex flex-col gap-2 text-destructive sm:flex-row sm:items-center sm:justify-between">
            <div className="text-sm">
                <p>
                    <span aria-hidden>{CROSS} </span>
                    The analysis failed: {report.error || "something went wrong on our side."}
                </p>
                <p className="text-xs text-muted-foreground">Reference id: {report.report_uuid}</p>
            </div>
            <Button variant="outline" onClick={onRetry} disabled={retrying} className="min-h-11 md:min-h-9">
                {retrying ? "Retrying…" : "Retry"}
            </Button>
        </div>
    );
}
