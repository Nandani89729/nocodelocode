"use client";

import Link from "next/link";

import { Button } from "@/components/ui/button";
import {
    callsAnalysed,
    formatReportTime,
    formatSeconds,
    layer0CheckLabel,
    medianReplySeconds,
    partialStopText,
    platformFailure,
    plural,
} from "@/lib/doctor/format";
import type { ReportResponse } from "@/lib/doctor/types";

import { CROSS, FailedNotice, FewCallsChip, StageChecklist, WARN } from "./ReportStates";

interface VerdictStripProps {
    report: ReportResponse;
    workflowId: number;
    resuming: boolean;
    onResume: () => void;
    onRetry: () => void;
    retrying: boolean;
}

/**
 * First thing on the page: "am I OK?" (1A). Plain section, no card (5A).
 * Sticky on phones (7A). Status text is announced politely (8A).
 */
export function VerdictStrip({ report, workflowId, resuming, onResume, onRetry, retrying }: VerdictStripProps) {
    const { result, status } = report;
    const calls = callsAnalysed(report);
    const layer0 = result.layer0;
    const platform = platformFailure(layer0);
    const median = medianReplySeconds(result.latency);
    const problems = result.clusters.length;

    const parts: string[] = [];
    if (layer0) {
        parts.push(
            platform
                ? `Platform issue ${layer0.runs_flagged}/${layer0.runs_checked}`
                : `Platform OK ${layer0.runs_checked - layer0.runs_flagged}/${layer0.runs_checked}`,
        );
    }
    if (result.ai_enabled && (status === "completed" || problems > 0)) parts.push(plural(problems, "problem"));
    if (median !== null) parts.push(`median reply ${formatSeconds(median)}`);
    if (calls > 0) parts.push(`${plural(calls, "call")} analysed`);
    parts.push(formatReportTime(report.created_at));

    return (
        <section
            aria-labelledby="doctor-verdict-heading"
            className="sticky top-0 z-10 space-y-2 border-b bg-background py-3 md:static md:z-auto"
        >
            <h2 id="doctor-verdict-heading" className="sr-only">
                Verdict
            </h2>
            <div aria-live="polite" className="space-y-2">
                {status === "queued" ? (
                    <p className="text-sm tabular-nums">
                        Queued: another analysis is running for your org
                        {report.queue_position ? ` · position ${report.queue_position} in queue` : ""}
                    </p>
                ) : null}

                {status === "running" || (status === "partial" && resuming) ? <StageChecklist report={report} /> : null}

                {status === "failed" ? <FailedNotice report={report} onRetry={onRetry} retrying={retrying} /> : null}

                {status === "completed" || status === "partial" ? (
                    <p className="text-base font-medium tabular-nums">{parts.join(" · ")}</p>
                ) : null}

                {platform && status !== "failed" ? (
                    <p className="text-sm text-destructive tabular-nums">
                        <span aria-hidden>{CROSS} </span>
                        Platform issue: {layer0CheckLabel(platform.check)} on {platform.runs}/{platform.runsChecked}{" "}
                        calls. Fix this first.{" "}
                        <Link
                            href={`/workflow/${workflowId}/run/${platform.runIds[0]}`}
                            className="underline underline-offset-2"
                        >
                            See call #{platform.runIds[0]}
                        </Link>
                    </p>
                ) : null}

                {status === "partial" && !resuming ? (
                    <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                        <p className="text-sm text-amber-700 dark:text-amber-400 tabular-nums">
                            <span aria-hidden>{WARN} </span>
                            {partialStopText(report)}
                        </p>
                        <Button variant="outline" onClick={onResume} className="min-h-11 md:min-h-9">
                            Resume
                        </Button>
                    </div>
                ) : null}

                {result.few_calls && calls > 0 && status !== "failed" ? <FewCallsChip calls={calls} /> : null}
            </div>
        </section>
    );
}
