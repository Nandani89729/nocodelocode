"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { createDoctorReport } from "@/lib/doctor/api";
import { callsAnalysed, isActive, platformFailure } from "@/lib/doctor/format";
import type { DoctorRunFilter, ReportResponse } from "@/lib/doctor/types";
import { DEFAULT_CALLS_PER_ANALYSIS } from "@/lib/doctor/types";

import { ProblemsTable } from "./ProblemsTable";
import { ReportRail } from "./ReportRail";
import { AiDisabledNotice, NoProblemsNotice, ProblemRowsSkeleton } from "./ReportStates";
import { ReportTabs } from "./ReportTabs";
import { useDoctorReport } from "./useDoctorReport";
import { VerdictStrip } from "./VerdictStrip";

function rerunRequest(report: ReportResponse) {
    const limit = typeof report.params.limit === "number" ? report.params.limit : DEFAULT_CALLS_PER_ANALYSIS;
    const filters = Array.isArray(report.params.filters) ? (report.params.filters as DoctorRunFilter[]) : null;
    return { limit, filters };
}

function ProblemsArea({
    report,
    workflowId,
    onAiEnabled,
}: {
    report: ReportResponse;
    workflowId: number;
    onAiEnabled: () => void;
}) {
    const { result, status } = report;
    const platform = platformFailure(result.layer0);

    if (!result.ai_enabled && result.ai_disabled_reason) {
        return <AiDisabledNotice reason={result.ai_disabled_reason} onEnabled={onAiEnabled} />;
    }
    if (result.clusters.length === 0) {
        if (status === "queued" || status === "running") return <ProblemRowsSkeleton />;
        if (status === "completed") return <NoProblemsNotice calls={callsAnalysed(report)} />;
        return <p className="py-4 text-sm text-muted-foreground">No problems found in the finished part of the analysis.</p>;
    }
    return (
        <>
            <ProblemsTable
                workflowId={workflowId}
                reportUuid={report.report_uuid}
                clusters={result.clusters}
                platformCaveat={Boolean(platform)}
                canPropose={!isActive(status)}
            />
            {status === "running" ? <ProblemRowsSkeleton rows={1} /> : null}
        </>
    );
}

export function DoctorReportView({ workflowId, reportUuid }: { workflowId: number; reportUuid: string }) {
    const router = useRouter();
    const { report, error, resuming, resume } = useDoctorReport(reportUuid);
    const [rerunning, setRerunning] = useState(false);
    const [rerunError, setRerunError] = useState<string | null>(null);

    const rerun = async () => {
        if (!report) return;
        setRerunning(true);
        setRerunError(null);
        try {
            const next = await createDoctorReport(workflowId, rerunRequest(report));
            router.push(`/workflow/${workflowId}/doctor/${next.report_uuid}`);
        } catch (err) {
            setRerunError(err instanceof Error ? err.message : "Could not start the analysis");
        } finally {
            setRerunning(false);
        }
    };

    return (
        <div className="container mx-auto max-w-7xl space-y-4 px-4 py-6 md:px-6">
            <div className="flex items-center justify-between gap-3">
                <nav aria-label="Breadcrumb" className="text-sm text-muted-foreground">
                    <ol className="flex flex-wrap items-center gap-1.5">
                        <li>
                            <Link href={`/workflow/${workflowId}`} className="hover:text-foreground">
                                Workflow
                            </Link>
                        </li>
                        <li aria-hidden>›</li>
                        <li>
                            <Link href={`/workflow/${workflowId}/runs`} className="hover:text-foreground">
                                Runs
                            </Link>
                        </li>
                        <li aria-hidden>›</li>
                        <li aria-current="page" className="text-foreground">
                            Doctor report
                        </li>
                    </ol>
                </nav>
                {report && !isActive(report.status) ? (
                    <Button variant="outline" onClick={rerun} disabled={rerunning} className="hidden md:inline-flex">
                        {rerunning ? "Starting…" : "Re-run"}
                    </Button>
                ) : null}
            </div>
            {rerunError ? <p className="text-sm text-destructive">{rerunError}</p> : null}

            {!report ? (
                error ? (
                    <p className="text-sm text-destructive" role="alert">{error}</p>
                ) : (
                    <div aria-busy="true">
                        <ProblemRowsSkeleton />
                    </div>
                )
            ) : (
                <>
                    <VerdictStrip
                        report={report}
                        workflowId={workflowId}
                        resuming={resuming}
                        onResume={resume}
                        onRetry={rerun}
                        retrying={rerunning}
                    />
                    {error ? <p className="text-sm text-destructive">{error}</p> : null}

                    {report.status !== "failed" ? (
                        <div className="grid gap-8 xl:grid-cols-3">
                            <section aria-labelledby="doctor-problems-heading" className="space-y-2 xl:col-span-2">
                                <h2 id="doctor-problems-heading" className="text-sm font-semibold">
                                    Problems
                                </h2>
                                <ProblemsArea report={report} workflowId={workflowId} onAiEnabled={rerun} />
                            </section>
                            <div className="hidden xl:block">
                                <ReportRail
                                    latency={report.result.latency}
                                    passed={report.result.passed_checks}
                                    layer0={report.result.layer0}
                                />
                            </div>
                        </div>
                    ) : null}

                    {report.status !== "failed" && report.status !== "queued" ? (
                        <ReportTabs workflowId={workflowId} result={report.result} />
                    ) : null}
                </>
            )}
        </div>
    );
}
