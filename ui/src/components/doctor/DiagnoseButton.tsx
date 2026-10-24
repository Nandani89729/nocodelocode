"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { getWorkflowRunsApiV1WorkflowWorkflowIdRunsGet } from "@/client/sdk.gen";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { detailFromError } from "@/lib/apiError";
import { createDoctorReport } from "@/lib/doctor/api";
import { filtersInWords, toDoctorFilters } from "@/lib/doctor/filters";
import { plural } from "@/lib/doctor/format";
import { DEFAULT_CALLS_PER_ANALYSIS, LAYER1_BATCH_SIZE, MAX_CALLS_PER_ANALYSIS } from "@/lib/doctor/types";
import type { ActiveFilter } from "@/types/filters";

interface DiagnoseButtonProps {
    workflowId: number;
    /** The runs page's applied (not draft) filters. */
    filters: ActiveFilter[];
    /** Total runs matching those filters, from the runs list. */
    totalCount: number;
}

interface Span {
    count: number;
    from: string | null;
    to: string | null;
}

const fmtDay = (iso: string) =>
    new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });

/**
 * Runs page entry point (D12): "Diagnose these calls (N)" with a confirm
 * popover showing count, date span, filters in words and the cost estimate.
 */
export function DiagnoseButton({ workflowId, filters, totalCount }: DiagnoseButtonProps) {
    const router = useRouter();
    const [open, setOpen] = useState(false);
    const [limit, setLimit] = useState(DEFAULT_CALLS_PER_ANALYSIS);
    const [span, setSpan] = useState<Span | null>(null);
    const [loadingSpan, setLoadingSpan] = useState(false);
    const [submitting, setSubmitting] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const safeLimit = Math.max(1, Math.min(MAX_CALLS_PER_ANALYSIS, Number.isFinite(limit) ? limit : DEFAULT_CALLS_PER_ANALYSIS));
    const buttonCount = Math.min(totalCount, DEFAULT_CALLS_PER_ANALYSIS);

    useEffect(() => {
        if (!open) return;
        let cancelled = false;
        setLoadingSpan(true);
        setError(null);
        const filterParam = filters.length
            ? JSON.stringify(filters.map((f) => ({ attribute: f.attribute.id, type: f.attribute.type, value: f.value })))
            : undefined;
        void (async () => {
            const res = await getWorkflowRunsApiV1WorkflowWorkflowIdRunsGet({
                path: { workflow_id: workflowId },
                query: {
                    page: 1,
                    limit: safeLimit,
                    sort_by: "created_at",
                    sort_order: "desc",
                    ...(filterParam ? { filters: filterParam } : {}),
                },
            });
            if (cancelled) return;
            setLoadingSpan(false);
            if (res.error) {
                setSpan(null);
                setError(detailFromError(res.error, "Could not load the calls"));
                return;
            }
            const runs = res.data?.runs ?? [];
            const dates = runs.map((r) => r.created_at).filter(Boolean).sort();
            setSpan({ count: runs.length, from: dates[0] ?? null, to: dates[dates.length - 1] ?? null });
        })();
        return () => {
            cancelled = true;
        };
    }, [open, filters, workflowId, safeLimit]);

    const confirm = async () => {
        setSubmitting(true);
        setError(null);
        try {
            const report = await createDoctorReport(workflowId, {
                limit: safeLimit,
                filters: filters.length ? toDoctorFilters(filters) : null,
            });
            router.push(`/workflow/${workflowId}/doctor/${report.report_uuid}`);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Could not start the analysis");
            setSubmitting(false);
        }
    };

    const count = span?.count ?? Math.min(totalCount, safeLimit);
    const batches = Math.max(1, Math.ceil(count / LAYER1_BATCH_SIZE));

    return (
        <Popover open={open} onOpenChange={setOpen}>
            <PopoverTrigger asChild>
                <Button variant="outline" disabled={totalCount === 0}>
                    Diagnose these calls <span className="tabular-nums">({buttonCount})</span>
                </Button>
            </PopoverTrigger>
            <PopoverContent align="end" className="w-80 space-y-3">
                <div className="space-y-1">
                    <h3 className="text-sm font-semibold">Diagnose calls</h3>
                    <p className="text-xs text-muted-foreground">Analyses the latest completed calls for problems and latency.</p>
                </div>
                <div className="flex items-center justify-between gap-2">
                    <Label htmlFor="doctor-limit" className="text-sm">
                        Calls (max {MAX_CALLS_PER_ANALYSIS})
                    </Label>
                    <Input
                        id="doctor-limit"
                        type="number"
                        min={1}
                        max={MAX_CALLS_PER_ANALYSIS}
                        value={limit}
                        onChange={(e) => setLimit(parseInt(e.target.value, 10))}
                        className="h-8 w-20 tabular-nums"
                    />
                </div>
                <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-sm" aria-live="polite">
                    <dt className="text-muted-foreground">Count</dt>
                    <dd className="tabular-nums">{loadingSpan ? "…" : plural(count, "call")}</dd>
                    <dt className="text-muted-foreground">Dates</dt>
                    <dd className="tabular-nums">
                        {loadingSpan ? "…" : span?.from && span.to ? `${fmtDay(span.from)} to ${fmtDay(span.to)}` : "–"}
                    </dd>
                    <dt className="text-muted-foreground">Filters</dt>
                    <dd>{filtersInWords(filters)}</dd>
                    <dt className="text-muted-foreground">Cost</dt>
                    <dd className="tabular-nums">
                        {plural(batches, "Claude analysis batch", "Claude analysis batches")} on your Anthropic key
                    </dd>
                </dl>
                {error ? <p className="text-sm text-destructive">{error}</p> : null}
                <div className="flex justify-end gap-2">
                    <Button variant="ghost" size="sm" onClick={() => setOpen(false)}>
                        Cancel
                    </Button>
                    <Button size="sm" onClick={confirm} disabled={submitting || count === 0}>
                        {submitting ? "Starting…" : "Diagnose"}
                    </Button>
                </div>
            </PopoverContent>
        </Popover>
    );
}
