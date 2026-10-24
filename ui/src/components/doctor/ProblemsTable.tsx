"use client";

import Link from "next/link";
import { type KeyboardEvent, useCallback, useRef, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent } from "@/components/ui/collapsible";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { MessageBubble } from "@/components/workflow/conversation/MessageBubble";
import {
    CATEGORY_LABEL,
    evidenceHref,
    evidenceLinkName,
    formatReportTime,
    frequencyLabel,
    rankClusters,
    SEVERITY_LABEL,
} from "@/lib/doctor/format";
import type { Cluster, Severity } from "@/lib/doctor/types";
import { cn } from "@/lib/utils";

import { VerifyPanel } from "./VerifyPanel";

const SEVERITY_VARIANT: Record<Severity, "destructive" | "secondary" | "outline"> = {
    critical: "destructive",
    major: "secondary",
    minor: "outline",
};

export function SeverityBadge({ severity }: { severity: Severity }) {
    return <Badge variant={SEVERITY_VARIANT[severity]}>{SEVERITY_LABEL[severity]}</Badge>;
}

function sinceFixText(sinceFix: Record<string, unknown> | null | undefined): string | null {
    if (!sinceFix) return null;
    const affected = sinceFix.runs_affected ?? sinceFix.affected;
    const analysed = sinceFix.runs_analysed ?? sinceFix.analysed ?? sinceFix.runs;
    if (typeof affected !== "number" || typeof analysed !== "number") return null;
    const published = typeof sinceFix.published_at === "string" ? ` (published ${formatReportTime(sinceFix.published_at)})` : "";
    return `Since the fix${published}: ${affected} of ${analysed} calls`;
}

interface ProblemsTableProps {
    workflowId: number;
    reportUuid: string;
    clusters: Cluster[];
    /** "may be symptoms of the platform issue" (2A platform failure). */
    platformCaveat?: boolean;
    /** Fixes need a finished report. */
    canPropose: boolean;
}

/**
 * Ranked problems (severity × frequency) with inline expandable evidence and
 * a per-row "Propose fix" (1A). Rows are buttons with aria-expanded; ↑/↓ move
 * between rows (8A). Each problem is its own <tbody> so a Collapsible can own
 * the summary row and the evidence row.
 */
export function ProblemsTable({ workflowId, reportUuid, clusters, platformCaveat, canPropose }: ProblemsTableProps) {
    const ranked = rankClusters(clusters);
    const [open, setOpen] = useState<Set<string>>(() => new Set(ranked[0] ? [ranked[0].cluster_id] : []));
    const [proposing, setProposing] = useState<string | null>(null);
    const rowButtons = useRef<Array<HTMLButtonElement | null>>([]);
    const proposeButtons = useRef<Record<string, HTMLButtonElement | null>>({});

    const toggle = useCallback((id: string) => {
        setOpen((prev) => {
            const next = new Set(prev);
            if (next.has(id)) next.delete(id);
            else next.add(id);
            return next;
        });
    }, []);

    const onRowKeyDown = (index: number) => (e: KeyboardEvent<HTMLButtonElement>) => {
        if (e.key === "ArrowDown" || e.key === "ArrowUp") {
            e.preventDefault();
            const next = e.key === "ArrowDown" ? index + 1 : index - 1;
            rowButtons.current[Math.max(0, Math.min(ranked.length - 1, next))]?.focus();
        }
    };

    return (
        <div className="space-y-2">
            {platformCaveat ? (
                <p className="text-sm text-muted-foreground">These may be symptoms of the platform issue.</p>
            ) : null}
            <Table>
                <TableHeader>
                    <TableRow>
                        <TableHead className="w-24">Severity</TableHead>
                        <TableHead>Problem</TableHead>
                        <TableHead className="hidden w-32 text-right sm:table-cell">Calls</TableHead>
                    </TableRow>
                </TableHeader>
                {ranked.map((cluster, index) => {
                    const isOpen = open.has(cluster.cluster_id);
                    const contentId = `problem-${cluster.cluster_id}-evidence`;
                    const since = sinceFixText(cluster.since_fix);
                    return (
                        <Collapsible key={cluster.cluster_id} asChild open={isOpen} onOpenChange={() => toggle(cluster.cluster_id)}>
                            <TableBody className="border-b">
                                <TableRow className="hover:bg-transparent">
                                    <TableCell className="align-top">
                                        <SeverityBadge severity={cluster.severity} />
                                    </TableCell>
                                    <TableCell className="p-0 align-top" colSpan={1}>
                                        <button
                                            type="button"
                                            ref={(el) => {
                                                rowButtons.current[index] = el;
                                            }}
                                            aria-expanded={isOpen}
                                            aria-controls={contentId}
                                            onClick={() => toggle(cluster.cluster_id)}
                                            onKeyDown={onRowKeyDown(index)}
                                            className="flex min-h-11 w-full flex-col items-start gap-1 rounded-sm p-2 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                                        >
                                            <span className="flex items-start gap-1.5 font-medium">
                                                <span aria-hidden className="text-muted-foreground">
                                                    {isOpen ? "▾" : "▸"}
                                                </span>
                                                {cluster.title}
                                            </span>
                                            <span className="flex flex-wrap gap-x-2 pl-4 text-xs text-muted-foreground tabular-nums">
                                                <span>{CATEGORY_LABEL[cluster.category] ?? cluster.category}</span>
                                                <span>[{cluster.fix_target}]</span>
                                                <span className="sm:hidden">{frequencyLabel(cluster)}</span>
                                                {cluster.watch_only ? <span>watch: seen in one call</span> : null}
                                            </span>
                                            {since ? <span className="pl-4 text-xs tabular-nums">{since}</span> : null}
                                        </button>
                                    </TableCell>
                                    <TableCell className="hidden text-right align-top tabular-nums sm:table-cell">
                                        {frequencyLabel(cluster)}
                                    </TableCell>
                                </TableRow>
                                <CollapsibleContent asChild>
                                    <TableRow id={contentId} className="hover:bg-transparent">
                                        <TableCell colSpan={3} className="pb-4">
                                            <div className="space-y-3 md:pl-24">
                                                <p className="text-sm text-muted-foreground">{cluster.root_cause}</p>
                                                <ul className="space-y-3">
                                                    {cluster.evidence.map((e, i) => (
                                                        <li key={`${e.run_id}-${e.turn ?? "x"}-${i}`} className="space-y-1">
                                                            <MessageBubble role="assistant" text={e.quote} />
                                                            <Link
                                                                href={evidenceHref(workflowId, e)}
                                                                className="inline-flex min-h-11 items-center pl-1 text-xs underline underline-offset-2 md:min-h-0"
                                                            >
                                                                {evidenceLinkName(e)}
                                                                {e.node_name ? ` (${e.node_name})` : ""}
                                                            </Link>
                                                        </li>
                                                    ))}
                                                </ul>
                                                {!cluster.watch_only ? (
                                                    <div>
                                                        <p className="text-sm text-muted-foreground md:hidden">Open on desktop to fix</p>
                                                        <Button
                                                            ref={(el) => {
                                                                proposeButtons.current[cluster.cluster_id] = el;
                                                            }}
                                                            className={cn("hidden md:inline-flex")}
                                                            disabled={!canPropose || proposing === cluster.cluster_id}
                                                            onClick={() => setProposing(cluster.cluster_id)}
                                                        >
                                                            Propose fix
                                                        </Button>
                                                    </div>
                                                ) : null}
                                                {proposing === cluster.cluster_id ? (
                                                    <div className="hidden md:block">
                                                        <VerifyPanel
                                                            workflowId={workflowId}
                                                            reportUuid={reportUuid}
                                                            cluster={cluster}
                                                            returnFocusRef={{
                                                                get current() {
                                                                    return proposeButtons.current[cluster.cluster_id] ?? null;
                                                                },
                                                            }}
                                                            onClose={() => {
                                                                setProposing(null);
                                                                setTimeout(() => proposeButtons.current[cluster.cluster_id]?.focus(), 0);
                                                            }}
                                                        />
                                                    </div>
                                                ) : null}
                                            </div>
                                        </TableCell>
                                    </TableRow>
                                </CollapsibleContent>
                            </TableBody>
                        </Collapsible>
                    );
                })}
            </Table>
        </div>
    );
}
