"use client";

import { useMemo } from "react";

import { unifiedDiff } from "@/lib/copilot/format";
import type { FieldDiff } from "@/lib/copilot/types";
import { cn } from "@/lib/utils";

const MARK: Record<"unchanged" | "removed" | "added", string> = { unchanged: " ", removed: "−", added: "+" };
const SR_LABEL: Record<"removed" | "added", string> = { removed: "removed: ", added: "added: " };

/**
 * One field's before/after as a single-column diff (fits the narrow panel).
 * Lines carry a −/+ mark and a screen-reader label, so the change never
 * relies on color alone; long values scroll inside the box.
 */
export function FieldDiffView({ diff }: { diff: FieldDiff }) {
    const lines = useMemo(() => unifiedDiff(diff.before, diff.after), [diff.before, diff.after]);
    return (
        <div className="space-y-1">
            <p className="font-mono text-xs text-muted-foreground">{diff.field}</p>
            <div
                className="max-h-72 overflow-auto rounded-md border bg-muted/30 py-1 font-mono text-xs leading-5"
                tabIndex={0}
                role="group"
                aria-label={`Changes to ${diff.field}`}
            >
                {lines.length === 0 ? <p className="px-2 text-muted-foreground">No content</p> : null}
                {lines.map((line, i) =>
                    line.kind === "skipped" ? (
                        <p key={i} className="px-2 text-muted-foreground tabular-nums">
                            … {line.count} unchanged {line.count === 1 ? "line" : "lines"}
                        </p>
                    ) : (
                        <p
                            key={i}
                            className={cn(
                                "flex gap-2 whitespace-pre-wrap break-words px-2",
                                line.kind === "removed" && "bg-red-500/10",
                                line.kind === "added" && "bg-emerald-500/10",
                            )}
                        >
                            <span aria-hidden className="w-3 shrink-0 select-none text-muted-foreground">
                                {MARK[line.kind]}
                            </span>
                            <span className="min-w-0 flex-1">
                                {line.kind !== "unchanged" ? <span className="sr-only">{SR_LABEL[line.kind]}</span> : null}
                                {line.segments.map((segment, j) => (
                                    <span
                                        key={j}
                                        className={cn(
                                            segment.kind === "removed" && "bg-red-500/25 line-through decoration-red-700/50",
                                            segment.kind === "added" && "bg-emerald-500/25 underline decoration-emerald-700/50",
                                        )}
                                    >
                                        {segment.text}
                                    </span>
                                ))}
                            </span>
                        </p>
                    ),
                )}
            </div>
        </div>
    );
}
