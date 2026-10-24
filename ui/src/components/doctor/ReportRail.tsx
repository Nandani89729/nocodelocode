"use client";

import { formatSeconds, layer0CheckLabel, plural, topLatency } from "@/lib/doctor/format";
import type { LatencyRow, Layer0Summary, PassedCheck } from "@/lib/doctor/types";

import { CHECK } from "./ReportStates";

export function LatencyList({ rows }: { rows: LatencyRow[] }) {
    if (rows.length === 0) return <p className="text-sm text-muted-foreground">No latency data yet.</p>;
    return (
        <ol className="space-y-2">
            {rows.map((r) => (
                <li key={r.node_name} className="flex items-baseline justify-between gap-3 text-sm">
                    <span className="min-w-0 truncate">{r.node_name}</span>
                    <span className="shrink-0 tabular-nums text-muted-foreground">
                        p50 {formatSeconds(r.p50_s)} · p90 {formatSeconds(r.p90_s)}
                    </span>
                </li>
            ))}
        </ol>
    );
}

/**
 * Passed checks, plus Layer 0 checks that legacy calls could not evaluate:
 * "Not enough data" is never shown as a failure (2A legacy calls).
 */
export function PassedList({ passed, layer0 }: { passed: PassedCheck[]; layer0?: Layer0Summary | null }) {
    const insufficient = Object.entries(layer0?.insufficient_data ?? {}).filter(([, n]) => n > 0);
    if (passed.length === 0 && insufficient.length === 0) {
        return <p className="text-sm text-muted-foreground">No passed checks yet.</p>;
    }
    return (
        <ul className="space-y-2 text-sm">
            {passed.map((p) => (
                <li key={p.title} className="flex items-baseline justify-between gap-3">
                    <span>
                        <span aria-hidden>{CHECK} </span>
                        {p.title}
                    </span>
                    <span className="shrink-0 tabular-nums text-muted-foreground">{plural(p.runs, "call")}</span>
                </li>
            ))}
            {insufficient.map(([check, n]) => (
                <li key={check} className="flex items-baseline justify-between gap-3 text-muted-foreground">
                    <span className="first-letter:uppercase">{layer0CheckLabel(check)}</span>
                    <span className="shrink-0 tabular-nums">Not enough data ({plural(n, "older call")})</span>
                </li>
            ))}
        </ul>
    );
}

/** Right rail at ≥1280px (7A): latency top 3 + passed checks. Landmark: complementary. */
export function ReportRail({
    latency,
    passed,
    layer0,
}: {
    latency: LatencyRow[];
    passed: PassedCheck[];
    layer0?: Layer0Summary | null;
}) {
    return (
        <aside aria-label="Latency and passed checks" className="space-y-6">
            <section aria-labelledby="rail-latency" className="space-y-2">
                <h2 id="rail-latency" className="text-sm font-semibold">
                    Latency by step (top 3)
                </h2>
                <LatencyList rows={topLatency(latency)} />
            </section>
            <section aria-labelledby="rail-passed" className="space-y-2">
                <h2 id="rail-passed" className="text-sm font-semibold">
                    Passed checks
                </h2>
                <PassedList passed={passed} layer0={layer0} />
            </section>
        </aside>
    );
}
