"use client";

import Link from "next/link";

import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { formatSeconds } from "@/lib/doctor/format";
import type { ReportResult } from "@/lib/doctor/types";

import { PassedList } from "./ReportRail";

/**
 * Details tabs: Latency · Passed · Calls analysed. At 768–1279px this is
 * where the rail content lives (7A); at ≥1280 it adds the full tables.
 */
export function ReportTabs({ workflowId, result }: { workflowId: number; result: ReportResult }) {
    const latency = [...result.latency].sort((a, b) => b.p90_s - a.p90_s);
    return (
        <section aria-labelledby="doctor-details-heading" className="space-y-2">
            <h2 id="doctor-details-heading" className="sr-only">
                Details
            </h2>
            <Tabs defaultValue="latency">
                <TabsList>
                    <TabsTrigger value="latency" className="min-h-11 md:min-h-0">Latency</TabsTrigger>
                    <TabsTrigger value="passed" className="min-h-11 md:min-h-0">Passed</TabsTrigger>
                    <TabsTrigger value="calls" className="min-h-11 md:min-h-0">
                        Calls analysed <span className="ml-1 tabular-nums">({result.run_ids.length})</span>
                    </TabsTrigger>
                </TabsList>
                <TabsContent value="latency">
                    {latency.length === 0 ? (
                        <p className="py-3 text-sm text-muted-foreground">No latency data yet.</p>
                    ) : (
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHead>Step</TableHead>
                                    <TableHead className="text-right">Turns</TableHead>
                                    <TableHead className="text-right">p50</TableHead>
                                    <TableHead className="text-right">p90</TableHead>
                                    <TableHead className="hidden text-right md:table-cell">LLM first token</TableHead>
                                    <TableHead className="hidden text-right md:table-cell">STT</TableHead>
                                    <TableHead className="hidden text-right md:table-cell">TTS first byte</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody className="tabular-nums">
                                {latency.map((r) => (
                                    <TableRow key={r.node_name}>
                                        <TableCell>{r.node_name}</TableCell>
                                        <TableCell className="text-right">{r.turns}</TableCell>
                                        <TableCell className="text-right">{formatSeconds(r.p50_s)}</TableCell>
                                        <TableCell className="text-right">{formatSeconds(r.p90_s)}</TableCell>
                                        <TableCell className="hidden text-right md:table-cell">{formatSeconds(r.llm_ttfb_p50_s)}</TableCell>
                                        <TableCell className="hidden text-right md:table-cell">{formatSeconds(r.stt_p50_s)}</TableCell>
                                        <TableCell className="hidden text-right md:table-cell">{formatSeconds(r.tts_ttfb_p50_s)}</TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    )}
                </TabsContent>
                <TabsContent value="passed" className="py-2">
                    <PassedList passed={result.passed_checks} layer0={result.layer0} />
                </TabsContent>
                <TabsContent value="calls" className="py-2">
                    {result.run_ids.length === 0 ? (
                        <p className="text-sm text-muted-foreground">No calls selected yet.</p>
                    ) : (
                        <ul className="flex flex-wrap gap-x-3 gap-y-1 text-sm tabular-nums">
                            {result.run_ids.map((id) => (
                                <li key={id}>
                                    <Link
                                        href={`/workflow/${workflowId}/run/${id}`}
                                        className="inline-flex min-h-11 items-center underline underline-offset-2 md:min-h-0"
                                    >
                                        Call #{id}
                                    </Link>
                                </li>
                            ))}
                        </ul>
                    )}
                </TabsContent>
            </Tabs>
        </section>
    );
}
