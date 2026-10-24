"use client";

import Link from "next/link";
import { type RefObject, useCallback, useEffect, useRef, useState } from "react";

import { WorkflowVersionDiffDialog } from "@/app/workflow/[workflowId]/components/WorkflowVersionDiffDialog";
import { getWorkflowVersionsApiV1WorkflowWorkflowIdVersionsGet } from "@/client/sdk.gen";
import type { WorkflowVersionResponse } from "@/client/types.gen";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { MessageBubble } from "@/components/workflow/conversation/MessageBubble";
import {
    discardVerification,
    DoctorApiError,
    getVerification,
    proposeFix,
    saveVerification,
    scenarioIdOf,
    startVerification,
} from "@/lib/doctor/api";
import {
    formatUsd,
    notKeptText,
    notReproducibleText,
    replayCountsText,
    scenarioOutcomeLines,
    verifyProgress,
} from "@/lib/doctor/format";
import type { Cluster, SampleResult, VerificationResponse, VerificationResult } from "@/lib/doctor/types";

import { CHECK, CROSS, WARN } from "./ReportStates";

export const VERIFY_POLL_MS = 3000;
/** Rough per-regeneration time at concurrency 5 (plan R8): ~3s per wave of 5. */
const SECONDS_PER_WAVE = 3;
const CONCURRENCY = 5;

type SaveState = "idle" | "saving" | "saved" | "conflict" | "hash_mismatch" | "error";

interface VerifyPanelProps {
    workflowId: number;
    reportUuid: string;
    cluster: Cluster;
    /** The row's "Propose fix" button: focus returns here when the diff dialog closes (8A). */
    returnFocusRef: RefObject<HTMLButtonElement | null>;
    onClose: () => void;
}

async function fetchVersions(workflowId: number): Promise<WorkflowVersionResponse[]> {
    const res = await getWorkflowVersionsApiV1WorkflowWorkflowIdVersionsGet({
        path: { workflow_id: workflowId },
        query: { limit: 10, offset: 0 },
    });
    return res.data ?? [];
}

function SampleList({ samples, label }: { samples: SampleResult[]; label: string }) {
    if (samples.length === 0) return null;
    return (
        <div className="space-y-2">
            <h4 className="text-xs font-medium text-muted-foreground">{label}</h4>
            {samples.map((s, i) => (
                <div key={i} className="space-y-1">
                    <MessageBubble role="assistant" text={s.text} />
                    {s.tool_calls.length > 0 ? (
                        <div className="flex flex-wrap gap-1 pl-1">
                            {s.tool_calls.map((t, j) => (
                                <span key={j} className="rounded border px-1.5 py-0.5 font-mono text-[11px]">
                                    {t}
                                </span>
                            ))}
                        </div>
                    ) : null}
                    {s.reason ? (
                        <p className="pl-1 text-xs text-muted-foreground">
                            <span aria-hidden>{WARN} </span>
                            {s.reason}
                        </p>
                    ) : null}
                </div>
            ))}
        </div>
    );
}

/**
 * Propose fix → verify → save (design 3A). Rendered as a scenario card under
 * the expanded problem row; desktop only (7A).
 */
function OutcomeLines({ verification }: { verification: VerificationResult }) {
    const lines = scenarioOutcomeLines(verification);
    if (lines.length === 0) return null;
    return (
        <ul className="space-y-1 text-sm tabular-nums text-muted-foreground">
            {lines.map((line) => (
                <li key={line}>{line}</li>
            ))}
        </ul>
    );
}

export function VerifyPanel({ workflowId, reportUuid, cluster, returnFocusRef, onClose }: VerifyPanelProps) {
    const [verification, setVerification] = useState<VerificationResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [busy, setBusy] = useState(false);
    const [saveState, setSaveState] = useState<SaveState>("idle");
    const [saveError, setSaveError] = useState<string | null>(null);
    const [diffPair, setDiffPair] = useState<{ previous: WorkflowVersionResponse; selected: WorkflowVersionResponse } | null>(null);
    const [savedVersion, setSavedVersion] = useState<number | null>(null);
    const expectedDraftVersion = useRef<number | null>(null);
    const cancelled = useRef(false);
    const attempt = useRef(0);

    const scenarioId = scenarioIdOf(verification);
    const status = verification?.status ?? "proposing";

    const propose = useCallback(async () => {
        const mine = ++attempt.current;
        cancelled.current = false;
        setError(null);
        setVerification(null);
        setSaveState("idle");
        try {
            // Optimistic lock baseline: the draft as it is when the user starts.
            const versions = await fetchVersions(workflowId).catch(() => []);
            expectedDraftVersion.current = versions.find((v) => v.status === "draft")?.version_number ?? null;
            const v = await proposeFix(reportUuid, cluster.cluster_id);
            if (cancelled.current || mine !== attempt.current) return;
            setVerification(v);
        } catch (err) {
            if (mine === attempt.current) setError(err instanceof Error ? err.message : "Could not write a fix");
        }
    }, [workflowId, reportUuid, cluster.cluster_id]);

    useEffect(() => {
        void propose();
        return () => {
            cancelled.current = true;
        };
    }, [propose]);

    // Poll while the backend is still proposing or replaying.
    useEffect(() => {
        if (scenarioId === null || (status !== "proposing" && status !== "verifying")) return;
        const id = setTimeout(async () => {
            try {
                const next = await getVerification(scenarioId);
                if (!cancelled.current) setVerification(next);
            } catch (err) {
                setError(err instanceof Error ? err.message : "Could not load verification");
            }
        }, VERIFY_POLL_MS);
        return () => clearTimeout(id);
    }, [scenarioId, status, verification]);

    const cancel = async () => {
        cancelled.current = true;
        attempt.current++;
        if (scenarioId !== null) {
            try {
                await discardVerification(scenarioId);
            } catch {
                // Discard is best-effort; the candidate stays hidden either way.
            }
        }
        onClose();
    };

    const start = async () => {
        if (scenarioId === null) return;
        setBusy(true);
        setError(null);
        setSaveState("idle");
        try {
            setVerification(await startVerification(scenarioId));
        } catch (err) {
            setError(err instanceof Error ? err.message : "Could not start verification");
        } finally {
            setBusy(false);
        }
    };

    const tryAnother = async () => {
        if (scenarioId !== null) {
            try {
                await discardVerification(scenarioId);
            } catch {
                // ignore
            }
        }
        void propose();
    };

    const save = async () => {
        if (scenarioId === null) return;
        setSaveState("saving");
        setSaveError(null);
        try {
            await saveVerification(scenarioId, { expected_draft_version: expectedDraftVersion.current });
            const versions = await fetchVersions(workflowId);
            const draft = versions.find((v) => v.status === "draft");
            const base =
                versions.find((v) => v.status === "published") ??
                versions.find((v) => draft && v.id !== draft.id);
            setSaveState("saved");
            setSavedVersion(draft?.version_number ?? null);
            if (draft) expectedDraftVersion.current = draft.version_number;
            if (draft && base) setDiffPair({ previous: base, selected: draft });
        } catch (err) {
            const message = err instanceof Error ? err.message : "Could not save the draft";
            if (err instanceof DoctorApiError && (err.status === 409 || err.status === 412)) {
                setSaveState(/hash|changed since|re-?verify/i.test(message) ? "hash_mismatch" : "conflict");
            } else {
                setSaveState("error");
                setSaveError(message);
            }
        }
    };

    const closeDiff = (open: boolean) => {
        if (open) return;
        setDiffPair(null);
        setTimeout(() => returnFocusRef.current?.focus(), 0);
    };

    const reopenDiff = async () => {
        const versions = await fetchVersions(workflowId);
        const draft = versions.find((v) => v.status === "draft");
        const base = versions.find((v) => v.status === "published") ?? versions.find((v) => draft && v.id !== draft.id);
        if (draft && base) setDiffPair({ previous: base, selected: draft });
    };

    const saveButton = (unverified: boolean) => (
        <Button onClick={save} disabled={saveState === "saving" || saveState === "saved"}>
            {saveState === "saving" ? "Saving…" : unverified ? "Save as draft (unverified)" : "Save as draft"}
        </Button>
    );

    const progress = verification ? verifyProgress(verification) : null;
    const remaining = progress ? progress.total * 2 - progress.baselineDone - progress.candidateDone : 0;
    const secondsLeft = Math.ceil(remaining / CONCURRENCY) * SECONDS_PER_WAVE;
    const failingCandidates = verification?.candidate_samples.filter((s) => !s.passed) ?? [];

    let body: React.ReactNode;
    if (error) {
        body = (
            <div className="space-y-3">
                <p className="text-sm text-destructive">
                    <span aria-hidden>{CROSS} </span>
                    {error}
                </p>
                <div className="flex gap-2">
                    <Button variant="outline" onClick={() => void propose()}>Retry</Button>
                    <Button variant="ghost" onClick={cancel}>Close</Button>
                </div>
            </div>
        );
    } else if (status === "proposing") {
        body = (
            <div className="space-y-3">
                <p className="text-sm">Writing a minimal fix for: {cluster.title}</p>
                <div className="space-y-2" aria-hidden>
                    <Skeleton className="h-4 w-3/4" />
                    <Skeleton className="h-4 w-1/2" />
                    <Skeleton className="h-4 w-2/3" />
                </div>
                <Button variant="ghost" onClick={cancel}>Cancel</Button>
            </div>
        );
    } else if (status === "awaiting_start") {
        const cost = formatUsd(verification?.cost_estimate_usd);
        body = (
            <div className="space-y-3">
                {verification?.diff_summary ? <p className="text-sm whitespace-pre-wrap">{verification.diff_summary}</p> : null}
                <p className="text-sm text-muted-foreground tabular-nums">
                    Verification replays the failing turn {progress?.total ?? 10} times on the old and the new version.
                    {cost ? ` Estimated cost: ${cost}, billed as replay usage.` : " Billed as replay usage."}
                </p>
                <div className="flex gap-2">
                    <Button variant="outline" onClick={start} disabled={busy}>{busy ? "Starting…" : "Start verification"}</Button>
                    <Button variant="ghost" onClick={cancel}>Cancel</Button>
                </div>
            </div>
        );
    } else if (status === "verifying" && progress) {
        body = (
            <div className="space-y-3">
                <p className="text-sm tabular-nums">
                    Old version {progress.baselineDone}/{progress.total} done · New version {progress.candidateDone}/
                    {progress.total} done
                    {remaining > 0 ? ` · about ${secondsLeft}s left` : ""}
                </p>
                <div className="space-y-2">
                    <Progress value={(progress.baselineDone / progress.total) * 100} aria-label="Old version replays" />
                    <Progress value={(progress.candidateDone / progress.total) * 100} aria-label="New version replays" />
                </div>
                <Button variant="ghost" onClick={cancel}>Cancel</Button>
            </div>
        );
    } else if (status === "kept" && verification) {
        body = (
            <div className="space-y-3">
                <p className="text-sm tabular-nums">
                    <span aria-hidden>{CHECK} </span>
                    {replayCountsText(verification)}
                </p>
                <OutcomeLines verification={verification} />
                <p className="text-sm text-muted-foreground">Confirm on a real call after publishing.</p>
                {verification.diff_summary ? <p className="text-sm whitespace-pre-wrap">{verification.diff_summary}</p> : null}
                <div className="flex gap-2">{saveButton(false)}</div>
            </div>
        );
    } else if (status === "not_kept" && verification) {
        body = (
            <div className="space-y-3">
                <p className="text-sm tabular-nums">
                    <span aria-hidden>{CROSS} </span>
                    {notKeptText(verification)}
                </p>
                <OutcomeLines verification={verification} />
                <SampleList samples={failingCandidates} label="Failing replies from the new version" />
                <div className="flex gap-2">
                    <Button variant="outline" onClick={tryAnother}>Try another fix</Button>
                    <Button asChild variant="ghost">
                        <Link href={`/workflow/${workflowId}`}>Edit manually</Link>
                    </Button>
                </div>
            </div>
        );
    } else if (status === "not_reproducible" && verification) {
        body = (
            <div className="space-y-3">
                <p className="text-sm tabular-nums">
                    <span aria-hidden>{WARN} </span>
                    {notReproducibleText(verification)}
                </p>
                {verification.diff_summary ? <p className="text-sm whitespace-pre-wrap">{verification.diff_summary}</p> : null}
                <div className="flex gap-2">{saveButton(true)}</div>
            </div>
        );
    } else if (status === "not_text_verifiable" && verification) {
        body = (
            <div className="space-y-3">
                <p className="text-sm">
                    <span aria-hidden>{WARN} </span>
                    This changes routing before the failing turn; verify on a real call.
                </p>
                {verification.diff_summary ? <p className="text-sm whitespace-pre-wrap">{verification.diff_summary}</p> : null}
                <div className="flex gap-2">{saveButton(true)}</div>
            </div>
        );
    } else {
        body = (
            <div className="space-y-3">
                <p className="text-sm text-destructive">
                    <span aria-hidden>{CROSS} </span>
                    {verification?.error || "Verification failed."}
                </p>
                <div className="flex gap-2">
                    <Button variant="outline" onClick={tryAnother}>Try again</Button>
                    <Button variant="ghost" onClick={cancel}>Close</Button>
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-3 rounded-lg border bg-background p-4">
            <div aria-live="polite" className="space-y-3">
                {body}
                {saveState === "saved" ? (
                    <div className="flex flex-wrap items-center gap-2 text-sm">
                        <span>
                            Saved as draft{savedVersion !== null ? ` v${savedVersion}` : ""}. Publish it, then confirm on a
                            real call.
                        </span>
                        <Button variant="link" className="h-auto p-0" onClick={() => void reopenDiff()}>
                            View changes
                        </Button>
                    </div>
                ) : null}
                {saveState === "conflict" ? (
                    <div className="flex flex-wrap items-center gap-2 text-sm">
                        <span>
                            <span aria-hidden>{WARN} </span>Someone edited this draft since you opened it.
                        </span>
                        <Button asChild variant="outline" size="sm">
                            <Link href={`/workflow/${workflowId}`} target="_blank" rel="noopener noreferrer">
                                Review their changes
                            </Link>
                        </Button>
                    </div>
                ) : null}
                {saveState === "hash_mismatch" ? (
                    <div className="flex flex-wrap items-center gap-2 text-sm">
                        <span>
                            <span aria-hidden>{WARN} </span>This version changed since it was tested.
                        </span>
                        <Button variant="outline" size="sm" onClick={start} disabled={busy}>
                            Re-verify
                        </Button>
                    </div>
                ) : null}
                {saveState === "error" && saveError ? <p className="text-sm text-destructive">{saveError}</p> : null}
            </div>
            {diffPair ? (
                <WorkflowVersionDiffDialog
                    open
                    onOpenChange={closeDiff}
                    previousVersion={diffPair.previous}
                    selectedVersion={diffPair.selected}
                />
            ) : null}
        </div>
    );
}
