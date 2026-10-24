"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { CROSS } from "@/components/doctor/ReportStates";
import { Skeleton } from "@/components/ui/skeleton";
import { draftBrief, getBrief, getProposal } from "@/lib/copilot/api";
import { briefDraftUnavailable, briefDraftUnavailableText } from "@/lib/copilot/brief";
import type { ApplyProposalResponse, BriefResponse, Proposal } from "@/lib/copilot/types";
import { listSelfImproveIterations,resumeSelfImprove, runSelfImproveNow, updateSelfImprove } from "@/lib/self-improve/api";
import { isProposalPublished, isRoundDone, noRoundsText, runNowText } from "@/lib/self-improve/format";
import type { IterationResponse } from "@/lib/self-improve/types";

import { BriefCard } from "./BriefCard";
import { type NoBriefChoice,NoBriefPrompt } from "./NoBriefPrompt";
import { RoundRow } from "./RoundRow";
import { RoundView } from "./RoundView";
import { StatusLine } from "./StatusLine";
import type { SelfImproveController } from "./useSelfImproveStatus";

const TERMINAL: ReadonlySet<Proposal["status"]> = new Set(["applied", "discarded", "stale"]);

export interface SelfImproveTabProps {
    workflowId: number;
    controller: SelfImproveController;
    /** Below 768px (DS-8): read + discard only. */
    mobile?: boolean;
    draftVersion?: number | string | null;
    hasUnsavedChanges?: boolean;
    /** A round's changes were saved as a draft: reload the canvas. */
    onApplied?: (proposal: Proposal, result: ApplyProposalResponse) => void | Promise<void>;
    onReloadCanvas?: () => void;
}

/** Changes whenever the rounds list may have changed, so it is reloaded then. */
function roundsSignature(controller: SelfImproveController): string {
    const s = controller.status;
    if (!s) return "";
    const it = s.current_iteration;
    return [s.state, s.to_review, it?.iteration_id, it?.status, it?.reviewed, it?.proposal_uuid, it?.updated_at].join("|");
}

/**
 * The Copilot panel's Self-improve tab, action first (DS-1):
 * 1. status line (switch + N behind the gear), 2. the round needing you,
 * expanded, 3. earlier rounds as collapsed rows, 4. the brief, collapsed.
 */
export function SelfImproveTab({
    workflowId,
    controller,
    mobile = false,
    draftVersion,
    hasUnsavedChanges,
    onApplied,
    onReloadCanvas,
}: SelfImproveTabProps) {
    const { status, refresh, replaceStatus } = controller;
    const [iterations, setIterations] = useState<IterationResponse[] | null>(null);
    const [roundsError, setRoundsError] = useState<string | null>(null);
    const [proposals, setProposals] = useState<Record<string, Proposal>>({});
    const [proposalErrors, setProposalErrors] = useState<Record<string, string>>({});
    const [brief, setBrief] = useState<BriefResponse | null>(null);
    const [briefLoading, setBriefLoading] = useState(true);
    const [briefError, setBriefError] = useState<string | null>(null);
    const [editingBrief, setEditingBrief] = useState(false);
    /** DS-6: turning on without a brief asks first; holds the N to turn on with. */
    const [askBrief, setAskBrief] = useState<{ everyN: number } | null>(null);
    const [actionError, setActionError] = useState<string | null>(null);
    const [busy, setBusy] = useState(false);
    /** The round to focus after apply/discard (DS-8). */
    const [focusId, setFocusId] = useState<number | null>(null);
    const proposalsRef = useRef(proposals);
    proposalsRef.current = proposals;

    const loadProposal = useCallback(async (uuid: string) => {
        try {
            const p = await getProposal(uuid);
            setProposals((all) => ({ ...all, [uuid]: p }));
            setProposalErrors((all) => {
                if (!(uuid in all)) return all;
                const rest = { ...all };
                delete rest[uuid];
                return rest;
            });
        } catch (e) {
            setProposalErrors((all) => ({ ...all, [uuid]: e instanceof Error ? e.message : "Could not load the changes" }));
        }
    }, []);

    const loadRounds = useCallback(async () => {
        try {
            const { iterations: list } = await listSelfImproveIterations(workflowId);
            setIterations(list);
            setRoundsError(null);
            // Proposals: fetch the ones not seen yet or not final (a final one never changes).
            const uuids = list
                .map((it) => it.proposal_uuid)
                .filter((u): u is string => !!u)
                .filter((u) => !proposalsRef.current[u] || !TERMINAL.has(proposalsRef.current[u].status));
            await Promise.all(uuids.map(loadProposal));
        } catch (e) {
            setRoundsError(e instanceof Error ? e.message : "Could not load rounds");
        }
    }, [workflowId, loadProposal]);

    const loadBrief = useCallback(async () => {
        setBriefLoading(true);
        try {
            setBrief(await getBrief(workflowId));
            setBriefError(null);
        } catch (e) {
            setBriefError(e instanceof Error ? e.message : "Could not load the brief");
        } finally {
            setBriefLoading(false);
        }
    }, [workflowId]);

    const signature = roundsSignature(controller);
    useEffect(() => {
        if (!signature) return;
        void loadRounds();
    }, [signature, loadRounds]);

    useEffect(() => {
        void loadBrief();
    }, [loadBrief]);

    // ---- current round vs earlier rounds ---------------------------------

    const current = status?.current_iteration ?? iterations?.[0] ?? null;
    const currentProposal = current?.proposal_uuid ? proposals[current.proposal_uuid] : undefined;
    const currentExpanded =
        !!current &&
        (!(current.reviewed || isRoundDone(current, currentProposal)) ||
            current.status === "no_changes" ||
            current.status === "failed");
    const earlier = useMemo(
        () => (iterations ?? []).filter((it) => !(currentExpanded && current && it.iteration_id === current.iteration_id)),
        [iterations, current, currentExpanded],
    );

    // ---- actions -----------------------------------------------------------

    const putSettings = useCallback(
        async (enabled: boolean, everyN: number) => {
            replaceStatus(await updateSelfImprove(workflowId, { enabled, every_n_calls: everyN }));
        },
        [workflowId, replaceStatus],
    );

    const saveSettings = async (enabled: boolean, everyN: number) => {
        if (!status) return;
        const turningOn = enabled && !status.settings.enabled;
        if (turningOn && !status.has_brief) {
            setAskBrief({ everyN });
            return;
        }
        await putSettings(enabled, everyN);
        setAskBrief(null);
    };

    const turnOn = async () => {
        if (!status) return;
        setActionError(null);
        if (!status.has_brief) {
            setAskBrief({ everyN: status.settings.every_n_calls });
            return;
        }
        setBusy(true);
        try {
            await putSettings(true, status.settings.every_n_calls);
        } catch (e) {
            setActionError(e instanceof Error ? e.message : "Could not turn on Self-improve");
        } finally {
            setBusy(false);
        }
    };

    const chooseBrief = async (choice: NoBriefChoice) => {
        const everyN = askBrief?.everyN ?? status?.settings.every_n_calls ?? 10;
        await putSettings(true, everyN);
        setAskBrief(null);
        if (choice === "write") {
            setEditingBrief(true);
        } else if (choice === "draft") {
            try {
                const { brief_change } = await draftBrief(workflowId);
                setBrief((b) =>
                    b
                        ? { ...b, pending_change: brief_change }
                        : {
                              workflow_id: workflowId,
                              brief: brief_change.proposed,
                              version: brief_change.base_version,
                              pending_change: brief_change,
                          },
                );
                // The server's view (pending_change, version) wins once it answers.
                void loadBrief();
            } catch (e) {
                const unavailable = briefDraftUnavailable(e);
                if (unavailable === "no_chat_history") setEditingBrief(true);
                setActionError(
                    unavailable
                        ? briefDraftUnavailableText(unavailable)
                        : e instanceof Error
                          ? e.message
                          : "Could not draft a brief from your chats",
                );
            }
        }
    };

    const resume = async () => {
        setActionError(null);
        try {
            replaceStatus(await resumeSelfImprove(workflowId));
        } catch (e) {
            setActionError(e instanceof Error ? e.message : "Could not resume Self-improve");
        }
    };

    const runNow = async () => {
        const result = await runSelfImproveNow(workflowId);
        void refresh();
        return runNowText(result);
    };

    const onProposalChange = (proposal: Proposal) => {
        setProposals((all) => ({ ...all, [proposal.proposal_uuid]: proposal }));
        if (proposal.status === "applied" && !proposal.accepted_change_ids?.length) {
            // The apply response carried no proposal: fetch it for "Applied 3 of 5 changes".
            void loadProposal(proposal.proposal_uuid);
        }
        if (TERMINAL.has(proposal.status)) {
            // Reviewed (D4): the next window starts; move focus to the round's row.
            const owner = (iterations ?? []).find((it) => it.proposal_uuid === proposal.proposal_uuid) ?? current;
            if (owner) setFocusId(owner.iteration_id);
            void refresh();
        }
    };

    const clearFocus = useCallback(() => setFocusId(null), []);

    if (!status) return null;

    const gated = status.state === "ai_disabled" || status.state === "no_key";
    const noRounds = iterations !== null && iterations.length === 0 && !current;

    return (
        <div>
            <StatusLine
                status={status}
                onSaveSettings={saveSettings}
                onTurnOn={turnOn}
                onResume={resume}
                onRunNow={runNow}
                onAiEnabled={() => void refresh()}
                busy={busy}
                error={actionError}
            />

            {askBrief ? <NoBriefPrompt onChoose={chooseBrief} /> : null}

            {!gated && current && currentExpanded ? (
                <RoundView
                    workflowId={workflowId}
                    iteration={current}
                    status={status}
                    proposal={current.proposal_uuid ? currentProposal : null}
                    proposalError={current.proposal_uuid ? proposalErrors[current.proposal_uuid] : null}
                    onRetryProposal={current.proposal_uuid ? () => void loadProposal(current.proposal_uuid!) : undefined}
                    onProposalChange={onProposalChange}
                    mobile={mobile}
                    draftVersion={draftVersion}
                    hasUnsavedChanges={hasUnsavedChanges}
                    onApplied={onApplied}
                    onReloadCanvas={onReloadCanvas}
                    focusRequested={focusId === current.iteration_id}
                />
            ) : null}

            {!gated && status.settings.enabled && noRounds ? (
                <p className="border-b px-4 py-3 text-sm tabular-nums">{noRoundsText(status.window_size)}</p>
            ) : null}

            {iterations === null && !roundsError && status.settings.enabled ? (
                <div className="space-y-2 border-b px-4 py-3" aria-hidden>
                    <Skeleton className="h-4 w-2/3" />
                    <Skeleton className="h-4 w-1/2" />
                </div>
            ) : null}

            {roundsError ? (
                <p className="border-b px-4 py-3 text-sm text-destructive">
                    <span aria-hidden>{CROSS} </span>
                    {roundsError}
                </p>
            ) : null}

            {earlier.length > 0 ? (
                <section aria-label="Earlier rounds" className="border-b px-4 py-3">
                    <h3 className="text-xs font-medium text-muted-foreground">Earlier rounds</h3>
                    <ul className="divide-y">
                        {earlier.map((it) => (
                            <RoundRow
                                key={it.iteration_id}
                                workflowId={workflowId}
                                iteration={it}
                                proposal={it.proposal_uuid ? proposals[it.proposal_uuid] : null}
                                published={!!it.proposal_uuid && isProposalPublished(it.proposal_uuid, iterations ?? [])}
                                focusRequested={focusId === it.iteration_id}
                                onFocused={clearFocus}
                            />
                        ))}
                    </ul>
                </section>
            ) : null}

            <BriefCard
                workflowId={workflowId}
                brief={brief}
                loading={briefLoading}
                error={briefError}
                onRetry={() => void loadBrief()}
                onBriefChange={(b) => {
                    setBrief(b);
                    if (!status.has_brief && !b.pending_change) void refresh();
                }}
                editing={editingBrief}
                onEditingChange={setEditingBrief}
            />
        </div>
    );
}
