"use client";

import Link from "next/link";
import { useId, useMemo, useState } from "react";

import { CHECK, CROSS, WARN } from "@/components/doctor/ReportStates";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { applyProposal, discardProposal } from "@/lib/copilot/api";
import {
    appliedText,
    applyButtonLabel,
    applyErrorText,
    changeFieldDiffs,
    dependencyLabels,
    groupChanges,
    OP_LABEL,
    plural,
    STALE_TEXT,
} from "@/lib/copilot/format";
import { orderedSelection, selectAll, toggleChange, validateSelection } from "@/lib/copilot/selection";
import type { ApplyProposalResponse, ChangeItem, Proposal } from "@/lib/copilot/types";
import { appliedRoundText } from "@/lib/self-improve/format";

import { FieldDiffView } from "./FieldDiffView";

export interface ProposalCardProps {
    proposal: Proposal;
    /**
     * Optimistic lock for apply: the draft version the user is looking at
     * (null when there is no draft yet). Ignored for create proposals.
     */
    draftVersion?: number | string | null;
    /** The canvas has unsaved edits: applying would reload over them. */
    hasUnsavedChanges?: boolean;
    /** A copilot turn is running: actions wait for it. */
    busy?: boolean;
    onProposalChange: (proposal: Proposal) => void;
    /** After a successful save (edit: reload the canvas; create: open the workflow). */
    onApplied?: (proposal: Proposal, result: ApplyProposalResponse) => void | Promise<void>;
    /** Ask the copilot to redo a stale proposal. */
    onAskRedo?: () => void;
    /** Reload the editor canvas from the saved working copy. */
    onReloadCanvas?: () => void;
}

type Phase = "idle" | "applying" | "discarding";

function ChangeRow({
    change,
    changes,
    selectable,
    checked,
    onCheckedChange,
}: {
    change: ChangeItem;
    changes: readonly ChangeItem[];
    selectable: boolean;
    checked: boolean;
    onCheckedChange: (checked: boolean) => void;
}) {
    const [open, setOpen] = useState(false);
    const baseId = useId();
    const checkboxId = `${baseId}-check`;
    const panelId = `${baseId}-diff`;
    const diffs = changeFieldDiffs(change);
    const needs = dependencyLabels(change, changes);

    return (
        <li className="space-y-2 py-2">
            <div className="flex items-start gap-2">
                {selectable ? (
                    <div className="flex min-h-11 items-center md:min-h-0 md:pt-0.5">
                        <Checkbox
                            id={checkboxId}
                            checked={checked}
                            onCheckedChange={(value) => onCheckedChange(value === true)}
                        />
                    </div>
                ) : null}
                <div className="min-w-0 flex-1 space-y-1">
                    <div className="flex flex-wrap items-center gap-2">
                        <Badge variant="outline" className="font-normal">
                            {OP_LABEL[change.op]}
                        </Badge>
                        {selectable ? (
                            <label htmlFor={checkboxId} className="min-w-0 flex-1 cursor-pointer break-words text-sm">
                                {change.label}
                            </label>
                        ) : (
                            <span className="min-w-0 flex-1 break-words text-sm">{change.label}</span>
                        )}
                    </div>
                    {needs.length > 0 ? (
                        <p className="break-words text-xs text-muted-foreground">needs: {needs.join(", ")}</p>
                    ) : null}
                    {diffs.length > 0 ? (
                        <Button
                            variant="link"
                            className="h-auto min-h-11 p-0 text-xs md:min-h-0"
                            aria-expanded={open}
                            aria-controls={panelId}
                            onClick={() => setOpen((v) => !v)}
                        >
                            {open ? "Hide changes" : `Show changes (${plural(diffs.length, "field")})`}
                        </Button>
                    ) : null}
                </div>
            </div>
            {open ? (
                <div id={panelId} className="space-y-3">
                    {diffs.map((diff, i) => (
                        <FieldDiffView key={`${diff.field}-${i}`} diff={diff} />
                    ))}
                </div>
            ) : null}
        </li>
    );
}

/**
 * A proposed set of workflow changes with per-change accept/reject. Edit
 * proposals save only the checked changes as a draft; create proposals are
 * all-or-nothing. Publishing always stays manual.
 */
export function ProposalCard({
    proposal,
    draftVersion = null,
    hasUnsavedChanges = false,
    busy = false,
    onProposalChange,
    onApplied,
    onAskRedo,
    onReloadCanvas,
}: ProposalCardProps) {
    const isCreate = proposal.kind === "create";
    const pending = proposal.status === "pending";
    /** Another apply holds the claim (M2b O2): show it, keep the buttons disabled. */
    const claimed = proposal.status === "applying";
    const acceptedIds = proposal.accepted_change_ids ?? [];
    const [selected, setSelected] = useState<ReadonlySet<string>>(() => selectAll(proposal.changes));
    const [phase, setPhase] = useState<Phase>("idle");
    const [failure, setFailure] = useState<ApplyProposalResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [appliedCount, setAppliedCount] = useState<number | null>(null);

    const groups = useMemo(() => groupChanges(proposal.changes), [proposal.changes]);
    const check = validateSelection(proposal.changes, selected);
    const selectedCount = orderedSelection(proposal.changes, selected).length;
    const selectable = pending && !isCreate;
    const canApply = pending && phase === "idle" && !busy && (isCreate || (!hasUnsavedChanges && check.ok));

    const apply = async () => {
        if (!canApply) return;
        setPhase("applying");
        setFailure(null);
        setError(null);
        const accepted = isCreate ? [] : orderedSelection(proposal.changes, selected);
        try {
            const result = await applyProposal(proposal.proposal_uuid, {
                accepted_change_ids: accepted,
                expected_draft_version: isCreate ? null : draftVersion,
            });
            if (result.saved || result.error_code === "already_applied") {
                const next: Proposal = result.proposal ?? {
                    ...proposal,
                    status: "applied",
                    workflow_id: result.workflow_id ?? proposal.workflow_id,
                };
                setAppliedCount(isCreate ? proposal.changes.length : accepted.length);
                onProposalChange(next);
                if (result.saved) await onApplied?.(next, result);
                return;
            }
            if (result.error_code === "stale") {
                onProposalChange(result.proposal ?? { ...proposal, status: "stale" });
                return;
            }
            setFailure(result);
        } catch (e) {
            setError(e instanceof Error ? e.message : "Could not apply the changes");
        } finally {
            setPhase("idle");
        }
    };

    const discard = async () => {
        setPhase("discarding");
        setFailure(null);
        setError(null);
        try {
            onProposalChange(await discardProposal(proposal.proposal_uuid));
        } catch (e) {
            setError(e instanceof Error ? e.message : "Could not discard the proposal");
        } finally {
            setPhase("idle");
        }
    };

    const workflowHref = proposal.workflow_id ? `/workflow/${proposal.workflow_id}` : null;

    return (
        <section aria-label="Proposed changes" className="space-y-3 rounded-lg border bg-background p-3">
            <div className="space-y-1">
                <h3 className="text-sm font-medium">
                    {isCreate ? `New workflow${proposal.workflow_name ? `: ${proposal.workflow_name}` : ""}` : "Proposed changes"}
                </h3>
                <p className="whitespace-pre-wrap break-words text-sm">{proposal.summary}</p>
                <p className="text-xs text-muted-foreground tabular-nums">
                    {plural(proposal.changes.length, "change")}
                    {selectable ? ` · ${selectedCount} selected` : ""}
                </p>
            </div>

            {groups.map((group) => (
                <div key={group.target} className="space-y-1">
                    <h4 className="text-xs font-medium text-muted-foreground tabular-nums">
                        {group.label} ({group.changes.length})
                    </h4>
                    <ul className="divide-y">
                        {group.changes.map((change) => (
                            <ChangeRow
                                key={change.change_id}
                                change={change}
                                changes={proposal.changes}
                                selectable={selectable}
                                checked={selected.has(change.change_id)}
                                onCheckedChange={(value) =>
                                    setSelected((s) => toggleChange(proposal.changes, s, change.change_id, value))
                                }
                            />
                        ))}
                    </ul>
                </div>
            ))}

            <div aria-live="polite" className="space-y-2 text-sm">
                {proposal.status === "applied" ? (
                    <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                        <span>
                            <span aria-hidden>{CHECK} </span>
                            {!isCreate && acceptedIds.length > 0
                                ? appliedRoundText(acceptedIds.length, proposal.changes.length)
                                : appliedCount !== null
                                ? appliedText(proposal, appliedCount)
                                : isCreate
                                  ? "Workflow created as a draft."
                                  : "Saved as a draft. Publish when ready."}
                        </span>
                        {!isCreate && onReloadCanvas ? (
                            <Button variant="link" className="h-auto min-h-11 p-0 md:min-h-0" onClick={onReloadCanvas}>
                                Reload canvas
                            </Button>
                        ) : workflowHref ? (
                            <Link href={workflowHref} className="underline underline-offset-2">
                                Open workflow
                            </Link>
                        ) : null}
                    </div>
                ) : null}
                {proposal.status === "discarded" ? <p className="text-muted-foreground">Discarded.</p> : null}
                {proposal.status === "stale" ? (
                    <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                        <span>
                            <span aria-hidden>{WARN} </span>
                            {STALE_TEXT}
                        </span>
                        {onAskRedo ? (
                            <Button
                                variant="outline"
                                size="sm"
                                className="min-h-11 md:min-h-8"
                                onClick={onAskRedo}
                                disabled={busy}
                            >
                                Ask to redo it
                            </Button>
                        ) : null}
                    </div>
                ) : null}
                {failure ? (
                    <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                        <span className="whitespace-pre-wrap break-words text-destructive">
                            <span aria-hidden>{failure.error_code === "draft_conflict" ? WARN : CROSS} </span>
                            {applyErrorText(failure)}
                        </span>
                        {failure.error_code === "draft_conflict" && onReloadCanvas ? (
                            <Button variant="outline" size="sm" className="min-h-11 md:min-h-8" onClick={onReloadCanvas}>
                                Reload canvas
                            </Button>
                        ) : null}
                    </div>
                ) : null}
                {error ? (
                    <p className="text-destructive">
                        <span aria-hidden>{CROSS} </span>
                        {error}
                    </p>
                ) : null}
                {pending && !isCreate && !check.ok && !check.empty ? (
                    <p className="text-destructive">
                        <span aria-hidden>{WARN} </span>
                        Some selected changes need changes that aren&apos;t selected.
                    </p>
                ) : null}
                {pending && hasUnsavedChanges && !isCreate ? (
                    <p className="text-muted-foreground">
                        <span aria-hidden>{WARN} </span>
                        The canvas has unsaved edits. Save them first: applying reloads the canvas.
                    </p>
                ) : null}
            </div>

            {pending || claimed ? (
                <div className="flex flex-wrap gap-2">
                    <Button className="min-h-11 md:min-h-9 tabular-nums" onClick={apply} disabled={!canApply}>
                        {phase === "applying" || claimed
                            ? isCreate
                                ? "Creating…"
                                : "Applying…"
                            : applyButtonLabel(proposal, selectedCount)}
                    </Button>
                    <Button
                        variant="outline"
                        className="min-h-11 md:min-h-9"
                        onClick={discard}
                        disabled={phase !== "idle" || busy || claimed}
                    >
                        {phase === "discarding" ? "Discarding…" : "Discard"}
                    </Button>
                </div>
            ) : null}
        </section>
    );
}
