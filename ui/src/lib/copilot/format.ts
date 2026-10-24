/**
 * Pure helpers for the copilot panel: labels, change grouping, readable field
 * diffs and the user-facing wording for apply results.
 */
import { buildSideBySideDiffRows, type DiffSegment } from "@/app/workflow/[workflowId]/utils/workflowVersionDiff";

import type { ApplyProposalResponse, ChangeItem, ChangeOp, ChangeTarget, FieldDiff, Proposal } from "./types";

export const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

export const OP_LABEL: Record<ChangeOp, string> = {
    add: "Add",
    update: "Update",
    remove: "Remove",
};

export const TARGET_GROUP_LABEL: Record<ChangeTarget, string> = {
    node: "Nodes",
    edge: "Edges",
    workflow: "Settings",
    tool: "New tools",
};

const GROUP_ORDER: readonly ChangeTarget[] = ["node", "edge", "workflow", "tool"];

export interface ChangeGroup {
    target: ChangeTarget;
    label: string;
    changes: ChangeItem[];
}

/** Nodes, then edges, then settings, then new tools; empty groups dropped. */
export function groupChanges(changes: readonly ChangeItem[]): ChangeGroup[] {
    return GROUP_ORDER.map((target) => ({
        target,
        label: TARGET_GROUP_LABEL[target],
        changes: changes.filter((c) => c.target === target),
    })).filter((g) => g.changes.length > 0);
}

/** "get_workflow_details" → "get workflow details" for tool chips. */
export const toolLabel = (name: string) => name.replace(/[_-]+/g, " ").trim();

/** Diff text for a field value: strings as-is (prompts stay readable), the rest as JSON. */
export function valueText(value: unknown): string {
    if (value === null || value === undefined) return "";
    if (typeof value === "string") return value;
    try {
        return JSON.stringify(value, null, 2);
    } catch {
        return String(value);
    }
}

/**
 * The diffs a change shows when expanded. Adds and removes usually carry no
 * field_diffs, so they show the whole node/edge instead.
 */
export function changeFieldDiffs(change: ChangeItem): FieldDiff[] {
    if (change.field_diffs.length > 0) return change.field_diffs;
    if (change.op === "add" && change.after) return [{ field: change.target, before: null, after: change.after }];
    if (change.op === "remove" && change.before) return [{ field: change.target, before: change.before, after: null }];
    if (change.before || change.after) return [{ field: change.target, before: change.before, after: change.after }];
    return [];
}

type ContentLine = { kind: "unchanged" | "removed" | "added"; segments: DiffSegment[] };

export type DiffLine = ContentLine | { kind: "skipped"; count: number };

/**
 * Unified (single-column) diff for the narrow panel, built on the version
 * diff dialog's line + inline diff. Removed lines of a hunk come before its
 * added lines; unchanged runs longer than 2×context collapse to a
 * "n unchanged lines" marker.
 */
export function unifiedDiff(before: unknown, after: unknown, context = 2): DiffLine[] {
    const rows = buildSideBySideDiffRows(valueText(before), valueText(after));
    const lines: ContentLine[] = [];
    let removed: ContentLine[] = [];
    let added: ContentLine[] = [];
    const flush = () => {
        lines.push(...removed, ...added);
        removed = [];
        added = [];
    };
    for (const row of rows) {
        const changed = row.left?.kind === "removed" || row.right?.kind === "added";
        if (!changed) {
            flush();
            const text = row.left?.text ?? row.right?.text ?? "";
            lines.push({ kind: "unchanged", segments: [{ kind: "unchanged", text }] });
            continue;
        }
        if (row.left) removed.push({ kind: "removed", segments: row.left.segments ?? [{ kind: "removed", text: row.left.text }] });
        if (row.right) added.push({ kind: "added", segments: row.right.segments ?? [{ kind: "added", text: row.right.text }] });
    }
    flush();

    // Collapse long unchanged runs, keeping `context` lines next to changes.
    const out: DiffLine[] = [];
    let i = 0;
    while (i < lines.length) {
        if (lines[i].kind !== "unchanged") {
            out.push(lines[i]);
            i++;
            continue;
        }
        let j = i;
        while (j < lines.length && lines[j].kind === "unchanged") j++;
        const run = lines.slice(i, j);
        const keepHead = i === 0 ? 0 : context;
        const keepTail = j === lines.length ? 0 : context;
        if (run.length > keepHead + keepTail + 1) {
            out.push(...run.slice(0, keepHead));
            out.push({ kind: "skipped", count: run.length - keepHead - keepTail });
            out.push(...run.slice(run.length - keepTail));
        } else {
            out.push(...run);
        }
        i = j;
    }
    return out;
}

/** "needs: Add node "Book demo"" hint for a change with dependencies. */
export function dependencyLabels(change: ChangeItem, changes: readonly ChangeItem[]): string[] {
    const index = new Map(changes.map((c) => [c.change_id, c]));
    return change.depends_on.map((id) => index.get(id)?.label ?? id);
}

export const applyButtonLabel = (proposal: Proposal, selectedCount: number) =>
    proposal.kind === "create" ? "Create workflow" : `Apply selected (${selectedCount})`;

export const STALE_TEXT = "The workflow changed since this was proposed. Ask the copilot to redo it.";
export const REDO_MESSAGE = "The workflow changed since your last proposal. Please redo it against the current version.";
export const PLAN_APPROVED_MESSAGE = "Plan approved. Go ahead.";

/** User-facing line for a failed apply. */
export function applyErrorText(result: ApplyProposalResponse): string {
    switch (result.error_code) {
        case "stale":
            return STALE_TEXT;
        case "draft_conflict":
            return "Someone saved this workflow since you opened it. Reload the canvas to see their changes, then ask the copilot again.";
        case "missing_dependency":
            return result.error || "Some selected changes need changes that aren't selected.";
        case "validation_error":
        case "graph_validation":
            return result.error || "The changed workflow didn't pass validation.";
        case "not_found":
            return "This proposal no longer exists.";
        case "already_applied":
            return "These changes were already applied.";
        default:
            return result.error || "Could not apply the changes.";
    }
}

export const appliedText = (proposal: Proposal, count: number) =>
    proposal.kind === "create"
        ? "Workflow created as a draft. Opening it…"
        : `Saved ${plural(count, "change")} as a draft. Publish when ready.`;
