/**
 * Pure helpers for the workflow brief (North Star, M2b R15/D2, DS-9):
 * normalising what the editor produces, and per-field diffs for the
 * brief-change card (rendered with FieldDiffView).
 */
import type { FieldDiff, WorkflowBrief } from "./types";

export type BriefListField = "must_do" | "must_never" | "accepted_tradeoffs";
export type BriefField = "goal" | BriefListField | "notes";

/** Display order and visible labels (DS-9: labels, never placeholder-as-label). */
export const BRIEF_FIELDS: readonly BriefField[] = ["goal", "must_do", "must_never", "accepted_tradeoffs", "notes"];

export const BRIEF_FIELD_LABEL: Record<BriefField, string> = {
    goal: "Goal",
    must_do: "Must do",
    must_never: "Must never",
    accepted_tradeoffs: "Accepted tradeoffs",
    notes: "Notes",
};

export const BRIEF_LIST_FIELDS: readonly BriefListField[] = ["must_do", "must_never", "accepted_tradeoffs"];

export const isListField = (field: BriefField): field is BriefListField =>
    (BRIEF_LIST_FIELDS as readonly string[]).includes(field);

export const emptyBrief = (): WorkflowBrief => ({
    goal: "",
    must_do: [],
    must_never: [],
    accepted_tradeoffs: [],
    notes: "",
});

/** Tolerates partial/missing data from the API (fields default like the contract). */
export function coerceBrief(brief: Partial<WorkflowBrief> | null | undefined): WorkflowBrief {
    return {
        goal: brief?.goal ?? "",
        must_do: [...(brief?.must_do ?? [])],
        must_never: [...(brief?.must_never ?? [])],
        accepted_tradeoffs: [...(brief?.accepted_tradeoffs ?? [])],
        notes: brief?.notes ?? "",
    };
}

/** Trim text, drop empty list items: what Save sends and what diffs compare. */
export function normalizeBrief(brief: Partial<WorkflowBrief> | null | undefined): WorkflowBrief {
    const b = coerceBrief(brief);
    const list = (items: string[]) => items.map((s) => s.trim()).filter((s) => s !== "");
    return {
        goal: b.goal.trim(),
        must_do: list(b.must_do),
        must_never: list(b.must_never),
        accepted_tradeoffs: list(b.accepted_tradeoffs),
        notes: b.notes.trim(),
    };
}

export function isBriefEmpty(brief: Partial<WorkflowBrief> | null | undefined): boolean {
    const b = normalizeBrief(brief);
    return BRIEF_FIELDS.every((f) => (isListField(f) ? b[f].length === 0 : b[f] === ""));
}

/** The text a field shows in a diff: lists one "- item" per line, so line diffs map to items. */
export function briefFieldText(brief: WorkflowBrief, field: BriefField): string {
    if (isListField(field)) return brief[field].map((item) => `- ${item}`).join("\n");
    return brief[field];
}

/**
 * Per-field diffs between two briefs, in display order; unchanged fields are
 * left out. `field` is the visible label (FieldDiffView shows it as the title).
 */
export function briefFieldDiffs(
    before: Partial<WorkflowBrief> | null | undefined,
    after: Partial<WorkflowBrief> | null | undefined,
): FieldDiff[] {
    const a = normalizeBrief(before);
    const b = normalizeBrief(after);
    const diffs: FieldDiff[] = [];
    for (const field of BRIEF_FIELDS) {
        const left = briefFieldText(a, field);
        const right = briefFieldText(b, field);
        if (left !== right) diffs.push({ field: BRIEF_FIELD_LABEL[field], before: left, after: right });
    }
    return diffs;
}

export const briefsEqual = (a: Partial<WorkflowBrief> | null | undefined, b: Partial<WorkflowBrief> | null | undefined) =>
    briefFieldDiffs(a, b).length === 0;

/** "Brief: <goal in one line>" for the collapsed card (DS-1, 4th). */
export function briefGoalLine(brief: Partial<WorkflowBrief> | null | undefined): string {
    const goal = normalizeBrief(brief).goal;
    const first = goal.split(/\r?\n/)[0] ?? "";
    return first;
}

/** Copy (DS-9). */
export const BRIEF_CONFLICT_TEXT = "Someone changed the brief since you opened it.";
export const BRIEF_CHANGE_STALE_TEXT = "The brief changed since this was suggested.";
export const BRIEF_NONE_TEXT = "No brief yet.";

/** The 422 `detail` codes POST /brief/draft returns when there is nothing to draft. */
export type BriefDraftUnavailable = "no_chat_history" | "no_new_information";

/** Reads a draft-unavailable code from a failed draft call, if that is why it failed. */
export function briefDraftUnavailable(error: unknown): BriefDraftUnavailable | null {
    const body = (error as { body?: { detail?: unknown } } | null)?.body;
    const detail = body?.detail;
    return detail === "no_chat_history" || detail === "no_new_information" ? detail : null;
}

/** User-facing copy for a draft that could not be made. */
export function briefDraftUnavailableText(reason: BriefDraftUnavailable): string {
    return reason === "no_chat_history"
        ? "There are no Copilot chats for this workflow yet. Write the brief yourself."
        : "Your chats don't add anything new to the brief.";
}
