/**
 * Per-change accept/reject logic for an edit proposal (pure, no React).
 *
 * `depends_on` points from a change to the changes it needs: an added edge
 * needs its added endpoint nodes; removing a node needs its edges removed.
 * Accepting a change therefore accepts its whole depends_on closure, and
 * un-accepting a change un-accepts everything that (transitively) needs it,
 * so the selection is always closed under depends_on. Dependencies that
 * aren't in the proposal can't be selected and are left for the backend to
 * judge (it answers `missing_dependency`).
 */
import type { ChangeItem } from "./types";

export type Selection = ReadonlySet<string>;

function byId(changes: readonly ChangeItem[]): Map<string, ChangeItem> {
    return new Map(changes.map((c) => [c.change_id, c]));
}

/** Every change selected: the default for a fresh proposal. */
export function selectAll(changes: readonly ChangeItem[]): Set<string> {
    return new Set(changes.map((c) => c.change_id));
}

/** `ids` plus everything they transitively depend on (known ids only). */
export function dependencyClosure(changes: readonly ChangeItem[], ids: Iterable<string>): Set<string> {
    const index = byId(changes);
    const out = new Set<string>();
    const stack = [...ids];
    while (stack.length > 0) {
        const id = stack.pop()!;
        if (out.has(id) || !index.has(id)) continue;
        out.add(id);
        for (const dep of index.get(id)!.depends_on) stack.push(dep);
    }
    return out;
}

/** `ids` plus every change that transitively depends on any of them. */
export function dependentClosure(changes: readonly ChangeItem[], ids: Iterable<string>): Set<string> {
    const dependents = new Map<string, string[]>();
    for (const c of changes) {
        for (const dep of c.depends_on) {
            const list = dependents.get(dep) ?? [];
            list.push(c.change_id);
            dependents.set(dep, list);
        }
    }
    const out = new Set<string>();
    const stack = [...ids];
    while (stack.length > 0) {
        const id = stack.pop()!;
        if (out.has(id)) continue;
        out.add(id);
        for (const next of dependents.get(id) ?? []) stack.push(next);
    }
    return out;
}

/** Check or uncheck one change, keeping the selection closed under depends_on. */
export function toggleChange(
    changes: readonly ChangeItem[],
    selected: Selection,
    changeId: string,
    accept: boolean,
): Set<string> {
    const next = new Set(selected);
    if (accept) {
        for (const id of dependencyClosure(changes, [changeId])) next.add(id);
    } else {
        for (const id of dependentClosure(changes, [changeId])) next.delete(id);
    }
    return next;
}

export interface SelectionProblem {
    change_id: string;
    /** depends_on ids that are not selected */
    missing: string[];
}

export interface SelectionCheck {
    ok: boolean;
    /** nothing selected */
    empty: boolean;
    problems: SelectionProblem[];
}

/**
 * Validate before apply: at least one change, and every selected change has
 * its in-proposal dependencies selected too.
 */
export function validateSelection(changes: readonly ChangeItem[], selected: Selection): SelectionCheck {
    const index = byId(changes);
    const problems: SelectionProblem[] = [];
    for (const id of selected) {
        const change = index.get(id);
        if (!change) continue;
        const missing = change.depends_on.filter((dep) => index.has(dep) && !selected.has(dep));
        if (missing.length > 0) problems.push({ change_id: id, missing });
    }
    const empty = ![...selected].some((id) => index.has(id));
    return { ok: !empty && problems.length === 0, empty, problems };
}

/** Selected ids in proposal order (stable request bodies and tests). */
export function orderedSelection(changes: readonly ChangeItem[], selected: Selection): string[] {
    return changes.filter((c) => selected.has(c.change_id)).map((c) => c.change_id);
}
