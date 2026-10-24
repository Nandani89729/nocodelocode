import { describe, expect, it } from "vitest";

import {
    dependencyClosure,
    dependentClosure,
    orderedSelection,
    selectAll,
    toggleChange,
    validateSelection,
} from "./selection";
import type { ChangeItem } from "./types";

const change = (change_id: string, depends_on: string[] = []): ChangeItem => ({
    change_id,
    target: change_id.startsWith("e:") ? "edge" : "node",
    op: "add",
    ref_id: change_id,
    label: change_id,
    field_diffs: [],
    depends_on,
});

// n:2 (new node) <- e:1-2 (edge to it) ; n:3 <- e:2-3 needs n:2 and n:3 ; n:4 independent
// remove-node r:5 needs its edge removal e:5-1
const changes: ChangeItem[] = [
    change("n:2"),
    change("n:3"),
    change("e:1-2", ["n:2"]),
    change("e:2-3", ["n:2", "n:3"]),
    change("n:4"),
    change("e:5-1"),
    change("r:5", ["e:5-1"]),
];

describe("closures", () => {
    it("collects transitive dependencies, ignoring unknown ids", () => {
        expect([...dependencyClosure(changes, ["e:2-3"])].sort()).toEqual(["e:2-3", "n:2", "n:3"]);
        expect([...dependencyClosure([change("e:9", ["ghost"])], ["e:9"])]).toEqual(["e:9"]);
    });

    it("collects transitive dependents", () => {
        expect([...dependentClosure(changes, ["n:2"])].sort()).toEqual(["e:1-2", "e:2-3", "n:2"]);
    });

    it("terminates on dependency cycles", () => {
        const cyclic = [change("a", ["b"]), change("b", ["a"])];
        expect([...dependencyClosure(cyclic, ["a"])].sort()).toEqual(["a", "b"]);
        expect([...dependentClosure(cyclic, ["a"])].sort()).toEqual(["a", "b"]);
    });
});

describe("toggleChange", () => {
    it("accepting a change selects its depends_on closure", () => {
        const next = toggleChange(changes, new Set(), "e:2-3", true);
        expect(orderedSelection(changes, next)).toEqual(["n:2", "n:3", "e:2-3"]);
    });

    it("un-accepting a change unselects the changes that depend on it", () => {
        const next = toggleChange(changes, selectAll(changes), "n:2", false);
        expect(orderedSelection(changes, next)).toEqual(["n:3", "n:4", "e:5-1", "r:5"]);
    });

    it("unchecking an edge removal also unchecks the node removal that needs it", () => {
        const next = toggleChange(changes, selectAll(changes), "e:5-1", false);
        expect(next.has("r:5")).toBe(false);
        expect(next.has("n:4")).toBe(true);
    });

    it("does not mutate the input selection", () => {
        const before = selectAll(changes);
        toggleChange(changes, before, "n:2", false);
        expect(before.size).toBe(changes.length);
    });
});

describe("validateSelection", () => {
    it("passes a closed selection", () => {
        expect(validateSelection(changes, selectAll(changes))).toEqual({ ok: true, empty: false, problems: [] });
    });

    it("reports missing in-proposal dependencies", () => {
        const check = validateSelection(changes, new Set(["e:2-3", "n:3"]));
        expect(check.ok).toBe(false);
        expect(check.problems).toEqual([{ change_id: "e:2-3", missing: ["n:2"] }]);
    });

    it("rejects an empty selection, and ids not in the proposal don't count", () => {
        expect(validateSelection(changes, new Set())).toMatchObject({ ok: false, empty: true });
        expect(validateSelection(changes, new Set(["nope"]))).toMatchObject({ ok: false, empty: true });
    });

    it("leaves dependencies outside the proposal to the backend", () => {
        expect(validateSelection([change("e:9", ["ghost"])], new Set(["e:9"])).ok).toBe(true);
    });
});
