import { describe, expect, it } from "vitest";

import {
    briefDraftUnavailable,
    briefDraftUnavailableText,
    briefFieldDiffs,
    briefGoalLine,
    briefsEqual,
    coerceBrief,
    emptyBrief,
    isBriefEmpty,
    normalizeBrief,
} from "./brief";
import type { WorkflowBrief } from "./types";

const brief = (over: Partial<WorkflowBrief> = {}): WorkflowBrief => ({ ...emptyBrief(), ...over });

describe("normalizeBrief", () => {
    it("trims text and drops empty list items", () => {
        expect(
            normalizeBrief({ goal: "  Book demos \n", must_do: [" Collect email ", "", "  "], notes: " n " }),
        ).toEqual({ goal: "Book demos", must_do: ["Collect email"], must_never: [], accepted_tradeoffs: [], notes: "n" });
    });

    it("fills missing fields like the contract defaults", () => {
        expect(coerceBrief(null)).toEqual(emptyBrief());
        expect(coerceBrief({ goal: "x" })).toEqual(brief({ goal: "x" }));
    });
});

describe("isBriefEmpty", () => {
    it("is empty when every field is blank after trimming", () => {
        expect(isBriefEmpty(brief({ goal: "  ", must_do: [""] }))).toBe(true);
        expect(isBriefEmpty(null)).toBe(true);
        expect(isBriefEmpty(brief({ must_never: ["Promise refunds"] }))).toBe(false);
    });
});

describe("briefFieldDiffs", () => {
    it("returns only changed fields, labelled, in display order", () => {
        const before = brief({ goal: "Book demos", must_do: ["Collect email"], notes: "Keep it short" });
        const after = brief({
            goal: "Book demos",
            must_do: ["Collect email", "Confirm time zone"],
            must_never: ["Quote prices"],
            notes: "Keep it short",
        });
        expect(briefFieldDiffs(before, after)).toEqual([
            { field: "Must do", before: "- Collect email", after: "- Collect email\n- Confirm time zone" },
            { field: "Must never", before: "", after: "- Quote prices" },
        ]);
    });

    it("diffs the goal and notes as plain text", () => {
        expect(briefFieldDiffs(brief({ goal: "A" }), brief({ goal: "B", notes: "N" }))).toEqual([
            { field: "Goal", before: "A", after: "B" },
            { field: "Notes", before: "", after: "N" },
        ]);
    });

    it("ignores whitespace-only and empty-item differences", () => {
        const a = brief({ goal: "Goal", accepted_tradeoffs: ["Slower replies"] });
        const b = brief({ goal: " Goal ", accepted_tradeoffs: ["Slower replies", " "] });
        expect(briefFieldDiffs(a, b)).toEqual([]);
        expect(briefsEqual(a, b)).toBe(true);
    });

    it("treats a missing brief as empty", () => {
        expect(briefFieldDiffs(null, brief({ goal: "New" }))).toEqual([{ field: "Goal", before: "", after: "New" }]);
    });
});

describe("briefGoalLine", () => {
    it("is the goal's first line", () => {
        expect(briefGoalLine(brief({ goal: "Qualify leads\nand book demos" }))).toBe("Qualify leads");
        expect(briefGoalLine(null)).toBe("");
    });
});

describe("brief draft unavailable", () => {
    it("reads the 422 detail code and maps it to copy", () => {
        const noChats = { body: { detail: "no_chat_history" } };
        expect(briefDraftUnavailable(noChats)).toBe("no_chat_history");
        expect(briefDraftUnavailableText("no_chat_history")).toBe(
            "There are no Copilot chats for this workflow yet. Write the brief yourself.",
        );
        expect(briefDraftUnavailable({ body: { detail: "no_new_information" } })).toBe("no_new_information");
        expect(briefDraftUnavailable(new Error("boom"))).toBeNull();
        expect(briefDraftUnavailable(null)).toBeNull();
    });
});
