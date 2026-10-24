import { describe, expect, it } from "vitest";

import { applyErrorText, changeFieldDiffs, groupChanges, unifiedDiff, valueText } from "./format";
import { parseInline, parseMarkdown } from "./markdown";
import type { ChangeItem } from "./types";

const item = (over: Partial<ChangeItem>): ChangeItem => ({
    change_id: "x",
    target: "node",
    op: "update",
    ref_id: "1",
    label: "x",
    field_diffs: [],
    depends_on: [],
    ...over,
});

describe("groupChanges", () => {
    it("orders Nodes, Edges, Settings, New tools and drops empty groups", () => {
        const groups = groupChanges([item({ change_id: "e", target: "edge" }), item({ change_id: "n" })]);
        expect(groups.map((g) => g.label)).toEqual(["Nodes", "Edges"]);
        const all = groupChanges([
            item({ change_id: "t:book_demo", target: "tool" }),
            item({ change_id: "s:max_call_duration", target: "workflow" }),
            item({ change_id: "n" }),
        ]);
        expect(all.map((g) => g.label)).toEqual(["Nodes", "Settings", "New tools"]);
    });
});

describe("changeFieldDiffs", () => {
    it("falls back to the whole node for adds and removes", () => {
        expect(changeFieldDiffs(item({ op: "add", after: { a: 1 } }))).toEqual([{ field: "node", before: null, after: { a: 1 } }]);
        expect(changeFieldDiffs(item({ op: "remove", before: { a: 1 } }))[0].before).toEqual({ a: 1 });
        const fd = [{ field: "prompt", before: "a", after: "b" }];
        expect(changeFieldDiffs(item({ field_diffs: fd }))).toBe(fd);
    });
});

describe("unifiedDiff", () => {
    it("keeps prompts as text and lists removed before added lines", () => {
        const lines = unifiedDiff("Greet the caller.\nAsk their name.", "Greet the caller.\nAsk their full name.");
        expect(lines.map((l) => l.kind)).toEqual(["unchanged", "removed", "added"]);
        const added = lines[2] as { segments: { kind: string; text: string }[] };
        expect(added.segments.some((s) => s.kind === "added" && s.text.includes("full"))).toBe(true);
    });

    it("collapses long unchanged runs around changes", () => {
        const before = Array.from({ length: 20 }, (_, i) => `line ${i}`).join("\n");
        const after = before.replace("line 10", "line ten");
        const lines = unifiedDiff(before, after);
        expect(lines.filter((l) => l.kind === "skipped")).toEqual([
            { kind: "skipped", count: 8 },
            { kind: "skipped", count: 7 },
        ]);
    });

    it("renders non-string values as JSON", () => {
        expect(valueText({ b: 1 })).toBe('{\n  "b": 1\n}');
        expect(valueText(null)).toBe("");
        expect(unifiedDiff(null, ["t1"]).every((l) => l.kind === "added")).toBe(true);
    });
});

describe("applyErrorText", () => {
    it("uses fixed wording for stale and conflict and the server message for validation", () => {
        expect(applyErrorText({ saved: false, error_code: "stale" })).toMatch(/changed since this was proposed/);
        expect(applyErrorText({ saved: false, error_code: "draft_conflict" })).toMatch(/Reload the canvas/);
        expect(applyErrorText({ saved: false, error_code: "graph_validation", error: "End node unreachable" })).toBe(
            "End node unreachable",
        );
    });
});

describe("markdown", () => {
    it("parses blocks", () => {
        const blocks = parseMarkdown("## Plan\nFirst line\nsecond\n\n- a\n- **b**\n\n1. one\n2. two\n\n```\ncode()\n```");
        expect(blocks.map((b) => b.type)).toEqual(["heading", "paragraph", "list", "list", "code"]);
        expect(blocks[3]).toMatchObject({ ordered: true, start: 1 });
        expect(blocks[4]).toEqual({ type: "code", text: "code()" });
    });

    it("parses inline code, bold, italic and only http(s) links, keeping snake_case literal", () => {
        expect(parseInline("use `end_call` **now** *maybe* [docs](https://x.y) tool_uuids")).toEqual([
            { type: "text", text: "use " },
            { type: "code", text: "end_call" },
            { type: "text", text: " " },
            { type: "strong", children: [{ type: "text", text: "now" }] },
            { type: "text", text: " " },
            { type: "em", children: [{ type: "text", text: "maybe" }] },
            { type: "text", text: " " },
            { type: "link", href: "https://x.y", children: [{ type: "text", text: "docs" }] },
            { type: "text", text: " tool_uuids" },
        ]);
        expect(parseInline("[x](javascript:alert(1))")).toEqual([{ type: "text", text: "[x](javascript:alert(1))" }]);
    });

    it("renders an unclosed fence (mid-stream) as code", () => {
        expect(parseMarkdown("```\npartial")).toEqual([{ type: "code", text: "partial" }]);
    });
});
