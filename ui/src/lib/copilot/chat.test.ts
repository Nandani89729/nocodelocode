import { describe, expect, it } from "vitest";

import { applyEvent, cancelTurn, chatFromSession, emptyChat, endTurn, isPlanActionable, startTurn } from "./chat";
import type { CopilotEvent, Proposal, SessionResponse } from "./types";

const proposal = (uuid: string, created_at?: string): Proposal => ({
    proposal_uuid: uuid,
    session_uuid: "s",
    kind: "edit",
    workflow_id: 1,
    summary: "Add a demo step",
    changes: [],
    status: "pending",
    created_at,
});

const run = (events: CopilotEvent[], state = startTurn(emptyChat(), "hi")) => events.reduce(applyEvent, state);

describe("applyEvent", () => {
    it("streams text into one assistant bubble and splits it at tool calls", () => {
        const state = run([
            { type: "text_delta", data: { text: "Let me " } },
            { type: "text_delta", data: { text: "look." } },
            { type: "tool_call", data: { id: "t1", name: "get_workflow", summary: "Reading the workflow" } },
            { type: "tool_result", data: { id: "t1", name: "get_workflow", ok: true, summary: "Read 5 nodes" } },
            { type: "text_delta", data: { text: "Done." } },
            { type: "message_done", data: {} },
        ]);
        expect(state.items.map((i) => i.kind)).toEqual(["user", "assistant", "tool", "assistant"]);
        expect(state.items[1]).toMatchObject({ text: "Let me look.", streaming: false });
        expect(state.items[2]).toMatchObject({ status: "ok", summary: "Read 5 nodes" });
        expect(state.items[3]).toMatchObject({ text: "Done.", streaming: false });
        expect(state.streaming).toBe(false);
    });

    it("marks failed tools and fails tools still running when the turn ends", () => {
        const state = run([
            { type: "tool_call", data: { id: "a", name: "x" } },
            { type: "tool_call", data: { id: "b", name: "y" } },
            { type: "tool_result", data: { id: "a", name: "x", ok: false, summary: "Not found" } },
        ]);
        const ended = endTurn(state, { tone: "info", text: "Stopped." });
        expect(ended.items.filter((i) => i.kind === "tool").map((i) => (i as { status: string }).status)).toEqual([
            "failed",
            "failed",
        ]);
        expect(ended.items.at(-1)).toMatchObject({ kind: "notice", text: "Stopped." });
    });

    it("stores proposals by uuid and updates without duplicating the card", () => {
        const p = proposal("p1");
        let state = run([{ type: "proposal", data: { proposal: p } }]);
        state = applyEvent(state, { type: "proposal", data: { proposal: { ...p, summary: "v2" } } });
        expect(state.items.filter((i) => i.kind === "proposal")).toHaveLength(1);
        expect(state.proposals.p1.summary).toBe("v2");
    });

    it("adds one brief-change card per change and updates it in place", () => {
        const change = {
            change_uuid: "b1",
            workflow_id: 1,
            base_version: 2,
            proposed: { goal: "Book demos", must_do: [], must_never: [], accepted_tradeoffs: [], notes: "" },
            reason: "You said demos are the goal",
            status: "pending" as const,
        };
        let state = run([
            { type: "text_delta", data: { text: "I suggest a brief." } },
            { type: "brief_change", data: { brief_change: change } },
        ]);
        state = applyEvent(state, { type: "brief_change", data: { brief_change: { ...change, status: "stale" } } });
        expect(state.items.map((i) => i.kind)).toEqual(["user", "assistant", "brief_change"]);
        expect(state.items[1]).toMatchObject({ streaming: false });
        expect(state.briefChanges.b1.status).toBe("stale");
    });

    it("shows errors as notices", () => {
        const state = run([{ type: "error", data: { message: "Model overloaded" } }]);
        expect(state.items.at(-1)).toMatchObject({ kind: "notice", tone: "error", text: "Model overloaded" });
    });
});

describe("turns", () => {
    it("cancelTurn removes the optimistic user message", () => {
        const state = cancelTurn(startTurn(emptyChat(), "hello"), "hello");
        expect(state.items).toEqual([]);
        expect(state.streaming).toBe(false);
    });

    it("plan buttons are actionable only while the plan is the newest answerable item", () => {
        let state = run([
            { type: "plan", data: { plan: { summary: "s", steps: [{ title: "a" }], questions: [] } } },
            { type: "message_done", data: {} },
        ]);
        const key = state.items.find((i) => i.kind === "plan")!.key;
        expect(isPlanActionable(state, key)).toBe(true);
        state = startTurn(state, "Plan approved. Go ahead.");
        expect(isPlanActionable(state, key)).toBe(false);
    });
});

describe("chatFromSession", () => {
    const session: SessionResponse = {
        session_uuid: "s",
        workflow_id: 1,
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:05:00Z",
        messages: [
            { role: "user", content: "Add a demo step", is_error: false, created_at: "2026-01-01T00:00:00Z" },
            { role: "assistant", content: "", is_error: false, created_at: "2026-01-01T00:00:01Z" },
            { role: "tool", content: "{...}", tool_name: "propose_changes", is_error: true, created_at: "2026-01-01T00:00:02Z" },
            { role: "assistant", content: "Here it is.", is_error: false, created_at: "2026-01-01T00:00:04Z" },
        ],
        proposals: [proposal("p1", "2026-01-01T00:00:03Z")],
        plan: { summary: "old plan", steps: [], questions: [] },
    };

    it("rebuilds the timeline with proposals in time order and tool chips", () => {
        const state = chatFromSession(session);
        expect(state.items.map((i) => i.kind)).toEqual(["user", "tool", "proposal", "assistant"]);
        expect(state.items[1]).toMatchObject({ name: "propose_changes", status: "failed" });
        expect(state.proposals.p1).toBeDefined();
    });

    it("shows the last plan only while nothing was proposed", () => {
        const state = chatFromSession({ ...session, proposals: [] });
        expect(state.items.at(-1)).toMatchObject({ kind: "plan" });
    });
});
