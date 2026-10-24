/**
 * Chat timeline for the copilot panel (pure, no React): builds it from a
 * saved session and folds streamed events into it.
 */
import type { BriefChange, CopilotEvent, Plan, Proposal, SessionResponse } from "./types";

export type ToolStatus = "running" | "ok" | "failed";

export type ChatItem =
    | { kind: "user"; key: string; text: string }
    | { kind: "assistant"; key: string; text: string; streaming: boolean }
    | { kind: "tool"; key: string; callId: string | null; name: string; summary: string; status: ToolStatus }
    | { kind: "plan"; key: string; plan: Plan }
    | { kind: "proposal"; key: string; proposalUuid: string }
    | { kind: "brief_change"; key: string; changeUuid: string }
    | { kind: "notice"; key: string; tone: "error" | "info"; text: string };

export interface ChatState {
    items: ChatItem[];
    /** proposals by uuid; cards read from here so apply/discard updates show */
    proposals: Record<string, Proposal>;
    /** copilot-proposed brief changes by uuid (M2b, O8) */
    briefChanges: Record<string, BriefChange>;
    streaming: boolean;
    seq: number;
}

export const emptyChat = (): ChatState => ({ items: [], proposals: {}, briefChanges: {}, streaming: false, seq: 0 });

const nextKey = (state: ChatState, prefix: string) => `${prefix}-${state.seq + 1}`;

function push(state: ChatState, item: ChatItem): ChatState {
    return { ...state, items: [...state.items, item], seq: state.seq + 1 };
}

/** Close the streaming assistant bubble (a tool call or the end of the turn splits text). */
function settleAssistant(items: ChatItem[]): ChatItem[] {
    const last = items.at(-1);
    if (last?.kind === "assistant" && last.streaming) {
        return [...items.slice(0, -1), { ...last, streaming: false }];
    }
    return items;
}

const toTime = (iso: string | null | undefined) => {
    const t = iso ? Date.parse(iso) : NaN;
    return Number.isNaN(t) ? null : t;
};

/**
 * History → timeline. Messages keep their order; proposals are merged in by
 * created_at (appended when it is missing). The session's last plan is shown
 * at the end only while nothing has been proposed yet, i.e. while it may
 * still be waiting for approval.
 */
export function chatFromSession(session: SessionResponse): ChatState {
    let state = emptyChat();
    const proposals = [...session.proposals];
    const flushProposalsBefore = (time: number | null) => {
        while (proposals.length > 0) {
            const t = toTime(proposals[0].created_at);
            if (time === null || t === null || t > time) break;
            const p = proposals.shift()!;
            state = push(state, { kind: "proposal", key: nextKey(state, "p"), proposalUuid: p.proposal_uuid });
        }
    };
    for (const m of session.messages) {
        flushProposalsBefore(toTime(m.created_at));
        if (m.role === "user") {
            state = push(state, { kind: "user", key: nextKey(state, "u"), text: m.content });
        } else if (m.role === "assistant") {
            if (m.content.trim() === "") continue;
            state = push(state, { kind: "assistant", key: nextKey(state, "a"), text: m.content, streaming: false });
        } else {
            state = push(state, {
                kind: "tool",
                key: nextKey(state, "t"),
                callId: null,
                name: m.tool_name ?? "tool",
                summary: "",
                status: m.is_error ? "failed" : "ok",
            });
        }
    }
    for (const p of proposals) {
        state = push(state, { kind: "proposal", key: nextKey(state, "p"), proposalUuid: p.proposal_uuid });
    }
    if (session.plan && session.proposals.length === 0) {
        state = push(state, { kind: "plan", key: nextKey(state, "plan"), plan: session.plan });
    }
    return {
        ...state,
        proposals: Object.fromEntries(session.proposals.map((p) => [p.proposal_uuid, p])),
    };
}

/** The user sent a message: show it and start a turn. */
export function startTurn(state: ChatState, text: string): ChatState {
    const next = push({ ...state, items: settleAssistant(state.items) }, { kind: "user", key: nextKey(state, "u"), text });
    return { ...next, streaming: true };
}

/** The send was refused before the turn started (409/403/503): drop the optimistic message. */
export function cancelTurn(state: ChatState, text: string): ChatState {
    const last = state.items.at(-1);
    const items = last?.kind === "user" && last.text === text ? state.items.slice(0, -1) : state.items;
    return { ...state, items, streaming: false };
}

/** The turn ended: by message_done, the stream closing, Stop, or a failure. */
export function endTurn(state: ChatState, notice?: { tone: "error" | "info"; text: string }): ChatState {
    const items = settleAssistant(state.items).map((item) =>
        item.kind === "tool" && item.status === "running" ? { ...item, status: "failed" as const } : item,
    );
    let next: ChatState = { ...state, items, streaming: false };
    if (notice) next = push(next, { kind: "notice", key: nextKey(next, "n"), ...notice });
    return next;
}

export function upsertProposal(state: ChatState, proposal: Proposal): ChatState {
    return { ...state, proposals: { ...state.proposals, [proposal.proposal_uuid]: proposal } };
}

export function upsertBriefChange(state: ChatState, change: BriefChange): ChatState {
    return { ...state, briefChanges: { ...state.briefChanges, [change.change_uuid]: change } };
}

export function applyEvent(state: ChatState, event: CopilotEvent): ChatState {
    switch (event.type) {
        case "text_delta": {
            const text = event.data.text ?? "";
            if (!text) return state;
            const last = state.items.at(-1);
            if (last?.kind === "assistant" && last.streaming) {
                return { ...state, items: [...state.items.slice(0, -1), { ...last, text: last.text + text }] };
            }
            return push(state, { kind: "assistant", key: nextKey(state, "a"), text, streaming: true });
        }
        case "tool_call":
            return push(
                { ...state, items: settleAssistant(state.items) },
                {
                    kind: "tool",
                    key: nextKey(state, "t"),
                    callId: event.data.id,
                    name: event.data.name,
                    summary: event.data.summary ?? "",
                    status: "running",
                },
            );
        case "tool_result": {
            const index = state.items.findLastIndex(
                (item) => item.kind === "tool" && item.callId === event.data.id,
            );
            const status: ToolStatus = event.data.ok ? "ok" : "failed";
            if (index === -1) {
                return push(
                    { ...state, items: settleAssistant(state.items) },
                    {
                        kind: "tool",
                        key: nextKey(state, "t"),
                        callId: event.data.id,
                        name: event.data.name,
                        summary: event.data.summary ?? "",
                        status,
                    },
                );
            }
            const items = [...state.items];
            const item = items[index] as Extract<ChatItem, { kind: "tool" }>;
            items[index] = { ...item, status, summary: event.data.summary || item.summary };
            return { ...state, items };
        }
        case "plan":
            return push(
                { ...state, items: settleAssistant(state.items) },
                { kind: "plan", key: nextKey(state, "plan"), plan: event.data.plan },
            );
        case "proposal": {
            const proposal = event.data.proposal;
            const known = proposal.proposal_uuid in state.proposals;
            const withProposal = upsertProposal({ ...state, items: settleAssistant(state.items) }, proposal);
            if (known) return withProposal;
            return push(withProposal, { kind: "proposal", key: nextKey(state, "p"), proposalUuid: proposal.proposal_uuid });
        }
        case "brief_change": {
            const change = event.data.brief_change;
            if (!change?.change_uuid) return state;
            const known = change.change_uuid in state.briefChanges;
            const withChange = upsertBriefChange({ ...state, items: settleAssistant(state.items) }, change);
            if (known) return withChange;
            return push(withChange, { kind: "brief_change", key: nextKey(state, "b"), changeUuid: change.change_uuid });
        }
        case "message_done":
            return endTurn(state);
        case "error":
            return push(
                { ...state, items: settleAssistant(state.items) },
                { kind: "notice", key: nextKey(state, "n"), tone: "error", text: event.data.message || "The copilot hit an error." },
            );
        default:
            return state;
    }
}

/**
 * A plan's buttons only make sense while it is the newest thing the user can
 * answer: no user message after it and no turn running.
 */
export function isPlanActionable(state: ChatState, key: string): boolean {
    if (state.streaming) return false;
    const index = state.items.findIndex((item) => item.key === key);
    if (index === -1) return false;
    return !state.items.slice(index + 1).some((item) => item.kind === "user" || item.kind === "plan");
}
