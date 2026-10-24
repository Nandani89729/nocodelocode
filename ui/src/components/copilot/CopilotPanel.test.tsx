import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { SelfImproveController } from "@/components/self-improve/useSelfImproveStatus";
import { CopilotApiError } from "@/lib/copilot/api";
import type { CopilotEvent, SessionResponse } from "@/lib/copilot/types";
import type { SelfImproveStatus } from "@/lib/self-improve/types";

import { CopilotPanel } from "./CopilotPanel";
import { BUSY_TEXT } from "./useCopilotChat";

const push = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));
vi.mock("@/lib/auth", () => ({ useAuth: () => ({ user: { id: "u1" }, loading: false }) }));
vi.mock("@/lib/doctor/api", () => ({ setDoctorAiOptIn: vi.fn().mockResolvedValue({ ai_opt_in: true }) }));

const api = vi.hoisted(() => ({
    listCopilotSessions: vi.fn(),
    getCopilotSession: vi.fn(),
    createCopilotSession: vi.fn(),
    streamCopilotMessage: vi.fn(),
    getBrief: vi.fn(),
    getProposal: vi.fn(),
    confirmBriefChange: vi.fn(),
    dismissBriefChange: vi.fn(),
}));
vi.mock("@/lib/copilot/api", async (importOriginal) => ({
    ...(await importOriginal<typeof import("@/lib/copilot/api")>()),
    ...api,
}));

const si = vi.hoisted(() => ({
    listSelfImproveIterations: vi.fn(),
    updateSelfImprove: vi.fn(),
    runSelfImproveNow: vi.fn(),
    resumeSelfImprove: vi.fn(),
    getSelfImproveStatus: vi.fn(),
}));
vi.mock("@/lib/self-improve/api", () => si);

const controller = (over: Partial<SelfImproveStatus> = {}): SelfImproveController => {
    const status: SelfImproveStatus = {
        workflow_id: 7,
        settings: { enabled: true, every_n_calls: 10 },
        state: "collecting",
        collected: 0,
        window_size: 10,
        window_multiplier: 1,
        to_review: 0,
        has_brief: true,
        ...over,
    };
    return { status, error: null, refresh: vi.fn(), replaceStatus: vi.fn(), toReview: status.to_review };
};

const session = (over: Partial<SessionResponse> = {}): SessionResponse => ({
    session_uuid: "s1",
    workflow_id: 7,
    messages: [],
    proposals: [],
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...over,
});

function streams(events: CopilotEvent[]) {
    api.streamCopilotMessage.mockImplementation(async (_s: string, _t: string, onEvent: (e: CopilotEvent) => void) => {
        for (const e of events) onEvent(e);
    });
}

const input = () => screen.getByLabelText("Message the copilot") as HTMLTextAreaElement;

describe("CopilotPanel", () => {
    beforeEach(() => {
        Object.values(api).forEach((fn) => fn.mockReset());
        api.listCopilotSessions.mockResolvedValue([]);
        api.createCopilotSession.mockResolvedValue(session());
        window.sessionStorage.clear();
    });

    it("shows editing examples and streams a reply after sending one", async () => {
        streams([
            { type: "text_delta", data: { text: "Looking at the **welcome** step." } },
            { type: "tool_call", data: { id: "t1", name: "get_workflow", summary: "Reading the workflow" } },
            { type: "tool_result", data: { id: "t1", name: "get_workflow", ok: true, summary: "Read 6 nodes" } },
            { type: "message_done", data: {} },
        ]);
        render(<CopilotPanel workflowId={7} />);
        const example = await screen.findByRole("button", { name: "Find and fix why calls hang in the welcome step" });
        fireEvent.click(example);
        expect(await screen.findByText("Read 6 nodes")).toBeTruthy();
        expect(screen.getByText("welcome").tagName).toBe("STRONG");
        expect(api.createCopilotSession).toHaveBeenCalledWith(7);
        expect(api.streamCopilotMessage.mock.calls[0][1]).toBe("Find and fix why calls hang in the welcome step");
        expect(screen.getByRole("log").getAttribute("aria-live")).toBe("polite");
    });

    it("resumes the latest session for the workflow", async () => {
        api.listCopilotSessions.mockResolvedValue([session()]);
        api.getCopilotSession.mockResolvedValue(
            session({ messages: [{ role: "user", content: "Earlier question", is_error: false }] }),
        );
        render(<CopilotPanel workflowId={7} />);
        expect(await screen.findByText("Earlier question")).toBeTruthy();
        expect(api.getCopilotSession).toHaveBeenCalledWith("s1");
    });

    it("sends on Enter, not on Shift+Enter", async () => {
        streams([{ type: "message_done", data: {} }]);
        render(<CopilotPanel workflowId={7} />);
        await screen.findByRole("button", { name: /Add a step that books a demo/ });
        fireEvent.change(input(), { target: { value: "Hello" } });
        fireEvent.keyDown(input(), { key: "Enter", shiftKey: true });
        expect(api.streamCopilotMessage).not.toHaveBeenCalled();
        fireEvent.keyDown(input(), { key: "Enter" });
        await waitFor(() => expect(api.streamCopilotMessage).toHaveBeenCalled());
        expect(input().value).toBe("");
    });

    it("on 409 says a reply is already running and gives the text back", async () => {
        api.streamCopilotMessage.mockRejectedValue(new CopilotApiError("busy", 409, {}));
        render(<CopilotPanel workflowId={7} />);
        await screen.findByRole("button", { name: /books a demo/ });
        fireEvent.change(input(), { target: { value: "Change the greeting" } });
        fireEvent.keyDown(input(), { key: "Enter" });
        expect(await screen.findByText(BUSY_TEXT)).toBeTruthy();
        expect(input().value).toBe("Change the greeting");
    });

    it("on 503 explains that the server needs an AI key", async () => {
        api.streamCopilotMessage.mockRejectedValue(new CopilotApiError("no key", 503, {}));
        render(<CopilotPanel workflowId={7} />);
        fireEvent.click(await screen.findByRole("button", { name: /books a demo/ }));
        expect(await screen.findByText(/this Dograh server has no AI API key/)).toBeTruthy();
        expect(input().disabled).toBe(true);
    });

    it("on 403 offers to enable AI, then retries the message", async () => {
        api.streamCopilotMessage.mockRejectedValueOnce(new CopilotApiError("not opted in", 403, {}));
        render(<CopilotPanel workflowId={7} />);
        await screen.findByRole("button", { name: /books a demo/ });
        fireEvent.change(input(), { target: { value: "Review prompts" } });
        fireEvent.keyDown(input(), { key: "Enter" });
        streams([{ type: "message_done", data: {} }]);
        fireEvent.click(await screen.findByRole("button", { name: "Enable AI features" }));
        await waitFor(() => expect(api.streamCopilotMessage).toHaveBeenCalledTimes(2));
        expect(api.streamCopilotMessage.mock.calls[1][1]).toBe("Review prompts");
    });

    it("shows Stop while streaming and aborts the request", async () => {
        let signal: AbortSignal | undefined;
        api.streamCopilotMessage.mockImplementation(
            (_s: string, _t: string, onEvent: (e: CopilotEvent) => void, sig?: AbortSignal) =>
                new Promise((_, reject) => {
                    signal = sig;
                    onEvent({ type: "text_delta", data: { text: "Working" } });
                    sig?.addEventListener("abort", () => reject(new DOMException("stop", "AbortError")));
                }),
        );
        render(<CopilotPanel workflowId={7} />);
        fireEvent.click(await screen.findByRole("button", { name: /books a demo/ }));
        fireEvent.click(await screen.findByRole("button", { name: "Stop" }));
        expect(await screen.findByText("Stopped.")).toBeTruthy();
        expect(signal?.aborted).toBe(true);
        expect(screen.getByRole("button", { name: "Send" })).toBeTruthy();
    });

    it("workflow panels with Self-improve get Chat | Self-improve (1) tabs and start on the waiting round", async () => {
        si.listSelfImproveIterations.mockResolvedValue({ iterations: [] });
        render(<CopilotPanel workflowId={7} selfImprove={controller({ to_review: 1, state: "waiting" })} />);
        const tab = screen.getByRole("tab", { name: "Self-improve (1)" });
        expect(tab.getAttribute("aria-selected")).toBe("true");
        expect(screen.getByRole("tab", { name: "Chat" })).toBeTruthy();
        expect(screen.getByText("1 round waiting for review")).toBeTruthy();
        // The chat stays mounted (hidden) so a draft message survives switching tabs.
        expect(screen.getByLabelText("Message the copilot")).toBeTruthy();
        expect(screen.queryByRole("button", { name: "New chat" })).toBeNull();
    });

    it("shows no count when nothing waits, and no tabs without Self-improve", async () => {
        const { unmount } = render(<CopilotPanel workflowId={7} selfImprove={controller({ to_review: 0 })} />);
        expect(screen.getByRole("tab", { name: "Self-improve" }).getAttribute("aria-selected")).toBe("false");
        unmount();
        render(<CopilotPanel workflowId={null} />);
        await screen.findByRole("button", { name: /dental clinic/ });
        expect(screen.queryByRole("tab")).toBeNull();
    });

    it("renders a brief_change event as a diff card with Confirm / Dismiss", async () => {
        const proposed = { goal: "Book demos", must_do: ["Collect email"], must_never: [], accepted_tradeoffs: [], notes: "" };
        streams([
            {
                type: "brief_change",
                data: {
                    brief_change: {
                        change_uuid: "b1",
                        workflow_id: 7,
                        base_version: 1,
                        proposed,
                        reason: "You said demos are the goal.",
                        status: "pending",
                    },
                },
            },
            { type: "message_done", data: {} },
        ]);
        api.getBrief.mockResolvedValue({
            workflow_id: 7,
            version: 1,
            brief: { goal: "Answer questions", must_do: ["Collect email"], must_never: [], accepted_tradeoffs: [], notes: "" },
        });
        api.dismissBriefChange.mockResolvedValue({ workflow_id: 7, version: 1, brief: proposed });
        render(<CopilotPanel workflowId={7} />);
        fireEvent.click(await screen.findByRole("button", { name: /books a demo/ }));
        const card = await screen.findByRole("region", { name: "Proposed brief change" });
        const goal = await screen.findByRole("group", { name: "Changes to Goal" });
        expect(goal.textContent).toContain("Book demos");
        expect(screen.queryByRole("group", { name: "Changes to Must do" })).toBeNull();
        expect(card.textContent).toContain("You said demos are the goal.");
        fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
        expect(await screen.findByText("Dismissed.")).toBeTruthy();
        expect(api.dismissBriefChange).toHaveBeenCalledWith(7, "b1");
    });

    it("marks a brief change out of date when the brief moved on", async () => {
        streams([
            {
                type: "brief_change",
                data: {
                    brief_change: {
                        change_uuid: "b2",
                        workflow_id: 7,
                        base_version: 1,
                        proposed: { goal: "X", must_do: [], must_never: [], accepted_tradeoffs: [], notes: "" },
                        reason: "",
                        status: "pending",
                    },
                },
            },
        ]);
        api.getBrief.mockResolvedValue({
            workflow_id: 7,
            version: 3,
            brief: { goal: "Y", must_do: [], must_never: [], accepted_tradeoffs: [], notes: "" },
        });
        render(<CopilotPanel workflowId={7} />);
        fireEvent.click(await screen.findByRole("button", { name: /books a demo/ }));
        expect(await screen.findByText("The brief changed since this was suggested.")).toBeTruthy();
        expect(screen.queryByRole("button", { name: "Confirm" })).toBeNull();
    });

    it("build-new shows build examples and approves plans with a message", async () => {
        streams([
            {
                type: "plan",
                data: { plan: { summary: "Dental inbound agent", steps: [{ title: "Greet" }], questions: ["Hours?"] } },
            },
            { type: "message_done", data: {} },
        ]);
        render(<CopilotPanel workflowId={null} />);
        fireEvent.click(await screen.findByRole("button", { name: /dental clinic/ }));
        const approve = await screen.findByRole("button", { name: "Approve plan" });
        expect(api.createCopilotSession).toHaveBeenCalledWith(null);
        expect(screen.getByText("Hours?")).toBeTruthy();
        streams([{ type: "message_done", data: {} }]);
        fireEvent.click(approve);
        await waitFor(() => expect(api.streamCopilotMessage).toHaveBeenCalledTimes(2));
        expect(api.streamCopilotMessage.mock.calls[1][1]).toBe("Plan approved. Go ahead.");
    });
});
