import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { CopilotApiError } from "@/lib/copilot/api";
import type { BriefChange, BriefResponse, Proposal } from "@/lib/copilot/types";
import type { IterationResponse, SelfImproveStatus } from "@/lib/self-improve/types";

import { SelfImproveTab } from "./SelfImproveTab";
import type { SelfImproveController } from "./useSelfImproveStatus";

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }));
vi.mock("@/lib/auth", () => ({ useAuth: () => ({ user: { id: "u1" }, loading: false }) }));
vi.mock("@/lib/doctor/api", () => ({ setDoctorAiOptIn: vi.fn().mockResolvedValue({ ai_opt_in: true }) }));

const si = vi.hoisted(() => ({
    updateSelfImprove: vi.fn(),
    runSelfImproveNow: vi.fn(),
    resumeSelfImprove: vi.fn(),
    listSelfImproveIterations: vi.fn(),
    getSelfImproveStatus: vi.fn(),
}));
vi.mock("@/lib/self-improve/api", () => si);

const cp = vi.hoisted(() => ({
    getProposal: vi.fn(),
    getBrief: vi.fn(),
    updateBrief: vi.fn(),
    draftBrief: vi.fn(),
    confirmBriefChange: vi.fn(),
    dismissBriefChange: vi.fn(),
    applyProposal: vi.fn(),
    discardProposal: vi.fn(),
}));
vi.mock("@/lib/copilot/api", async (importOriginal) => ({
    ...(await importOriginal<typeof import("@/lib/copilot/api")>()),
    ...cp,
}));

const OCT2 = "2026-10-02T12:00:00Z";
const tenRuns = Array.from({ length: 10 }, (_, i) => i + 1);

const iteration = (over: Partial<IterationResponse> = {}): IterationResponse => ({
    iteration_id: 30,
    number: 3,
    status: "ready",
    run_ids: tenRuns,
    report_uuid: "r3",
    proposal_uuid: "p3",
    reviewed: false,
    carryover: [],
    created_at: OCT2,
    ...over,
});

const status = (over: Partial<SelfImproveStatus> = {}): SelfImproveStatus => ({
    workflow_id: 7,
    settings: { enabled: true, every_n_calls: 10 },
    state: "collecting",
    collected: 7,
    window_size: 10,
    window_multiplier: 1,
    to_review: 0,
    has_brief: true,
    ...over,
});

const proposal = (over: Partial<Proposal> = {}): Proposal => ({
    proposal_uuid: "p3",
    session_uuid: "s-si",
    kind: "edit",
    workflow_id: 7,
    summary: "Move pricing out of the Global prompt",
    status: "pending",
    iteration_id: 30,
    changes: [
        {
            change_id: "n:1",
            target: "node",
            op: "update",
            ref_id: "1",
            label: 'Update node "Welcome" (prompt)',
            field_diffs: [{ field: "prompt", before: "a", after: "b" }],
            depends_on: [],
        },
        {
            change_id: "n:2",
            target: "node",
            op: "update",
            ref_id: "2",
            label: 'Update node "Global" (prompt)',
            field_diffs: [{ field: "prompt", before: "c", after: "d" }],
            depends_on: [],
        },
    ],
    ...over,
});

const brief = (over: Partial<BriefResponse> = {}): BriefResponse => ({
    workflow_id: 7,
    brief: { goal: "Book demos", must_do: ["Collect email"], must_never: [], accepted_tradeoffs: [], notes: "" },
    version: 2,
    ...over,
});

const refresh = vi.fn();

function Harness({ initial, mobile = false }: { initial: SelfImproveStatus; mobile?: boolean }) {
    const [current, setCurrent] = useState(initial);
    const controller: SelfImproveController = {
        status: current,
        error: null,
        refresh: async () => {
            refresh();
            return current;
        },
        replaceStatus: setCurrent,
        toReview: current.to_review,
    };
    return <SelfImproveTab workflowId={7} controller={controller} mobile={mobile} />;
}

const waitingRound = () => {
    const it = iteration({
        carryover: [
            {
                proposal_uuid: "p1",
                change_id: "n:1",
                title: "Welcome step answers pricing",
                outcome: "fixed",
                eligible_calls: 9,
                affected_calls: 0,
                published_at: OCT2,
            },
        ],
    });
    si.listSelfImproveIterations.mockResolvedValue({ iterations: [it, iteration({ iteration_id: 20, number: 2, status: "no_changes", proposal_uuid: null, reviewed: true })] });
    return status({ state: "waiting", to_review: 1, collected: 0, current_iteration: it });
};

describe("SelfImproveTab", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        si.listSelfImproveIterations.mockResolvedValue({ iterations: [] });
        cp.getBrief.mockResolvedValue(brief());
        cp.getProposal.mockResolvedValue(proposal());
    });

    it("Off: says so; turning on with no brief asks first and Skip still turns it on", async () => {
        si.updateSelfImprove.mockResolvedValue(status());
        render(
            <Harness
                initial={status({ state: "off", has_brief: false, settings: { enabled: false, every_n_calls: 10 } })}
            />,
        );
        expect(
            screen.getByText("Self-improve is off. Turn it on to review every 10 real calls automatically."),
        ).toBeTruthy();
        fireEvent.click(screen.getByRole("button", { name: "Turn on" }));
        expect(screen.getByText("Rounds follow your brief. You don't have one yet.")).toBeTruthy();
        expect(si.updateSelfImprove).not.toHaveBeenCalled();
        fireEvent.click(screen.getByRole("button", { name: "Skip" }));
        await waitFor(() => expect(si.updateSelfImprove).toHaveBeenCalledWith(7, { enabled: true, every_n_calls: 10 }));
        expect(await screen.findByText("On · every 10 calls · 7/10 collected")).toBeTruthy();
        expect(screen.queryByText(/You don't have one yet/)).toBeNull();
    });

    it("Draft from my chats turns it on and shows the drafted brief to confirm", async () => {
        const change: BriefChange = {
            change_uuid: "b1",
            workflow_id: 7,
            base_version: 0,
            proposed: { goal: "Qualify leads", must_do: [], must_never: ["Quote prices"], accepted_tradeoffs: [], notes: "" },
            reason: "From your chats",
            status: "pending",
        };
        const empty = brief({ brief: { goal: "", must_do: [], must_never: [], accepted_tradeoffs: [], notes: "" }, version: 0 });
        cp.getBrief.mockResolvedValueOnce(empty).mockResolvedValue({ ...empty, pending_change: change });
        si.updateSelfImprove.mockResolvedValue(status({ has_brief: false }));
        cp.draftBrief.mockResolvedValue({ brief_change: change });
        cp.confirmBriefChange.mockResolvedValue(brief({ brief: change.proposed, version: 1 }));
        render(
            <Harness
                initial={status({ state: "off", has_brief: false, settings: { enabled: false, every_n_calls: 10 } })}
            />,
        );
        fireEvent.click(screen.getByRole("button", { name: "Turn on" }));
        fireEvent.click(screen.getByRole("button", { name: "Draft from my chats" }));
        const card = await screen.findByRole("region", { name: "Proposed brief change" });
        expect(si.updateSelfImprove).toHaveBeenCalledWith(7, { enabled: true, every_n_calls: 10 });
        expect(within(card).getByRole("group", { name: "Changes to Goal" }).textContent).toContain("Qualify leads");
        expect(within(card).getByRole("group", { name: "Changes to Must never" })).toBeTruthy();
        fireEvent.click(within(card).getByRole("button", { name: "Confirm" }));
        await waitFor(() => expect(cp.confirmBriefChange).toHaveBeenCalledWith(7, "b1"));
        expect(await screen.findByText("Qualify leads")).toBeTruthy();
    });

    it("AI not enabled shows the Enable AI analysis notice", () => {
        render(<Harness initial={status({ state: "ai_disabled" })} />);
        expect(screen.getByRole("button", { name: "Enable AI analysis" })).toBeTruthy();
    });

    it("No key", () => {
        render(<Harness initial={status({ state: "no_key" })} />);
        expect(screen.getByText("This server has no AI key.")).toBeTruthy();
    });

    it("On with no rounds yet explains what counts", async () => {
        render(<Harness initial={status()} />);
        expect(screen.getByRole("status").textContent).toBe("On · every 10 calls · 7/10 collected");
        expect(
            await screen.findByText(
                "No rounds yet. The first one runs after 10 real calls (phone or web, where the caller speaks). Text-chat tests don't count.",
            ),
        ).toBeTruthy();
    });

    it("Waiting: the round first, with since-earlier-fixes lines, the proposal and the report link", async () => {
        render(<Harness initial={waitingRound()} />);
        expect(
            screen.getByText("Waiting for your review. Next round starts after you apply or discard Round 3."),
        ).toBeTruthy();
        expect(screen.getByRole("heading", { name: "Round 3 · waiting for review" })).toBeTruthy();
        expect(screen.getByText("Since earlier fixes")).toBeTruthy();
        expect(screen.getByText("Welcome step answers pricing: 0 of 9 calls since the fix (published 2 Oct)")).toBeTruthy();
        expect(await screen.findByRole("button", { name: "Apply selected (2)" })).toBeTruthy();
        expect(screen.getByRole("link", { name: "Open the Doctor report" }).getAttribute("href")).toBe(
            "/workflow/7/doctor/r3",
        );
        expect(await screen.findByText("Round 2 · No changes needed in 10 calls · 2 Oct")).toBeTruthy();
        expect(document.body.textContent?.toLowerCase()).not.toContain("verified");
    });

    it("after Discard the round becomes a row, takes focus, and the status refreshes", async () => {
        cp.discardProposal.mockResolvedValue(proposal({ status: "discarded" }));
        render(<Harness initial={waitingRound()} />);
        fireEvent.click(await screen.findByRole("button", { name: "Discard" }));
        const row = await screen.findByRole("button", { name: /Round 3 · Discarded · 2 Oct/ });
        await waitFor(() => expect(document.activeElement).toBe(row));
        expect(refresh).toHaveBeenCalled();
        expect(screen.queryByRole("heading", { name: /waiting for review/ })).toBeNull();
    });

    it("after Apply the row says how many changes were applied", async () => {
        cp.applyProposal.mockResolvedValue({
            saved: true,
            workflow_id: 7,
            proposal: proposal({ status: "applied", accepted_change_ids: ["n:1"] }),
        });
        render(<Harness initial={waitingRound()} />);
        fireEvent.click(await screen.findByRole("checkbox", { name: /Global/ }));
        fireEvent.click(screen.getByRole("button", { name: "Apply selected (1)" }));
        const row = await screen.findByRole("button", { name: /Round 3 · Applied 1 of 2 changes · draft · 2 Oct/ });
        await waitFor(() => expect(document.activeElement).toBe(row));
    });

    it("queued", () => {
        render(<Harness initial={status({ state: "running", current_iteration: iteration({ number: 4, status: "queued", report_uuid: null, proposal_uuid: null }) })} />);
        expect(screen.getByRole("heading", { name: "Round 4 · starting" })).toBeTruthy();
        expect(screen.getByRole("status").textContent).toBe("On · every 10 calls");
    });

    it("diagnosing links the live Doctor report", () => {
        render(<Harness initial={status({ state: "running", current_iteration: iteration({ status: "diagnosing", proposal_uuid: null }) })} />);
        expect(screen.getByText("Analysing 10 calls…")).toBeTruthy();
        expect(screen.getByRole("link", { name: "Watch the live Doctor report" }).getAttribute("href")).toBe(
            "/workflow/7/doctor/r3",
        );
    });

    it("proposing", () => {
        render(
            <Harness
                initial={status({
                    state: "running",
                    current_iteration: iteration({ status: "proposing", proposal_uuid: null, problem_count: 2 }),
                })}
            />,
        );
        expect(screen.getByText("Writing fixes for 2 problems…")).toBeTruthy();
    });

    it("no_changes says when the next check runs", () => {
        render(
            <Harness
                initial={status({
                    collected: 0,
                    window_size: 20,
                    window_multiplier: 2,
                    current_iteration: iteration({ status: "no_changes", proposal_uuid: null, reviewed: true }),
                })}
            />,
        );
        expect(screen.getByText("No changes needed in 10 calls. Next check after 20 calls.")).toBeTruthy();
        expect(screen.getByRole("status").textContent).toBe("On · every 20 calls (backed off) · 0/20 collected");
    });

    it("failed says when the next round runs", () => {
        render(
            <Harness
                initial={status({
                    collected: 0,
                    current_iteration: iteration({ status: "failed", error: "the analysis timed out", proposal_uuid: null, reviewed: true }),
                })}
            />,
        );
        expect(screen.getByText("Round failed: the analysis timed out. Next round runs after 10 more calls.")).toBeTruthy();
    });

    it("Paused offers Resume", async () => {
        si.resumeSelfImprove.mockResolvedValue(status({ collected: 0 }));
        render(
            <Harness
                initial={status({
                    state: "paused",
                    paused_reason: "the AI provider is down",
                    current_iteration: iteration({ status: "failed", error: "provider down", proposal_uuid: null, reviewed: true }),
                })}
            />,
        );
        expect(screen.getAllByText("Paused after 2 failed rounds: the AI provider is down.")).toHaveLength(2);
        fireEvent.click(screen.getByRole("button", { name: "Resume" }));
        await waitFor(() => expect(si.resumeSelfImprove).toHaveBeenCalledWith(7));
        expect(await screen.findByText("On · every 10 calls · 0/10 collected")).toBeTruthy();
    });

    it("mobile: change titles, Open on desktop to apply, and Discard round works", async () => {
        cp.discardProposal.mockResolvedValue(proposal({ status: "discarded" }));
        render(<Harness initial={waitingRound()} mobile />);
        expect(await screen.findByText('Update node "Welcome" (prompt)')).toBeTruthy();
        expect(screen.getByText("Open on desktop to apply")).toBeTruthy();
        expect((screen.getByRole("button", { name: "Review changes" }) as HTMLButtonElement).disabled).toBe(true);
        expect(screen.queryByRole("button", { name: /Apply/ })).toBeNull();
        fireEvent.click(screen.getByRole("button", { name: "Discard round" }));
        await waitFor(() => expect(cp.discardProposal).toHaveBeenCalledWith("p3"));
        expect(await screen.findByRole("button", { name: /Round 3 · Discarded/ })).toBeTruthy();
    });

    it("gear: changing N explains what happens to counted calls before Save (DS-10)", async () => {
        si.updateSelfImprove.mockResolvedValue(status({ settings: { enabled: true, every_n_calls: 5 }, window_size: 5 }));
        render(<Harness initial={status()} />);
        fireEvent.click(screen.getByRole("button", { name: "Self-improve settings" }));
        fireEvent.click(await screen.findByRole("button", { name: "5 calls" }));
        expect(screen.getByText("7 calls already counted. With N=5 a round starts now.")).toBeTruthy();
        fireEvent.click(screen.getByRole("button", { name: "20 calls" }));
        expect(screen.getByText("7 calls already counted. With N=20 a round starts after 13 more calls.")).toBeTruthy();
        expect(screen.getByRole("button", { name: "20 calls" }).getAttribute("aria-pressed")).toBe("true");
        fireEvent.click(screen.getByRole("button", { name: "Save" }));
        await waitFor(() => expect(si.updateSelfImprove).toHaveBeenCalledWith(7, { enabled: true, every_n_calls: 20 }));
    });

    it("brief: collapsed to its goal; Edit shows labelled fields; a version conflict is explained", async () => {
        cp.updateBrief.mockRejectedValue(new CopilotApiError("brief_version_conflict", 409, { detail: "brief_version_conflict" }));
        render(<Harness initial={status()} />);
        expect(await screen.findByText("Book demos")).toBeTruthy();
        fireEvent.click(screen.getByRole("button", { name: "Edit" }));
        const goal = screen.getByLabelText("Goal") as HTMLInputElement;
        expect(goal.value).toBe("Book demos");
        expect((screen.getByLabelText("Must do 1") as HTMLInputElement).value).toBe("Collect email");
        expect(screen.getByLabelText("Notes")).toBeTruthy();
        fireEvent.click(screen.getByRole("button", { name: "Add to Must never" }));
        fireEvent.change(screen.getByLabelText("Must never 1"), { target: { value: "Quote prices" } });
        await act(async () => {
            fireEvent.click(screen.getByRole("button", { name: "Save brief" }));
        });
        expect(cp.updateBrief).toHaveBeenCalledWith(7, {
            brief: { goal: "Book demos", must_do: ["Collect email"], must_never: ["Quote prices"], accepted_tradeoffs: [], notes: "" },
            expected_version: 2,
        });
        expect(await screen.findByText("Someone changed the brief since you opened it.")).toBeTruthy();
    });
});
