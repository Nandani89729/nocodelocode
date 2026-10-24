import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ApplyProposalResponse, Proposal } from "@/lib/copilot/types";

import { ProposalCard, type ProposalCardProps } from "./ProposalCard";

const applyProposal = vi.fn();
const discardProposal = vi.fn();
vi.mock("@/lib/copilot/api", () => ({
    applyProposal: (...args: unknown[]) => applyProposal(...args),
    discardProposal: (...args: unknown[]) => discardProposal(...args),
}));

const editProposal: Proposal = {
    proposal_uuid: "p1",
    session_uuid: "s1",
    kind: "edit",
    workflow_id: 7,
    summary: "Add a demo booking step after the greeting",
    status: "pending",
    changes: [
        {
            change_id: "n:9",
            target: "node",
            op: "add",
            ref_id: "9",
            label: 'Add node "Book demo"',
            after: { id: "9", type: "agentNode", data: { name: "Book demo", prompt: "Offer two slots." } },
            field_diffs: [],
            depends_on: [],
        },
        {
            change_id: "n:1",
            target: "node",
            op: "update",
            ref_id: "1",
            label: 'Update node "Welcome" (prompt)',
            field_diffs: [{ field: "prompt", before: "Greet the caller.", after: "Greet the caller, then offer a demo." }],
            depends_on: [],
        },
        {
            change_id: "e:1-9",
            target: "edge",
            op: "add",
            ref_id: "1-9",
            label: "Add edge Welcome → Book demo",
            field_diffs: [],
            depends_on: ["n:9"],
        },
    ],
};

function Harness(props: Partial<ProposalCardProps> & { initial?: Proposal }) {
    const [proposal, setProposal] = useState(props.initial ?? editProposal);
    return <ProposalCard draftVersion={4} {...props} proposal={proposal} onProposalChange={setProposal} />;
}

const checkbox = (name: RegExp) => screen.getByRole("checkbox", { name });

describe("ProposalCard", () => {
    beforeEach(() => {
        applyProposal.mockReset();
        discardProposal.mockReset();
    });

    it("groups changes, labels ops in text and starts with everything selected", () => {
        render(<Harness />);
        expect(screen.getByText("Nodes (2)")).toBeTruthy();
        expect(screen.getByText("Edges (1)")).toBeTruthy();
        expect(screen.getAllByText("Add")).toHaveLength(2);
        expect(screen.getByText("Update")).toBeTruthy();
        expect(screen.getByRole("button", { name: "Apply selected (3)" })).toBeTruthy();
        expect(screen.getByText(/needs: Add node "Book demo"/)).toBeTruthy();
    });

    it("unchecking a node unchecks the edge that needs it; re-checking the edge re-checks the node", () => {
        render(<Harness />);
        fireEvent.click(checkbox(/Book demo"$/));
        expect(checkbox(/Welcome → Book demo/).getAttribute("aria-checked")).toBe("false");
        expect(screen.getByRole("button", { name: "Apply selected (1)" })).toBeTruthy();
        fireEvent.click(checkbox(/Welcome → Book demo/));
        expect(checkbox(/Book demo"$/).getAttribute("aria-checked")).toBe("true");
        expect(screen.getByRole("button", { name: "Apply selected (3)" })).toBeTruthy();
    });

    it("expands field diffs with aria-expanded", () => {
        render(<Harness />);
        const toggle = screen.getAllByRole("button", { name: /Show changes/ })[1];
        expect(toggle.getAttribute("aria-expanded")).toBe("false");
        fireEvent.click(toggle);
        expect(toggle.getAttribute("aria-expanded")).toBe("true");
        expect(screen.getByRole("group", { name: "Changes to prompt" }).textContent).toContain("offer a demo");
    });

    it("applies only the selected changes with the draft version, then reports the save", async () => {
        const onApplied = vi.fn();
        applyProposal.mockResolvedValue({ saved: true, workflow_id: 7, draft_hash: "h" } satisfies ApplyProposalResponse);
        render(<Harness onApplied={onApplied} />);
        fireEvent.click(checkbox(/Welcome" \(prompt\)/));
        fireEvent.click(screen.getByRole("button", { name: "Apply selected (2)" }));
        await waitFor(() => expect(onApplied).toHaveBeenCalled());
        expect(applyProposal).toHaveBeenCalledWith("p1", {
            accepted_change_ids: ["n:9", "e:1-9"],
            expected_draft_version: 4,
        });
        expect(screen.getByText(/Saved 2 changes as a draft. Publish when ready./)).toBeTruthy();
        expect(screen.queryByRole("checkbox")).toBeNull();
    });

    it("shows the stale state with a redo action", async () => {
        const onAskRedo = vi.fn();
        applyProposal.mockResolvedValue({ saved: false, error_code: "stale", error: "hash mismatch" });
        render(<Harness onAskRedo={onAskRedo} />);
        fireEvent.click(screen.getByRole("button", { name: "Apply selected (3)" }));
        await screen.findByText(/The workflow changed since this was proposed/);
        fireEvent.click(screen.getByRole("button", { name: "Ask to redo it" }));
        expect(onAskRedo).toHaveBeenCalled();
    });

    it("shows a draft conflict with a canvas reload, and validation messages verbatim", async () => {
        const onReloadCanvas = vi.fn();
        applyProposal.mockResolvedValueOnce({ saved: false, error_code: "draft_conflict" });
        render(<Harness onReloadCanvas={onReloadCanvas} />);
        fireEvent.click(screen.getByRole("button", { name: "Apply selected (3)" }));
        fireEvent.click(await screen.findByRole("button", { name: "Reload canvas" }));
        expect(onReloadCanvas).toHaveBeenCalled();

        applyProposal.mockResolvedValueOnce({ saved: false, error_code: "graph_validation", error: "End call is unreachable" });
        fireEvent.click(screen.getByRole("button", { name: "Apply selected (3)" }));
        expect(await screen.findByText("End call is unreachable")).toBeTruthy();
    });

    it("holds apply while the canvas has unsaved edits", () => {
        render(<Harness hasUnsavedChanges />);
        expect((screen.getByRole("button", { name: "Apply selected (3)" }) as HTMLButtonElement).disabled).toBe(true);
        expect(screen.getByText(/unsaved edits/)).toBeTruthy();
    });

    it("create proposals have no checkboxes and a Create workflow action", async () => {
        const onApplied = vi.fn();
        applyProposal.mockResolvedValue({ saved: true, workflow_id: 42 });
        render(
            <Harness
                onApplied={onApplied}
                initial={{ ...editProposal, kind: "create", workflow_id: null, workflow_name: "Dental support" }}
            />,
        );
        expect(screen.queryByRole("checkbox")).toBeNull();
        expect(screen.getByText("New workflow: Dental support")).toBeTruthy();
        fireEvent.click(screen.getByRole("button", { name: "Create workflow" }));
        await waitFor(() => expect(onApplied).toHaveBeenCalled());
        expect(applyProposal).toHaveBeenCalledWith("p1", { accepted_change_ids: [], expected_draft_version: null });
        expect(onApplied.mock.calls[0][0]).toMatchObject({ status: "applied", workflow_id: 42 });
    });

    it("discards", async () => {
        discardProposal.mockResolvedValue({ ...editProposal, status: "discarded" });
        render(<Harness />);
        fireEvent.click(screen.getByRole("button", { name: "Discard" }));
        expect(await screen.findByText("Discarded.")).toBeTruthy();
    });

    it("shows another apply in progress with the buttons disabled", () => {
        render(<Harness initial={{ ...editProposal, status: "applying" }} />);
        const applying = screen.getByRole("button", { name: "Applying…" }) as HTMLButtonElement;
        expect(applying.disabled).toBe(true);
        expect((screen.getByRole("button", { name: "Discard" }) as HTMLButtonElement).disabled).toBe(true);
        expect(screen.queryByRole("checkbox")).toBeNull();
    });

    it("reports how many changes were applied from accepted_change_ids", async () => {
        applyProposal.mockResolvedValue({
            saved: true,
            workflow_id: 7,
            proposal: { ...editProposal, status: "applied", accepted_change_ids: ["n:9", "e:1-9"] },
        } satisfies ApplyProposalResponse);
        render(<Harness />);
        fireEvent.click(checkbox(/Welcome" \(prompt\)/));
        fireEvent.click(screen.getByRole("button", { name: "Apply selected (2)" }));
        expect(await screen.findByText("Applied 2 of 3 changes · draft")).toBeTruthy();
    });

    it("never shows the word Verified", () => {
        const { container } = render(<Harness />);
        expect(container.textContent?.toLowerCase()).not.toContain("verified");
    });
});
