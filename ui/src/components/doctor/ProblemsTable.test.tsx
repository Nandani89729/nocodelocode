import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { Cluster } from "@/lib/doctor/types";

import { ProblemsTable } from "./ProblemsTable";

vi.mock("./VerifyPanel", () => ({
    VerifyPanel: () => <div data-testid="verify-panel" />,
}));

const clusters: Cluster[] = [
    {
        cluster_id: "C2",
        title: "Agrees when caller says a competitor is better",
        category: "hallucination",
        severity: "major",
        fix_target: "Pricing",
        root_cause: "No guidance on competitors",
        runs_affected: 1,
        runs_analysed: 2,
        evidence: [{ run_id: 9, turn: 3, quote: "Yes, they are better." }],
        watch_only: false,
    },
    {
        cluster_id: "C1",
        title: "Answers questions in the welcome step and never moves on",
        category: "workflow_error",
        severity: "critical",
        fix_target: "Welcome and Source",
        root_cause: "Welcome prompt answers questions",
        runs_affected: 2,
        runs_analysed: 2,
        evidence: [{ run_id: 16, turn: 5, quote: "Sure, let me explain our pricing." }],
        watch_only: false,
    },
];

describe("ProblemsTable", () => {
    it("ranks rows, expands the top one, and links evidence descriptively", () => {
        render(<ProblemsTable workflowId={7} reportUuid="r" clusters={clusters} canPropose />);
        const rows = screen.getAllByRole("button", { expanded: true });
        expect(rows).toHaveLength(1);
        expect(rows[0].textContent).toContain("welcome step");
        expect(screen.getByRole("link", { name: /Call #16, turn 5/ })).toBeTruthy();
        expect(screen.getByText("Critical")).toBeTruthy();
    });

    it("toggles rows and moves focus with arrow keys", () => {
        render(<ProblemsTable workflowId={7} reportUuid="r" clusters={clusters} canPropose />);
        const top = screen.getByRole("button", { name: /welcome step/ });
        const second = screen.getByRole("button", { name: /competitor/ });
        expect(second.getAttribute("aria-expanded")).toBe("false");
        fireEvent.click(second);
        expect(second.getAttribute("aria-expanded")).toBe("true");
        top.focus();
        fireEvent.keyDown(top, { key: "ArrowDown" });
        expect(document.activeElement).toBe(second);
    });

    it("never shows the word Verified", () => {
        const { container } = render(<ProblemsTable workflowId={7} reportUuid="r" clusters={clusters} canPropose />);
        expect(container.textContent?.toLowerCase()).not.toContain("verified");
    });
});
