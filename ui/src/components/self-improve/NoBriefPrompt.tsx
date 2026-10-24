"use client";

import { useState } from "react";

import { CROSS } from "@/components/doctor/ReportStates";
import { Button } from "@/components/ui/button";
import {
    DRAFT_BRIEF_LABEL,
    NO_BRIEF_TEXT,
    SKIP_BRIEF_LABEL,
    WRITE_BRIEF_LABEL,
} from "@/lib/self-improve/format";

export type NoBriefChoice = "draft" | "write" | "skip";

/**
 * DS-6: turning Self-improve on without a brief asks once. Every choice turns
 * it on; Draft also asks the copilot for a brief to confirm, Write opens the
 * editor.
 */
export function NoBriefPrompt({ onChoose }: { onChoose: (choice: NoBriefChoice) => Promise<void> }) {
    const [busy, setBusy] = useState<NoBriefChoice | null>(null);
    const [error, setError] = useState<string | null>(null);

    const choose = async (choice: NoBriefChoice) => {
        setBusy(choice);
        setError(null);
        try {
            await onChoose(choice);
        } catch (e) {
            setError(e instanceof Error ? e.message : "Something went wrong");
        } finally {
            setBusy(null);
        }
    };

    return (
        <section aria-label="Brief needed" className="space-y-2 border-b px-4 py-3">
            <p className="text-sm">{NO_BRIEF_TEXT}</p>
            <div className="flex flex-wrap gap-2">
                <Button className="min-h-11 md:min-h-9" onClick={() => choose("draft")} disabled={busy !== null}>
                    {busy === "draft" ? "Drafting…" : DRAFT_BRIEF_LABEL}
                </Button>
                <Button
                    variant="outline"
                    className="min-h-11 md:min-h-9"
                    onClick={() => choose("write")}
                    disabled={busy !== null}
                >
                    {WRITE_BRIEF_LABEL}
                </Button>
                <Button
                    variant="ghost"
                    className="min-h-11 md:min-h-9"
                    onClick={() => choose("skip")}
                    disabled={busy !== null}
                >
                    {SKIP_BRIEF_LABEL}
                </Button>
            </div>
            {error ? (
                <p role="alert" className="text-sm text-destructive">
                    <span aria-hidden>{CROSS} </span>
                    {error}
                </p>
            ) : null}
        </section>
    );
}
