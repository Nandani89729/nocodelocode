"use client";

import { Settings } from "lucide-react";
import { useId, useState } from "react";

import { type AiDisabledCopy, AiDisabledNotice, CROSS } from "@/components/doctor/ReportStates";
import { Button } from "@/components/ui/button";
import { ChoiceChips } from "@/components/ui/choice-chips";
import { Label } from "@/components/ui/label";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Progress } from "@/components/ui/progress";
import { Switch } from "@/components/ui/switch";
import {
    ENABLE_AI_LABEL,
    nChangeText,
    NO_KEY_DETAIL,
    NO_KEY_TEXT,
    statusLineText,
} from "@/lib/self-improve/format";
import { ALLOWED_EVERY_N_CALLS, type SelfImproveStatus } from "@/lib/self-improve/types";

const SELF_IMPROVE_AI_COPY: AiDisabledCopy = {
    noKeyTitle: NO_KEY_TEXT,
    noKeyDetail: NO_KEY_DETAIL,
    optInTitle: "AI analysis is off for your organisation. Self-improve needs it.",
    optInDetail:
        "When enabled, call transcripts are sent to Anthropic for analysis. Phone numbers, emails and card numbers are masked first; names and addresses spoken in calls may not be.",
    enableLabel: ENABLE_AI_LABEL,
};

const N_OPTIONS = ALLOWED_EVERY_N_CALLS.map((n) => ({ value: String(n), label: `${n} calls` }));

export interface StatusLineProps {
    status: SelfImproveStatus;
    /** Save the gear popover's settings (turning on may first ask about the brief, DS-6). */
    onSaveSettings: (enabled: boolean, everyN: number) => Promise<void>;
    onTurnOn: () => void;
    onResume: () => Promise<void>;
    onRunNow: () => Promise<string>;
    /** The org opted in to AI from the notice. */
    onAiEnabled: () => void;
    busy?: boolean;
    error?: string | null;
}

function SettingsPopover({
    status,
    onSave,
    onRunNow,
}: {
    status: SelfImproveStatus;
    onSave: (enabled: boolean, everyN: number) => Promise<void>;
    onRunNow: () => Promise<string>;
}) {
    const [open, setOpen] = useState(false);
    const [enabled, setEnabled] = useState(status.settings.enabled);
    const [everyN, setEveryN] = useState(status.settings.every_n_calls);
    const [saving, setSaving] = useState(false);
    const [running, setRunning] = useState(false);
    const [message, setMessage] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const switchId = useId();
    const chipsLabelId = useId();
    const changed = enabled !== status.settings.enabled || everyN !== status.settings.every_n_calls;
    // DS-10: counted calls are kept and the new N applies at once.
    const nNote =
        enabled && status.state === "collecting"
            ? nChangeText(status.collected, status.settings.every_n_calls, everyN, status.window_multiplier)
            : null;

    const onOpenChange = (next: boolean) => {
        if (next) {
            setEnabled(status.settings.enabled);
            setEveryN(status.settings.every_n_calls);
            setMessage(null);
            setError(null);
        }
        setOpen(next);
    };

    const save = async () => {
        setSaving(true);
        setError(null);
        try {
            await onSave(enabled, everyN);
            setOpen(false);
        } catch (e) {
            setError(e instanceof Error ? e.message : "Could not save Self-improve settings");
        } finally {
            setSaving(false);
        }
    };

    const runNow = async () => {
        setRunning(true);
        setError(null);
        setMessage(null);
        try {
            setMessage(await onRunNow());
        } catch (e) {
            setError(e instanceof Error ? e.message : "Could not start a round");
        } finally {
            setRunning(false);
        }
    };

    return (
        <Popover open={open} onOpenChange={onOpenChange}>
            <PopoverTrigger asChild>
                <Button
                    variant="ghost"
                    size="icon"
                    className="size-11 shrink-0 md:size-8"
                    aria-label="Self-improve settings"
                >
                    <Settings aria-hidden />
                </Button>
            </PopoverTrigger>
            <PopoverContent align="end" className="w-80 max-w-[calc(100vw-2rem)] space-y-4">
                <div className="flex min-h-11 items-center justify-between gap-3 md:min-h-0">
                    <Label htmlFor={switchId}>Self-improve</Label>
                    <Switch id={switchId} checked={enabled} onCheckedChange={setEnabled} disabled={saving} />
                </div>
                <div className="space-y-2">
                    <p id={chipsLabelId} className="text-sm font-medium">
                        Run a round every
                    </p>
                    <ChoiceChips
                        aria-labelledby={chipsLabelId}
                        options={N_OPTIONS}
                        value={String(everyN)}
                        onChange={(v) => setEveryN(Number(v))}
                        disabled={saving}
                        className="flex-wrap p-0"
                        chipClassName="min-h-11 px-3 py-1 tabular-nums md:min-h-8"
                    />
                    <p className="text-xs text-muted-foreground">
                        Real calls only: phone or web, where the caller speaks. Text-chat tests don&apos;t count.
                    </p>
                </div>
                <div aria-live="polite" className="space-y-1 text-sm tabular-nums">
                    {nNote ? <p>{nNote}</p> : null}
                    {message ? <p>{message}</p> : null}
                    {error ? (
                        <p className="text-destructive">
                            <span aria-hidden>{CROSS} </span>
                            {error}
                        </p>
                    ) : null}
                </div>
                <div className="flex flex-wrap gap-2">
                    <Button className="min-h-11 md:min-h-9" onClick={save} disabled={!changed || saving}>
                        {saving ? "Saving…" : "Save"}
                    </Button>
                    {status.settings.enabled ? (
                        <Button variant="outline" className="min-h-11 md:min-h-9" onClick={runNow} disabled={running}>
                            {running ? "Starting…" : "Run now"}
                        </Button>
                    ) : null}
                </div>
            </PopoverContent>
        </Popover>
    );
}

/**
 * First thing in the tab (DS-1): one line saying what Self-improve is doing,
 * with the on/off switch and N behind the gear (DS-3 status-line rows).
 */
export function StatusLine({
    status,
    onSaveSettings,
    onTurnOn,
    onResume,
    onRunNow,
    onAiEnabled,
    busy = false,
    error,
}: StatusLineProps) {
    const [resuming, setResuming] = useState(false);

    if (status.state === "ai_disabled" || status.state === "no_key") {
        return (
            <div className="border-b px-4">
                <AiDisabledNotice
                    reason={status.state === "no_key" ? "no_key" : "not_opted_in"}
                    copy={SELF_IMPROVE_AI_COPY}
                    onEnabled={onAiEnabled}
                />
            </div>
        );
    }

    const text = statusLineText(status);
    const showProgress = status.state === "collecting" && status.window_size > 0;

    return (
        <div className="space-y-2 border-b px-4 py-3">
            <div className="flex items-start gap-2">
                <p role="status" aria-live="polite" className="min-w-0 flex-1 pt-1.5 text-sm tabular-nums md:pt-1">
                    {text}
                </p>
                <SettingsPopover status={status} onSave={onSaveSettings} onRunNow={onRunNow} />
            </div>
            {showProgress ? (
                <Progress
                    aria-label="Calls collected"
                    value={Math.min(100, (status.collected / status.window_size) * 100)}
                    className="h-1.5"
                />
            ) : null}
            {status.state === "off" ? (
                <Button className="min-h-11 md:min-h-9" onClick={onTurnOn} disabled={busy}>
                    Turn on
                </Button>
            ) : null}
            {status.state === "paused" ? (
                <Button
                    className="min-h-11 md:min-h-9"
                    disabled={busy || resuming}
                    onClick={async () => {
                        setResuming(true);
                        try {
                            await onResume();
                        } finally {
                            setResuming(false);
                        }
                    }}
                >
                    {resuming ? "Resuming…" : "Resume"}
                </Button>
            ) : null}
            {error ? (
                <p className="text-sm text-destructive">
                    <span aria-hidden>{CROSS} </span>
                    {error}
                </p>
            ) : null}
        </div>
    );
}
