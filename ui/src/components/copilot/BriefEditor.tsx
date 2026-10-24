"use client";

import { useEffect, useId, useRef, useState } from "react";

import { CROSS, WARN } from "@/components/doctor/ReportStates";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { getBrief, isBriefConflict, updateBrief } from "@/lib/copilot/api";
import {
    BRIEF_CONFLICT_TEXT,
    BRIEF_FIELD_LABEL,
    BRIEF_LIST_FIELDS,
    type BriefListField,
    briefsEqual,
    coerceBrief,
    normalizeBrief,
} from "@/lib/copilot/brief";
import type { BriefResponse, WorkflowBrief } from "@/lib/copilot/types";

export interface BriefEditorProps {
    workflowId: number;
    /** The brief as loaded; its version is the optimistic lock for Save. */
    brief: BriefResponse | null;
    onSaved: (brief: BriefResponse) => void;
    onCancel: () => void;
}

function ListField({
    field,
    items,
    onChange,
    disabled,
}: {
    field: BriefListField;
    items: string[];
    onChange: (items: string[]) => void;
    disabled: boolean;
}) {
    const label = BRIEF_FIELD_LABEL[field];
    const legendId = useId();
    const refs = useRef<(HTMLInputElement | null)[]>([]);
    const [focusIndex, setFocusIndex] = useState<number | null>(null);

    useEffect(() => {
        if (focusIndex === null) return;
        refs.current[focusIndex]?.focus();
        setFocusIndex(null);
    }, [focusIndex, items.length]);

    return (
        <fieldset className="space-y-2" aria-labelledby={legendId}>
            <legend id={legendId} className="text-sm font-medium">
                {label}
            </legend>
            {items.length === 0 ? <p className="text-xs text-muted-foreground">None yet.</p> : null}
            <ul className="space-y-2">
                {items.map((item, i) => (
                    <li key={i} className="flex items-center gap-2">
                        <Input
                            ref={(el) => {
                                refs.current[i] = el;
                            }}
                            aria-label={`${label} ${i + 1}`}
                            value={item}
                            disabled={disabled}
                            onChange={(e) => onChange(items.map((v, j) => (j === i ? e.target.value : v)))}
                            className="min-h-11 text-sm md:min-h-9"
                        />
                        <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            className="min-h-11 shrink-0 md:min-h-8"
                            aria-label={`Remove ${label} ${i + 1}`}
                            disabled={disabled}
                            onClick={() => {
                                onChange(items.filter((_, j) => j !== i));
                                setFocusIndex(Math.max(0, Math.min(i, items.length - 2)));
                            }}
                        >
                            Remove
                        </Button>
                    </li>
                ))}
            </ul>
            <Button
                type="button"
                variant="outline"
                size="sm"
                className="min-h-11 md:min-h-8"
                aria-label={`Add to ${label}`}
                disabled={disabled}
                onClick={() => {
                    onChange([...items, ""]);
                    setFocusIndex(items.length);
                }}
            >
                Add
            </Button>
        </fieldset>
    );
}

/**
 * Direct brief editing (DS-9): Goal (one input), Must do / Must never /
 * Accepted tradeoffs (one input per item), Notes. Visible labels; Save sends
 * the version it was opened at, and a conflict says so instead of overwriting.
 */
export function BriefEditor({ workflowId, brief, onSaved, onCancel }: BriefEditorProps) {
    const [form, setForm] = useState<WorkflowBrief>(() => coerceBrief(brief?.brief));
    const [version, setVersion] = useState(brief?.version ?? 0);
    const [saving, setSaving] = useState(false);
    const [conflict, setConflict] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const goalId = useId();
    const notesId = useId();
    const unchanged = briefsEqual(form, brief?.brief);

    const set = <K extends keyof WorkflowBrief>(key: K, value: WorkflowBrief[K]) =>
        setForm((f) => ({ ...f, [key]: value }));

    const save = async () => {
        setSaving(true);
        setError(null);
        try {
            const saved = await updateBrief(workflowId, { brief: normalizeBrief(form), expected_version: version });
            setConflict(false);
            onSaved(saved);
        } catch (e) {
            if (isBriefConflict(e)) setConflict(true);
            else setError(e instanceof Error ? e.message : "Could not save the brief");
        } finally {
            setSaving(false);
        }
    };

    /** Conflict: take the latest brief (drops these edits) or keep the edits on top of it. */
    const loadLatest = async (keepEdits: boolean) => {
        setError(null);
        try {
            const latest = await getBrief(workflowId);
            setVersion(latest.version);
            if (!keepEdits) setForm(coerceBrief(latest.brief));
            setConflict(false);
        } catch (e) {
            setError(e instanceof Error ? e.message : "Could not load the brief");
        }
    };

    return (
        <form
            aria-label="Edit brief"
            className="space-y-4"
            onSubmit={(e) => {
                e.preventDefault();
                void save();
            }}
        >
            <div className="space-y-2">
                <Label htmlFor={goalId}>{BRIEF_FIELD_LABEL.goal}</Label>
                <Input
                    id={goalId}
                    value={form.goal}
                    disabled={saving}
                    onChange={(e) => set("goal", e.target.value)}
                    className="min-h-11 text-sm md:min-h-9"
                />
            </div>
            {BRIEF_LIST_FIELDS.map((field) => (
                <ListField
                    key={field}
                    field={field}
                    items={form[field]}
                    disabled={saving}
                    onChange={(items) => set(field, items)}
                />
            ))}
            <div className="space-y-2">
                <Label htmlFor={notesId}>{BRIEF_FIELD_LABEL.notes}</Label>
                <Textarea
                    id={notesId}
                    value={form.notes}
                    disabled={saving}
                    rows={3}
                    onChange={(e) => set("notes", e.target.value)}
                    className="text-sm"
                />
            </div>

            <div aria-live="polite" className="space-y-2 text-sm">
                {conflict ? (
                    <div className="space-y-2">
                        <p>
                            <span aria-hidden>{WARN} </span>
                            {BRIEF_CONFLICT_TEXT}
                        </p>
                        <div className="flex flex-wrap gap-2">
                            <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                className="min-h-11 md:min-h-8"
                                onClick={() => void loadLatest(false)}
                            >
                                Load their version
                            </Button>
                            <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                className="min-h-11 md:min-h-8"
                                onClick={() => void loadLatest(true)}
                            >
                                Keep my edits
                            </Button>
                        </div>
                    </div>
                ) : null}
                {error ? (
                    <p className="text-destructive">
                        <span aria-hidden>{CROSS} </span>
                        {error}
                    </p>
                ) : null}
            </div>

            <div className="flex flex-wrap gap-2">
                <Button type="submit" className="min-h-11 md:min-h-9" disabled={saving || conflict || unchanged}>
                    {saving ? "Saving…" : "Save brief"}
                </Button>
                <Button type="button" variant="outline" className="min-h-11 md:min-h-9" onClick={onCancel} disabled={saving}>
                    Cancel
                </Button>
            </div>
        </form>
    );
}
