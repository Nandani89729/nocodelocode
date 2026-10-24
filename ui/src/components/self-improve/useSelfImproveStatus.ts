"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { REPORT_POLL_MS } from "@/components/doctor/useDoctorReport";
import { useAuth } from "@/lib/auth";
import { getSelfImproveStatus } from "@/lib/self-improve/api";
import type { SelfImproveStatus } from "@/lib/self-improve/types";

/** Same cadence as the Doctor report page (DS-2). */
export const SELF_IMPROVE_POLL_MS = REPORT_POLL_MS;

function pageVisible() {
    return typeof document === "undefined" || document.visibilityState !== "hidden";
}

/** Nothing changes on its own while Self-improve is off or AI-gated: no need to poll. */
export function shouldPollStatus(status: SelfImproveStatus | null): boolean {
    if (!status) return false;
    if (!status.settings.enabled) return false;
    return status.state !== "off" && status.state !== "ai_disabled" && status.state !== "no_key";
}

/**
 * Self-improve status for the open editor (DS-2): drives "Copilot · 1 to
 * review" in the header and the panel's tab label. Polls every 3s while
 * Self-improve is on and the browser tab is visible; refreshes once when the
 * tab becomes visible again. Pass null to disable (no workflow).
 */
export function useSelfImproveStatus(workflowId: number | null) {
    const { user, loading: authLoading } = useAuth();
    const [status, setStatus] = useState<SelfImproveStatus | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [visible, setVisible] = useState(pageVisible);
    const inFlight = useRef(false);
    const generation = useRef(0);

    /**
     * Load the status. A forced refresh (the default; used after the user acts)
     * supersedes a poll already in flight, whose older answer is then dropped.
     */
    const refresh = useCallback(async (force = true) => {
        if (workflowId === null) return null;
        if (inFlight.current && !force) return null;
        if (force) generation.current++;
        inFlight.current = true;
        const mine = generation.current;
        try {
            const next = await getSelfImproveStatus(workflowId);
            if (mine !== generation.current) return null;
            setStatus(next);
            setError(null);
            return next;
        } catch (err) {
            if (mine !== generation.current) return null;
            setError(err instanceof Error ? err.message : "Could not load Self-improve");
            return null;
        } finally {
            if (mine === generation.current) inFlight.current = false;
        }
    }, [workflowId]);

    /** Use a status returned by a mutation (PUT / resume) without another GET. */
    const replaceStatus = useCallback((next: SelfImproveStatus) => {
        generation.current++;
        inFlight.current = false;
        setStatus(next);
        setError(null);
    }, []);

    // First load once auth is ready (the Bearer interceptor registers then), once per workflow.
    const loadedFor = useRef<number | null>(null);
    useEffect(() => {
        if (authLoading || !user || workflowId === null || loadedFor.current === workflowId) return;
        loadedFor.current = workflowId;
        void refresh();
    }, [authLoading, user, workflowId, refresh]);

    useEffect(() => {
        const onVisibility = () => {
            const now = pageVisible();
            setVisible(now);
            if (now) void refresh();
        };
        document.addEventListener("visibilitychange", onVisibility);
        return () => document.removeEventListener("visibilitychange", onVisibility);
    }, [refresh]);

    const polling = visible && shouldPollStatus(status);
    useEffect(() => {
        if (!polling) return;
        const id = setInterval(() => void refresh(false), SELF_IMPROVE_POLL_MS);
        return () => clearInterval(id);
    }, [polling, refresh]);

    return { status, error, refresh, replaceStatus, toReview: status?.to_review ?? 0 };
}

export type SelfImproveController = ReturnType<typeof useSelfImproveStatus>;
