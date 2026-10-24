"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { useAuth } from "@/lib/auth";
import { getDoctorReport, resumeDoctorReport } from "@/lib/doctor/api";
import type { ReportResponse } from "@/lib/doctor/types";

export const REPORT_POLL_MS = 3000;

/**
 * Loads a Doctor report and polls every 3s while it is queued, running, or a
 * partial report is being resumed (design 2A).
 */
export function useDoctorReport(reportUuid: string) {
    const { user, loading: authLoading } = useAuth();
    const [report, setReport] = useState<ReportResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [resuming, setResuming] = useState(false);
    const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

    const load = useCallback(async () => {
        try {
            const next = await getDoctorReport(reportUuid);
            setReport(next);
            setError(null);
            if (next.status !== "partial") setResuming(false);
            return next;
        } catch (err) {
            setError(err instanceof Error ? err.message : "Could not load the report");
            return null;
        }
    }, [reportUuid]);

    const shouldPoll = report
        ? report.status === "queued" || report.status === "running" || (report.status === "partial" && resuming)
        : false;

    useEffect(() => {
        if (authLoading || !user) return;
        void load();
    }, [authLoading, user, load]);

    useEffect(() => {
        if (!shouldPoll) return;
        timer.current = setTimeout(() => {
            void load();
        }, REPORT_POLL_MS);
        return () => {
            if (timer.current) clearTimeout(timer.current);
        };
    }, [shouldPoll, report, load]);

    const resume = useCallback(async () => {
        setResuming(true);
        try {
            const next = await resumeDoctorReport(reportUuid);
            setReport(next);
        } catch (err) {
            setResuming(false);
            setError(err instanceof Error ? err.message : "Could not resume the analysis");
        }
    }, [reportUuid]);

    return { report, error, resuming, reload: load, resume };
}
