"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { useAuth } from "@/lib/auth";
import {
    CopilotApiError,
    type CopilotGate,
    createCopilotSession,
    gateOf,
    getCopilotSession,
    isAbortError,
    isBusyError,
    listCopilotSessions,
    streamCopilotMessage,
} from "@/lib/copilot/api";
import {
    applyEvent,
    cancelTurn,
    chatFromSession,
    type ChatState,
    emptyChat,
    endTurn,
    startTurn,
    upsertBriefChange,
    upsertProposal,
} from "@/lib/copilot/chat";
import type { BriefChange, Proposal } from "@/lib/copilot/types";

export const BUSY_TEXT =
    "The copilot is already working on a reply in this chat. Wait for it to finish, or start a new chat.";

export type LoadState = "loading" | "ready" | "error";

/**
 * Build-new chats have no workflow to list sessions by, so the open one is
 * remembered for this tab: closing and reopening the panel resumes it.
 */
const BUILD_SESSION_KEY = "dograh.copilot.buildSession";

function readBuildSession(): string | null {
    try {
        return window.sessionStorage.getItem(BUILD_SESSION_KEY);
    } catch {
        return null;
    }
}

/** After a build-new chat created its workflow, the next visit starts fresh. */
export const forgetBuildSession = () => writeBuildSession(null);

function writeBuildSession(uuid: string | null) {
    try {
        if (uuid) window.sessionStorage.setItem(BUILD_SESSION_KEY, uuid);
        else window.sessionStorage.removeItem(BUILD_SESSION_KEY);
    } catch {
        // storage unavailable: the chat just won't resume
    }
}

export interface SendResult {
    /** the message was not accepted: put this text back in the input */
    restore?: string;
}

/**
 * Session + streaming state for the copilot panel.
 *
 * Editing: on open, loads the workflow's latest session (if any). Build-new
 * (workflowId null) starts empty. A session is created lazily on the first
 * message, so opening the panel never leaves empty sessions behind.
 */
export function useCopilotChat(workflowId: number | null) {
    const { user, loading: authLoading } = useAuth();
    const [chat, setChat] = useState<ChatState>(emptyChat);
    const [loadState, setLoadState] = useState<LoadState>("loading");
    const [loadError, setLoadError] = useState<string | null>(null);
    const [gate, setGate] = useState<CopilotGate | null>(null);
    /** transient problem shown above the input (busy, failed send) */
    const [banner, setBanner] = useState<string | null>(null);
    const sessionUuid = useRef<string | null>(null);
    const abortRef = useRef<AbortController | null>(null);
    const generation = useRef(0);

    const load = useCallback(async () => {
        const mine = ++generation.current;
        setLoadState("loading");
        setLoadError(null);
        setGate(null);
        try {
            if (workflowId === null) {
                const saved = readBuildSession();
                const session = saved ? await getCopilotSession(saved).catch(() => null) : null;
                if (mine !== generation.current) return;
                if (saved && !session) writeBuildSession(null);
                sessionUuid.current = session?.session_uuid ?? null;
                setChat(session ? chatFromSession(session) : emptyChat());
            } else {
                // Chat sessions only: the backend leaves out Self-improve sessions
                // (kind="self_improve", RI-1), so a round never hijacks the chat.
                const sessions = await listCopilotSessions(workflowId);
                const latest = sessions[0];
                const session = latest ? await getCopilotSession(latest.session_uuid) : null;
                if (mine !== generation.current) return;
                sessionUuid.current = session?.session_uuid ?? null;
                setChat(session ? chatFromSession(session) : emptyChat());
            }
            setLoadState("ready");
        } catch (err) {
            if (mine !== generation.current) return;
            const g = gateOf(err);
            if (g) {
                setGate(g);
                setLoadState("ready");
                return;
            }
            setLoadError(err instanceof Error ? err.message : "Could not load the copilot chat");
            setLoadState("error");
        }
    }, [workflowId]);

    // Wait for auth: the Bearer interceptor is only registered once it's ready.
    const hasLoaded = useRef(false);
    useEffect(() => {
        if (authLoading || !user || hasLoaded.current) return;
        hasLoaded.current = true;
        void load();
    }, [authLoading, user, load]);

    // Stop the stream when the panel unmounts.
    useEffect(() => () => abortRef.current?.abort(), []);

    const send = useCallback(
        async (raw: string): Promise<SendResult> => {
            const text = raw.trim();
            if (!text || abortRef.current) return { restore: raw };
            setBanner(null);
            setChat((s) => startTurn(s, text));
            const controller = new AbortController();
            abortRef.current = controller;
            let started = false;
            try {
                if (!sessionUuid.current) {
                    const session = await createCopilotSession(workflowId);
                    sessionUuid.current = session.session_uuid;
                    if (workflowId === null) writeBuildSession(session.session_uuid);
                }
                await streamCopilotMessage(
                    sessionUuid.current,
                    text,
                    (event) => {
                        started = true;
                        setChat((s) => applyEvent(s, event));
                    },
                    controller.signal,
                );
                // The stream closed; make sure the turn is over even without message_done.
                setChat((s) => (s.streaming ? endTurn(s) : s));
                return {};
            } catch (err) {
                if (controller.signal.aborted || isAbortError(err)) {
                    setChat((s) => endTurn(s, { tone: "info", text: "Stopped." }));
                    return {};
                }
                if (err instanceof CopilotApiError && !started) {
                    // Refused before the turn began: nothing was said, give the text back.
                    setChat((s) => cancelTurn(s, text));
                    const g = gateOf(err);
                    if (g) {
                        setGate(g);
                    } else {
                        setBanner(isBusyError(err) ? BUSY_TEXT : err.message);
                    }
                    return { restore: raw };
                }
                setChat((s) =>
                    endTurn(s, { tone: "error", text: "The connection dropped before the reply finished. Try again." }),
                );
                return {};
            } finally {
                if (abortRef.current === controller) abortRef.current = null;
            }
        },
        [workflowId],
    );

    const stop = useCallback(() => {
        abortRef.current?.abort();
    }, []);

    const newChat = useCallback(() => {
        abortRef.current?.abort();
        generation.current++;
        sessionUuid.current = null;
        if (workflowId === null) writeBuildSession(null);
        setChat(emptyChat());
        setBanner(null);
        setGate(null);
        setLoadError(null);
        setLoadState("ready");
    }, [workflowId]);

    /** After the org opts in: the panel re-sends the restored input, or reloads. */
    const clearGate = useCallback(() => setGate(null), []);

    const updateProposal = useCallback((proposal: Proposal) => {
        setChat((s) => upsertProposal(s, proposal));
    }, []);

    const updateBriefChange = useCallback((change: BriefChange) => {
        setChat((s) => upsertBriefChange(s, change));
    }, []);

    return {
        chat,
        loadState,
        loadError,
        gate,
        banner,
        dismissBanner: () => setBanner(null),
        send,
        stop,
        newChat,
        reload: load,
        clearGate,
        updateProposal,
        updateBriefChange,
    };
}
