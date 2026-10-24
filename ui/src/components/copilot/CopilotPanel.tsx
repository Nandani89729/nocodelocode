"use client";

import { useRouter } from "next/navigation";
import { type KeyboardEvent, useCallback, useEffect, useId, useRef, useState } from "react";

import { type AiDisabledCopy, AiDisabledNotice, CHECK, CROSS } from "@/components/doctor/ReportStates";
import { SelfImproveTab } from "@/components/self-improve/SelfImproveTab";
import type { SelfImproveController } from "@/components/self-improve/useSelfImproveStatus";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { type ChatItem, isPlanActionable } from "@/lib/copilot/chat";
import { PLAN_APPROVED_MESSAGE, REDO_MESSAGE, toolLabel } from "@/lib/copilot/format";
import { type ApplyProposalResponse, MAX_MESSAGE_CHARS, type Proposal } from "@/lib/copilot/types";
import { readyAnnouncement, selfImproveTabLabel } from "@/lib/self-improve/format";
import { cn } from "@/lib/utils";

import { BriefChangeCard } from "./BriefChangeCard";
import { Markdown } from "./Markdown";
import { PlanCard } from "./PlanCard";
import { ProposalCard } from "./ProposalCard";
import { forgetBuildSession, useCopilotChat } from "./useCopilotChat";

export const EDIT_EXAMPLES = [
    "Find and fix why calls hang in the welcome step",
    "Add a step that books a demo",
    "Review my prompts for contradictions",
] as const;

export const BUILD_EXAMPLES = [
    "Build an inbound support agent for a dental clinic",
    "Build an outbound agent that qualifies real-estate leads",
    "Build an agent that confirms or reschedules appointments",
] as const;

const COPILOT_AI_COPY: AiDisabledCopy = {
    noKeyTitle: "The copilot is not available: this Dograh server has no AI API key.",
    noKeyDetail: "Ask your administrator to set ANTHROPIC_API_KEY on the server.",
    optInTitle: "AI features are off for your organisation.",
    optInDetail:
        "When enabled, your workflow definitions and the messages you send here are sent to Anthropic. The same setting turns on AI analysis in Agent Doctor.",
    enableLabel: "Enable AI features",
};

export interface CopilotPanelProps {
    /** null = build a new workflow (from /workflow/create) */
    workflowId: number | null;
    /** The draft version shown on the canvas, sent as the apply optimistic lock. */
    draftVersion?: number | string | null;
    /** The canvas has unsaved edits (apply is held until they're saved). */
    hasUnsavedChanges?: boolean;
    /** An edit proposal was saved: reload the working copy on the canvas. */
    onApplied?: (proposal: Proposal, result: ApplyProposalResponse) => void | Promise<void>;
    /** Reload the canvas on request (after a draft conflict). */
    onReloadCanvas?: () => void;
    /** Shows a Close button when given (omit inside a Sheet, which has its own). */
    onClose?: () => void;
    className?: string;
    /** e.g. room for a Sheet's own close button */
    headerClassName?: string;
    /**
     * Self-improve status for this workflow (polled by the editor). When given
     * (workflow-bound panels only) the panel shows Chat | Self-improve tabs.
     */
    selfImprove?: SelfImproveController;
    /** The panel is the full-screen sheet below 768px (DS-8). */
    mobile?: boolean;
}

type PanelTab = "chat" | "self-improve";

function ToolChip({ item }: { item: Extract<ChatItem, { kind: "tool" }> }) {
    const label = item.summary || toolLabel(item.name);
    return (
        <p className="flex">
            <span
                className={cn(
                    "inline-flex max-w-full items-center gap-1 rounded-md border px-2 py-0.5 text-xs text-muted-foreground",
                    item.status === "failed" && "border-destructive/40 text-destructive",
                )}
            >
                <span className="truncate">{label}</span>
                {item.status === "ok" ? <span aria-hidden>{CHECK}</span> : null}
                {item.status === "failed" ? <span aria-hidden>{CROSS}</span> : null}
                <span className="sr-only">
                    {item.status === "running" ? "(running)" : item.status === "ok" ? "(done)" : "(failed)"}
                </span>
                {item.status === "running" ? <span aria-hidden>…</span> : null}
            </span>
        </p>
    );
}

/**
 * The Agent Copilot chat: plans, builds and edits workflows through
 * reviewable proposals. Publishing always stays manual.
 */
export function CopilotPanel({
    workflowId,
    draftVersion,
    hasUnsavedChanges = false,
    onApplied,
    onReloadCanvas,
    onClose,
    className,
    headerClassName,
    selfImprove,
    mobile = false,
}: CopilotPanelProps) {
    const router = useRouter();
    const {
        chat,
        loadState,
        loadError,
        gate,
        banner,
        dismissBanner,
        send,
        stop,
        newChat,
        reload,
        clearGate,
        updateProposal,
        updateBriefChange,
    } = useCopilotChat(workflowId);
    const hasSelfImprove = !!selfImprove && workflowId !== null;
    const toReview = selfImprove?.toReview ?? 0;
    // Opened from "Copilot · 1 to review": start on the round waiting for review.
    const [tab, setTab] = useState<PanelTab>(() => (hasSelfImprove && toReview > 0 ? "self-improve" : "chat"));
    const [input, setInput] = useState("");
    const inputRef = useRef<HTMLTextAreaElement>(null);
    const logRef = useRef<HTMLDivElement>(null);
    const stickToBottom = useRef(true);
    const inputId = useId();
    const isBuildNew = workflowId === null;
    const examples = isBuildNew ? BUILD_EXAMPLES : EDIT_EXAMPLES;
    const streaming = chat.streaming;
    const inputDisabled = loadState !== "ready" || gate !== null;

    // Follow new output while the user is at the bottom of the log.
    useEffect(() => {
        const el = logRef.current;
        if (el && stickToBottom.current) el.scrollTop = el.scrollHeight;
    }, [chat.items]);

    /** Send typed text (clears the input, restores it if refused) or a canned message. */
    const submit = useCallback(
        async (text: string, fromInput = false) => {
            if (!text.trim() || streaming) return;
            if (fromInput) setInput("");
            stickToBottom.current = true;
            const result = await send(text);
            if (fromInput && result.restore !== undefined) setInput((current) => current || result.restore!);
        },
        [send, streaming],
    );

    const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            void submit(input, true);
        }
    };

    const handleApplied = useCallback(
        async (proposal: Proposal, result: ApplyProposalResponse) => {
            if (proposal.kind === "create") {
                const id = result.workflow_id ?? proposal.workflow_id;
                if (id) {
                    // The build chat now belongs to the new workflow (the server
                    // rebinds it); open its editor with the Copilot showing it.
                    forgetBuildSession();
                    router.push(`/workflow/${id}?copilot=1`);
                }
                return;
            }
            await onApplied?.(proposal, result);
        },
        [onApplied, router],
    );

    const handleNewChat = () => {
        newChat();
        setInput("");
        inputRef.current?.focus();
    };

    const renderItem = (item: ChatItem) => {
        switch (item.kind) {
            case "user":
                return (
                    <div key={item.key} className="flex justify-end">
                        <p className="max-w-[85%] whitespace-pre-wrap break-words rounded-lg bg-muted px-3 py-2 text-sm">
                            <span className="sr-only">You: </span>
                            {item.text}
                        </p>
                    </div>
                );
            case "assistant":
                return (
                    <div key={item.key} className="max-w-full">
                        <span className="sr-only">Copilot: </span>
                        <Markdown text={item.text} />
                    </div>
                );
            case "tool":
                return <ToolChip key={item.key} item={item} />;
            case "plan":
                return (
                    <PlanCard
                        key={item.key}
                        plan={item.plan}
                        actionable={isPlanActionable(chat, item.key) && gate === null}
                        onApprove={() => void submit(PLAN_APPROVED_MESSAGE)}
                        onChangeSomething={() => inputRef.current?.focus()}
                    />
                );
            case "proposal": {
                const proposal = chat.proposals[item.proposalUuid];
                if (!proposal) return null;
                return (
                    <ProposalCard
                        key={item.key}
                        proposal={proposal}
                        draftVersion={draftVersion}
                        hasUnsavedChanges={hasUnsavedChanges}
                        busy={streaming}
                        onProposalChange={updateProposal}
                        onApplied={handleApplied}
                        onAskRedo={() => void submit(REDO_MESSAGE)}
                        onReloadCanvas={onReloadCanvas}
                    />
                );
            }
            case "brief_change": {
                const change = chat.briefChanges[item.changeUuid];
                if (!change || workflowId === null) return null;
                return (
                    <BriefChangeCard
                        key={item.key}
                        workflowId={workflowId}
                        change={change}
                        onChange={updateBriefChange}
                        onResolved={() => void selfImprove?.refresh()}
                    />
                );
            }
            case "notice":
                return (
                    <p
                        key={item.key}
                        className={cn("text-sm", item.tone === "error" ? "text-destructive" : "text-muted-foreground")}
                    >
                        {item.tone === "error" ? <span aria-hidden>{CROSS} </span> : null}
                        {item.text}
                    </p>
                );
        }
    };

    let body: React.ReactNode;
    if (loadState === "loading") {
        body = (
            <div className="space-y-3" aria-hidden>
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-4 w-1/2" />
                <Skeleton className="h-16 w-full" />
            </div>
        );
    } else if (loadState === "error") {
        body = (
            <div className="space-y-2 text-sm">
                <p className="text-destructive">
                    <span aria-hidden>{CROSS} </span>
                    {loadError}
                </p>
                <Button variant="outline" className="min-h-11 md:min-h-9" onClick={() => void reload()}>
                    Retry
                </Button>
            </div>
        );
    } else if (chat.items.length === 0 && gate === null) {
        body = (
            <div className="space-y-3">
                <p className="text-sm text-muted-foreground">
                    {isBuildNew
                        ? "Describe the agent you want. The copilot plans it with you, then builds it as a draft workflow."
                        : "Ask for a change to this workflow. The copilot proposes edits you can review one by one; nothing is published."}
                </p>
                <div className="space-y-2">
                    <p className="text-xs font-medium text-muted-foreground">Try</p>
                    <ul className="space-y-2">
                        {examples.map((example) => (
                            <li key={example}>
                                <Button
                                    variant="outline"
                                    className="h-auto min-h-11 w-full justify-start whitespace-normal py-2 text-left font-normal md:min-h-9"
                                    onClick={() => void submit(example)}
                                >
                                    {example}
                                </Button>
                            </li>
                        ))}
                    </ul>
                </div>
            </div>
        );
    } else {
        body = <div className="space-y-4">{chat.items.map(renderItem)}</div>;
    }

    const chatPane = (
        <>
            <div
                ref={logRef}
                role="log"
                aria-live="polite"
                aria-busy={streaming || loadState === "loading"}
                aria-label="Copilot conversation"
                className="min-h-0 flex-1 overflow-y-auto px-4 py-4"
                onScroll={(e) => {
                    const el = e.currentTarget;
                    stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
                }}
            >
                {gate ? (
                    <div className="mb-4 border-b">
                        <AiDisabledNotice
                            reason={gate}
                            copy={COPILOT_AI_COPY}
                            onEnabled={() => {
                                clearGate();
                                // The refused message was put back in the input: send it now.
                                if (input.trim()) void submit(input, true);
                                else void reload();
                            }}
                        />
                    </div>
                ) : null}
                {body}
            </div>

            <div className="space-y-2 border-t px-4 py-3">
                {banner ? (
                    <div role="status" className="flex items-start gap-2 text-sm">
                        <p className="flex-1 text-destructive">{banner}</p>
                        <Button variant="ghost" size="sm" className="min-h-11 md:min-h-8" onClick={dismissBanner}>
                            Dismiss
                        </Button>
                    </div>
                ) : null}
                <form
                    className="space-y-2"
                    onSubmit={(e) => {
                        e.preventDefault();
                        void submit(input, true);
                    }}
                >
                    <Label htmlFor={inputId} className="sr-only">
                        Message the copilot
                    </Label>
                    <Textarea
                        id={inputId}
                        ref={inputRef}
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={onKeyDown}
                        disabled={inputDisabled}
                        maxLength={MAX_MESSAGE_CHARS}
                        rows={3}
                        placeholder={isBuildNew ? "Describe the agent to build" : "Ask for a change"}
                        className="max-h-48 min-h-[4.5rem] resize-none text-sm"
                    />
                    <div className="flex items-center justify-between gap-2">
                        <p className="text-xs text-muted-foreground">Enter to send, Shift+Enter for a new line</p>
                        {streaming ? (
                            <Button type="button" variant="outline" className="min-h-11 md:min-h-9" onClick={stop}>
                                Stop
                            </Button>
                        ) : (
                            <Button
                                type="submit"
                                variant="outline"
                                className="min-h-11 md:min-h-9"
                                disabled={inputDisabled || input.trim() === ""}
                            >
                                Send
                            </Button>
                        )}
                    </div>
                </form>
            </div>
        </>
    );

    return (
        <section aria-label="Copilot" className={cn("flex h-full min-h-0 flex-col bg-background", className)}>
            <header className={cn("flex items-center gap-2 border-b px-4 py-3", headerClassName)}>
                <div className="min-w-0 flex-1">
                    <h2 className="text-sm font-medium">Copilot</h2>
                    <p className="truncate text-xs text-muted-foreground">
                        {isBuildNew ? "New workflow" : "Editing this workflow"}
                    </p>
                </div>
                {!hasSelfImprove || tab === "chat" ? (
                    <Button
                        variant="ghost"
                        size="sm"
                        className="min-h-11 md:min-h-8"
                        onClick={handleNewChat}
                        disabled={loadState === "loading"}
                    >
                        New chat
                    </Button>
                ) : null}
                {onClose ? (
                    <Button variant="ghost" size="sm" className="min-h-11 md:min-h-8" onClick={onClose}>
                        Close
                    </Button>
                ) : null}
            </header>

            {hasSelfImprove && selfImprove && workflowId !== null ? (
                <Tabs
                    value={tab}
                    onValueChange={(value) => setTab(value as PanelTab)}
                    className="min-h-0 flex-1 gap-0"
                >
                    <div className="border-b px-4 py-2">
                        <TabsList className="h-11 w-full md:h-9">
                            <TabsTrigger value="chat">Chat</TabsTrigger>
                            <TabsTrigger value="self-improve" className="tabular-nums">
                                {selfImproveTabLabel(toReview)}
                            </TabsTrigger>
                        </TabsList>
                    </div>
                    {/* Kept mounted while hidden: the draft message and scroll position survive a tab switch. */}
                    <TabsContent value="chat" forceMount className="flex min-h-0 flex-col data-[state=inactive]:hidden">
                        {chatPane}
                    </TabsContent>
                    <TabsContent value="self-improve" className="min-h-0 overflow-y-auto">
                        <SelfImproveTab
                            workflowId={workflowId}
                            controller={selfImprove}
                            mobile={mobile}
                            draftVersion={draftVersion}
                            hasUnsavedChanges={hasUnsavedChanges}
                            onApplied={onApplied}
                            onReloadCanvas={onReloadCanvas}
                        />
                    </TabsContent>
                    <p className="sr-only" aria-live="polite">
                        {readyAnnouncement(toReview)}
                    </p>
                </Tabs>
            ) : (
                chatPane
            )}
        </section>
    );
}
