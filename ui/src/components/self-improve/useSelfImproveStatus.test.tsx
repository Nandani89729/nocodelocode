import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { SelfImproveStatus } from "@/lib/self-improve/types";

import { SELF_IMPROVE_POLL_MS, shouldPollStatus, useSelfImproveStatus } from "./useSelfImproveStatus";

vi.mock("@/lib/auth", () => ({ useAuth: () => ({ user: { id: "u1" }, loading: false }) }));

const getSelfImproveStatus = vi.hoisted(() => vi.fn());
vi.mock("@/lib/self-improve/api", () => ({ getSelfImproveStatus }));

const status = (over: Partial<SelfImproveStatus> = {}): SelfImproveStatus => ({
    workflow_id: 7,
    settings: { enabled: true, every_n_calls: 10 },
    state: "collecting",
    collected: 3,
    window_size: 10,
    window_multiplier: 1,
    to_review: 0,
    has_brief: true,
    ...over,
});

let visibility: DocumentVisibilityState = "visible";

/** Let the pending GET resolve and React apply it. */
const flush = () => act(async () => {});

describe("useSelfImproveStatus", () => {
    beforeEach(() => {
        vi.useFakeTimers();
        getSelfImproveStatus.mockReset();
        visibility = "visible";
        Object.defineProperty(document, "visibilityState", { configurable: true, get: () => visibility });
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it("polls every 3s while on, and exposes the ready count", async () => {
        getSelfImproveStatus.mockResolvedValue(status({ to_review: 1, state: "waiting" }));
        const { result } = renderHook(() => useSelfImproveStatus(7));
        await flush();
        expect(result.current.toReview).toBe(1);
        expect(getSelfImproveStatus).toHaveBeenCalledTimes(1);
        await act(async () => {
            vi.advanceTimersByTime(SELF_IMPROVE_POLL_MS);
        });
        expect(getSelfImproveStatus).toHaveBeenCalledTimes(2);
    });

    it("pauses while the browser tab is hidden and refreshes when it comes back", async () => {
        getSelfImproveStatus.mockResolvedValue(status());
        renderHook(() => useSelfImproveStatus(7));
        await flush();
        visibility = "hidden";
        await act(async () => {
            document.dispatchEvent(new Event("visibilitychange"));
        });
        const calls = getSelfImproveStatus.mock.calls.length;
        await act(async () => {
            vi.advanceTimersByTime(SELF_IMPROVE_POLL_MS * 3);
        });
        expect(getSelfImproveStatus.mock.calls.length).toBe(calls);
        visibility = "visible";
        await act(async () => {
            document.dispatchEvent(new Event("visibilitychange"));
        });
        expect(getSelfImproveStatus.mock.calls.length).toBe(calls + 1);
    });

    it("does not poll while off or AI-gated", () => {
        expect(shouldPollStatus(status({ state: "off", settings: { enabled: false, every_n_calls: 10 } }))).toBe(false);
        expect(shouldPollStatus(status({ state: "ai_disabled" }))).toBe(false);
        expect(shouldPollStatus(status({ state: "no_key" }))).toBe(false);
        expect(shouldPollStatus(status({ state: "running" }))).toBe(true);
        expect(shouldPollStatus(null)).toBe(false);
    });

    it("does nothing without a workflow", async () => {
        renderHook(() => useSelfImproveStatus(null));
        await flush();
        expect(getSelfImproveStatus).not.toHaveBeenCalled();
    });
});
