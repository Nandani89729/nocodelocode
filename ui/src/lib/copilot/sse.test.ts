import { describe, expect, it } from "vitest";

import { readCopilotEvents, SseParser, toCopilotEvent } from "./sse";

function streamOf(chunks: Array<string | Uint8Array>): ReadableStream<Uint8Array> {
    const encoder = new TextEncoder();
    return new ReadableStream({
        start(controller) {
            for (const c of chunks) controller.enqueue(typeof c === "string" ? encoder.encode(c) : c);
            controller.close();
        },
    });
}

async function collect(stream: ReadableStream<Uint8Array>) {
    const out = [];
    for await (const e of readCopilotEvents(stream)) out.push(e);
    return out;
}

describe("SseParser", () => {
    it("parses complete frames", () => {
        const p = new SseParser();
        const frames = p.feed('event: text_delta\ndata: {"text":"Hi"}\n\nevent: message_done\ndata: {}\n\n');
        expect(frames).toEqual([
            { event: "text_delta", data: '{"text":"Hi"}' },
            { event: "message_done", data: "{}" },
        ]);
    });

    it("handles a frame split at every possible chunk boundary", () => {
        const raw = 'event: tool_call\ndata: {"id":"t1","name":"get_workflow","summary":"Reading"}\n\n';
        for (let cut = 0; cut <= raw.length; cut++) {
            const p = new SseParser();
            const frames = [...p.feed(raw.slice(0, cut)), ...p.feed(raw.slice(cut))];
            expect(frames).toHaveLength(1);
            expect(frames[0].event).toBe("tool_call");
        }
    });

    it("joins multi-line data with newlines", () => {
        const p = new SseParser();
        expect(p.feed("event: x\ndata: line one\ndata: line two\ndata:\n\n")).toEqual([
            { event: "x", data: "line one\nline two\n" },
        ]);
    });

    it("accepts CRLF and bare CR line endings, including CRLF split across chunks", () => {
        const p = new SseParser();
        const frames = [
            ...p.feed("event: a\r"),
            ...p.feed("\ndata: 1\r\n\r"),
            ...p.feed("\nevent: b\rdata: 2\r\r"),
            // the final CR could still be half of a CRLF until the stream ends
            ...p.flush(),
        ];
        expect(frames).toEqual([
            { event: "a", data: "1" },
            { event: "b", data: "2" },
        ]);
    });

    it("ignores comments, unknown fields and blank-only dispatches", () => {
        const p = new SseParser();
        expect(p.feed(": keep-alive\n\nid: 7\nretry: 100\n\nevent: e\nfoo: bar\ndata: ok\n\n")).toEqual([
            { event: "e", data: "ok" },
        ]);
    });

    it("strips only one space after the colon and defaults the event name", () => {
        const p = new SseParser();
        expect(p.feed("data:  two spaces\n\n")).toEqual([{ event: "message", data: " two spaces" }]);
    });

    it("drops an unterminated trailing event on flush", () => {
        const p = new SseParser();
        expect(p.feed("event: x\ndata: partial")).toEqual([]);
        expect(p.flush()).toEqual([]);
    });
});

describe("toCopilotEvent", () => {
    it("types known events and skips unknown or malformed ones", () => {
        expect(toCopilotEvent({ event: "text_delta", data: '{"text":"a"}' })).toEqual({
            type: "text_delta",
            data: { text: "a" },
        });
        expect(toCopilotEvent({ event: "message_done", data: "" })).toEqual({ type: "message_done", data: {} });
        expect(toCopilotEvent({ event: "future_thing", data: "{}" })).toBeNull();
        expect(toCopilotEvent({ event: "text_delta", data: "not json" })).toBeNull();
        expect(toCopilotEvent({ event: "text_delta", data: "[1]" })).toBeNull();
    });
});

describe("readCopilotEvents", () => {
    it("decodes multi-byte characters split across chunks", async () => {
        const bytes = new TextEncoder().encode('event: text_delta\ndata: {"text":"café ✓"}\n\n');
        const split = bytes.indexOf(0xc3) + 1; // inside "é"
        const events = await collect(streamOf([bytes.slice(0, split), bytes.slice(split)]));
        expect(events).toEqual([{ type: "text_delta", data: { text: "café ✓" } }]);
    });

    it("yields events in order across many chunks", async () => {
        const events = await collect(
            streamOf([
                "event: text_delta\nda",
                'ta: {"text":"Hel"}\n\nevent: text_delta\ndata: {"text":"lo"}\n',
                '\nevent: tool_result\ndata: {"id":"t1","name":"n","ok":false,"summary":"Failed"}\n\n',
                "event: message_done\ndata: {}\n\n",
            ]),
        );
        expect(events.map((e) => e.type)).toEqual(["text_delta", "text_delta", "tool_result", "message_done"]);
    });
});
