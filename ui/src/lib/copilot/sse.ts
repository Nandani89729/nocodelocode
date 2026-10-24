/**
 * Minimal Server-Sent Events parser for the copilot message stream.
 *
 * The stream is a POST, so EventSource can't be used; the response body is
 * read with fetch and fed through this parser. It follows the WHATWG
 * event-stream rules that matter here: CR, LF and CRLF line endings (also
 * when a CRLF pair is split across chunks), multi-line `data:` fields joined
 * with "\n", `:` comment lines, a single optional space after the colon, and
 * dispatch on a blank line. `id` and `retry` are ignored (no reconnects).
 */
import type { CopilotEvent, CopilotEventType } from "./types";
import { COPILOT_EVENT_TYPES } from "./types";

export interface SseFrame {
    /** `event:` field, "message" when absent */
    event: string;
    /** `data:` lines joined with "\n" */
    data: string;
}

export class SseParser {
    private buffer = "";
    private eventName = "";
    private dataLines: string[] = [];
    private hasData = false;

    /** Feed a decoded text chunk; returns the frames completed by it. */
    feed(chunk: string): SseFrame[] {
        this.buffer += chunk;
        const frames: SseFrame[] = [];
        let start = 0;
        for (let i = 0; i < this.buffer.length; i++) {
            const ch = this.buffer[i];
            if (ch !== "\n" && ch !== "\r") continue;
            if (ch === "\r" && i === this.buffer.length - 1) {
                // A CR at the end of the chunk may be the first half of CRLF:
                // wait for the next chunk before treating it as a line end.
                break;
            }
            this.processLine(this.buffer.slice(start, i), frames);
            if (ch === "\r" && this.buffer[i + 1] === "\n") i++;
            start = i + 1;
        }
        this.buffer = this.buffer.slice(start);
        return frames;
    }

    /** End of stream: a trailing event without its blank line is dropped (per spec). */
    flush(): SseFrame[] {
        const frames: SseFrame[] = [];
        if (this.buffer.endsWith("\r")) {
            this.processLine(this.buffer.slice(0, -1), frames);
            this.buffer = "";
        }
        this.reset();
        return frames;
    }

    private reset() {
        this.eventName = "";
        this.dataLines = [];
        this.hasData = false;
    }

    private processLine(line: string, frames: SseFrame[]) {
        if (line === "") {
            if (this.hasData) {
                frames.push({ event: this.eventName || "message", data: this.dataLines.join("\n") });
            }
            this.reset();
            return;
        }
        if (line.startsWith(":")) return;
        const colon = line.indexOf(":");
        const field = colon === -1 ? line : line.slice(0, colon);
        let value = colon === -1 ? "" : line.slice(colon + 1);
        if (value.startsWith(" ")) value = value.slice(1);
        if (field === "event") {
            this.eventName = value;
        } else if (field === "data") {
            this.dataLines.push(value);
            this.hasData = true;
        }
    }
}

const KNOWN_TYPES = new Set<string>(COPILOT_EVENT_TYPES);

/**
 * Turn a frame into a typed copilot event. Unknown event names and data that
 * isn't a JSON object are skipped (returns null) so a newer backend can add
 * event types without breaking older UIs.
 */
export function toCopilotEvent(frame: SseFrame): CopilotEvent | null {
    if (!KNOWN_TYPES.has(frame.event)) return null;
    let data: unknown;
    try {
        data = frame.data === "" ? {} : JSON.parse(frame.data);
    } catch {
        return null;
    }
    if (!data || typeof data !== "object" || Array.isArray(data)) return null;
    return { type: frame.event as CopilotEventType, data } as CopilotEvent;
}

/** Read a byte stream to the end, yielding typed copilot events as they complete. */
export async function* readCopilotEvents(stream: ReadableStream<Uint8Array>): AsyncGenerator<CopilotEvent> {
    const reader = stream.getReader();
    const decoder = new TextDecoder();
    const parser = new SseParser();
    try {
        while (true) {
            const { done, value } = await reader.read();
            const text = done ? decoder.decode() : decoder.decode(value, { stream: true });
            const frames = text ? parser.feed(text) : [];
            if (done) frames.push(...parser.flush());
            for (const frame of frames) {
                const event = toCopilotEvent(frame);
                if (event) yield event;
            }
            if (done) return;
        }
    } finally {
        reader.releaseLock();
    }
}
