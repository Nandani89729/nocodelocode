/**
 * A deliberately small Markdown subset for copilot replies (the app ships no
 * Markdown renderer). Parses to plain data that the React renderer maps to
 * elements, so model output is never injected as HTML.
 *
 * Blocks: paragraphs, `#`–`###` headings, `-`/`*` and `1.` lists, fenced
 * code. Inline: `code`, **bold**, *italic*, [text](http(s) link).
 * Anything else renders as literal text.
 */

export type Inline =
    | { type: "text"; text: string }
    | { type: "code"; text: string }
    | { type: "strong"; children: Inline[] }
    | { type: "em"; children: Inline[] }
    | { type: "link"; href: string; children: Inline[] };

export type Block =
    | { type: "paragraph"; children: Inline[] }
    | { type: "heading"; level: 1 | 2 | 3; children: Inline[] }
    | { type: "list"; ordered: boolean; start: number; items: Inline[][] }
    | { type: "code"; text: string };

// `_` emphasis is intentionally unsupported: snake_case field and tool names
// (tool_uuids, end_call) are common in replies and must stay literal.
const INLINE_PATTERN = /`([^`\n]+)`|\*\*([^*\n]+?)\*\*|\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)|\*([^*\s](?:[^*\n]*[^*\s])?)\*/g;

export function parseInline(text: string): Inline[] {
    const out: Inline[] = [];
    let last = 0;
    for (const match of text.matchAll(INLINE_PATTERN)) {
        const index = match.index ?? 0;
        if (index > last) out.push({ type: "text", text: text.slice(last, index) });
        const [, code, strong, linkText, href, em] = match;
        if (code !== undefined) out.push({ type: "code", text: code });
        else if (strong !== undefined) out.push({ type: "strong", children: parseInline(strong) });
        else if (linkText !== undefined && href !== undefined) out.push({ type: "link", href, children: parseInline(linkText) });
        else out.push({ type: "em", children: parseInline(em) });
        last = index + match[0].length;
    }
    if (last < text.length) out.push({ type: "text", text: text.slice(last) });
    return out;
}

const HEADING = /^(#{1,3})\s+(.*)$/;
const BULLET = /^\s*[-*]\s+(.*)$/;
const ORDERED = /^\s*(\d+)[.)]\s+(.*)$/;
const FENCE = /^\s*```/;

export function parseMarkdown(source: string): Block[] {
    const lines = source.replace(/\r\n?/g, "\n").split("\n");
    const blocks: Block[] = [];
    let paragraph: string[] = [];

    const flushParagraph = () => {
        if (paragraph.length === 0) return;
        blocks.push({ type: "paragraph", children: parseInline(paragraph.join("\n")) });
        paragraph = [];
    };

    let i = 0;
    while (i < lines.length) {
        const line = lines[i];
        if (FENCE.test(line)) {
            flushParagraph();
            const body: string[] = [];
            i++;
            while (i < lines.length && !FENCE.test(lines[i])) body.push(lines[i++]);
            i++; // closing fence (or end of a still-streaming block)
            blocks.push({ type: "code", text: body.join("\n") });
            continue;
        }
        if (line.trim() === "") {
            flushParagraph();
            i++;
            continue;
        }
        const heading = HEADING.exec(line);
        if (heading) {
            flushParagraph();
            blocks.push({
                type: "heading",
                level: heading[1].length as 1 | 2 | 3,
                children: parseInline(heading[2]),
            });
            i++;
            continue;
        }
        const bullet = BULLET.exec(line);
        const ordered = ORDERED.exec(line);
        if (bullet || ordered) {
            flushParagraph();
            const isOrdered = !bullet;
            const pattern = isOrdered ? ORDERED : BULLET;
            const items: Inline[][] = [];
            const start = isOrdered ? Number(ordered![1]) : 1;
            while (i < lines.length) {
                const m = pattern.exec(lines[i]);
                if (!m) break;
                items.push(parseInline(isOrdered ? m[2] : m[1]));
                i++;
            }
            blocks.push({ type: "list", ordered: isOrdered, start, items });
            continue;
        }
        paragraph.push(line);
        i++;
    }
    flushParagraph();
    return blocks;
}
