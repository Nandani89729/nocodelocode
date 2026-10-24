"use client";

import { Fragment, useMemo } from "react";

import { type Block, type Inline, parseMarkdown } from "@/lib/copilot/markdown";

function InlineNodes({ nodes }: { nodes: Inline[] }) {
    return (
        <>
            {nodes.map((node, i) => {
                switch (node.type) {
                    case "text":
                        return <Fragment key={i}>{node.text}</Fragment>;
                    case "code":
                        return (
                            <code key={i} className="rounded bg-muted px-1 py-0.5 font-mono text-[0.85em]">
                                {node.text}
                            </code>
                        );
                    case "strong":
                        return (
                            <strong key={i} className="font-semibold">
                                <InlineNodes nodes={node.children} />
                            </strong>
                        );
                    case "em":
                        return (
                            <em key={i}>
                                <InlineNodes nodes={node.children} />
                            </em>
                        );
                    case "link":
                        return (
                            <a
                                key={i}
                                href={node.href}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="underline underline-offset-2"
                            >
                                <InlineNodes nodes={node.children} />
                            </a>
                        );
                }
            })}
        </>
    );
}

function BlockNode({ block }: { block: Block }) {
    switch (block.type) {
        case "paragraph":
            return (
                <p className="whitespace-pre-wrap break-words">
                    <InlineNodes nodes={block.children} />
                </p>
            );
        case "heading":
            return (
                <p className="font-semibold">
                    <InlineNodes nodes={block.children} />
                </p>
            );
        case "list": {
            const items = block.items.map((item, i) => (
                <li key={i} className="break-words">
                    <InlineNodes nodes={item} />
                </li>
            ));
            return block.ordered ? (
                <ol start={block.start} className="list-decimal space-y-1 pl-5 tabular-nums">
                    {items}
                </ol>
            ) : (
                <ul className="list-disc space-y-1 pl-5">{items}</ul>
            );
        }
        case "code":
            return (
                <pre className="max-h-64 overflow-auto rounded-md bg-muted p-2 font-mono text-xs leading-5">
                    <code>{block.text}</code>
                </pre>
            );
    }
}

/** Renders the copilot's Markdown subset as React elements (never as HTML). */
export function Markdown({ text }: { text: string }) {
    const blocks = useMemo(() => parseMarkdown(text), [text]);
    return (
        <div className="space-y-2 text-sm leading-relaxed">
            {blocks.map((block, i) => (
                <BlockNode key={i} block={block} />
            ))}
        </div>
    );
}
