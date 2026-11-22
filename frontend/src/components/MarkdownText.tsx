import { Fragment, type ReactNode } from "react";

// Minimal markdown for LLM text: **bold**, "- " / "* " / "•" bullet lines,
// blank-line paragraphs and single line breaks. Builds React elements only,
// so no HTML from the text is ever injected.

type Block = { kind: "p"; lines: string[] } | { kind: "ul"; items: string[] };

const BULLET = /^[-*•]\s+(.*)$/;

export function parseMarkdownBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  let current: Block | null = null;

  for (const raw of text.replace(/\r\n?/g, "\n").split("\n")) {
    const line = raw.trim();
    if (!line) {
      current = null;
      continue;
    }
    const bullet = BULLET.exec(line);
    if (bullet) {
      if (current?.kind !== "ul") {
        current = { kind: "ul", items: [] };
        blocks.push(current);
      }
      current.items.push(bullet[1]);
    } else {
      if (current?.kind !== "p") {
        current = { kind: "p", lines: [] };
        blocks.push(current);
      }
      current.lines.push(line);
    }
  }
  return blocks;
}

// Splitting on a capture group puts the bold text at odd indexes
function renderInline(line: string): ReactNode[] {
  return line.split(/\*\*(.+?)\*\*/g).map((part, i) =>
    i % 2 === 1 ? (
      <strong key={i} className="font-semibold">{part}</strong>
    ) : (
      <Fragment key={i}>{part}</Fragment>
    )
  );
}

export function MarkdownText({ text, className }: { text: string; className?: string }) {
  const blocks = parseMarkdownBlocks(text);
  return (
    <div className={className}>
      {blocks.map((block, i) =>
        block.kind === "ul" ? (
          <ul key={i} className="list-disc pl-5 space-y-1">
            {block.items.map((item, j) => (
              <li key={j}>{renderInline(item)}</li>
            ))}
          </ul>
        ) : (
          <p key={i}>
            {block.lines.map((line, j) => (
              <Fragment key={j}>
                {j > 0 && <br />}
                {renderInline(line)}
              </Fragment>
            ))}
          </p>
        )
      )}
    </div>
  );
}
