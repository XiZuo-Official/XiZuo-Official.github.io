function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function normalizeImageSrc(src) {
  if (src.startsWith("./pic/") || src.startsWith("pic/")) {
    return `/${src.replace(/^\.\//, "")}`;
  }
  return src;
}

function withPreservedRaw(text, formatter) {
  const tokens = [];
  const preserved = text.replace(/(<[^>]+>|&[a-zA-Z0-9#]+;)/g, (match) => {
    const token = `__RAW_${tokens.length}__`;
    tokens.push(match);
    return token;
  });

  let result = formatter(preserved);

  tokens.forEach((raw, index) => {
    result = result.replaceAll(`__RAW_${index}__`, raw);
  });

  return result;
}

function parseInline(text) {
  return withPreservedRaw(text, (safeText) => {
    let parsed = escapeHtml(safeText);

    parsed = parsed.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (_, alt, src) => {
      const normalized = normalizeImageSrc(src);
      return `<img src="${normalized}" alt="${escapeHtml(alt)}" />`;
    });

    parsed = parsed.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer noopener">$1</a>');
    parsed = parsed.replace(/`([^`]+)`/g, "<code>$1</code>");
    parsed = parsed.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    parsed = parsed.replace(/\*([^*]+)\*/g, "<em>$1</em>");

    return parsed;
  });
}

export function markdownToHtml(markdown) {
  const lines = markdown.replaceAll("\r\n", "\n").split("\n");
  const html = [];

  let inCode = false;
  let inList = false;
  let listType = "ul";
  let inMath = false;
  let mathTerminator = "";
  const mathBuffer = [];

  function closeListIfOpen() {
    if (inList) {
      inList = false;
      html.push(`</${listType}>`);
    }
  }

  function flushMath() {
    const body = mathBuffer.join("\n");
    if (mathTerminator === "\\]") {
      html.push(`<div class="math-block">\\[\n${body}\n\\]</div>`);
    } else {
      html.push(`<div class="math-block">$$\n${body}\n$$</div>`);
    }
    mathBuffer.length = 0;
  }

  for (const line of lines) {
    const trimmed = line.trim();

    if (trimmed.startsWith("```")) {
      closeListIfOpen();

      if (!inCode) {
        inCode = true;
        html.push("<pre><code>");
      } else {
        inCode = false;
        html.push("</code></pre>");
      }
      continue;
    }

    if (inCode) {
      html.push(`${escapeHtml(line)}\n`);
      continue;
    }

    if (!inMath && (trimmed === "\\[" || trimmed === "$$")) {
      closeListIfOpen();
      inMath = true;
      mathTerminator = trimmed === "\\[" ? "\\]" : "$$";
      continue;
    }

    if (inMath) {
      if (trimmed === mathTerminator) {
        inMath = false;
        flushMath();
        continue;
      }
      mathBuffer.push(line);
      continue;
    }

    const unordered = /^\s*[-*+]\s+/.test(line);
    const ordered = /^\s*\d+\.\s+/.test(line);

    if (unordered || ordered) {
      const nextType = ordered ? "ol" : "ul";
      if (!inList) {
        inList = true;
        listType = nextType;
        html.push(`<${listType}>`);
      } else if (nextType !== listType) {
        html.push(`</${listType}>`);
        listType = nextType;
        html.push(`<${listType}>`);
      }

      const cleaned = line.replace(/^\s*(?:[-*+]|\d+\.)\s+/, "");
      html.push(`<li>${parseInline(cleaned)}</li>`);
      continue;
    }

    closeListIfOpen();

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      const level = heading[1].length;
      html.push(`<h${level}>${parseInline(heading[2])}</h${level}>`);
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      html.push(`<blockquote>${parseInline(line.replace(/^\s*>\s?/, ""))}</blockquote>`);
      continue;
    }

    if (!trimmed) {
      html.push("");
      continue;
    }

    if (/^<[^>]+>$/.test(trimmed)) {
      html.push(trimmed);
      continue;
    }

    html.push(`<p>${parseInline(line)}</p>`);
  }

  closeListIfOpen();

  if (inMath) {
    flushMath();
  }

  return html.join("\n");
}

export function stripMarkdown(markdown) {
  return markdown
    .replaceAll(/```[\s\S]*?```/g, " ")
    .replaceAll(/\$\$[\s\S]*?\$\$/g, " ")
    .replaceAll(/\\\[[\s\S]*?\\\]/g, " ")
    .replaceAll(/<[^>]+>/g, " ")
    .replaceAll(/&[a-zA-Z0-9#]+;/g, " ")
    .replaceAll(/!\[[^\]]*\]\([^)]+\)/g, " ")
    .replaceAll(/\[[^\]]+\]\(([^)]+)\)/g, " ")
    .replaceAll(/[`*_>#-]/g, " ")
    .replaceAll(/\s+/g, " ")
    .trim();
}
