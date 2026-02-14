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
  let codeLang = "";
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

  function splitTableRow(line) {
    const trimmed = line.trim().replace(/^\|/, "").replace(/\|$/, "");
    return trimmed.split("|").map((cell) => cell.trim());
  }

  function isTableSeparator(line) {
    const cells = splitTableRow(line);
    if (!cells.length) return false;
    return cells.every((cell) => /^:?-{3,}:?$/.test(cell));
  }

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    const trimmed = line.trim();

    if (trimmed.startsWith("```")) {
      closeListIfOpen();

      if (!inCode) {
        inCode = true;
        codeLang = trimmed.slice(3).trim().toLowerCase();
        if (codeLang === "mysql") {
          codeLang = "sql";
        }
        const classAttr = codeLang ? ` class="language-${escapeHtml(codeLang)}"` : "";
        html.push(`<pre><code${classAttr}>`);
      } else {
        inCode = false;
        codeLang = "";
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

    const nextLine = i + 1 < lines.length ? lines[i + 1] : "";
    if (trimmed.includes("|") && nextLine.trim() && isTableSeparator(nextLine)) {
      closeListIfOpen();
      const tableLeadingSpaces = (line.match(/^\s*/)?.[0] || "").replaceAll("\t", "    ").length;
      const tableIndentLevel = Math.floor(tableLeadingSpaces / 2);
      const tableIndentRem = Math.min(tableIndentLevel * 0.65, 4);
      const headers = splitTableRow(line);
      const alignSpec = splitTableRow(nextLine);
      const aligns = alignSpec.map((cell) => {
        if (cell.startsWith(":") && cell.endsWith(":")) return "center";
        if (cell.endsWith(":")) return "right";
        return "left";
      });

      html.push(`<table style="margin-left:${tableIndentRem}rem;width:calc(100% - ${tableIndentRem}rem)">`);
      html.push("<thead><tr>");
      headers.forEach((header, idx) => {
        const align = aligns[idx] || "left";
        html.push(`<th style="text-align:${align}">${parseInline(header)}</th>`);
      });
      html.push("</tr></thead>");
      html.push("<tbody>");

      i += 2;
      for (; i < lines.length; i += 1) {
        const rowLine = lines[i];
        const rowTrimmed = rowLine.trim();
        if (!rowTrimmed || !rowTrimmed.includes("|")) {
          i -= 1;
          break;
        }

        const cells = splitTableRow(rowLine);
        html.push("<tr>");
        headers.forEach((_, idx) => {
          const align = aligns[idx] || "left";
          const value = cells[idx] || "";
          html.push(`<td style="text-align:${align}">${parseInline(value)}</td>`);
        });
        html.push("</tr>");
      }

      html.push("</tbody></table>");
      continue;
    }

    const listMatch = line.match(/^(\s*)([-*+]|\d+\.)\s+(.*)$/);
    const unordered = !!listMatch && /[-*+]/.test(listMatch[2]);
    const ordered = !!listMatch && /\d+\./.test(listMatch[2]);

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

      const leadingSpaces = (listMatch?.[1] || "").replaceAll("\t", "    ").length;
      const level = Math.max(0, Math.floor(leadingSpaces / 2));
      const cleaned = listMatch?.[3] || line.replace(/^\s*(?:[-*+]|\d+\.)\s+/, "");
      html.push(`<li class="list-level-${Math.min(level, 3)}">${parseInline(cleaned)}</li>`);
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
      const leadingSpaces = line.match(/^\s*/)?.[0].length || 0;
      const indentLevel = Math.floor(leadingSpaces / 2);
      const indentRem = Math.min(indentLevel * 0.65, 4);
      html.push(`<blockquote style="margin-left:${indentRem}rem">${parseInline(line.replace(/^\s*>\s?/, ""))}</blockquote>`);
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
    .replaceAll(/[`*_>#|\-]/g, " ")
    .replaceAll(/\s+/g, " ")
    .trim();
}
