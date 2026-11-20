/**
 * Pure, DOM-free helpers for the Evidentia UI.
 *
 * Everything here is deterministic and side-effect free, so it is unit-tested
 * directly with Node's built-in test runner (tests/js/lib.test.js).
 */

export const MIN_LEN = 10;
export const MAX_LEN = 500;

const REPORT_ID = /^[0-9a-f]{20}$/;

export const VERDICT_LABEL = Object.freeze({
  supported: "Supported",
  disputed: "Disputed",
  refuted: "Refuted",
  unverified: "Unverified",
});

export const VERDICT_ICON = Object.freeze({
  supported: "✓",
  disputed: "⚠",
  refuted: "✗",
  unverified: "?",
});

export const REASON_LABEL = Object.freeze({
  methodology: "Different methodology",
  timeframe: "Different timeframe",
  definition: "Different definitions",
  source_bias: "Possible source bias",
  outdated: "Outdated data",
  other: "Other",
});

/** A normalised href for http(s) URLs only; anything else (javascript:, data:) is null. */
export function safeUrl(url) {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "http:" || parsed.protocol === "https:" ? parsed.href : null;
  } catch {
    return null;
  }
}

/** Client-side mirror of the server's validation, for instant accessible feedback. */
export function validateQuestion(raw) {
  const question = String(raw ?? "").trim();
  if (question.length < MIN_LEN) {
    return { ok: false, error: `Please enter at least ${MIN_LEN} characters.` };
  }
  if (question.length > MAX_LEN) {
    return { ok: false, error: `Please keep your question under ${MAX_LEN} characters.` };
  }
  return { ok: true, question };
}

export function isReportId(id) {
  return typeof id === "string" && REPORT_ID.test(id);
}

export function percent(fraction) {
  return Math.round(fraction * 100);
}

/**
 * Incremental Server-Sent Events parser. Feed it text chunks as they arrive;
 * it returns the complete events parsed so far and keeps any partial tail.
 */
export function createSseParser() {
  let buffer = "";
  return function feed(chunk) {
    buffer += chunk;
    const events = [];
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const data = buffer
        .slice(0, boundary)
        .split("\n")
        .filter((line) => line.startsWith("data: "))
        .map((line) => line.slice("data: ".length))
        .join("\n");
      if (data) events.push(JSON.parse(data));
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");
    }
    return events;
  };
}

export function verdictCounts(claims) {
  const counts = { supported: 0, disputed: 0, refuted: 0, unverified: 0 };
  for (const claim of claims) {
    if (claim.verdict in counts) counts[claim.verdict] += 1;
  }
  return counts;
}

/** Headline numbers shown above the report, as [label, value] pairs. */
export function reportStats(report) {
  const counts = verdictCounts(report.claims);
  const highTrust = report.sources.filter((source) => source.tier === "high").length;
  return [
    ["Claims checked", String(report.claims.length)],
    ["Supported", String(counts.supported)],
    ["Disputed / refuted", String(counts.disputed + counts.refuted)],
    ["Conflicts", String(report.conflicts.length)],
    ["Sources (high-trust)", `${report.sources.length} (${highTrust})`],
  ];
}

/** One-line description of how and when a report was produced. */
export function reportMeta(report, formatDate = (iso) => new Date(iso).toLocaleString()) {
  const depth = report.depth === "quick" ? "Quick" : "Standard";
  const backend = report.search_backend === "google_search" ? "Google Search" : "Wikipedia";
  return (
    `Question: “${report.question}” · ${depth} depth · ${formatDate(report.created_at)} · ` +
    `evidence via ${backend} · ${report.duration_s}s`
  );
}

const escapeMarkdown = (text) => String(text).replace(/([\\`*_[\]|<>])/g, "\\$1");

/** Render a report as portable Markdown (model text is escaped; only safe URLs become links). */
export function reportToMarkdown(report) {
  const md = escapeMarkdown;
  const lines = [`# ${md(report.title)}`, "", `> ${md(report.question)}`, ""];
  lines.push("## Summary", "", md(report.summary), "", "## Key findings", "");
  report.key_findings.forEach((finding) => lines.push(`- ${md(finding)}`));

  for (const section of report.sections) {
    lines.push("", `## ${md(section.heading)}`, "");
    for (const paragraph of section.paragraphs) {
      const cites = paragraph.citations.map((id) => `[${id}]`).join("");
      lines.push(`${md(paragraph.text)} ${cites}`.trim(), "");
    }
  }

  if (report.conflicts.length) {
    lines.push("## Conflicting evidence", "");
    for (const conflict of report.conflicts) {
      lines.push(
        `- **${md(conflict.topic)}** (${conflict.reason}): ${md(conflict.explanation)} ` +
          `(${conflict.claim_ids.join(", ")})`,
      );
    }
    lines.push("");
  }

  lines.push("## Claims", "", "| ID | Claim | Verdict | Confidence |", "|---|---|---|---|");
  for (const claim of report.claims) {
    lines.push(
      `| ${claim.id} | ${md(claim.statement)} | ${claim.verdict} | ${percent(claim.confidence)}% |`,
    );
  }

  lines.push("", "## Sources", "");
  for (const source of report.sources) {
    const href = safeUrl(source.url);
    const title = md(source.title);
    const link = href ? `[${title}](${href})` : title;
    lines.push(`- [${source.id}] ${link} (${source.domain}, ${source.tier} credibility)`);
  }
  return lines.join("\n");
}
