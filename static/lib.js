// Pure, DOM-free helpers shared by the UI (window.EvidentiaLib) and the Node
// unit tests (module.exports). Keeping them pure makes them easy to test.
"use strict";

(function (root, factory) {
  const lib = factory();
  if (typeof module === "object" && module.exports) module.exports = lib;
  else root.EvidentiaLib = lib;
})(typeof self !== "undefined" ? self : this, () => {
  const MIN_LEN = 10;
  const MAX_LEN = 500;
  const REPORT_ID = /^[0-9a-f]{20}$/;

  /** Return a normalised href for http(s) URLs only; anything else (javascript:, data:) is null. */
  function safeUrl(url) {
    try {
      const parsed = new URL(url);
      return parsed.protocol === "http:" || parsed.protocol === "https:" ? parsed.href : null;
    } catch {
      return null;
    }
  }

  /** Client-side mirror of the server's validation, for instant accessible feedback. */
  function validateQuestion(raw) {
    const question = String(raw ?? "").trim();
    if (question.length < MIN_LEN) return { ok: false, error: `Please enter at least ${MIN_LEN} characters.` };
    if (question.length > MAX_LEN) return { ok: false, error: `Please keep your question under ${MAX_LEN} characters.` };
    return { ok: true, question };
  }

  function isReportId(id) {
    return typeof id === "string" && REPORT_ID.test(id);
  }

  /**
   * Incremental Server-Sent Events parser. Feed it text chunks as they arrive;
   * it returns the complete events parsed so far and keeps any partial tail.
   */
  function createSseParser() {
    let buffer = "";
    return function feed(chunk) {
      buffer += chunk;
      const events = [];
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const data = block
          .split("\n")
          .filter((line) => line.startsWith("data: "))
          .map((line) => line.slice(6))
          .join("\n");
        if (data) events.push(JSON.parse(data));
      }
      return events;
    };
  }

  function verdictCounts(claims) {
    const counts = { supported: 0, disputed: 0, refuted: 0, unverified: 0 };
    for (const claim of claims) if (claim.verdict in counts) counts[claim.verdict] += 1;
    return counts;
  }

  const escapeMd = (text) => String(text).replace(/([\\`*_[\]|<>])/g, "\\$1");

  /** Render a report as portable Markdown (model text is escaped; only safe URLs become links). */
  function reportToMarkdown(r) {
    const lines = [`# ${escapeMd(r.title)}`, "", `> ${escapeMd(r.question)}`, "", "## Summary", "", escapeMd(r.summary), ""];
    lines.push("## Key findings", "");
    r.key_findings.forEach((f) => lines.push(`- ${escapeMd(f)}`));
    r.sections.forEach((s) => {
      lines.push("", `## ${escapeMd(s.heading)}`, "");
      s.paragraphs.forEach((p) => lines.push(`${escapeMd(p.text)} ${p.citations.map((c) => `[${c}]`).join("")}`.trim(), ""));
    });
    if (r.conflicts.length) {
      lines.push("## Conflicting evidence", "");
      r.conflicts.forEach((k) => lines.push(`- **${escapeMd(k.topic)}** (${k.reason}): ${escapeMd(k.explanation)} (${k.claim_ids.join(", ")})`));
      lines.push("");
    }
    lines.push("## Claims", "", "| ID | Claim | Verdict | Confidence |", "|---|---|---|---|");
    r.claims.forEach((c) => lines.push(`| ${c.id} | ${escapeMd(c.statement)} | ${c.verdict} | ${Math.round(c.confidence * 100)}% |`));
    lines.push("", "## Sources", "");
    r.sources.forEach((s) => {
      const href = safeUrl(s.url);
      const title = escapeMd(s.title);
      lines.push(`- [${s.id}] ${href ? `[${title}](${href})` : title} (${s.domain}, ${s.tier} credibility)`);
    });
    return lines.join("\n");
  }

  return { safeUrl, validateQuestion, isReportId, createSseParser, verdictCounts, reportToMarkdown, MIN_LEN, MAX_LEN };
});
