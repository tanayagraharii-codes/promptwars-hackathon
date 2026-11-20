/**
 * DOM rendering for the Evidentia report.
 *
 * Security: model and web text is only ever assigned through textContent
 * (never innerHTML), and only http(s) URLs become links, so untrusted content
 * cannot inject markup or script.
 */

import {
  REASON_LABEL,
  VERDICT_ICON,
  VERDICT_LABEL,
  percent,
  reportMeta,
  reportStats,
  safeUrl,
} from "./lib.js";

/**
 * Create an element. `attrs.text` sets textContent and `attrs.class` the class
 * name; null/false attributes are skipped. Children may be nodes, strings or arrays.
 */
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child !== null && child !== undefined) node.append(child);
  }
  return node;
}

const byId = (id) => document.getElementById(id);

function citeLinks(ids) {
  return ids.map((id) =>
    el("a", { class: "cite", href: `#src-${id}`, "aria-label": `Source ${id}`, text: `[${id}]` }),
  );
}

export function renderPlan(subQuestions) {
  byId("plan-list").replaceChildren(...subQuestions.map((q) => el("li", { text: q.question })));
  byId("plan").hidden = false;
}

function renderStats(report) {
  byId("stats").replaceChildren(
    ...reportStats(report).map(([label, value]) =>
      el("div", {}, el("dt", { text: label }), el("dd", { text: value })),
    ),
  );
}

function renderSections(sections) {
  byId("report-sections").replaceChildren(
    ...sections.map((section) =>
      el(
        "section",
        {},
        el("h3", { text: section.heading }),
        section.paragraphs.map((p) => el("p", {}, p.text, " ", citeLinks(p.citations))),
      ),
    ),
  );
}

function renderConflicts(conflicts, claims) {
  const statementById = Object.fromEntries(claims.map((c) => [c.id, c.statement]));
  byId("conflicts-block").hidden = conflicts.length === 0;
  byId("conflicts").replaceChildren(
    ...conflicts.map((conflict) =>
      el(
        "div",
        { class: "conflict" },
        el("h4", { text: conflict.topic }),
        el(
          "p",
          {},
          el("strong", { text: `${REASON_LABEL[conflict.reason] ?? "Other"}: ` }),
          conflict.explanation,
        ),
        el(
          "ul",
          {},
          conflict.claim_ids.map((id) => el("li", { text: `${id}: ${statementById[id] ?? ""}` })),
        ),
      ),
    ),
  );
}

function evidenceCell(claim) {
  const parts = [];
  if (claim.supporting_source_ids.length) {
    parts.push("For: ", ...citeLinks(claim.supporting_source_ids), " ");
  }
  if (claim.contradicting_source_ids.length) {
    parts.push("Against: ", ...citeLinks(claim.contradicting_source_ids));
  }
  return el("td", {}, parts.length ? parts : "None found");
}

function claimRow(claim) {
  const confidence = percent(claim.confidence);
  return el(
    "tr",
    { id: `claim-${claim.id}` },
    el("td", { text: claim.id }),
    el(
      "td",
      {},
      claim.statement,
      claim.rationale ? el("span", { class: "rationale", text: claim.rationale }) : null,
    ),
    el(
      "td",
      {},
      el(
        "span",
        { class: `badge badge-${claim.verdict}` },
        el("span", { "aria-hidden": "true", text: VERDICT_ICON[claim.verdict] }),
        VERDICT_LABEL[claim.verdict],
      ),
    ),
    el(
      "td",
      {},
      el("meter", {
        min: "0",
        max: "100",
        value: String(confidence),
        "aria-label": `Confidence ${confidence}%`,
      }),
      `${confidence}%`,
    ),
    evidenceCell(claim),
  );
}

function sourceItem(source) {
  const href = safeUrl(source.url);
  const title = href
    ? el("a", { href, target: "_blank", rel: "noopener noreferrer nofollow", text: source.title })
    : el("span", { text: source.title });
  const credibility = `${source.tier} credibility (${percent(source.credibility)}%)`;
  return el(
    "li",
    { id: `src-${source.id}` },
    el("strong", { text: `[${source.id}] ` }),
    title,
    el("span", { class: `tier-${source.tier}`, text: ` · ${source.domain} · ${credibility}` }),
    href ? el("span", { class: "visually-hidden", text: " (opens in a new tab)" }) : null,
  );
}

/** Render a complete report into the page and move focus to its title. */
export function renderReport(report) {
  byId("report-title").textContent = report.title || report.question;
  byId("report-meta").textContent = reportMeta(report);
  renderStats(report);
  byId("report-summary").textContent = report.summary;
  byId("key-findings").replaceChildren(...report.key_findings.map((f) => el("li", { text: f })));
  renderSections(report.sections);
  renderConflicts(report.conflicts, report.claims);
  byId("claims-body").replaceChildren(...report.claims.map(claimRow));
  byId("limitations").replaceChildren(...report.limitations.map((l) => el("li", { text: l })));
  byId("sources").replaceChildren(...report.sources.map(sourceItem));
  byId("report-section").hidden = false;
  byId("report-title").focus();
}
