// Evidentia frontend. All model/web text is rendered with textContent (never
// innerHTML) and only http(s) links are created, so untrusted content cannot
// inject markup or script.
"use strict";

(() => {
  const { safeUrl, validateQuestion, isReportId, createSseParser, verdictCounts, reportToMarkdown, MAX_LEN } =
    window.EvidentiaLib;
  const $ = (id) => document.getElementById(id);
  const form = $("research-form");
  const input = $("question");
  const submitBtn = $("submit-btn");
  const errorBox = $("question-error");
  const liveStatus = $("live-status");
  const VERDICT_ICON = { supported: "✓", disputed: "⚠", refuted: "✗", unverified: "?" };
  const VERDICT_LABEL = { supported: "Supported", disputed: "Disputed", refuted: "Refuted", unverified: "Unverified" };
  const REASON_LABEL = {
    methodology: "Different methodology", timeframe: "Different timeframe", definition: "Different definitions",
    source_bias: "Possible source bias", outdated: "Outdated data", other: "Other",
  };
  let currentReport = null;

  function el(tag, attrs = {}, ...children) {
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

  function citeLinks(ids) {
    return ids.map((id) => el("a", { class: "cite", href: `#src-${id}`, "aria-label": `Source ${id}`, text: `[${id}]` }));
  }

  function announce(message) { liveStatus.textContent = message; }

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
    input.setAttribute("aria-invalid", "true");
    input.focus();
  }

  function clearError() {
    errorBox.hidden = true;
    errorBox.textContent = "";
    input.removeAttribute("aria-invalid");
  }

  function setBusy(busy) {
    submitBtn.disabled = busy;
    submitBtn.textContent = busy ? "Researching…" : "Research & verify";
    form.setAttribute("aria-busy", String(busy));
  }

  function resetAgents() {
    for (const li of document.querySelectorAll("#agent-list li")) {
      li.dataset.state = "waiting";
      li.querySelector(".agent-status").textContent = "Waiting";
      li.querySelector(".agent-detail").textContent = "";
    }
    $("plan").hidden = true;
    $("plan-list").replaceChildren();
  }

  function updateAgent({ agent, status, detail }) {
    const li = document.querySelector(`#agent-list li[data-agent="${CSS.escape(agent)}"]`);
    if (!li) return;
    const state = status === "progress" ? "running" : status;
    li.dataset.state = state;
    li.querySelector(".agent-status").textContent =
      state === "done" ? "Done" : state === "error" ? "Failed" : "Working";
    li.querySelector(".agent-detail").textContent = detail || "";
    announce(`${li.querySelector("strong").textContent}: ${detail}`);
  }

  // ---------------------------------------------------------------- report

  function renderReport(report) {
    currentReport = report;

    $("report-title").textContent = report.title || report.question;
    $("report-meta").textContent =
      `Question: “${report.question}” · ${report.depth === "quick" ? "Quick" : "Standard"} depth · ` +
      `${new Date(report.created_at).toLocaleString()} · ` +
      `evidence via ${report.search_backend === "google_search" ? "Google Search" : "Wikipedia"} · ${report.duration_s}s`;

    const counts = verdictCounts(report.claims);
    const highTrust = report.sources.filter((s) => s.tier === "high").length;
    $("stats").replaceChildren(
      ...[
        ["Claims checked", report.claims.length],
        ["Supported", counts.supported],
        ["Disputed / refuted", counts.disputed + counts.refuted],
        ["Conflicts", report.conflicts.length],
        ["Sources (high-trust)", `${report.sources.length} (${highTrust})`],
      ].map(([label, value]) => el("div", {}, el("dt", { text: label }), el("dd", { text: String(value) }))),
    );

    $("report-summary").textContent = report.summary;
    $("key-findings").replaceChildren(...report.key_findings.map((f) => el("li", { text: f })));

    $("report-sections").replaceChildren(
      ...report.sections.map((section) =>
        el("section", {},
          el("h3", { text: section.heading }),
          section.paragraphs.map((p) => el("p", {}, p.text, " ", citeLinks(p.citations))),
        ),
      ),
    );

    const claimById = Object.fromEntries(report.claims.map((c) => [c.id, c]));
    $("conflicts-block").hidden = report.conflicts.length === 0;
    $("conflicts").replaceChildren(
      ...report.conflicts.map((k) =>
        el("div", { class: "conflict" },
          el("h4", { text: k.topic }),
          el("p", {}, el("strong", { text: `${REASON_LABEL[k.reason] || "Other"}: ` }), k.explanation),
          el("ul", {}, k.claim_ids.map((id) => el("li", { text: `${id}: ${claimById[id]?.statement ?? ""}` }))),
        ),
      ),
    );

    $("claims-body").replaceChildren(
      ...report.claims.map((c) => {
        const pct = Math.round(c.confidence * 100);
        const evidence = [];
        if (c.supporting_source_ids.length) evidence.push("For: ", ...citeLinks(c.supporting_source_ids), " ");
        if (c.contradicting_source_ids.length) evidence.push("Against: ", ...citeLinks(c.contradicting_source_ids));
        if (!evidence.length) evidence.push("None found");
        return el("tr", { id: `claim-${c.id}` },
          el("td", { text: c.id }),
          el("td", {}, c.statement, c.rationale ? el("span", { class: "rationale", text: c.rationale }) : null),
          el("td", {}, el("span", { class: `badge badge-${c.verdict}` },
            el("span", { "aria-hidden": "true", text: VERDICT_ICON[c.verdict] }), VERDICT_LABEL[c.verdict])),
          el("td", {}, el("meter", { min: "0", max: "100", value: String(pct), "aria-label": `Confidence ${pct}%` }), `${pct}%`),
          el("td", {}, evidence),
        );
      }),
    );

    $("limitations").replaceChildren(...report.limitations.map((l) => el("li", { text: l })));

    $("sources").replaceChildren(
      ...report.sources.map((s) => {
        const href = safeUrl(s.url);
        const title = href
          ? el("a", { href, target: "_blank", rel: "noopener noreferrer nofollow", text: s.title })
          : el("span", { text: s.title });
        return el("li", { id: `src-${s.id}` },
          el("strong", { text: `[${s.id}] ` }), title,
          el("span", { class: `tier-${s.tier}`, text: ` · ${s.domain} · ${s.tier} credibility (${Math.round(s.credibility * 100)}%)` }),
          href ? el("span", { class: "visually-hidden", text: " (opens in a new tab)" }) : null,
        );
      }),
    );

    $("report-section").hidden = false;
    $("report-title").focus();
    announce("Report ready.");
  }

  // ---------------------------------------------------------------- network

  async function runResearch(question, depth) {
    clearError();
    setBusy(true);
    resetAgents();
    $("report-section").hidden = true;
    $("progress-section").hidden = false;
    announce("Research started.");

    try {
      const response = await fetch("/api/research", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({ question, depth }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${response.status}).`);
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      const parse = createSseParser();
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        parse(decoder.decode(value, { stream: true })).forEach(handleEvent);
      }
    } catch (err) {
      showError(err.message || "Network error. Please try again.");
      document.querySelectorAll('#agent-list li[data-state="running"]').forEach((li) => { li.dataset.state = "error"; });
    } finally {
      setBusy(false);
    }
  }

  function handleEvent(event) {
    switch (event.type) {
      case "agent": updateAgent(event); break;
      case "plan":
        $("plan-list").replaceChildren(...event.sub_questions.map((q) => el("li", { text: q.question })));
        $("plan").hidden = false;
        break;
      case "cached":
        document.querySelectorAll("#agent-list li").forEach((li) => {
          li.dataset.state = "done";
          li.querySelector(".agent-status").textContent = "Done (cached)";
        });
        announce(event.detail);
        break;
      case "report":
        renderReport(event.report);
        history.replaceState(null, "", `?report=${encodeURIComponent(event.report.id)}`);
        break;
      case "error": throw new Error(event.message);
      default: break;
    }
  }

  async function loadSharedReport(id) {
    if (!isReportId(id)) return;
    try {
      const response = await fetch(`/api/reports/${id}`);
      if (response.ok) renderReport(await response.json());
    } catch { /* a missing shared report is not fatal */ }
  }

  // ---------------------------------------------------------------- wiring

  input.addEventListener("input", () => {
    $("char-count").textContent = `${input.value.length} / ${MAX_LEN}`;
    if (!errorBox.hidden) clearError();
  });

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const result = validateQuestion(input.value);
    if (!result.ok) return showError(result.error);
    const depth = form.querySelector('input[name="depth"]:checked')?.value || "standard";
    runResearch(result.question, depth);
  });

  document.querySelectorAll(".chip").forEach((chip) =>
    chip.addEventListener("click", () => {
      input.value = chip.textContent;
      input.dispatchEvent(new Event("input"));
      input.focus();
    }),
  );

  $("copy-link").addEventListener("click", async () => {
    if (!currentReport) return;
    const url = `${location.origin}/?report=${currentReport.id}`;
    try { await navigator.clipboard.writeText(url); announce("Share link copied."); $("copy-link").textContent = "Link copied ✓"; }
    catch { announce(`Share link: ${url}`); }
  });

  $("download-md").addEventListener("click", () => {
    if (!currentReport) return;
    const blob = new Blob([reportToMarkdown(currentReport)], { type: "text/markdown" });
    const a = el("a", { href: URL.createObjectURL(blob), download: `evidentia-${currentReport.id}.md` });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  const shared = new URLSearchParams(location.search).get("report");
  if (shared) loadSharedReport(shared);
})();
