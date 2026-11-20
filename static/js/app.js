/**
 * Evidentia UI controller: form handling, the live agent timeline and the
 * Server-Sent Events stream from POST /api/research.
 */

import { MAX_LEN, createSseParser, isReportId, reportToMarkdown, validateQuestion } from "./lib.js";
import { el, renderPlan, renderReport } from "./render.js";

const byId = (id) => document.getElementById(id);

const ui = {
  form: byId("research-form"),
  input: byId("question"),
  submit: byId("submit-btn"),
  error: byId("question-error"),
  charCount: byId("char-count"),
  liveStatus: byId("live-status"),
  progress: byId("progress-section"),
  report: byId("report-section"),
  agents: () => document.querySelectorAll("#agent-list li"),
};

const AGENT_STATUS_TEXT = { running: "Working", done: "Done", error: "Failed", waiting: "Waiting" };

let currentReport = null;

// ------------------------------------------------------------------ feedback

function announce(message) {
  ui.liveStatus.textContent = message;
}

function showError(message) {
  ui.error.textContent = message;
  ui.error.hidden = false;
  ui.input.setAttribute("aria-invalid", "true");
  ui.input.focus();
}

function clearError() {
  ui.error.hidden = true;
  ui.error.textContent = "";
  ui.input.removeAttribute("aria-invalid");
}

function setBusy(busy) {
  ui.submit.disabled = busy;
  ui.submit.textContent = busy ? "Researching…" : "Research & verify";
  ui.form.setAttribute("aria-busy", String(busy));
}

// ------------------------------------------------------------------ agent timeline

function setAgentState(item, state, detail = "") {
  item.dataset.state = state;
  item.querySelector(".agent-status").textContent = AGENT_STATUS_TEXT[state] ?? state;
  item.querySelector(".agent-detail").textContent = detail;
}

function resetAgents() {
  ui.agents().forEach((item) => setAgentState(item, "waiting"));
  byId("plan").hidden = true;
  byId("plan-list").replaceChildren();
}

function updateAgent({ agent, status, detail }) {
  const item = document.querySelector(`#agent-list li[data-agent="${CSS.escape(agent)}"]`);
  if (!item) return;
  setAgentState(item, status === "progress" ? "running" : status, detail);
  announce(`${item.querySelector("strong").textContent}: ${detail}`);
}

function markRunningAgentsFailed() {
  ui.agents().forEach((item) => {
    if (item.dataset.state === "running") setAgentState(item, "error");
  });
}

// ------------------------------------------------------------------ report

function showReport(report) {
  currentReport = report;
  renderReport(report);
  announce("Report ready.");
}

function handleEvent(event) {
  switch (event.type) {
    case "agent":
      updateAgent(event);
      break;
    case "plan":
      renderPlan(event.sub_questions);
      break;
    case "cached":
      ui.agents().forEach((item) => setAgentState(item, "done", "Served from cache."));
      announce(event.detail);
      break;
    case "report":
      showReport(event.report);
      history.replaceState(null, "", `?report=${encodeURIComponent(event.report.id)}`);
      break;
    case "error":
      throw new Error(event.message);
    default:
      break;
  }
}

// ------------------------------------------------------------------ network

async function streamResearch(question, depth) {
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
}

async function runResearch(question, depth) {
  clearError();
  setBusy(true);
  resetAgents();
  ui.report.hidden = true;
  ui.progress.hidden = false;
  announce("Research started.");
  try {
    await streamResearch(question, depth);
  } catch (error) {
    showError(error.message || "Network error. Please try again.");
    markRunningAgentsFailed();
  } finally {
    setBusy(false);
  }
}

async function loadSharedReport(id) {
  if (!isReportId(id)) return;
  try {
    const response = await fetch(`/api/reports/${id}`);
    if (response.ok) showReport(await response.json());
  } catch {
    // A missing or expired shared report is not an error worth interrupting the user for.
  }
}

// ------------------------------------------------------------------ actions

async function copyShareLink() {
  if (!currentReport) return;
  const url = `${location.origin}/?report=${currentReport.id}`;
  try {
    await navigator.clipboard.writeText(url);
    byId("copy-link").textContent = "Link copied ✓";
    announce("Share link copied.");
  } catch {
    announce(`Share link: ${url}`);
  }
}

function downloadMarkdown() {
  if (!currentReport) return;
  const blob = new Blob([reportToMarkdown(currentReport)], { type: "text/markdown" });
  const link = el("a", {
    href: URL.createObjectURL(blob),
    download: `evidentia-${currentReport.id}.md`,
  });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

// ------------------------------------------------------------------ wiring

ui.input.addEventListener("input", () => {
  ui.charCount.textContent = `${ui.input.value.length} / ${MAX_LEN}`;
  if (!ui.error.hidden) clearError();
});

ui.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const result = validateQuestion(ui.input.value);
  if (!result.ok) {
    showError(result.error);
    return;
  }
  const depth = ui.form.querySelector('input[name="depth"]:checked')?.value ?? "standard";
  runResearch(result.question, depth);
});

document.querySelectorAll(".chip").forEach((chip) =>
  chip.addEventListener("click", () => {
    ui.input.value = chip.textContent;
    ui.input.dispatchEvent(new Event("input"));
    ui.input.focus();
  }),
);

byId("copy-link").addEventListener("click", copyShareLink);
byId("download-md").addEventListener("click", downloadMarkdown);

const sharedReportId = new URLSearchParams(location.search).get("report");
if (sharedReportId) loadSharedReport(sharedReportId);
