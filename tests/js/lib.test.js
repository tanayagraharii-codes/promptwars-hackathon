// Frontend unit tests for the pure helpers. Run with: npm test
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  createSseParser,
  isReportId,
  percent,
  reportMeta,
  reportStats,
  reportToMarkdown,
  safeUrl,
  validateQuestion,
  verdictCounts,
} from "../../static/js/lib.js";

const sampleReport = () => ({
  id: "0123456789abcdef0123",
  title: "EVs *really* cleaner?",
  question: "Do EVs emit less?",
  depth: "quick",
  created_at: "2026-11-20T10:00:00+00:00",
  search_backend: "google_search",
  duration_s: 42.5,
  summary: "Mostly yes | with caveats",
  key_findings: ["Lifecycle emissions are lower"],
  sections: [
    { heading: "Findings", paragraphs: [{ text: "Lower on average.", citations: ["S1", "S2"] }] },
  ],
  conflicts: [
    {
      topic: "Battery",
      reason: "methodology",
      explanation: "Different grids.",
      claim_ids: ["C1", "C2"],
    },
  ],
  claims: [
    { id: "C1", statement: "EVs | emit less", verdict: "supported", confidence: 0.81 },
    { id: "C2", statement: "Batteries are dirty", verdict: "disputed", confidence: 0.5 },
  ],
  sources: [
    {
      id: "S1",
      url: "https://iea.org/report",
      title: "IEA [report]",
      domain: "iea.org",
      tier: "high",
    },
    { id: "S2", url: "javascript:alert(1)", title: "Evil", domain: "evil.test", tier: "low" },
  ],
});

test("safeUrl allows only http(s)", () => {
  assert.equal(safeUrl("https://who.int/a"), "https://who.int/a");
  assert.equal(safeUrl("http://example.com"), "http://example.com/");
  assert.equal(safeUrl("javascript:alert(1)"), null);
  assert.equal(safeUrl("data:text/html,<b>x</b>"), null);
  assert.equal(safeUrl("not a url"), null);
  assert.equal(safeUrl(undefined), null);
});

test("validateQuestion mirrors the server limits", () => {
  assert.deepEqual(validateQuestion("   short  "), {
    ok: false,
    error: "Please enter at least 10 characters.",
  });
  assert.equal(validateQuestion("x".repeat(501)).ok, false);
  assert.deepEqual(validateQuestion("  Is coffee healthy?  "), {
    ok: true,
    question: "Is coffee healthy?",
  });
  assert.equal(validateQuestion(null).ok, false);
});

test("isReportId accepts only 20 lowercase hex characters", () => {
  assert.equal(isReportId("0123456789abcdef0123"), true);
  assert.equal(isReportId("../../etc/passwd"), false);
  assert.equal(isReportId("0123456789ABCDEF0123"), false);
  assert.equal(isReportId(42), false);
});

test("percent rounds a fraction", () => {
  assert.equal(percent(0.814), 81);
  assert.equal(percent(1), 100);
});

test("SSE parser handles events split across chunks", () => {
  const feed = createSseParser();
  assert.deepEqual(feed('data: {"type":"agent","agent":"pl'), []);
  assert.deepEqual(feed('anner"}\n\ndata: {"type":"plan"}\n'), [
    { type: "agent", agent: "planner" },
  ]);
  assert.deepEqual(feed("\n"), [{ type: "plan" }]);
  assert.deepEqual(feed(": keep-alive comment\n\n"), []);
});

test("SSE parser surfaces malformed JSON as an error", () => {
  assert.throws(() => createSseParser()("data: {not json}\n\n"), SyntaxError);
});

test("verdictCounts tallies known verdicts and ignores unknown ones", () => {
  const counts = verdictCounts([...sampleReport().claims, { verdict: "weird" }]);
  assert.deepEqual(counts, { supported: 1, disputed: 1, refuted: 0, unverified: 0 });
});

test("reportStats summarises claims, conflicts and source trust", () => {
  assert.deepEqual(reportStats(sampleReport()), [
    ["Claims checked", "2"],
    ["Supported", "1"],
    ["Disputed / refuted", "1"],
    ["Conflicts", "1"],
    ["Sources (high-trust)", "2 (1)"],
  ]);
});

test("reportMeta describes depth, backend and duration", () => {
  const meta = reportMeta(sampleReport(), () => "20 Nov 2026");
  assert.equal(
    meta,
    "Question: “Do EVs emit less?” · Quick depth · 20 Nov 2026 · evidence via Google Search · 42.5s",
  );
  const fallback = reportMeta({
    ...sampleReport(),
    depth: "standard",
    search_backend: "wikipedia",
  });
  assert.match(fallback, /Standard depth .* evidence via Wikipedia/);
});

test("reportToMarkdown escapes model text and drops unsafe links", () => {
  const md = reportToMarkdown(sampleReport());
  assert.match(md, /^# EVs \\\*really\\\* cleaner\?/);
  assert.match(md, /Mostly yes \\\| with caveats/);
  assert.match(md, /\| C1 \| EVs \\\| emit less \| supported \| 81% \|/);
  assert.match(md, /\[S1\] \[IEA \\\[report\\\]\]\(https:\/\/iea\.org\/report\)/);
  assert.doesNotMatch(md, /javascript:/);
  assert.match(md, /- \[S2\] Evil \(evil\.test, low credibility\)/);
  assert.match(md, /## Conflicting evidence/);
  assert.match(md, /Lower on average\. \[S1\]\[S2\]/);
});

test("reportToMarkdown omits the conflicts section when there are none", () => {
  assert.doesNotMatch(
    reportToMarkdown({ ...sampleReport(), conflicts: [] }),
    /Conflicting evidence/,
  );
});
