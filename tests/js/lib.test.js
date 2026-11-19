// Frontend unit tests. Run with: node --test tests/js/
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

const lib = require(path.join(__dirname, "..", "..", "static", "lib.js"));

const sampleReport = () => ({
  id: "0123456789abcdef0123",
  title: "EVs *really* cleaner?",
  question: "Do EVs emit less?",
  summary: "Mostly yes | with caveats",
  key_findings: ["Lifecycle emissions are lower"],
  sections: [{ heading: "Findings", paragraphs: [{ text: "Lower on average.", citations: ["S1", "S2"] }] }],
  conflicts: [{ topic: "Battery", reason: "methodology", explanation: "Different grids.", claim_ids: ["C1", "C2"] }],
  claims: [
    { id: "C1", statement: "EVs | emit less", verdict: "supported", confidence: 0.81 },
    { id: "C2", statement: "Batteries are dirty", verdict: "disputed", confidence: 0.5 },
  ],
  sources: [
    { id: "S1", url: "https://iea.org/report", title: "IEA [report]", domain: "iea.org", tier: "high" },
    { id: "S2", url: "javascript:alert(1)", title: "Evil", domain: "evil.test", tier: "low" },
  ],
});

test("safeUrl allows only http(s)", () => {
  assert.equal(lib.safeUrl("https://who.int/a"), "https://who.int/a");
  assert.equal(lib.safeUrl("http://example.com"), "http://example.com/");
  assert.equal(lib.safeUrl("javascript:alert(1)"), null);
  assert.equal(lib.safeUrl("data:text/html,<b>x</b>"), null);
  assert.equal(lib.safeUrl("not a url"), null);
  assert.equal(lib.safeUrl(undefined), null);
});

test("validateQuestion mirrors server limits", () => {
  assert.deepEqual(lib.validateQuestion("   short  "), { ok: false, error: "Please enter at least 10 characters." });
  assert.equal(lib.validateQuestion("x".repeat(501)).ok, false);
  assert.deepEqual(lib.validateQuestion("  Is coffee healthy?  "), { ok: true, question: "Is coffee healthy?" });
  assert.equal(lib.validateQuestion(null).ok, false);
});

test("isReportId accepts only 20 lowercase hex chars", () => {
  assert.equal(lib.isReportId("0123456789abcdef0123"), true);
  assert.equal(lib.isReportId("../../etc/passwd"), false);
  assert.equal(lib.isReportId("0123456789ABCDEF0123"), false);
  assert.equal(lib.isReportId(42), false);
});

test("SSE parser handles events split across chunks", () => {
  const feed = lib.createSseParser();
  assert.deepEqual(feed('data: {"type":"agent","agent":"pl'), []);
  assert.deepEqual(feed('anner"}\n\ndata: {"type":"plan"}\n'), [{ type: "agent", agent: "planner" }]);
  assert.deepEqual(feed("\n"), [{ type: "plan" }]);
  assert.deepEqual(feed(": keep-alive comment\n\n"), []);
});

test("SSE parser surfaces malformed JSON as an error", () => {
  const feed = lib.createSseParser();
  assert.throws(() => feed("data: {not json}\n\n"), SyntaxError);
});

test("verdictCounts tallies known verdicts and ignores unknown ones", () => {
  const counts = lib.verdictCounts([...sampleReport().claims, { verdict: "weird" }]);
  assert.deepEqual(counts, { supported: 1, disputed: 1, refuted: 0, unverified: 0 });
});

test("reportToMarkdown escapes model text and drops unsafe links", () => {
  const md = lib.reportToMarkdown(sampleReport());
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
  const report = { ...sampleReport(), conflicts: [] };
  assert.doesNotMatch(lib.reportToMarkdown(report), /Conflicting evidence/);
});
