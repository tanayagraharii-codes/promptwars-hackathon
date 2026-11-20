# Evidentia: Multi-Agent Research & Fact-Checking

> **PromptWars × Agent X 2.0 · PS03: Multi-Agent Research & Fact-Checking**
> A multi-agent system that **researches complex topics**, **verifies information using reliable
> sources**, **identifies conflicting claims**, and **generates evidence-backed reports**.

Built on **Gemini** (Google Gen AI SDK) with **Google Search grounding**, deployed on **Cloud Run**,
with optional **Cloud Firestore**, **Secret Manager**, **Cloud Build** and **Cloud Logging**.

---

## How it maps to the problem statement

| PS03 requirement | How Evidentia does it |
|---|---|
| Multi-agent system | Five specialised agents: **Planner → Researchers (parallel) → Fact-Checkers (parallel) → Conflict Detector → Report Writer**, coordinated by `app/orchestrator.py`. |
| Researches complex topics | The **Planner** splits the topic into focused sub-questions, including one that looks for counter-evidence. Each **Researcher** gathers evidence with Google Search grounding and extracts specific, checkable claims tied to their sources. |
| Verifies using reliable sources | The **Fact-Checker** runs an *independent* verification search per claim and gives a verdict (supported / disputed / refuted / unverified). Every source gets a **credibility score** (`app/sources.py`: .gov/.edu/WHO/peer-reviewed rank high, social media and blogs rank low), and the final confidence blends the model's judgement with the credibility and number of corroborating sources. |
| Identifies conflicting claims | The **Conflict Detector** groups contradictory claims and explains *why* sources disagree (methodology, timeframe, definitions, bias, outdated data). |
| Evidence-backed reports | The **Report Writer** produces a structured report in which every paragraph carries citations. Citations are **validated in code**, so the model cannot cite sources that do not exist. Reports can be shared by link and downloaded as Markdown. |

```
            ┌──────────┐
 question → │ Planner  │ → 2–4 sub-questions (+ keyword queries)
            └────┬─────┘
       ┌─────────┼─────────┐              Google Search grounding
       ▼         ▼         ▼              (fallback: Wikipedia API)
  ┌─────────┐┌─────────┐┌─────────┐
  │Research ││Research ││Research │  → claims + cited sources [S1..Sn]
  └────┬────┘└────┬────┘└────┬────┘
       ▼          ▼          ▼
  ┌─────────┐┌─────────┐┌─────────┐
  │FactCheck││FactCheck││FactCheck│  → verdict + confidence per claim
  └────┬────┘└────┬────┘└────┬────┘
       └──────────┼──────────┘
                  ▼
        ┌──────────────────┐
        │ Conflict Detector│ → contradictions + reason
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │  Report Writer   │ → cited report (validated)
        └──────────────────┘
```

Progress is streamed to the browser live (Server-Sent Events), so users see each agent working.

## Google services used

| Service | Use |
|---|---|
| **Gemini API** (`google-genai`) | All five agents; structured JSON output validated against Pydantic schemas; tiered models (Flash-Lite for extraction, Flash for judgement) with automatic fallback. |
| **Grounding with Google Search** | Primary evidence source; grounding metadata is turned into numbered, clickable citations. |
| **Vertex AI** (default in production) | `build_genai_client()` authenticates with the Cloud Run service account through IAM, so no API key exists to leak. |
| **Cloud Run** | Serverless hosting (`Dockerfile`, non-root container, dedicated service account). |
| **Cloud Firestore** | Durable, shareable reports behind an in-memory cache. Turned on automatically on Cloud Run (detected from `K_SERVICE`). |
| **IAM** | A least-privilege runtime service account (`aiplatform.user`, `datastore.user`, `logging.logWriter`), created by `deploy.sh`. |
| **Secret Manager** | API-key mode only: holds `GEMINI_API_KEY`, with access granted on that one secret only. |
| **Cloud Build / Artifact Registry** | `cloudbuild.yaml`: lint, test, build, push, deploy. |
| **Cloud Logging** | Structured JSON logs with `severity`, parsed natively on Cloud Run. |

## Engineering qualities

**Security**
- Strict Content-Security-Policy (no inline script), HSTS, `X-Frame-Options: DENY`, `nosniff`, Referrer-Policy and Permissions-Policy.
- Input validation and normalisation (length limits, control-character stripping), plus a 4 KB request-body cap.
- Per-client rate limiting and a global cap on concurrent jobs.
- **Prompt-injection containment**: all web content is fenced as untrusted data, and fence-breaking tags are stripped.
- Model output is never trusted blindly: source IDs, claim IDs and citations are all validated in code.
- Frontend renders only via `textContent`, allows only `http(s)` links (with `rel="noopener noreferrer"`), and uses no `innerHTML`.
- Error messages never leak internals. The container runs as non-root, and secrets come from env or Secret Manager.
- **Supply chain**: pinned dependencies, plus `pip-audit` and security linting in CI on every push and weekly. See [SECURITY.md](SECURITY.md) for the threat model.

**Efficiency**
- **Research depth**: *Quick* mode (2 sub-questions × 2 claims) makes roughly 40% of the model calls of *Standard* mode and finishes about twice as fast. Each depth is cached separately.
- Researchers and fact-checkers run **concurrently** (`asyncio`), with a semaphore to respect quotas.
- Claims are fact-checked **in batches per sub-question** (one search and one judgement call per batch, not one per claim).
- Repeated questions are served from a **TTL/LRU cache** (and Firestore) at zero model cost.
- The Wikipedia fallback paces requests, honours `Retry-After`, caches results, and sends only the article paragraphs relevant to each sub-question to the model, which keeps prompts small.
- **Model fallback chain** with a per-model cooldown, so an overloaded model is skipped instead of retried. A **circuit breaker** switches to Wikipedia when Search grounding is out of quota.
- Responses are streamed and GZip-compressed. The frontend uses no framework or build step and loads no third-party scripts.

**Accessibility (WCAG 2.2 AA-minded)**
- Semantic landmarks, a skip link, one `h1`, labelled form controls, and `aria-describedby` hints and errors.
- Live agent progress through `role="status"` / `aria-live` regions. Focus moves to the report when it is ready.
- Verdicts use **text + icon + colour** (never colour alone). Confidence uses `<meter>` with a label.
- Fully keyboard operable, visible focus ring, 44 px touch targets.
- Light and dark themes with AA contrast; `prefers-reduced-motion` respected.

**Testing**: about 155 automated tests, all offline (Gemini and the web are faked), run in CI on every push:
- **Python (146 tests, ~97% coverage)**: agents, orchestrator, API (streaming, validation, rate limits, security headers), retrieval (including passage selection, caching and backoff), citation mapping, credibility scoring, caching, Firestore, and Vertex AI client selection.
- **Accessibility (included above)**: landmarks, heading order, labels, ARIA references, live regions and table semantics, plus **WCAG AA contrast ratios computed for every colour pair in both light and dark themes**.
- **Frontend (8 tests, Node's built-in runner)**: SSE stream parsing across chunk boundaries, URL sanitising, input validation and Markdown export escaping.

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
export GEMINI_API_KEY=...            # from https://aistudio.google.com/apikey
uvicorn app.main:app --reload --port 8080
# open http://localhost:8080
```

Run the checks:

```bash
pytest --cov                      # Python + accessibility tests, with coverage
node --test "tests/js/*.test.js"  # frontend unit tests
ruff check app tests              # lint (includes security rules)
pip-audit -r requirements.txt     # known-vulnerability scan
```

## Deploy to Cloud Run

One idempotent script sets up the APIs, a least-privilege service account and Firestore, then deploys:

```bash
gcloud auth login
PROJECT_ID=your-project-id ./deploy.sh                       # Vertex AI (recommended: no API key)
PROJECT_ID=your-project-id AUTH_MODE=apikey ./deploy.sh      # or: Gemini API key via Secret Manager
```

With Vertex AI, Google Search grounding is billed to the project, so there are no free-tier grounding limits.
For CI/CD after the first deploy, run `gcloud builds submit --config cloudbuild.yaml`.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | Gemini Developer API key (not needed with Vertex AI). |
| `GOOGLE_GENAI_USE_VERTEXAI` / `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION` | `false` / — / `us-central1` | Use Vertex AI with the service account. |
| `SEARCH_BACKEND` | `auto` | `auto` / `google` / `wikipedia` |
| `GEMINI_FAST_MODELS` | `gemini-3.5-flash-lite,gemini-3.1-flash-lite` | Planner and researcher models (fallback order). |
| `GEMINI_SMART_MODELS` | `gemini-3.5-flash,…` | Fact-checker, conflict detector and writer models. |
| `GEMINI_SEARCH_MODELS` | `gemini-3.5-flash,gemini-3.5-flash-lite` | Models used with Search grounding. |
| `MAX_SUB_QUESTIONS` / `MAX_CLAIMS_PER_QUESTION` | `4` / `3` | Depth vs. cost. |
| `RATE_LIMIT_PER_MINUTE` | `5` | Per-client requests per minute. |
| `WIKIPEDIA_CONTACT` | — | Contact URL for the Wikipedia fallback's User-Agent (Wikimedia API policy), e.g. your repo URL. |
| `FIRESTORE_ENABLED` | `true` on Cloud Run, else `false` | Persist reports in Firestore. |

## Project layout

```
app/
  agents/            planner, researcher, fact_checker, conflict_detector, report_writer
  orchestrator.py    pipeline + streaming events
  llm.py             Gemini wrapper: JSON schemas, grounding, retries, fallback
  search.py          Google Search grounding, Wikipedia fallback, circuit breaker
  sources.py         credibility scoring, source registry
  security.py        sanitisation, prompt-injection fencing, rate limiting
  storage.py         cache + Firestore
  main.py            FastAPI app, security headers, SSE endpoint
static/              accessible single-page UI (HTML/CSS/JS, no build step)
  lib.js             pure, unit-tested helpers (SSE parser, sanitising, Markdown export)
tests/               offline pytest suite (incl. accessibility + contrast)
  js/                frontend unit tests (node --test)
deploy.sh            one-command Cloud Run deploy (APIs, IAM, Firestore)
```

## Limitations

- Credibility scores are domain-based priors. They do not replace reading the source.
- In Wikipedia fallback mode the evidence is less diverse; enable Search grounding (billing-enabled key or Vertex AI) for best results.
- AI-generated research can contain mistakes. Every claim links to its sources so readers can check.
