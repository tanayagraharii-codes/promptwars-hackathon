# Security

## Reporting a vulnerability

Please open a private security advisory on this repository (**Security → Report a vulnerability**)
instead of a public issue. We aim to respond within 72 hours.

## Threat model and controls

| Threat | Control | Where |
|---|---|---|
| Leaked credentials | No secrets in code or image. The API key comes from Secret Manager, or Vertex AI uses the service account with no key at all. `.env` is git-ignored. | `config.py`, `deploy.sh` |
| Prompt injection from web pages | Third-party text is fenced as untrusted data, and delimiter tags are stripped so content cannot escape the fence. | `security.wrap_untrusted` |
| Hallucinated or forged citations | Every source, claim and citation ID the model returns is checked against the registry; unknown IDs are dropped. | agents, `report_writer._validate_citations` |
| XSS through model or web content | Rendering uses only `textContent`, links are limited to `http(s)`, and the strict CSP blocks inline script. | `static/app.js`, `static/lib.js`, `main.SECURITY_HEADERS` |
| Clickjacking / MIME sniffing | `frame-ancestors 'none'`, `X-Frame-Options: DENY`, `nosniff` | `main.py` |
| Abuse / cost exhaustion | Per-client rate limit, global concurrent-job cap, 4 KB body limit, input length limits, response cache | `security.RateLimiter`, `main.py` |
| Path traversal on report IDs | Report IDs must match `^[0-9a-f]{20}$` before lookup | `main.get_report` |
| Information disclosure | Generic error messages; details only in server logs | `orchestrator.run` |
| Container escape impact | Non-root user, slim base image, no build tools at runtime | `Dockerfile` |
| Vulnerable dependencies | Pinned versions, `pip-audit` in CI (weekly too), Dependabot | `.github/` |

The Cloud Run service account has only the roles the app needs (`aiplatform.user`,
`datastore.user`, `secretmanager.secretAccessor`, `logging.logWriter`). See `deploy.sh`.
