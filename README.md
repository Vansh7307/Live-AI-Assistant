---
title: Live AI Assistant
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# Live AI Assistant

A production-grade, **live** streaming AI assistant.

- **Live web search** for current context (Tavily) with inline citations
- **Streams answers token-by-token** over Server-Sent Events (SSE) for a real-time chat feel
- **Verifies/corrects** the draft answer against the latest sources
- **Multi-provider LLM** with automatic failover (Gemini → OpenAI → Anthropic)
- **Long-term memory** per session with automatic summarization
  (SQLite for single-instance, PostgreSQL for multi-instance production)
- **Observable**: Prometheus `/metrics`, `/health/live`, `/health/ready`, request tracing
- **Safety layer**: configurable prompt-injection + output-content guardrails
- **Degrades gracefully**: a flaky search or verification call never fails the whole request

## Features at a glance

| Capability | Details |
| --- | --- |
| Streaming | `POST /chat/stream` (SSE) plus classic `POST /chat` |
| LLM failover | Gemini, OpenAI, Anthropic — tried in `LLM_PROVIDERS` order |
| Memory | SQLite (default) or PostgreSQL (`DATABASE_URL`), per-session, + summaries |
| Observability | Prometheus `/metrics`, structured logs, W3C `traceparent` |
| Safety | Prompt-injection detection + output blocklist |
| Hardening | API-key auth, per-IP rate limiting, non-root Docker, healthchecks |
| DevOps | CI (lint/type/sec/deps/tests), secret scanning, k8s manifests |

## Tech
- Backend: Python + FastAPI (async)
- Agent orchestration: LangGraph
- LLMs: Google Gemini, OpenAI, Anthropic (provider-agnostic with failover)
- Web search: Tavily
- Memory: SQLAlchemy (SQLite or PostgreSQL), singleton connection pool
- Streaming: Server-Sent Events
- Observability: prometheus_client
- Rate limiting / auth: slowapi + optional `X-API-Key`

## ⚠️ Security first

**Never commit a real `.env` file or paste real API keys into chat, issues, or
screenshots.** If a key has ever been exposed (including just pasted somewhere
outside your local machine), rotate it immediately in the provider's console —
treat "someone else saw it" as "assume it's compromised," regardless of
whether it made it into git history.

This repo's `.gitignore` already excludes `.env`, `*.log`, and `memory.sqlite3`.
Keep it that way, and double check `git status` before every commit.

Before deploying publicly:
- Set `APP_API_KEY` so `/chat` requires an `X-API-Key` header. Without it,
  anyone with your URL can spend your Gemini/Tavily quota.
- Set `CHAT_RATE_LIMIT` (default `20/minute` per IP) to a value that fits
  your budget.
- Set `CORS_ORIGINS` to your real frontend origin(s) only.

## Setup
1. Create API keys (you need **at least one** LLM provider + Tavily):
   - `GOOGLE_API_KEY` (Gemini) — https://aistudio.google.com/
   - `TAVILY_API_KEY` — https://tavily.com/
   - Optional fallback providers for automatic failover:
     - `OPENAI_API_KEY` — https://platform.openai.com/
     - `ANTHROPIC_API_KEY` — https://console.anthropic.com/

2. Install dependencies:
   ```bash
   cd backend
   python -m venv .venv
   # Windows: .venv\Scripts\activate | macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Configure environment:
   - Copy `backend/.env.example` to `backend/.env` and fill in your keys.

## Run
```bash
cd backend
uvicorn main:app --reload --port 8000
```
Open `frontend/index.html` in a browser (or serve it with any static file
server). If your backend requires `APP_API_KEY`, open the page as
`frontend/index.html?apiKey=YOUR_KEY`.

For a streaming/localhost demo simply open the page; it auto-detects
`http://localhost:8000` and uses the SSE `/chat/stream` endpoint by default,
falling back to `POST /chat` if streaming is unavailable.

## Tests
```bash
cd backend
python -m unittest discover -s tests -v
```
Test coverage includes: API contract & validation, per-session memory
isolation, LLM retry/timeout/quota-error handling, web-search graceful
degradation, and the agent graph's end-to-end orchestration (all external
calls mocked - no real API keys needed to run the suite).

## Development tooling

```bash
cd backend
pip install -r requirements-dev.txt
ruff check .          # lint
mypy .                 # static type checking
bandit -r . -x ./tests # security lint
pip-audit -r requirements.txt  # known-CVE dependency scan
coverage run -m unittest discover -s tests && coverage report
```
All of the above run in CI on every push/PR (`.github/workflows/ci.yml`),
along with a Docker build to catch container-level breakage early.

## Deployment (GitHub + Render + Vercel)

See the step-by-step walkthrough below. Short version: push to GitHub →
deploy `backend/` to Render → deploy `frontend/` to Vercel → point
`frontend/config.js` at your Render URL → update `CORS_ORIGINS` on Render to
your Vercel URL.

## API

### `POST /chat` (non-streaming)
```json
{ "message": "What is ...?", "session_id": "optional-existing-session-id" }
```
Response:
```json
{
  "answer": "...",
  "sources": [{ "title": "...", "url": "..." }],
  "session_id": "the session id to reuse on the next request"
}
```
Omit `session_id` on the first call; the server generates one and returns it.
Send it back on every subsequent call so conversation memory stays scoped to
that user. Requires header `X-API-Key: <APP_API_KEY>` if that variable is set
on the server.

### `POST /chat/stream` (SSE)
Same request body. Returns a `text/event-stream` with named events:

| event | data |
| --- | --- |
| `session` | `{ "session_id": "..." }` — reuse on the next request |
| `sources` | `{ "sources": [...] }` — search results for citation badges |
| `token` | `{ "token": "..." }` — one chunk of the answer (streamed live) |
| `done` | `{}` — stream complete |
| `error` | `{ "detail": "..." }` — a recoverable error |

### `GET /sessions`
Returns `{ "sessions": [...] }` — recent sessions + summaries for the
frontend history sidebar.

### Health & operations
- `GET /health` → `{ "status": "ok" }`
- `GET /health/live` → `{ "status": "alive" }` (liveness probe)
- `GET /health/ready` → `{ "status": "ready" }` or HTTP 503 if no LLM provider
  is configured (readiness probe)
- `GET /metrics` → Prometheus metrics (scrape target)
