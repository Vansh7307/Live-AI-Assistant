---
title: Live AI Assistant
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# Live AI Assistant

AI assistant that:
- Searches the live web for context (Tavily)
- Answers using that context with inline citations
- Verifies/corrects the draft answer against the latest sources
- Remembers each user's own conversation (SQLite, scoped per session)
- Degrades gracefully: a flaky search or verification call never fails the whole request

## Tech
- Backend: Python + FastAPI
- Agent orchestration: LangGraph
- LLM: Google Gemini
- Web search: Tavily
- Memory: SQLite via SQLAlchemy (per-session, singleton connection pool)
- Rate limiting: slowapi

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
1. Create API keys:
   - `GOOGLE_API_KEY` (Gemini) — https://aistudio.google.com/
   - `TAVILY_API_KEY` — https://tavily.com/

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

`POST /chat`
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

`GET /health` → `{ "status": "ok" }`
