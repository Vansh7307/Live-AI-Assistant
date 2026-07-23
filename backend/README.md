---
title: Live AI Assistant API
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# Backend (FastAPI)

Runs the Live AI Assistant API. This folder is self-contained and deployable
on its own — the YAML block above is Hugging Face Spaces config; it's
harmless metadata everywhere else (Render, local, etc. all ignore it).

- `POST /chat` takes `{ "message": "...", "session_id": "..." }` and returns
  `{ "answer": "...", "sources": [...], "session_id": "..." }`.
- Uses a LangGraph agent: load per-session memory → web search (Tavily) →
  draft answer → verify against sources → persist memory.
- Persists memory in SQLite, scoped per `session_id` (see `memory/sqlite_memory.py`).
- Optional API-key auth (`APP_API_KEY`) and per-IP rate limiting
  (`CHAT_RATE_LIMIT`) — see the root README's Security section before
  deploying publicly.

## Required environment variables / secrets
Set these in whichever platform hosts this folder (Render dashboard, HF
Space Settings → Variables and secrets, etc.):
- `GOOGLE_API_KEY`, `TAVILY_API_KEY` — required
- `CORS_ORIGINS` — your frontend's real URL(s), comma-separated
- `APP_API_KEY` — optional, gates `/chat` behind an `X-API-Key` header
- `CHAT_RATE_LIMIT` — optional, default `20/minute`
