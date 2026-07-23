# Live AI Assistant - Implementation Checklist

- [x] Create project scaffold (backend + frontend)
- [x] Implement FastAPI server with /chat endpoint
- [x] Implement agent graph (tool calling + web search + verification)
- [x] Implement web search tool using Tavily (with timeout + graceful fallback)
- [x] Implement verification step using latest search results + citations
- [x] Add memory using SQLite, scoped per session (not global)
- [x] Add simple frontend chat UI with persistent per-browser session id
- [x] Add .env.example and README with run instructions
- [x] Smoke test: run server, send a query, confirm citations + memory
- [x] Harden for production: optional API-key auth, per-IP rate limiting,
      singleton DB engine/LLM client, retries + timeouts on external calls,
      graceful degradation on search/verification failure
- [x] Real unit test coverage (mocked LLM/search/memory) for retry, timeout,
      quota-error, and session-isolation behavior - not just API smoke tests
- [x] Typed exceptions instead of string-sniffing error messages (`LLMQuotaError`)
- [x] Request-id correlation + structured logging + per-request timeout
- [x] Non-root Docker user, HEALTHCHECK, `.dockerignore`
- [x] CI: lint (ruff), type-check (mypy), security lint (bandit), dependency
      CVE scan (pip-audit), coverage gate, Docker build verification
- [x] SECURITY.md documenting secret handling and known scaling limitations
