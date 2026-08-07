# Security Policy

## Reporting a vulnerability

If you discover a security issue in this project (e.g. an auth bypass, an
injection vector, a way to exfiltrate another user's session data), please
**do not open a public GitHub issue**. Instead, email the maintainer directly
(see repository contact info) with:

- A description of the issue and its potential impact
- Steps to reproduce
- Any relevant logs or proof-of-concept code

We aim to acknowledge reports within 3 business days.

## Secrets handling

This project loads secrets (`GOOGLE_API_KEY`, `TAVILY_API_KEY`, `APP_API_KEY`)
from environment variables / a local `.env` file that is excluded from git via
`.gitignore`. If you ever paste a real key into a chat tool, a shared
document, a public repo, or any other non-secrets-manager destination,
**treat it as compromised and rotate it immediately** — don't wait to confirm
whether it was actually misused.

Recommended (not yet wired into CI due to sandbox network restrictions when
this project was scaffolded): add [gitleaks](https://github.com/gitleaks/gitleaks)
or [detect-secrets](https://github.com/Yelp/detect-secrets) as a pre-commit
hook and a CI job, so an accidental key commit is caught before it reaches a
remote.

## Known limitations (by design, tracked here rather than hidden)

- `/chat` and `/chat/stream` have no authentication unless `APP_API_KEY` is
  set. Set it before exposing this service publicly.
- Memory defaults to a single SQLite file. That's fine for a single-instance
  deployment; for multiple replicas, set `DATABASE_URL` to PostgreSQL (the
  app auto-upgrades `postgres://` to the psycopg driver). Concurrent writers
  to a single SQLite file will contend/lock, so SQLite is not safe for
  scaled-out multi-instance production.
- Conversation history has no automatic retention/expiry policy by default.
  A `Memory.cleanup_old_sessions(max_age_days=...)` method is provided for a
  scheduled job; wire it to your own scheduler and document it in a privacy
  policy before launch.
- The built-in safety layer is a heuristic, defense-in-depth guardrail
  (prompt-injection score + optional output blocklist). It is **not** a
  substitute for a dedicated moderation service at scale. For stricter
  enforcement, layer in a commercial moderation API and/or the provider's own
  safety filters.
- The `/sessions` endpoint (used by the frontend history sidebar) is
  unauthenticated unless `APP_API_KEY` is set. If you deploy it publicly,
  set `APP_API_KEY` so the session list is also protected.
- Secret scanning is wired into CI via gitleaks. For local development we
  recommend also installing [pre-commit](https://pre-commit.com/) with a
  gitleaks hook so an accidental key commit is caught before it reaches a
  remote.
