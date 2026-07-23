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

- `/chat` has no authentication unless `APP_API_KEY` is set. Set it before
  exposing this service publicly.
- Memory is a single SQLite file. That's fine for a single-instance
  deployment; it is **not** safe to run multiple API replicas against the
  same file without moving to a real database (e.g. Postgres) - concurrent
  writers to SQLite will contend/lock.
- Conversation history has no automatic retention/expiry policy. If you
  operate this for real users, add a data-retention job and document it in
  a privacy policy before launch.
