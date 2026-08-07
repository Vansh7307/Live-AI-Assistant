"""Safety / moderation layer.

Provides configurable input (prompt-injection) and output (content) guardrails
for the chat API. Enabled by default with conservative heuristics; can be
tuned via environment variables:

  - ``SAFETY_ENABLED`` (default ``true``): master switch.
  - ``INPUT_INJECTION_THRESHOLD`` (default ``0.6``): 0..1 score above which an
    incoming message is flagged as a likely prompt injection.
  - ``OUTPUT_BLOCKLIST``: comma-separated terms; if an assistant answer contains
    one, the answer is replaced with a refusal.

The goal is a reasonable defense-in-depth layer on top of the API-key auth and
rate limiting, not a replacement for a dedicated moderation service (e.g. the
provider's own safety filters or a commercial moderation API).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Callable

logger = logging.getLogger(__name__)

_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompt|directions)", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)", re.IGNORECASE),
    re.compile(r"you\s+are\s+(now\s+)?(a\s+)?(the\s+)?(system|assistant)\s*prompt", re.IGNORECASE),
    re.compile(r"reveal\s+(your|the)\s+(system|hidden|internal)\s*(prompt|instructions)", re.IGNORECASE),
    re.compile(r"new\s*system\s*(prompt|instruction)", re.IGNORECASE),
    re.compile(r"roleplay\s*as", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"\[system\]", re.IGNORECASE),
]

# Dangerous-content keywords. These are intentionally narrow to avoid false
# positives on legitimate educational/medical discussions.
_OUTPUT_BLOCKLIST = [
    # Do not include real harmful terms here; populate via env if you want
    # stricter filtering. This is a demonstration of the mechanism.
]


class SafetyError(Exception):
    """Raised when a message is blocked by the safety layer."""

    def __init__(self, reason: str, detail: str = "Your message was blocked by the safety filter."):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def _load_blocklist() -> list[str]:
    raw = os.getenv("OUTPUT_BLOCKLIST", "")
    return [term.strip().lower() for term in raw.split(",") if term.strip()]


def _injection_score(text: str) -> float:
    """Return a 0..1 heuristic score of how likely ``text`` is a prompt
    injection. Simple and deterministic: fraction of known patterns matched
    plus bonus for meta-language about the system prompt."""
    score = 0.0
    matched = 0
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            matched += 1
    if matched:
        score += min(0.5, matched * 0.15)
    # Meta-language about instructions/system.
    if re.search(r"\b(system prompt|instructions|developer|jailbreak|prompt cache)\b", text, re.IGNORECASE):
        score += 0.2
    return min(1.0, score)


def check_input(message: str) -> None:
    """Validate an incoming user message. Raises ``SafetyError`` if blocked."""
    if os.getenv("SAFETY_ENABLED", "true").lower() not in ("1", "true", "yes"):
        return
    threshold = float(os.getenv("INPUT_INJECTION_THRESHOLD", "0.6"))
    score = _injection_score(message)
    if score >= threshold:
        logger.warning("Input blocked by safety filter (score=%.2f)", score)
        raise SafetyError("injection")


def check_output(answer: str) -> str:
    """Post-process an assistant answer. Returns the answer unchanged unless
    a blocked term is present, in which case returns a safe refusal."""
    if os.getenv("SAFETY_ENABLED", "true").lower() not in ("1", "true", "yes"):
        return answer
    blocklist = _load_blocklist()
    if any(term in answer.lower() for term in blocklist):
        logger.warning("Output blocked: matched blocklist term")
        return (
            "I can't provide that content. If this relates to a legitimate "
            "question, please rephrase it and I'll be happy to help."
        )
    return answer
