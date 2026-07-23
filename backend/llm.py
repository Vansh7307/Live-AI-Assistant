from __future__ import annotations

import asyncio
import logging
import os

from google import genai

logger = logging.getLogger(__name__)

_CLIENT: genai.Client | None = None

_QUOTA_MARKERS = ("429", "quota", "rate limit", "resource_exhausted")


class LLMError(Exception):
    """Base class for LLM-layer failures, so callers can branch on error
    type instead of grep-ing exception messages for magic strings."""


class LLMQuotaError(LLMError):
    """Raised when the upstream model provider signals a rate limit or
    quota exhaustion (HTTP 429 / RESOURCE_EXHAUSTED)."""


def _get_client() -> genai.Client:
    global _CLIENT
    if _CLIENT is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not configured")
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


async def generate_text(
    prompt: str,
    temperature: float,
    retries: int = 2,
    timeout_seconds: float = 30.0,
) -> str:
    """Call Gemini with a bounded timeout and a small retry budget so a
    transient network blip or rate-limit doesn't take down the whole
    request chain."""
    client = _get_client()
    model = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.models.generate_content,
                    model=model,
                    contents=prompt,
                    config={"temperature": temperature},
                ),
                timeout=timeout_seconds,
            )
            if not response.text:
                raise RuntimeError("Gemini returned an empty response")
            return response.text
        except Exception as exc:  # noqa: BLE001 - we deliberately retry on any failure
            last_error = exc
            is_quota_error = any(marker in str(exc).lower() for marker in _QUOTA_MARKERS)
            is_last_attempt = attempt == retries
            if is_last_attempt:
                break
            wait = 0.5 * (2**attempt)
            logger.warning("Gemini call failed (attempt %s/%s): %s", attempt + 1, retries + 1, exc)
            # Quota errors won't resolve themselves within a single request's
            # retry budget - fail fast instead of burning the retry window.
            if is_quota_error:
                break
            await asyncio.sleep(wait)

    assert last_error is not None
    if any(marker in str(last_error).lower() for marker in _QUOTA_MARKERS):
        raise LLMQuotaError(str(last_error)) from last_error
    raise last_error
