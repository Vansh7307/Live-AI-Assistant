"""Multi-provider LLM layer with automatic failover.

Supported providers:
  - Google Gemini (default, requires GOOGLE_API_KEY)
  - OpenAI (requires OPENAI_API_KEY)
  - Anthropic (requires ANTHROPIC_API_KEY)

Probe order is configured via ``LLM_PROVIDERS`` (comma-separated list, e.g.
``gemini,openai,anthropic``). If the first provider fails (quota, network,
timeout, etc.) the next configured provider is tried in order. Gemini is
included by default if ``GOOGLE_API_KEY`` is set; others are added on demand
when their key is present.

Every call is bounded by a timeout and a small per-provider retry budget so a
transient blip never takes down the whole request chain.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

_QUOTA_MARKERS = ("429", "quota", "rate limit", "resource_exhausted", "insufficient_quota")


class LLMError(Exception):
    """Base class for LLM-layer failures, so callers can branch on error
    type instead of grep-ing exception messages for magic strings."""


class LLMQuotaError(LLMError):
    """Raised when the model provider signals a rate limit or quota
    exhaustion (HTTP 429 / RESOURCE_EXHAUSTED)."""


class LLMProviderError(LLMError):
    """Raised when a provider is misconfigured or unreachable (bad key,
    DNS failure, network error)."""


class _GeminiProvider:
    def __init__(self, api_key: str, model: str):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, prompt: str, temperature: float) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config={"temperature": temperature},
        )
        if not response.text:
            raise RuntimeError("Gemini returned an empty response")
        return response.text

    def stream(self, prompt: str, temperature: float):
        from google import genai

        for chunk in self._client.models.generate_content_stream(
            model=self._model,
            contents=prompt,
            config={"temperature": temperature},
        ):
            if chunk.text:
                yield chunk.text


class _OpenAIProvider:
    def __init__(self, api_key: str, model: str):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, prompt: str, temperature: float) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        text = response.choices[0].message.content or ""
        if not text:
            raise RuntimeError("OpenAI returned an empty response")
        return text

    def stream(self, prompt: str, temperature: float):
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta


class _AnthropicProvider:
    def __init__(self, api_key: str, model: str):
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(self, prompt: str, temperature: float) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
        if not text:
            raise RuntimeError("Anthropic returned an empty response")
        return text

    def stream(self, prompt: str, temperature: float):
        with self._client.messages.stream(
            model=self._model,
            max_tokens=4096,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text


def _provider_from_env() -> list[Any]:
    """Build the ordered list of configured providers from environment.

    Order comes from ``LLM_PROVIDERS`` (defaults to ``gemini,openai,anthropic``),
    but only providers whose API key is present are instantiated. This makes
    failover additive: enable a second key and you automatically gain a
    fallback without any code change.
    """
    order = [p.strip().lower() for p in os.getenv("LLM_PROVIDERS", "gemini,openai,anthropic").split(",") if p.strip()]
    providers: list[Any] = []

    for name in order:
        if name == "gemini":
            key = os.getenv("GOOGLE_API_KEY")
            if key:
                providers.append(_GeminiProvider(key, os.getenv("GEMINI_MODEL", "gemini-flash-latest")))
        elif name == "openai":
            key = os.getenv("OPENAI_API_KEY")
            if key:
                providers.append(_OpenAIProvider(key, os.getenv("OPENAI_MODEL", "gpt-4o-mini")))
        elif name == "anthropic":
            key = os.getenv("ANTHROPIC_API_KEY")
            if key:
                providers.append(_AnthropicProvider(key, os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")))
        else:
            logger.warning("Unknown LLM provider in LLM_PROVIDERS: %s", name)

    return providers


def _is_quota(exc: Exception) -> bool:
    return any(marker in str(exc).lower() for marker in _QUOTA_MARKERS)


async def _run_with_timeout(
    fn: Callable[[], Awaitable[str]], timeout_seconds: float
) -> str:
    return await asyncio.wait_for(asyncio.to_thread(fn), timeout=timeout_seconds)


async def generate_text(
    prompt: str,
    temperature: float,
    retries: int = 2,
    timeout_seconds: float = 30.0,
) -> str:
    """Call the best available LLM provider with bounded timeout and retry.

    Providers are tried in order. Within a provider a small retry budget is
    used for transient failures. Quota errors fail fast (they won't resolve
    within a single request's budget) and move on to the next provider.
    """
    providers = _provider_from_env()
    if not providers:
        raise LLMProviderError(
            "No LLM provider configured. Set GOOGLE_API_KEY, OPENAI_API_KEY, "
            "or ANTHROPIC_API_KEY."
        )

    last_error: Exception | None = None
    for provider in providers:
        for attempt in range(retries + 1):
            try:
                return await _run_with_timeout(
                    lambda p=provider: p.generate(prompt, temperature),
                    timeout_seconds,
                )
            except Exception as exc:  # noqa: BLE001 - deliberate retry/failover
                last_error = exc
                is_quota = _is_quota(exc)
                is_last_attempt = attempt == retries
                logger.warning(
                    "LLM provider %s failed (attempt %s/%s): %s",
                    provider.__class__.__name__,
                    attempt + 1,
                    retries + 1,
                    exc,
                )
                if is_quota or is_last_attempt:
                    break
                await asyncio.sleep(0.5 * (2**attempt))
        # Move on to the next provider regardless of why this one failed.
        logger.info("Failing over to next LLM provider after %s", provider.__class__.__name__)

    assert last_error is not None
    if _is_quota(last_error):
        raise LLMQuotaError(str(last_error)) from last_error
    raise last_error


async def stream_text(
    prompt: str,
    temperature: float,
    timeout_seconds: float = 60.0,
):
    """Stream tokens from the best available provider.

    Yields ``str`` chunks. If the first provider fails mid-stream we do not
    attempt a mid-stream failover (the partial response is already sent);
    instead we surface the error to the caller so it can send an error event.
    """
    providers = _provider_from_env()
    if not providers:
        raise LLMProviderError(
            "No LLM provider configured. Set GOOGLE_API_KEY, OPENAI_API_KEY, "
            "or ANTHROPIC_API_KEY."
        )

    last_error: Exception | None = None
    for provider in providers:
        try:
            # Run the generator in a thread so blocking provider SDKs don't
            # stall the event loop; pull chunks as they arrive.
            def _iter():
                return provider.stream(prompt, temperature)

            gen = await asyncio.to_thread(_iter)
            while True:
                chunk = await asyncio.wait_for(asyncio.to_thread(next, gen), timeout=timeout_seconds)
                yield chunk
            # unreachable
        except StopIteration:
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "LLM provider %s stream failed, trying next: %s",
                provider.__class__.__name__,
                exc,
            )
            continue

    if last_error is not None:
        if _is_quota(last_error):
            raise LLMQuotaError(str(last_error)) from last_error
        raise last_error
