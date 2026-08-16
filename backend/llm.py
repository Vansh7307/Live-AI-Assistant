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
import time
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

_QUOTA_MARKERS = ("429", "quota", "rate limit", "resource_exhausted", "insufficient_quota")
_CIRCUIT_FAILURE_THRESHOLD = int(os.getenv("LLM_CIRCUIT_FAILURE_THRESHOLD", "3"))
_CIRCUIT_RESET_SECONDS = float(os.getenv("LLM_CIRCUIT_RESET_SECONDS", "30"))
_CIRCUITS: dict[str, dict[str, float]] = {}


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
        interaction = self._client.interactions.create(
            model=self._model,
            input=prompt,
            generation_config={"temperature": temperature},
        )
        if not interaction.output_text:
            raise RuntimeError("Gemini returned an empty response")
        return interaction.output_text

    def stream(self, prompt: str, temperature: float):
        stream = self._client.interactions.create(
            model=self._model,
            input=prompt,
            generation_config={"temperature": temperature},
            stream=True,
        )
        for event in stream:
            if (
                getattr(event, "event_type", None) == "step.delta"
                and getattr(event, "delta", None)
                and getattr(event.delta, "type", None) == "text"
                and getattr(event.delta, "text", None)
            ):
                yield event.delta.text


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
                providers.append(_GeminiProvider(key, os.getenv("GEMINI_MODEL", "gemini-3.6-flash")))
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


def _provider_key(provider: Any) -> str:
    return provider.__class__.__name__


def _circuit_open(provider: Any) -> bool:
    state = _CIRCUITS.get(_provider_key(provider))
    if not state:
        return False
    if state.get("opened_until", 0) <= time.monotonic():
        _CIRCUITS.pop(_provider_key(provider), None)
        return False
    return True


def _record_provider_failure(provider: Any) -> None:
    key = _provider_key(provider)
    state = _CIRCUITS.setdefault(key, {"failures": 0, "opened_until": 0})
    state["failures"] += 1
    if state["failures"] >= _CIRCUIT_FAILURE_THRESHOLD:
        state["opened_until"] = time.monotonic() + _CIRCUIT_RESET_SECONDS
        logger.warning("Circuit opened for provider %s", key)


def _record_provider_success(provider: Any) -> None:
    _CIRCUITS.pop(_provider_key(provider), None)


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
        if _circuit_open(provider):
            logger.warning("Skipping provider %s because its circuit is open", _provider_key(provider))
            continue
        for attempt in range(retries + 1):
            try:
                result = await _run_with_timeout(
                    lambda p=provider: p.generate(prompt, temperature),
                    timeout_seconds,
                )
                _record_provider_success(provider)
                return result
            except Exception as exc:  # noqa: BLE001 - deliberate retry/failover
                last_error = exc
                _record_provider_failure(provider)
                is_quota = _is_quota(exc)
                is_last_attempt = attempt == retries
                logger.error(f"LLM Provider execution error: {str(exc)}")
                if is_quota or is_last_attempt:
                    break
                await asyncio.sleep(0.5 * (2**attempt))
        # Move on to the next provider regardless of why this one failed.
        logger.info("Failing over to next LLM provider after %s", provider.__class__.__name__)

    if last_error is None:
        raise LLMProviderError("All configured AI providers are temporarily unavailable.")
    if _is_quota(last_error):
        raise LLMQuotaError(f"Provider quota or rate limit error: {last_error}") from last_error
    raise LLMProviderError(
        f"All configured LLM providers failed. Upstream diagnostic: {last_error}"
    ) from last_error


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
        if _circuit_open(provider):
            logger.warning("Skipping provider %s because its circuit is open", _provider_key(provider))
            continue
        emitted_tokens = False
        try:
            # Run the generator in a thread so blocking provider SDKs don't
            # stall the event loop; pull chunks as they arrive.
            def _iter():
                return provider.stream(prompt, temperature)

            gen = await asyncio.to_thread(_iter)
            while True:
                chunk = await asyncio.wait_for(asyncio.to_thread(next, gen), timeout=timeout_seconds)
                emitted_tokens = True
                yield chunk
            # unreachable
        except StopIteration:
            _record_provider_success(provider)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            _record_provider_failure(provider)
            if emitted_tokens:
                # Retrying after yielding text would produce a duplicate or
                # contradictory answer, so the SSE boundary reports failure.
                raise
            logger.error(f"LLM Provider execution error: {str(exc)}")
            continue

    if last_error is None:
        raise LLMProviderError("All configured AI providers are temporarily unavailable.")
    if _is_quota(last_error):
        raise LLMQuotaError(f"Provider quota or rate limit error: {last_error}") from last_error
    raise LLMProviderError(
        f"All configured LLM providers failed. Upstream diagnostic: {last_error}"
    ) from last_error
