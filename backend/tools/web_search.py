from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from tavily import TavilyClient

from observability import CHAT_SEARCH_CALLS, CHAT_SEARCH_FAILURES

logger = logging.getLogger(__name__)

_CLIENT: TavilyClient | None = None
_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_CACHE_TTL_SECONDS = max(0, int(os.getenv("SEARCH_CACHE_TTL_SECONDS", "60")))


def _get_client() -> TavilyClient:
    global _CLIENT
    if _CLIENT is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY is not configured")
        _CLIENT = TavilyClient(api_key=api_key)
    return _CLIENT


async def tavily_search(
    query: str, max_results: int = 5, timeout_seconds: float = 15.0
) -> list[dict[str, Any]]:
    """Run a Tavily search. Returns an empty list (rather than raising) on
    timeout or transient failure, so the assistant can still fall back to
    answering from the model's own knowledge instead of hard-failing the
    whole request."""
    normalized_query = " ".join(query.lower().split())
    cached = _CACHE.get(normalized_query)
    if cached and cached[0] > time.monotonic():
        return cached[1]

    try:
        client = _get_client()
        CHAT_SEARCH_CALLS.inc()
        res = await asyncio.wait_for(
            asyncio.to_thread(
                client.search, query=query, max_results=max_results, search_depth="advanced"
            ),
            timeout=timeout_seconds,
        )
    except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
        CHAT_SEARCH_FAILURES.inc()
        logger.warning("Tavily search failed for query %r: %s", query, exc)
        return []

    results: list[dict[str, Any]] = []
    for item in res.get("results", []):
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
            }
        )
    if _CACHE_TTL_SECONDS:
        # Small bounded in-process cache: suitable for duplicate live queries,
        # while keeping time-sensitive search results fresh.
        if len(_CACHE) >= 256:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[normalized_query] = (time.monotonic() + _CACHE_TTL_SECONDS, results)
    return results
