from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from tavily import TavilyClient

logger = logging.getLogger(__name__)

_CLIENT: TavilyClient | None = None


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
    client = _get_client()
    try:
        res = await asyncio.wait_for(
            asyncio.to_thread(
                client.search, query=query, max_results=max_results, search_depth="advanced"
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning("Tavily search timed out for query: %s", query)
        return []
    except Exception as exc:  # noqa: BLE001
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
    return results
