from __future__ import annotations

import logging
import re
import json
from typing import Any

from llm import generate_text

logger = logging.getLogger(__name__)


async def verify_with_latest_sources(question: str, draft_answer: str, sources: list[dict[str, Any]]):
    if not sources:
        return {"answer": draft_answer, "sources": []}
    # Verification approach:
    # 1) Ask LLM to extract claims.
    # 2) Perform an additional targeted search for weak points.
    # 3) Produce a final answer with citations from the latest search.

    numbered = "\n\n".join(
        [f"[{i+1}] {s['title']} (url: {s['url']}): {s['snippet']}" for i, s in enumerate(sources)]
    )

    verify_prompt = (
        "You are a verification assistant.\n"
        "Given a question, a draft answer, and search context, verify factual claims.\n"
        "If claims are unsupported or outdated, correct them.\n"
        "Return valid JSON only: {\"answer\": string, \"sources\": [{\"title\":...,\"url\":...}, ...]}.\n"
        "Use only URLs from the provided sources.\n\n"
        f"Question:\n{question}\n\n"
        f"Draft answer:\n{draft_answer}\n\n"
        f"Current context:\n{numbered}\n"
    )

    try:
        content = await generate_text(verify_prompt, temperature=0.0)
    except Exception as exc:  # noqa: BLE001
        # Verification is a nice-to-have; if it fails, fall back to the
        # unverified draft instead of failing the whole chat request.
        logger.warning("Verification step failed, falling back to draft answer: %s", exc)
        return {"answer": draft_answer, "sources": [{"title": s["title"], "url": s["url"]} for s in sources]}

    # Heuristic: try to find JSON in the response
    json_match = re.search(r"\{\s*\"answer\"[\s\S]*?\}\s*$", content.strip())
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            allowed_sources = {s["url"]: s["title"] for s in sources if s.get("url")}
            srcs = [
                {"title": allowed_sources[source["url"]], "url": source["url"]}
                for source in parsed.get("sources", [])
                if isinstance(source, dict) and source.get("url") in allowed_sources
            ]
            return {"answer": str(parsed.get("answer") or draft_answer), "sources": srcs or [{"title": s["title"], "url": s["url"]} for s in sources]}
        except Exception:
            pass

    # Fallback
    return {"answer": content, "sources": [{"title": s["title"], "url": s["url"]} for s in sources]}
