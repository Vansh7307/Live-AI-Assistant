from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, TypedDict

from langgraph.graph import StateGraph, END

from tools.web_search import tavily_search
from tools.verify import verify_with_latest_sources
from memory.sqlite_memory import Memory
from llm import generate_text, stream_text
from observability import CHAT_LLM_CALLS, CHAT_LLM_FAILURES

logger = logging.getLogger(__name__)

SUMMARIZE_THRESHOLD = 12  # messages before we roll history into a summary


class AgentState(TypedDict, total=False):
    message: str
    session_id: str
    history: List[Dict[str, Any]]
    summary: str
    search_results: List[Dict[str, Any]]
    answer: str
    verified_answer: str
    sources: List[Dict[str, Any]]


async def node_load_memory(state: AgentState) -> AgentState:
    mem = Memory()
    state["history"] = await asyncio.to_thread(
        mem.get_recent_messages, state["session_id"], 8
    )
    state["summary"] = await asyncio.to_thread(mem.get_summary, state["session_id"])
    return state


async def node_search(state: AgentState) -> AgentState:
    query = state["message"]
    # tavily_search already degrades to [] on failure/timeout rather than
    # raising, so a flaky search provider never takes down the whole chat.
    results = await tavily_search(query, max_results=5)
    state["search_results"] = results
    return state


def _build_prompt(state: AgentState) -> str:
    system = (
        "You are a helpful, accurate AI assistant. "
        "Use the provided web search context when it is relevant. "
        "When you use facts from the context, cite the matching URLs inline like [1], [2]. "
        "If the context is insufficient or empty, answer from your own knowledge and say so."
    )

    sources = state.get("search_results", [])
    numbered_context = (
        "\n\n".join(
            f"[{i + 1}] {s['title']} - {s['snippet']} (url: {s['url']})"
            for i, s in enumerate(sources)
        )
        or "(no search results available)"
    )

    summary = state.get("summary", "")
    summary_block = f"Conversation summary so far:\n{summary}\n\n" if summary else ""

    recent_history = (
        "\n".join(f"{item['role']}: {item['content']}" for item in state.get("history", []))
        or "(no recent messages)"
    )

    prompt = (
        f"{summary_block}"
        f"Recent conversation:\n{recent_history}\n\n"
        f"User question:\n{state['message']}\n\n"
        f"Web search context:\n{numbered_context}\n\n"
        "Write the best answer using the context."
    )

    return f"{system}\n\n{prompt}"


async def node_answer(state: AgentState) -> AgentState:
    prompt = _build_prompt(state)
    CHAT_LLM_CALLS.labels("generate").inc()
    try:
        state["answer"] = await generate_text(prompt, temperature=0.2)
    except Exception as exc:  # noqa: BLE001
        CHAT_LLM_FAILURES.labels("generate").inc()
        raise
    return state


async def node_verify(state: AgentState) -> AgentState:
    sources = state.get("search_results", [])
    verified = await verify_with_latest_sources(state["message"], state.get("answer", ""), sources)
    state["verified_answer"] = verified["answer"]
    state["sources"] = verified["sources"]
    return state


async def node_compact_memory(state: AgentState) -> AgentState:
    """If a session has grown long, roll older context into a summary and
    keep the recent window small so prompts stay bounded."""
    mem = Memory()
    count = await asyncio.to_thread(mem.get_message_count, state["session_id"])
    if count >= SUMMARIZE_THRESHOLD:
        history = await asyncio.to_thread(
            mem.get_recent_messages, state["session_id"], 20
        )
        if history:
            lines = "\n".join(f"{m['role']}: {m['content']}" for m in history)
            try:
                summary = await generate_text(
                    (
                        "Summarize the following conversation into 2-4 concise bullet "
                        "points capturing the user's interests, questions, and any "
                        "conclusions. Keep it factual.\n\n"
                        f"{lines}"
                    ),
                    temperature=0.0,
                )
                await asyncio.to_thread(mem.set_summary, state["session_id"], summary)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Summary compaction failed: %s", exc)
    return state


async def node_store_memory(state: AgentState) -> AgentState:
    mem = Memory()
    final_answer = state.get("verified_answer") or state.get("answer") or ""
    await asyncio.to_thread(mem.append_user_message, state["session_id"], state["message"])
    if final_answer:
        await asyncio.to_thread(mem.append_assistant_message, state["session_id"], final_answer)
    return state


graph = StateGraph(AgentState)

graph.add_node("load_memory", node_load_memory)
graph.add_node("search", node_search)
graph.add_node("produce_answer", node_answer)
graph.add_node("verify", node_verify)
graph.add_node("compact_memory", node_compact_memory)
graph.add_node("store_memory", node_store_memory)

graph.set_entry_point("load_memory")
graph.add_edge("load_memory", "search")
graph.add_edge("search", "produce_answer")
graph.add_edge("produce_answer", "verify")
graph.add_edge("verify", "compact_memory")
graph.add_edge("compact_memory", "store_memory")
graph.add_edge("store_memory", END)

compiled = graph.compile()


async def build_and_run(message: str, session_id: str) -> dict[str, Any]:
    state: AgentState = AgentState(message=message, session_id=session_id)
    out = await compiled.ainvoke(state)
    answer = out.get("verified_answer") or out.get("answer")
    if not answer:
        raise RuntimeError("No answer was generated")
    return {"answer": answer, "sources": out.get("sources", out.get("search_results", []))}


async def stream_answer(message: str, session_id: str) -> AsyncGenerator[dict[str, Any], None]:
    """Run the same pipeline but stream the final (verified) answer to the
    caller as a sequence of events. Emits metadata events first, then token
    events, then a final event with sources.

    Yields dicts with a ``type`` key:
      - {"type": "metadata", "session_id": ...}
      - {"type": "sources", "sources": [...]}   (before tokens, so the UI can show linking)
      - {"type": "token", "token": "..."}        (streamed answer text)
      - {"type": "done"}                          (end of stream)
    """
    state: AgentState = AgentState(message=message, session_id=session_id)
    # Load memory + search first (non-streamed), reuse the graph nodes.
    await node_load_memory(state)
    await node_search(state)

    yield {"type": "metadata", "session_id": session_id}
    yield {"type": "sources", "sources": state.get("search_results", [])}

    prompt = _build_prompt(state)
    CHAT_LLM_CALLS.labels("stream").inc()
    try:
        full_answer = ""
        async for chunk in stream_text(prompt, temperature=0.2):
            full_answer += chunk
            yield {"type": "token", "token": chunk}
    except Exception as exc:  # noqa: BLE001
        CHAT_LLM_FAILURES.labels("stream").inc()
        logger.exception("Streaming failed")
        raise

    state["answer"] = full_answer
    # Verify and persist after delivery; neither operation delays first token.
    try:
        await node_verify(state)
        await node_compact_memory(state)
        await node_store_memory(state)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Post-stream persistence failed: %s", exc)
    yield {"type": "done"}
