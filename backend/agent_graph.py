from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, TypedDict

from langgraph.graph import StateGraph, END

from tools.web_search import tavily_search
from tools.verify import verify_with_latest_sources
from memory.sqlite_memory import SQLiteMemory
from llm import generate_text

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    message: str
    session_id: str
    history: List[Dict[str, Any]]
    search_results: List[Dict[str, Any]]
    answer: str
    verified_answer: str
    sources: List[Dict[str, Any]]


async def node_load_memory(state: AgentState) -> AgentState:
    mem = SQLiteMemory()
    state["history"] = await asyncio.to_thread(
        mem.get_recent_messages, state["session_id"], 8
    )
    return state


async def node_search(state: AgentState) -> AgentState:
    query = state["message"]
    # tavily_search already degrades to [] on failure/timeout rather than
    # raising, so a flaky search provider never takes down the whole chat.
    results = await tavily_search(query, max_results=5)
    state["search_results"] = results
    return state


async def node_answer(state: AgentState) -> AgentState:
    system = (
        "You are a helpful AI assistant. "
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
    recent_history = (
        "\n".join(f"{item['role']}: {item['content']}" for item in state.get("history", []))
        or "(no previous conversation)"
    )

    prompt = (
        f"Recent conversation:\n{recent_history}\n\n"
        f"User question:\n{state['message']}\n\n"
        f"Web search context:\n{numbered_context}\n\n"
        "Write the best answer using the context."
    )

    state["answer"] = await generate_text(f"{system}\n\n{prompt}", temperature=0.2)
    return state


async def node_verify(state: AgentState) -> AgentState:
    sources = state.get("search_results", [])
    verified = await verify_with_latest_sources(state["message"], state.get("answer", ""), sources)
    state["verified_answer"] = verified["answer"]
    state["sources"] = verified["sources"]
    return state


async def node_store_memory(state: AgentState) -> AgentState:
    mem = SQLiteMemory()
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
graph.add_node("store_memory", node_store_memory)

graph.set_entry_point("load_memory")
graph.add_edge("load_memory", "search")
graph.add_edge("search", "produce_answer")
graph.add_edge("produce_answer", "verify")
graph.add_edge("verify", "store_memory")
graph.add_edge("store_memory", END)

compiled = graph.compile()


async def build_and_run(message: str, session_id: str) -> dict[str, Any]:
    state: AgentState = AgentState(message=message, session_id=session_id)
    out = await compiled.ainvoke(state)
    answer = out.get("verified_answer") or out.get("answer")
    if not answer:
        raise RuntimeError("No answer was generated")
    return {"answer": answer, "sources": out.get("sources", out.get("search_results", []))}
