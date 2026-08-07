import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import agent_graph


def _make_memory():
    mem = MagicMock()
    mem.get_recent_messages.return_value = []
    mem.get_summary.return_value = ""
    mem.get_message_count.return_value = 0
    return mem


class BuildAndRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_happy_path_returns_verified_answer_and_sources(self):
        fake_memory = _make_memory()

        with patch("agent_graph.Memory", return_value=fake_memory), patch(
            "agent_graph.tavily_search",
            new=AsyncMock(return_value=[{"title": "T", "url": "https://x.test", "snippet": "s"}]),
        ), patch("agent_graph.generate_text", new=AsyncMock(return_value="draft answer")), patch(
            "agent_graph.verify_with_latest_sources",
            new=AsyncMock(
                return_value={
                    "answer": "verified answer",
                    "sources": [{"title": "T", "url": "https://x.test"}],
                }
            ),
        ):
            result = await agent_graph.build_and_run("What is X?", session_id="s1")

        self.assertEqual(result["answer"], "verified answer")
        self.assertEqual(result["sources"], [{"title": "T", "url": "https://x.test"}])
        fake_memory.append_user_message.assert_called_once_with("s1", "What is X?")
        fake_memory.append_assistant_message.assert_called_once_with("s1", "verified answer")

    async def test_falls_back_to_draft_answer_when_search_finds_nothing(self):
        fake_memory = _make_memory()

        with patch("agent_graph.Memory", return_value=fake_memory), patch(
            "agent_graph.tavily_search", new=AsyncMock(return_value=[])
        ), patch("agent_graph.generate_text", new=AsyncMock(return_value="draft answer")), patch(
            "agent_graph.verify_with_latest_sources",
            new=AsyncMock(return_value={"answer": "draft answer", "sources": []}),
        ):
            result = await agent_graph.build_and_run("hello", session_id="s2")

        self.assertEqual(result["answer"], "draft answer")
        self.assertEqual(result["sources"], [])

    async def test_raises_runtime_error_when_no_answer_produced(self):
        fake_memory = _make_memory()

        with patch("agent_graph.Memory", return_value=fake_memory), patch(
            "agent_graph.tavily_search", new=AsyncMock(return_value=[])
        ), patch("agent_graph.generate_text", new=AsyncMock(return_value="")), patch(
            "agent_graph.verify_with_latest_sources",
            new=AsyncMock(return_value={"answer": "", "sources": []}),
        ):
            with self.assertRaises(RuntimeError):
                await agent_graph.build_and_run("hello", session_id="s3")

    async def test_session_id_is_threaded_through_to_memory_lookup(self):
        fake_memory = _make_memory()
        fake_memory.get_recent_messages.return_value = [{"role": "user", "content": "earlier"}]

        with patch("agent_graph.Memory", return_value=fake_memory), patch(
            "agent_graph.tavily_search", new=AsyncMock(return_value=[])
        ), patch("agent_graph.generate_text", new=AsyncMock(return_value="draft")), patch(
            "agent_graph.verify_with_latest_sources",
            new=AsyncMock(return_value={"answer": "draft", "sources": []}),
        ):
            await agent_graph.build_and_run("follow-up question", session_id="session-xyz")

        fake_memory.get_recent_messages.assert_called_once_with("session-xyz", 8)


class StreamAnswerTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_emits_events_in_order(self):
        fake_memory = _make_memory()

        async def _fake_stream(prompt, temperature, timeout_seconds=60.0):
            yield "Hel"
            yield "lo "
            yield "world"

        with patch("agent_graph.Memory", return_value=fake_memory), patch(
            "agent_graph.tavily_search", new=AsyncMock(return_value=[])
        ), patch(
            "agent_graph.stream_text", new=_fake_stream
        ), patch(
            "agent_graph.verify_with_latest_sources",
            new=AsyncMock(return_value={"answer": "Hello world", "sources": []}),
        ), patch(
            "agent_graph.generate_text", new=AsyncMock(return_value="summary")
        ):
            events = [event async for event in agent_graph.stream_answer("hi", session_id="s1")]

        types = [e["type"] for e in events]
        self.assertEqual(types, ["metadata", "sources", "token", "token", "token", "done"])
        self.assertEqual(events[2]["token"], "Hel")
        self.assertEqual(events[3]["token"], "lo ")
        self.assertEqual(events[4]["token"], "world")


if __name__ == "__main__":
    unittest.main()

