import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import main
from main import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        # Ensure auth is disabled unless a test explicitly enables it, so the
        # default suite is not affected by a developer's local .env.
        self._api_key_patch = patch.object(main, "APP_API_KEY", None)
        self._api_key_patch.start()
        # Mock the agent pipeline so no real network/LLM calls are made.
        self._agent_patch = patch.object(
            main,
            "build_and_run",
            new=AsyncMock(
                return_value={
                    "answer": "mocked answer",
                    "sources": [],
                }
            ),
        )
        self._agent_patch.start()

        async def _fake_stream(message, session_id):
            yield {"type": "metadata", "session_id": session_id}
            yield {"type": "sources", "sources": []}
            yield {"type": "token", "token": "Hello "}
            yield {"type": "token", "token": "world"}
            yield {"type": "done"}

        self._stream_patch = patch.object(main, "stream_answer", new=_fake_stream)
        self._stream_patch.start()
        self.client = TestClient(app)

    def tearDown(self):
        self._api_key_patch.stop()
        self._agent_patch.stop()
        self._stream_patch.stop()

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("uptime_seconds", data)
        self.assertIn("latency_ms", data)

    def test_session_history_is_scoped_to_requested_session(self):
        with patch("memory.sqlite_memory.Memory.get_recent_messages", return_value=[]):
            response = self.client.get("/sessions/session-123")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"session_id": "session-123", "messages": []})

    def test_health_ready_without_provider_returns_503(self):
        # Force all LLM provider keys to be absent.
        with patch.dict(
            "os.environ",
            {"GOOGLE_API_KEY": "", "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""},
            clear=False,
        ):
            response = self.client.get("/health/ready")
        self.assertEqual(response.status_code, 503)

    def test_blank_message_is_rejected(self):
        response = self.client.post("/chat", json={"message": "   "})
        self.assertEqual(response.status_code, 422)

    def test_chat_returns_mocked_answer_and_session_id(self):
        response = self.client.post("/chat", json={"message": "Hello"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["answer"], "mocked answer")
        self.assertIn("session_id", data)

    def test_chat_accepts_and_echoes_session_id(self):
        # An explicit session_id round trips and is returned in the response.
        response = self.client.post(
            "/chat", json={"message": "Hello", "session_id": "my-session-123"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["session_id"], "my-session-123")

    def test_stream_endpoint_returns_sse(self):
        response = self.client.post("/chat/stream", json={"message": "Hello"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream; charset=utf-8")

    def test_invalid_api_key_is_rejected_when_configured(self):
        original = main.APP_API_KEY
        main.APP_API_KEY = "secret-test-key"
        try:
            response = self.client.post(
                "/chat",
                json={"message": "Hello"},
                headers={"X-API-Key": "wrong-key"},
            )
            self.assertEqual(response.status_code, 401)
        finally:
            main.APP_API_KEY = original


if __name__ == "__main__":
    unittest.main()
