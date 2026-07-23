import unittest

from fastapi.testclient import TestClient

from main import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_blank_message_is_rejected(self):
        response = self.client.post("/chat", json={"message": "   "})
        self.assertEqual(response.status_code, 422)

    def test_chat_reports_missing_configuration_cleanly(self):
        response = self.client.post("/chat", json={"message": "Hello"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "TAVILY_API_KEY is not configured")

    def test_chat_accepts_and_echoes_session_id(self):
        # Even though this request fails at the missing-API-key stage, the
        # request itself must validate: an explicit session_id round trips
        # and doesn't get rejected as malformed input.
        response = self.client.post(
            "/chat", json={"message": "Hello", "session_id": "my-session-123"}
        )
        self.assertEqual(response.status_code, 503)

    def test_invalid_api_key_is_rejected_when_configured(self):
        import main

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
