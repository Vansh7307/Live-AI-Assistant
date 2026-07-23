import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import llm


class FakeResponse:
    def __init__(self, text):
        self.text = text


class GenerateTextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        llm._CLIENT = None

    async def test_success_returns_text(self):
        fake_client = MagicMock()
        fake_client.models.generate_content.return_value = FakeResponse("ok")
        with patch("llm._get_client", return_value=fake_client):
            result = await llm.generate_text("hi", temperature=0.2)
        self.assertEqual(result, "ok")

    async def test_transient_failure_then_success_recovers(self):
        fake_client = MagicMock()
        fake_client.models.generate_content.side_effect = [
            RuntimeError("temporary network blip"),
            FakeResponse("ok after retry"),
        ]
        with patch("llm._get_client", return_value=fake_client), patch(
            "asyncio.sleep", new=AsyncMock()
        ):
            result = await llm.generate_text("hi", temperature=0.2, retries=2)
        self.assertEqual(result, "ok after retry")
        self.assertEqual(fake_client.models.generate_content.call_count, 2)

    async def test_persistent_failure_raises_after_exhausting_retries(self):
        fake_client = MagicMock()
        fake_client.models.generate_content.side_effect = RuntimeError("boom")
        with patch("llm._get_client", return_value=fake_client), patch(
            "asyncio.sleep", new=AsyncMock()
        ):
            with self.assertRaises(RuntimeError):
                await llm.generate_text("hi", temperature=0.2, retries=1)
        self.assertEqual(fake_client.models.generate_content.call_count, 2)

    async def test_quota_error_raises_typed_exception_and_fails_fast(self):
        fake_client = MagicMock()
        fake_client.models.generate_content.side_effect = RuntimeError("429 RESOURCE_EXHAUSTED")
        with patch("llm._get_client", return_value=fake_client), patch(
            "asyncio.sleep", new=AsyncMock()
        ) as sleep_mock:
            with self.assertRaises(llm.LLMQuotaError):
                await llm.generate_text("hi", temperature=0.2, retries=3)
        # Fail-fast: a quota error should not burn the whole retry/backoff budget.
        sleep_mock.assert_not_called()
        self.assertEqual(fake_client.models.generate_content.call_count, 1)

    async def test_empty_response_text_is_treated_as_failure(self):
        fake_client = MagicMock()
        fake_client.models.generate_content.return_value = FakeResponse("")
        with patch("llm._get_client", return_value=fake_client), patch(
            "asyncio.sleep", new=AsyncMock()
        ):
            with self.assertRaises(RuntimeError):
                await llm.generate_text("hi", temperature=0.2, retries=0)


if __name__ == "__main__":
    unittest.main()
