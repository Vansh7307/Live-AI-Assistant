import unittest
from unittest.mock import AsyncMock, MagicMock, call, patch

import llm


class FakeGeminiInteraction:
    def __init__(self, text):
        self.output_text = text


class FakeOpenAIResponse:
    def __init__(self, text):
        self.message = type("Msg", (), {"content": text})()
        self.choices = [type("Ch", (), {"message": self.message})()]


class FakeAnthropicBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeAnthropicResponse:
    def __init__(self, text):
        self.content = [FakeAnthropicBlock(text)]


class GenerateTextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        pass

    async def test_success_gemini_returns_text(self):
        fake_client = MagicMock()
        fake_client.interactions.create.return_value = FakeGeminiInteraction("ok")
        with patch.dict(
            "os.environ",
            {"GOOGLE_API_KEY": "test-key", "LLM_PROVIDERS": "gemini"},
            clear=False,
        ):
            with patch("llm._GeminiProvider") as mock_provider:
                mock_provider.return_value.generate.return_value = "ok"
                result = await llm.generate_text("hi", temperature=0.2)
        self.assertEqual(result, "ok")

    def test_gemini_uses_current_interactions_model_by_default(self):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}, clear=True), patch(
            "llm._GeminiProvider"
        ) as mock_provider:
            llm._provider_from_env()
        mock_provider.assert_called_once_with("test-key", "gemini-3.6-flash")

    def test_gemini_generate_uses_interactions_api(self):
        provider = object.__new__(llm._GeminiProvider)
        provider._client = MagicMock()
        provider._model = "gemini-3.6-flash"
        provider._client.interactions.create.return_value = FakeGeminiInteraction("ok")

        self.assertEqual(provider.generate("hi", 0.2), "ok")
        provider._client.interactions.create.assert_called_once_with(
            model="gemini-3.6-flash",
            input="hi",
            generation_config={"temperature": 0.2},
        )

    def test_gemini_stream_uses_interaction_step_delta_text(self):
        provider = object.__new__(llm._GeminiProvider)
        provider._client = MagicMock()
        provider._model = "gemini-3.6-flash"
        text_event = MagicMock(event_type="step.delta")
        text_event.delta.type = "text"
        text_event.delta.text = "hello"
        ignored_event = MagicMock(event_type="interaction.completed")
        provider._client.interactions.create.return_value = [text_event, ignored_event]

        self.assertEqual(list(provider.stream("hi", 0.2)), ["hello"])
        provider._client.interactions.create.assert_called_once_with(
            model="gemini-3.6-flash",
            input="hi",
            generation_config={"temperature": 0.2},
            stream=True,
        )

    def test_gemini_retries_model_not_found_with_interactions_fallback(self):
        provider = object.__new__(llm._GeminiProvider)
        provider._client = MagicMock()
        provider._model = "invalid-model"
        provider._client.interactions.create.side_effect = [
            RuntimeError("404 NOT_FOUND: model does not exist"),
            FakeGeminiInteraction("fallback response"),
        ]

        self.assertEqual(provider.generate("hi", 0.2), "fallback response")
        self.assertEqual(
            provider._client.interactions.create.call_args_list,
            [
                call(
                    model="invalid-model", input="hi", generation_config={"temperature": 0.2}
                ),
                call(
                    model="gemini-3.5-flash", input="hi", generation_config={"temperature": 0.2}
                ),
            ],
        )

    async def test_transient_failure_then_success_recovers(self):
        provider = MagicMock()
        provider.generate.side_effect = [RuntimeError("temporary network blip"), "ok after retry"]
        with patch("llm._provider_from_env", return_value=[provider]), patch(
            "asyncio.sleep", new=AsyncMock()
        ):
            result = await llm.generate_text("hi", temperature=0.2, retries=2)
        self.assertEqual(result, "ok after retry")
        self.assertEqual(provider.generate.call_count, 2)

    async def test_persistent_failure_raises_after_exhausting_retries(self):
        provider = MagicMock()
        provider.generate.side_effect = RuntimeError("boom")
        with patch("llm._provider_from_env", return_value=[provider]), patch(
            "asyncio.sleep", new=AsyncMock()
        ):
            with self.assertRaisesRegex(llm.LLMProviderError, "Upstream diagnostic: boom"):
                await llm.generate_text("hi", temperature=0.2, retries=1)
        self.assertEqual(provider.generate.call_count, 2)

    async def test_quota_error_raises_typed_exception_and_fails_fast(self):
        provider = MagicMock()
        provider.generate.side_effect = RuntimeError("429 RESOURCE_EXHAUSTED")
        with patch("llm._provider_from_env", return_value=[provider]), patch(
            "asyncio.sleep", new=AsyncMock()
        ) as sleep_mock:
            with self.assertRaises(llm.LLMQuotaError):
                await llm.generate_text("hi", temperature=0.2, retries=3)
        # Fail-fast: a quota error should not burn the whole retry/backoff budget.
        sleep_mock.assert_not_called()
        self.assertEqual(provider.generate.call_count, 1)

    async def test_failover_to_second_provider(self):
        failing = MagicMock()
        failing.generate.side_effect = RuntimeError("gemini down")
        succeeding = MagicMock()
        succeeding.generate.return_value = "openai ok"
        with patch("llm._provider_from_env", return_value=[failing, succeeding]), patch(
            "asyncio.sleep", new=AsyncMock()
        ):
            result = await llm.generate_text("hi", temperature=0.2, retries=0)
        self.assertEqual(result, "openai ok")

    async def test_empty_response_text_is_treated_as_failure(self):
        provider = MagicMock()
        provider.generate.side_effect = RuntimeError("empty response")
        with patch("llm._provider_from_env", return_value=[provider]), patch(
            "asyncio.sleep", new=AsyncMock()
        ):
            with self.assertRaises(llm.LLMProviderError):
                await llm.generate_text("hi", temperature=0.2, retries=0)


if __name__ == "__main__":
    unittest.main()
