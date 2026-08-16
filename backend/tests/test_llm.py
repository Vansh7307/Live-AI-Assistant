import unittest
from unittest.mock import AsyncMock, MagicMock, call, patch

import llm


class FakeGeminiResponse:
    def __init__(self, text):
        self.text = text


class FakeGeminiModel:
    def __init__(self, name, methods):
        self.name = name
        self.supported_generation_methods = methods


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
        fake_client.models.generate_content.return_value = FakeGeminiResponse("ok")
        with patch.dict(
            "os.environ",
            {"GOOGLE_API_KEY": "test-key", "LLM_PROVIDERS": "gemini"},
            clear=False,
        ):
            with patch("llm._GeminiProvider") as mock_provider:
                mock_provider.return_value.generate.return_value = "ok"
                result = await llm.generate_text("hi", temperature=0.2)
        self.assertEqual(result, "ok")

    def test_gemini_uses_first_priority_model_by_default(self):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}, clear=True), patch(
            "llm._GeminiProvider"
        ) as mock_provider:
            llm._provider_from_env()
        mock_provider.assert_called_once_with("test-key", "gemini-2.5-flash")

    def test_gemini_discovers_generate_content_models_after_primary_models(self):
        client = MagicMock()
        client.models.list.return_value = [
            FakeGeminiModel("models/gemini-2.5-flash", ["generateContent"]),
            FakeGeminiModel("models/key-enabled-model", ["generateContent"]),
            FakeGeminiModel("models/embedding-only", ["embedContent"]),
        ]

        models = llm.get_working_models(client)

        self.assertEqual(models[: len(llm.PRIMARY_MODELS)], llm.PRIMARY_MODELS)
        self.assertEqual(models[-1], "key-enabled-model")
        self.assertNotIn("embedding-only", models)

    def test_gemini_model_discovery_failure_keeps_primary_models(self):
        client = MagicMock()
        client.models.list.side_effect = RuntimeError("listing unavailable")

        self.assertEqual(llm.get_working_models(client), llm.PRIMARY_MODELS)

    def test_gemini_generate_uses_generate_content_api(self):
        provider = object.__new__(llm._GeminiProvider)
        provider._client = MagicMock()
        provider._model = "gemini-2.5-flash"
        provider._client.models.generate_content.return_value = FakeGeminiResponse("ok")

        self.assertEqual(provider.generate("hi", 0.2), "ok")
        provider._client.models.generate_content.assert_called_once_with(
            model="gemini-2.5-flash",
            contents="hi",
            config={"temperature": 0.2},
        )

    def test_gemini_stream_uses_generate_content_stream_text(self):
        provider = object.__new__(llm._GeminiProvider)
        provider._client = MagicMock()
        provider._model = "gemini-2.5-flash"
        provider._client.models.generate_content_stream.return_value = [
            FakeGeminiResponse("hello"),
            FakeGeminiResponse(""),
        ]

        self.assertEqual(list(provider.stream("hi", 0.2)), ["hello"])
        provider._client.models.generate_content_stream.assert_called_once_with(
            model="gemini-2.5-flash",
            contents="hi",
            config={"temperature": 0.2},
        )

    def test_gemini_retries_model_not_found_with_compatible_fallback(self):
        provider = object.__new__(llm._GeminiProvider)
        provider._client = MagicMock()
        provider._model = "invalid-model"
        provider._client.models.generate_content.side_effect = [
            RuntimeError("404 NOT_FOUND: model does not exist"),
            FakeGeminiResponse("fallback response"),
        ]

        self.assertEqual(provider.generate("hi", 0.2), "fallback response")
        self.assertEqual(
            provider._client.models.generate_content.call_args_list,
            [
                call(
                    model="invalid-model", contents="hi", config={"temperature": 0.2}
                ),
                call(
                    model="gemini-2.5-flash", contents="hi", config={"temperature": 0.2}
                ),
            ],
        )

    def test_gemini_stream_retries_model_not_found_with_next_fallback(self):
        provider = object.__new__(llm._GeminiProvider)
        provider._client = MagicMock()
        provider._model = "invalid-model"
        provider._client.models.generate_content_stream.side_effect = [
            RuntimeError("404 NOT_FOUND: model does not exist"),
            [FakeGeminiResponse("fallback stream")],
        ]

        self.assertEqual(list(provider.stream("hi", 0.2)), ["fallback stream"])
        self.assertEqual(
            provider._client.models.generate_content_stream.call_args_list,
            [
                call(
                    model="invalid-model", contents="hi", config={"temperature": 0.2}
                ),
                call(
                    model="gemini-2.5-flash", contents="hi", config={"temperature": 0.2}
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
