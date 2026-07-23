import time
import unittest
from unittest.mock import MagicMock, patch

from tools import web_search


class TavilySearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        web_search._CLIENT = None

    async def test_success_maps_results_to_expected_shape(self):
        fake_client = MagicMock()
        fake_client.search.return_value = {
            "results": [{"title": "T", "url": "https://x.test", "content": "snippet"}]
        }
        with patch("tools.web_search._get_client", return_value=fake_client):
            results = await web_search.tavily_search("q")
        self.assertEqual(results, [{"title": "T", "url": "https://x.test", "snippet": "snippet"}])

    async def test_provider_failure_degrades_to_empty_list(self):
        fake_client = MagicMock()
        fake_client.search.side_effect = RuntimeError("network down")
        with patch("tools.web_search._get_client", return_value=fake_client):
            results = await web_search.tavily_search("q")
        self.assertEqual(results, [])

    async def test_timeout_degrades_to_empty_list_instead_of_hanging(self):
        fake_client = MagicMock()

        def slow_search(*args, **kwargs):
            time.sleep(1)
            return {"results": []}

        fake_client.search.side_effect = slow_search
        with patch("tools.web_search._get_client", return_value=fake_client):
            results = await web_search.tavily_search("q", timeout_seconds=0.1)
        self.assertEqual(results, [])

    async def test_missing_api_key_raises_value_error(self):
        web_search._CLIENT = None
        with patch.dict("os.environ", {"TAVILY_API_KEY": ""}, clear=False):
            with self.assertRaises(ValueError):
                await web_search.tavily_search("q")


if __name__ == "__main__":
    unittest.main()
