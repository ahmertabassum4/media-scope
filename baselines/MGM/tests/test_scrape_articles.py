import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

MGM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MGM_DIR))

from scrape_articles import (  # noqa: E402
    booster_links,
    canonical_outlet_rows,
    compact_outlet_records,
    fetch,
)


class ScrapeArticlesTests(unittest.TestCase):
    def test_ssl_verification_is_not_bypassed_by_default(self):
        class Session:
            def __init__(self):
                self.calls = []

            def get(self, url, **kwargs):
                self.calls.append((url, kwargs))
                raise requests.exceptions.SSLError("certificate failure")

        session, notes = Session(), set()
        text, final_url = fetch(session, "https://example.test", timeout=1, notes=notes)

        self.assertIsNone(text)
        self.assertIsNone(final_url)
        self.assertEqual(notes, {"ssl_error"})
        self.assertEqual(len(session.calls), 1)
        self.assertNotIn("verify", session.calls[0][1])

    def test_insecure_ssl_requires_explicit_opt_in(self):
        class Response:
            status_code = 200
            text = "page body"
            url = "https://example.test/final"

        class Session:
            def __init__(self):
                self.calls = []

            def get(self, url, **kwargs):
                self.calls.append((url, kwargs))
                if len(self.calls) == 1:
                    raise requests.exceptions.SSLError("certificate failure")
                return Response()

        session, notes = Session(), set()
        text, final_url = fetch(
            session, "https://example.test", timeout=1, notes=notes, allow_insecure_ssl=True
        )

        self.assertEqual(text, "page body")
        self.assertEqual(final_url, "https://example.test/final")
        self.assertEqual(notes, {"ssl_insecure"})
        self.assertIs(session.calls[1][1]["verify"], False)

    def test_booster_never_fetches_an_off_domain_sitemap_index(self):
        calls = []

        def fake_fetch(session, url, timeout, notes=None, allow_insecure_ssl=False):
            calls.append(url)
            if url == "https://example.com/sitemap.xml":
                return (
                    "<loc>https://evil.example/attack.xml</loc>"
                    "<loc>https://example.com/news.xml</loc>",
                    url,
                )
            if url == "https://example.com/news.xml":
                return "<loc>https://example.com/article/long-story</loc>", url
            raise AssertionError(f"unexpected network request: {url}")

        with patch("scrape_articles.fetch", side_effect=fake_fetch):
            links = booster_links(
                object(), "https://example.com", {"example.com"}, pool=1, notes=set()
            )

        self.assertEqual(links, ["https://example.com/article/long-story"])
        self.assertNotIn("https://evil.example/attack.xml", calls)

    def test_compaction_keeps_the_same_richest_outlet_record_as_the_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "articles.jsonl"
            rows = [
                {"key": "one", "n_articles": 1, "status": "low_coverage"},
                {"key": "two", "n_articles": 2, "status": "ok"},
                {"key": "one", "n_articles": 3, "status": "ok"},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))

            self.assertEqual(compact_outlet_records(path), 1)
            compacted, raw_count = canonical_outlet_rows(path)

        self.assertEqual(raw_count, 2)
        self.assertEqual({row["key"]: row["n_articles"] for row in compacted}, {"one": 3, "two": 2})


if __name__ == "__main__":
    unittest.main()
