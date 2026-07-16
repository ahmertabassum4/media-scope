import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

MGM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MGM_DIR))

import train_svm
from extract_wiki import (
    CACHE_SCHEMA,
    api_get,
    cache_compatible,
    match_outlet,
    select_wiki_pages,
    selected_keys,
)
from train_svm import require_complete_wiki_features


class GroupCPipelineTests(unittest.TestCase):
    def test_api_client_honors_wikimedia_maxlag_retry_after(self):
        class Response:
            status_code = 200

            def __init__(self, payload, headers=None):
                self.payload = payload
                self.headers = headers or {}

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        lagged = Response({"error": {"code": "maxlag"}}, {"Retry-After": "7"})
        ready = Response({"query": {}})
        fake_session = type("Session", (), {"get": lambda self, *args, **kwargs: None})()
        with patch.object(fake_session, "get", side_effect=[lagged, ready]), \
             patch("extract_wiki.session", return_value=fake_session), \
             patch("extract_wiki.throttle_requests"), \
             patch("extract_wiki.time.sleep") as sleep:
            data = api_get("https://example.test", {"action": "query"}, tries=2)

        self.assertEqual(data, {"query": {}})
        sleep.assert_called_once_with(7.0)

    def test_wikidata_official_domain_match_is_selected(self):
        outlet = {
            "key": "cnn",
            "media_name": "CNN",
            "url": "https://cnn.com",
            "split": "train",
        }
        with patch("extract_wiki.search_candidates", return_value=["Wrong page", "CNN"]), \
             patch("extract_wiki.titles_to_qids", return_value={"Wrong page": "Q1", "CNN": "Q2"}), \
             patch("extract_wiki.official_domains", return_value={"Q1": {"wrong.example"}, "Q2": {"cnn.com"}}), \
             patch("extract_wiki.page_text", return_value="CNN is a news organization."):
            rec = match_outlet(outlet, strict=True)

        self.assertTrue(rec["found"])
        self.assertEqual(rec["matched_title"], "CNN")
        self.assertEqual(rec["qid"], "Q2")
        self.assertEqual(rec["match_reason"], "wikidata_domain")
        self.assertEqual(rec["schema"], CACHE_SCHEMA)

    def test_exact_name_fallback_is_disabled_in_strict_mode(self):
        outlet = {
            "key": "example-news",
            "media_name": "Example News",
            "url": "https://example-news.invalid",
            "split": "train",
        }
        patches = (
            patch("extract_wiki.search_candidates", return_value=["Example News"]),
            patch("extract_wiki.titles_to_qids", return_value={"Example News": "Q1"}),
            patch("extract_wiki.official_domains", return_value={"Q1": set()}),
            patch("extract_wiki.page_text", return_value="Example News is a newspaper."),
        )
        with patches[0], patches[1], patches[2], patches[3]:
            fallback = match_outlet(outlet, strict=False)
        with patches[0], patches[1], patches[2], patches[3]:
            strict = match_outlet(outlet, strict=True)

        self.assertTrue(fallback["found"])
        self.assertEqual(fallback["match_reason"], "name_verified")
        self.assertFalse(strict["found"])
        self.assertIsNone(strict["match_reason"])

    def test_cache_policy_and_selection_are_deterministic(self):
        rec = {"schema": CACHE_SCHEMA, "strict": False}
        self.assertTrue(cache_compatible(rec, strict=False))
        self.assertFalse(cache_compatible(rec, strict=True))

        outlets = {"c": {}, "a": {}, "b": {}}
        args = SimpleNamespace(keys="c,a", limit=1)
        self.assertEqual(selected_keys(outlets, args), ["a"])

    def test_wikipedia_page_deduplication_prefers_test_outlet(self):
        cache = {
            "train-alias": {
                "found": True,
                "split": "train",
                "qid": "Q1",
                "matched_title": "Example News",
                "text": "Example News is a newspaper.",
            },
            "test-alias": {
                "found": True,
                "split": "test",
                "qid": "Q1",
                "matched_title": "Example News",
                "text": "Example News is a newspaper.",
            },
            "unique": {
                "found": True,
                "split": "train",
                "qid": "Q2",
                "matched_title": "Unique News",
                "text": "Unique News is a magazine.",
            },
        }
        keys, summary = select_wiki_pages(sorted(cache), cache)
        self.assertEqual(keys, ["test-alias", "unique"])
        self.assertEqual(summary["duplicate_components"], 1)
        self.assertEqual(summary["duplicate_pages_removed"], 1)

    def test_svm_rejects_partial_wikipedia_features(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "feats_wiki.npz"
            np.savez_compressed(
                path,
                keys=np.array(["outlet"]),
                wiki=np.zeros((1, 768), dtype=np.float32),
                wiki_feature_schema=np.array("mgm_wiki_features_v3"),
                wiki_complete=np.array(False),
            )
            with np.load(path, allow_pickle=False) as archive:
                with self.assertRaises(SystemExit):
                    require_complete_wiki_features(archive, path)

            np.savez_compressed(
                path,
                keys=np.array(["outlet"]),
                wiki=np.zeros((1, 768), dtype=np.float32),
                wiki_feature_schema=np.array("mgm_wiki_features_v3"),
                wiki_complete=np.array(True),
            )
            with np.load(path, allow_pickle=False) as archive:
                require_complete_wiki_features(archive, path)

    def test_feature_loader_uses_safe_numpy_deserialization(self):
        keys = np.array(["first", "second"])
        content_filter = {"schema": "test-filter", "fingerprint": "test-fingerprint"}
        with tempfile.TemporaryDirectory() as temp_dir:
            feature_dir = Path(temp_dir)
            common = {
                "keys": keys,
                "content_filter_schema": np.array(content_filter["schema"]),
                "content_filter_fingerprint": np.array(content_filter["fingerprint"]),
            }
            np.savez_compressed(
                feature_dir / "feats_factuality.npz",
                bert_repr=np.ones((2, 2), dtype=np.float32),
                bert_prob=np.ones((2, 3), dtype=np.float32),
                **common,
            )
            np.savez_compressed(
                feature_dir / "nela_features.npz",
                nela=np.ones((2, 4), dtype=np.float32),
                # The loader never reads this legacy object array, so it must
                # remain safe to load the archive with allow_pickle=False.
                names=np.array(["nela_one", "nela_two"], dtype=object),
                **common,
            )
            np.savez_compressed(
                feature_dir / "feats_wiki.npz",
                keys=keys,
                wiki=np.ones((2, 768), dtype=np.float32),
                wiki_feature_schema=np.array("mgm_wiki_features_v3"),
                wiki_complete=np.array(True),
            )

            sets, index = train_svm.load_feature_sets("factuality", feature_dir, content_filter)

        self.assertEqual(set(sets), {"bert_repr", "bert_prob", "nela", "wiki"})
        self.assertEqual(sets["wiki"].shape, (2, 768))
        self.assertEqual(index, {"first": 0, "second": 1})

    def test_svm_adds_all_group_c_rows_when_wiki_is_available(self):
        class FakeClassifier:
            classes_ = np.array(["LOW", "MIXED", "HIGH"])

            def predict(self, X):
                return np.full(len(X), "LOW", dtype=object)

            def predict_proba(self, X):
                return np.full((len(X), 3), 1 / 3, dtype=float)

        keys = ["a", "b", "c", "d", "e", "f"]
        outlets = {
            key: {"label_3class": label}
            for key, label in zip(keys, ["LOW", "MIXED", "HIGH", "LOW", "MIXED", "HIGH"])
        }
        sets = {
            "bert_repr": np.ones((len(keys), 2), dtype=np.float32),
            "bert_prob": np.ones((len(keys), 3), dtype=np.float32),
            "wiki": np.ones((len(keys), 4), dtype=np.float32),
        }
        index = {key: i for i, key in enumerate(keys)}
        roles = {"bert_tune": [], "svm_train": keys[:3], "test": keys[3:]}

        def fake_run_svm(X_train, y_train, X_test, grid):
            return FakeClassifier(), X_test, 0.5, {"gamma": 0.1, "C": 1.0}

        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "results.csv"
            with patch.object(train_svm, "load_outlets", return_value=outlets), \
                 patch.object(train_svm, "build_content_filter", return_value={"summary": {}, "schema": "s", "fingerprint": "f"}), \
                 patch.object(train_svm, "load_roles", return_value=roles), \
                 patch.object(train_svm, "load_feature_sets", return_value=(sets, index)), \
                 patch.object(train_svm, "run_svm", side_effect=fake_run_svm), \
                 patch.object(sys, "argv", ["train_svm.py", "--task", "factuality", "--out", str(out)]), \
                 contextlib.redirect_stdout(io.StringIO()):
                train_svm.main()

            rows = pd.read_csv(out)

        self.assertEqual(len(rows), 8)
        self.assertIn("Wikipedia: BERT", rows["experiment"].tolist())
        self.assertIn("Articles+Wikipedia (c)", rows["experiment"].tolist())
        self.assertIn("Articles+Wikipedia (en)", rows["experiment"].tolist())
        article_concat = rows.loc[rows["experiment"] == "Articles: ALL (c)", "components"].item()
        wiki_concat = rows.loc[rows["experiment"] == "Articles+Wikipedia (c)", "components"].item()
        self.assertIn("bert_repr+bert_prob", article_concat)
        self.assertIn("bert_repr+bert_prob+wiki", wiki_concat)


if __name__ == "__main__":
    unittest.main()
