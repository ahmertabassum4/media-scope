import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

MGM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MGM_DIR))

from extract_features import (
    matching_nela_cache,
    mean_wordpiece_hidden,
    require_matching_model_content_filter,
    truncate_nela_text,
)
from finetune_bert import remove_training_content_seen_in_validation
from pipeline import (
    DEFAULT_BERT_FRAC,
    article_body_prefix_hash,
    article_text_hash,
    article_url_key,
    build_content_filter,
    build_roles,
    filtered_articles,
    split_fine_tune_outlets,
    validate_roles,
)


def synthetic_outlets():
    outlets = {}
    for split, per_label in (("train", 30), ("test", 6)):
        for label in ("LOW", "MIXED", "HIGH"):
            for i in range(per_label):
                key = f"{split}-{label.lower()}-{i}"
                outlets[key] = {
                    "key": key,
                    "split": split,
                    "label_3class": label,
                    "bias_rating": "LEAST BIASED",
                    "articles": [{"text": f"{key} article text"}],
                }
    return outlets


class GroupAPipelineTests(unittest.TestCase):
    def setUp(self):
        self.outlets = synthetic_outlets()

    def test_one_third_roles_are_complete_and_disjoint(self):
        roles = build_roles("factuality", self.outlets)
        self.assertEqual(roles["bert_frac"], DEFAULT_BERT_FRAC)
        self.assertEqual(len(roles["bert_tune"]), 30)
        self.assertEqual(len(roles["svm_train"]), 60)
        self.assertEqual(len(roles["test"]), 18)
        self.assertTrue(validate_roles("factuality", roles, self.outlets))

    def test_outlet_level_validation_is_disjoint(self):
        roles = build_roles("factuality", self.outlets)
        train_keys, val_keys = split_fine_tune_outlets(
            "factuality", roles, self.outlets, val_frac=0.2, seed=42)
        self.assertFalse(set(train_keys) & set(val_keys))
        self.assertFalse(set(train_keys) & set(roles["svm_train"]))
        self.assertFalse(set(val_keys) & set(roles["test"]))
        self.assertTrue(all(self.outlets[key]["label_3class"] != "MIXED"
                            for key in train_keys + val_keys))

    def test_invalid_role_overlap_is_rejected(self):
        roles = build_roles("factuality", self.outlets)
        broken = copy.deepcopy(roles)
        broken["svm_train"].append(broken["bert_tune"][0])
        with self.assertRaises(ValueError):
            validate_roles("factuality", broken, self.outlets)

    def test_wordpiece_mean_excludes_special_and_padding_tokens(self):
        hidden = torch.tensor([[[10.0], [2.0], [4.0], [20.0], [99.0]]])
        attention = torch.tensor([[1, 1, 1, 1, 0]])
        special = torch.tensor([[1, 0, 0, 1, 1]])
        mean = mean_wordpiece_hidden(hidden, attention, special)
        self.assertTrue(torch.equal(mean, torch.tensor([[3.0]])))

    def test_validation_text_is_removed_from_training(self):
        train_texts = ["unique train", "Shared   Article", "URL-only collision", "other train"]
        labels = ["LOW", "LOW", "HIGH", "HIGH"]
        kept_texts, kept_labels, kept_urls, removed = remove_training_content_seen_in_validation(
            train_texts, labels, ["one", "two", "shared-url", "four"],
            ["shared article", "different validation article"], ["unused", "shared-url"])
        self.assertEqual(removed, 2)
        self.assertEqual(kept_texts, ["unique train", "other train"])
        self.assertEqual(kept_labels, ["LOW", "HIGH"])
        self.assertEqual(kept_urls, ["one", "four"])
        self.assertNotIn(article_text_hash("shared article"),
                         {article_text_hash(text) for text in kept_texts})

    def test_content_filter_keeps_test_copy_and_removes_other_duplicates(self):
        shared_body = " ".join(f"word{index}" for index in range(90))
        outlets = {
            "outlet-a": {
                "split": "train",
                "articles": [
                    {"text": "Shared, story!", "url": "https://source-a.example/reprint"},
                    {"text": "URL source one", "url": "https://www.shared.example/story?id=7&utm_source=x"},
                    {"text": "Unique A", "url": "https://a.example/unique"},
                    {"text": "Repeated in one outlet", "url": "https://a.example/repeated"},
                    {"text": "Repeated in one outlet", "url": "https://a.example/repeated"},
                    {"text": f"Headline A\n{shared_body} ending-a", "url": "https://a.example/body-copy"},
                ],
            },
            "outlet-b": {
                "split": "test",
                "articles": [
                    {"text": "shared story", "url": "https://source-b.example/reprint"},
                    {"text": "URL source two", "url": "https://shared.example/story?fbclid=x&id=7"},
                    {"text": "Unique B", "url": "https://b.example/unique"},
                    {"text": f"Headline B\n{shared_body} ending-b", "url": "https://b.example/body-copy"},
                ],
            },
        }
        content_filter = build_content_filter(outlets)
        left = filtered_articles(outlets, "outlet-a", content_filter)
        right = filtered_articles(outlets, "outlet-b", content_filter)

        self.assertEqual([article["text"] for article in left], ["Unique A", "Repeated in one outlet"])
        self.assertEqual(
            [article["text"] for article in right],
            ["shared story", "URL source two", "Unique B", f"Headline B\n{shared_body} ending-b"],
        )
        self.assertEqual(content_filter["summary"]["cross_outlet_articles_removed"], 3)
        self.assertEqual(content_filter["summary"]["cross_outlet_body_prefix_groups"], 1)
        self.assertEqual(content_filter["summary"]["within_outlet_duplicates_removed"], 1)
        self.assertEqual(
            article_url_key("https://www.shared.example/story?id=7&utm_source=x"),
            article_url_key("https://shared.example/story?fbclid=x&id=7"),
        )
        self.assertEqual(
            article_url_key("http://shared.example/story?id=7"),
            article_url_key("https://shared.example/story?id=7"),
        )

        left_texts = {article_text_hash(article["text"]) for article in left}
        right_texts = {article_text_hash(article["text"]) for article in right}
        left_urls = {article_url_key(article["url"]) for article in left}
        right_urls = {article_url_key(article["url"]) for article in right}
        left_prefixes = {
            prefix for article in left
            if (prefix := article_body_prefix_hash(article["text"]))
        }
        right_prefixes = {
            prefix for article in right
            if (prefix := article_body_prefix_hash(article["text"]))
        }
        self.assertFalse(left_texts & right_texts)
        self.assertFalse(left_urls & right_urls)
        self.assertFalse(left_prefixes & right_prefixes)

    def test_feature_extraction_rejects_a_checkpoint_with_a_stale_filter(self):
        content_filter = build_content_filter(self.outlets)
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            split_path = model_dir / "finetune_split.json"
            split_path.write_text(json.dumps({
                "content_filter_schema": content_filter["schema"],
                "content_filter_fingerprint": content_filter["fingerprint"],
            }))
            require_matching_model_content_filter(model_dir, content_filter)

            split_path.write_text(json.dumps({
                "content_filter_schema": content_filter["schema"],
                "content_filter_fingerprint": "stale",
            }))
            with self.assertRaises(SystemExit):
                require_matching_model_content_filter(model_dir, content_filter)

    def test_nela_cache_requires_full_safe_keys(self):
        content_filter = build_content_filter(self.outlets)
        keys = sorted(self.outlets)
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            cache_path = out_dir / "nela_features.npz"
            np.savez_compressed(
                cache_path,
                keys=np.array(keys),
                nela=np.ones((len(keys), 3), dtype=np.float32),
                names=np.array(["feature"], dtype=str),
                content_filter_schema=np.array(content_filter["schema"]),
                content_filter_fingerprint=np.array(content_filter["fingerprint"]),
                nela_max_words=np.array(10_000),
            )
            self.assertTrue(matching_nela_cache(cache_path, keys, content_filter))
            with np.load(cache_path, allow_pickle=False) as archive:
                self.assertEqual(archive["names"].tolist(), ["feature"])

            unsafe = out_dir / "unsafe.npz"
            np.savez_compressed(
                unsafe,
                keys=np.array(keys),
                nela=np.ones((len(keys), 3), dtype=np.float32),
                names=np.array(["feature"], dtype=object),
                content_filter_schema=np.array(content_filter["schema"]),
                content_filter_fingerprint=np.array(content_filter["fingerprint"]),
                nela_max_words=np.array(10_000),
            )
            self.assertFalse(matching_nela_cache(unsafe, keys, content_filter))

            partial = out_dir / "partial.npz"
            np.savez_compressed(
                partial,
                keys=np.array(keys[:3]),
                nela=np.ones((3, 3), dtype=np.float32),
                names=np.array(["feature"], dtype=str),
                content_filter_schema=np.array(content_filter["schema"]),
                content_filter_fingerprint=np.array(content_filter["fingerprint"]),
                nela_max_words=np.array(10_000),
            )
            self.assertFalse(matching_nela_cache(partial, keys, content_filter))

    def test_nela_text_window_is_bounded_and_deterministic(self):
        self.assertEqual(truncate_nela_text("one  two\nthree four", 3), "one two three")
        self.assertEqual(truncate_nela_text("one two", 0), "one two")


if __name__ == "__main__":
    unittest.main()
