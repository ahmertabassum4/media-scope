import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from baselines.text_only_llm.train import (  # noqa: E402
    add_model_input_fingerprints,
    deduplicate_partition,
    make_macro_f1_metric,
    remove_training_inputs_seen_in_test,
    validate_partitions,
)


class FakeTokenizer:
    model_max_length = 32

    def __call__(self, texts, truncation, max_length, add_special_tokens=True):
        if isinstance(texts, str):
            texts = [texts]
        input_ids = []
        for text in texts:
            token_ids = [ord(char) for char in text][:max_length]
            if add_special_tokens:
                token_ids = [101] + token_ids + [102]
            input_ids.append(token_ids[:max_length])
        return {"input_ids": input_ids}


def row(key, label, text, source_index):
    return {
        "key": key,
        "label": label,
        "text": text,
        "media_name": key,
        "image_path": f"{key}.png",
        "source_key": key,
        "source_index": source_index,
    }


class TextOnlyPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def fingerprint(self, rows):
        return add_model_input_fingerprints(rows, self.tokenizer, max_len=32)

    def test_train_dedup_removes_conflicting_labels_and_test_keeps_one_copy(self):
        train_rows = self.fingerprint([
            row("same-a", "left", "same input", 0),
            row("same-b", "left", "same input", 1),
            row("conflict-a", "left", "contradictory input", 2),
            row("conflict-b", "right", "contradictory input", 3),
            row("held-out-copy", "center", "held out input", 4),
            row("unique", "right", "unique train input", 5),
        ])
        test_rows = self.fingerprint([
            row("held-out", "center", "held out input", 0),
            row("held-out-duplicate", "center", "held out input", 1),
        ])

        clean_test, _, test_report = deduplicate_partition(
            test_rows, "test", reject_conflicting_labels=True
        )
        clean_train, train_dropped, train_report = deduplicate_partition(
            train_rows, "train"
        )
        clean_train, cross_dropped, cross_report = remove_training_inputs_seen_in_test(
            clean_train, clean_test
        )

        self.assertEqual([item["key"] for item in clean_test], ["held-out"])
        self.assertEqual({item["key"] for item in clean_train}, {"same-a", "unique"})
        self.assertEqual(test_report["duplicate_rows_removed"], 1)
        self.assertEqual(train_report["conflicting_label_groups"], 1)
        self.assertEqual(train_report["conflicting_rows_removed"], 2)
        self.assertEqual(cross_report["rows_removed"], 1)
        self.assertEqual(
            {item["reason"] for item in train_dropped + cross_dropped},
            {
                "duplicate_model_input",
                "conflicting_labels_for_identical_input",
                "matches_held_out_test_input",
            },
        )

    def test_test_partition_rejects_conflicting_labels_for_one_model_input(self):
        test_rows = self.fingerprint([
            row("one", "left", "same screenshot text", 0),
            row("two", "right", "same screenshot text", 1),
        ])
        with self.assertRaises(ValueError):
            deduplicate_partition(test_rows, "test", reject_conflicting_labels=True)

    def test_partition_validator_rejects_shared_model_input(self):
        rows = self.fingerprint([
            row("one", "left", "one input", 0),
            row("two", "right", "two input", 1),
        ])
        validate_partitions({"train": [rows[0]], "validation": [], "test": [rows[1]]})
        with self.assertRaises(ValueError):
            validate_partitions({"train": [rows[0]], "validation": [rows[0]], "test": []})

    def test_validation_metric_scores_absent_classes_as_zero(self):
        metric = make_macro_f1_metric(3)
        result = metric((
            np.array([[4.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            np.array([0, 1]),
        ))
        self.assertAlmostEqual(result["macro_f1"], 2.0 / 9.0)


if __name__ == "__main__":
    unittest.main()
