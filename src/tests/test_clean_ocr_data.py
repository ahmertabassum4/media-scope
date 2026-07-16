import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clean_ocr_data import clean_rows, validate_clean_rows  # noqa: E402


def row(index, split, key, factuality, bias, **identifiers):
    defaults = {
        "outlet_key": key,
        "ocr": f"ocr-{index}",
        "features": f"features-{index}",
        "screenshot": f"screenshot-{index}",
        "embedding": f"embedding-{index}",
    }
    defaults.update(identifiers)
    return {
        "row_id": index,
        "split": split,
        "source_index": index,
        "key": key,
        "media_name": key,
        "image_file": f"{key}.png",
        "factuality": factuality,
        "bias": bias,
        "identifiers": defaults,
        "csv_record": {},
    }


class CleanOcrDataTests(unittest.TestCase):
    def test_same_label_duplicate_keeps_test_copy(self):
        rows = [
            row(0, "train", "train-copy", "HIGH", "center", ocr="shared"),
            row(1, "test", "test-copy", "HIGH", "center", ocr="shared"),
        ]
        clean, dropped, components = clean_rows(rows)
        self.assertEqual([item["key"] for item in clean], ["test-copy"])
        self.assertEqual(dropped[0]["reason"], "duplicate_model_input")
        self.assertEqual(dropped[0]["canonical_key"], "test-copy")
        self.assertEqual(components[0]["match_types"], ["ocr"])

    def test_conflicting_labels_drop_complete_connected_component(self):
        rows = [
            row(0, "train", "one", "HIGH", "center", embedding="shared"),
            row(1, "train", "two", "LOW", "right", embedding="shared"),
        ]
        clean, dropped, components = clean_rows(rows)
        self.assertEqual(clean, [])
        self.assertEqual(len(dropped), 2)
        self.assertTrue(all(item["reason"] == "conflicting_labels" for item in dropped))
        self.assertEqual(components[0]["conflicting_tasks"], ["bias", "factuality"])

    def test_missing_ocr_values_do_not_form_a_duplicate_group(self):
        rows = [
            row(0, "train", "one", "HIGH", "center", ocr=None),
            row(1, "train", "two", "LOW", "right", ocr=None),
            row(2, "test", "three", "MIXED", "left", ocr=None),
        ]
        clean, dropped, components = clean_rows(rows)
        self.assertEqual(len(clean), 3)
        self.assertEqual(dropped, [])
        self.assertEqual(components, [])
        validate_clean_rows(clean)


if __name__ == "__main__":
    unittest.main()
