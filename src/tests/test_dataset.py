import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import BIAS_CLASSES, load_embeddings, record_label  # noqa: E402


class DatasetTests(unittest.TestCase):
    def test_bias_label_mapping_merges_extreme_classes(self):
        self.assertEqual(record_label({"bias_rating": "EXTREME LEFT"}, "bias"), "left")
        self.assertEqual(record_label({"bias_rating": "RIGHT"}, "bias"), "right")
        self.assertIsNone(record_label({"bias_rating": "UNRATED"}, "bias"))
        self.assertEqual(
            BIAS_CLASSES,
            ("left", "left-center", "center", "right-center", "right"),
        )

    def test_embedding_loader_rejects_pickle_backed_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.npz"
            np.savez(
                path,
                embeddings=np.ones((1, 2), dtype=np.float32),
                keys=np.asarray(["outlet"], dtype=object),
            )
            with self.assertRaises(ValueError):
                load_embeddings(path)

    def test_embedding_loader_accepts_aligned_unicode_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "safe.npz"
            np.savez(
                path,
                embeddings=np.asarray([[3.0, 4.0]], dtype=np.float32),
                keys=np.asarray(["outlet"], dtype=str),
            )
            result = load_embeddings(path)
            self.assertEqual(result.index.tolist(), ["outlet"])
            np.testing.assert_allclose(result.iloc[0].to_numpy(), [0.6, 0.8])


if __name__ == "__main__":
    unittest.main()
