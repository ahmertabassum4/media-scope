import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import metrics  # noqa: E402
from dataset import TASKS  # noqa: E402
from src.train import OUTPUT_FILES  # noqa: E402


class MetricsTests(unittest.TestCase):
    def test_bias_outputs_do_not_overwrite_factuality_outputs(self):
        factuality = set(OUTPUT_FILES["factuality"].values())
        bias = set(OUTPUT_FILES["bias"].values())
        self.assertFalse(factuality & bias)

    def test_bias_scores_use_five_class_ordinal_mapping(self):
        classes = TASKS["bias"]["classes"]
        ordinal = TASKS["bias"]["ordinal"]
        truth = list(classes)
        predictions = ["left-center", "left-center", "center", "right", "right"]
        result = metrics.scores(
            "test", "Test", "test components", truth, predictions, classes, ordinal,
        )

        self.assertEqual(result["n_test"], 5)
        self.assertAlmostEqual(result["accuracy"], 60.0)
        self.assertAlmostEqual(result["mae"], 0.4)
        self.assertAlmostEqual(result["mse"], 0.4)
        self.assertTrue(all(f"{label}_f1" in result for label in classes))


if __name__ == "__main__":
    unittest.main()
