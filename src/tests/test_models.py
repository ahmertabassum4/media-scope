import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import models  # noqa: E402


FACTUALITY_CLASSES = ("LOW", "MIXED", "HIGH")
BIAS_CLASSES = ("left", "left-center", "center", "right-center", "right")


class _FakeEstimator:
    classes_ = np.asarray(FACTUALITY_CLASSES, dtype=object)

    def fit(self, X, y):
        return self

    def decision_function(self, X):
        return np.zeros((len(X), len(self.classes_)), dtype=float)


class ModelTests(unittest.TestCase):
    def test_feature_oof_tunes_inside_each_outer_fold(self):
        X = pd.DataFrame({"x": np.arange(18, dtype=float)})
        y = pd.Series(FACTUALITY_CLASSES * 6)
        tune_sizes = []

        def fake_tune(
            fold_X, fold_y, inner_splits, class_order, selection_label=None, seed=0,
        ):
            tune_sizes.append(len(fold_y))
            return {"kind": "logreg"}

        with patch.object(models, "_tune", side_effect=fake_tune), patch.object(
            models, "_estimator", return_value=_FakeEstimator()
        ):
            oof, test = models.feature_expert(
                X, y, X.iloc[:2], inner_splits=3,
                class_order=FACTUALITY_CLASSES, selection_label="MIXED",
            )

        self.assertEqual(tune_sizes, [12, 12, 12, 18])
        self.assertEqual(oof.shape, (18, len(FACTUALITY_CLASSES)))
        self.assertEqual(test.shape, (2, len(FACTUALITY_CLASSES)))

    def test_aligned_scores_supports_five_bias_classes(self):
        estimator = _FakeEstimator()
        estimator.classes_ = np.asarray(["left", "center", "right"], dtype=object)
        scores = models.aligned_scores(estimator, np.zeros((2, 1)), BIAS_CLASSES)
        self.assertEqual(scores.shape, (2, 5))
        np.testing.assert_array_equal(scores[:, [1, 3]], 0.0)


if __name__ == "__main__":
    unittest.main()
