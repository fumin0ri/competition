import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble import (
    blend_test_predictions,
    evaluate_fold_auc,
    hill_climb_auc,
    make_submission,
    save_ensemble_outputs,
)


class EnsembleTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.Index([10, 20, 30, 40, 50, 60])
        self.target = pd.Series([0, 1, 0, 0, 1, 1], index=self.index)
        self.oof = pd.DataFrame(
            {
                "model_a": [np.nan, np.nan, 0.1, 0.7, 0.6, 0.9],
                "model_b": [np.nan, np.nan, 0.7, 0.1, 0.9, 0.6],
            },
            index=self.index,
        )
        self.folds = [
            (pd.Index([10, 20]), pd.Index([30, 50])),
            (pd.Index([10, 20, 30, 50]), pd.Index([40, 60])),
        ]

    def test_hill_climbing_improves_auc_and_preserves_unscored_nan(self):
        result = hill_climb_auc(
            self.oof,
            self.target,
            weight_grid=[0.05, 0.10, 0.15, 0.20],
        )
        self.assertAlmostEqual(result.score, 1.0)
        self.assertGreater(result.score, result.individual_scores.max())
        self.assertAlmostEqual(result.weights.sum(), 1.0)
        self.assertTrue((result.weights > 0).all())
        self.assertTrue(result.oof_prediction.loc[[10, 20]].isna().all())
        self.assertTrue(result.oof_prediction.loc[[30, 40, 50, 60]].notna().all())
        self.assertEqual(result.n_scored_rows, 4)

    def test_rejects_different_oof_masks(self):
        mismatched = self.oof.copy()
        mismatched.loc[20, "model_b"] = 0.4
        with self.assertRaisesRegex(ValueError, "same OOF rows"):
            hill_climb_auc(mismatched, self.target)

    def test_rejects_target_index_mismatch(self):
        with self.assertRaisesRegex(ValueError, "target index"):
            hill_climb_auc(self.oof, self.target.reset_index(drop=True))

    def test_evaluate_fold_auc_uses_label_indices_and_reports_year(self):
        result = hill_climb_auc(self.oof, self.target, weight_grid=[0.15])
        years = pd.Series([2019, 2019, 2020, 2021, 2020, 2021], index=self.index)
        scores = evaluate_fold_auc(
            result.oof_prediction,
            self.target,
            self.folds,
            years=years,
        )
        self.assertEqual(scores["validation_year"].tolist(), [2020, 2021])
        self.assertEqual(scores["n_valid"].tolist(), [2, 2])
        self.assertTrue((scores["roc_auc"] == 1.0).all())

    def test_mean_fold_auc_ignores_cross_year_prediction_ranges(self):
        result = hill_climb_auc(
            self.oof,
            self.target,
            folds=self.folds,
            objective="mean_fold_auc",
        )
        self.assertAlmostEqual(result.score, 1.0)
        self.assertAlmostEqual(result.individual_scores["model_a"], 1.0)
        self.assertAlmostEqual(result.pooled_auc, 0.75)
        self.assertTrue((result.fold_scores["roc_auc"] == 1.0).all())

    def test_weighted_fold_auc_normalizes_weights_and_rank_transforms_per_fold(self):
        result = hill_climb_auc(
            self.oof[["model_a"]],
            self.target,
            folds=self.folds,
            objective="weighted_fold_auc",
            fold_weights=[2, 3],
            blend_mode="rank",
        )
        np.testing.assert_allclose(result.fold_weights.to_numpy(), [0.4, 0.6])
        np.testing.assert_allclose(
            result.oof_prediction.loc[[30, 50, 40, 60]],
            [0.5, 1.0, 0.5, 1.0],
        )
        self.assertEqual(result.objective, "weighted_fold_auc")
        self.assertEqual(result.blend_mode, "rank")

    def test_fold_objective_requires_exact_fold_rows(self):
        incomplete_folds = [self.folds[0]]
        with self.assertRaisesRegex(ValueError, "exactly match"):
            hill_climb_auc(
                self.oof,
                self.target,
                folds=incomplete_folds,
                objective="mean_fold_auc",
            )

    def test_weighted_fold_auc_rejects_wrong_weight_count(self):
        with self.assertRaisesRegex(ValueError, "exactly 2"):
            hill_climb_auc(
                self.oof,
                self.target,
                folds=self.folds,
                objective="weighted_fold_auc",
                fold_weights=[1.0],
            )

    def test_save_ensemble_outputs_includes_objective_diagnostics(self):
        result = hill_climb_auc(
            self.oof,
            self.target,
            folds=self.folds,
            objective="weighted_fold_auc",
            fold_weights=[0.4, 0.6],
            blend_mode="rank",
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = save_ensemble_outputs(result, directory)
            self.assertTrue(all(path.exists() for path in paths.values()))
            summary = pd.read_csv(paths["summary"])
            fold_scores = pd.read_csv(paths["fold_scores"])
        self.assertEqual(summary.loc[0, "objective"], "weighted_fold_auc")
        self.assertEqual(summary.loc[0, "blend_mode"], "rank")
        self.assertEqual(len(fold_scores), 2)

    def test_blend_test_predictions_uses_named_normalized_weights(self):
        prediction = blend_test_predictions(
            {
                "model_a": np.array([0.2, 0.8]),
                "model_b": np.array([0.6, 0.4]),
            },
            {"model_a": 3.0, "model_b": 1.0, "unused": 0.0},
        )
        np.testing.assert_allclose(prediction, [0.3, 0.7])

    def test_rank_blend_ranks_each_test_model_before_weighting(self):
        prediction = blend_test_predictions(
            {
                "model_a": np.array([0.2, 0.8, 0.5]),
                "model_b": np.array([0.9, 0.1, 0.4]),
            },
            {"model_a": 1.0, "model_b": 1.0},
            blend_mode="rank",
        )
        np.testing.assert_allclose(prediction, [2 / 3, 2 / 3, 2 / 3])

    def test_make_submission_preserves_id_order_and_writes_two_columns(self):
        test = pd.DataFrame({"project_id": ["z", "a", "m"], "feature": [1, 2, 3]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            submission = make_submission(
                test,
                [0.2, 0.7, 0.4],
                id_col="project_id",
                prediction_col="target",
                output_path=path,
            )
            loaded = pd.read_csv(path)
        self.assertEqual(submission.columns.tolist(), ["project_id", "target"])
        self.assertEqual(submission["project_id"].tolist(), ["z", "a", "m"])
        pd.testing.assert_frame_equal(loaded, submission)

    def test_make_submission_rejects_duplicate_ids(self):
        test = pd.DataFrame({"project_id": ["a", "a"]})
        with self.assertRaisesRegex(ValueError, "unique"):
            make_submission(test, [0.2, 0.8], id_col="project_id")


if __name__ == "__main__":
    unittest.main()
