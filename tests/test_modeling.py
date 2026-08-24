import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from modeling import (
    build_experiment_summary,
    fit_full_and_predict_test,
    prepare_tabular_features,
    run_all_experiments,
    run_e1_embedding_lr,
    run_e5_tabular_catboost,
    run_e6_embedding_tabular_xgboost,
    validate_embedding_input,
)
from validation import make_time_series_cv


class ModelingTests(unittest.TestCase):
    def setUp(self):
        rows = []
        index = []
        position = 0
        for year in range(2017, 2023):
            for target in (0, 1):
                rows.append(
                    {
                        "project_id": f"p{position}",
                        "project_name": "継続事業" if target == 0 else f"新規{year}",
                        "project_start_year": year,
                        "project_end_year": year + 2,
                        "project_fiscal_year": year,
                        "budget": np.nan if position == 3 else 100 + position,
                        "responsible_ministry": "A" if year < 2021 else "B",
                        "target": target,
                    }
                )
                index.append(100 + position * 3)
                position += 1
        self.train = pd.DataFrame(rows, index=index)
        rng = np.random.default_rng(42)
        self.embeddings = rng.normal(size=(len(self.train), 8)).astype(np.float32)
        self.metadata = pd.DataFrame(
            {
                "split": "train",
                "original_index": self.train.index,
                "project_id": self.train["project_id"].to_numpy(),
            }
        )
        self.folds, _ = make_time_series_cv(
            self.train,
            year_col="project_start_year",
            project_col="project_name",
            target_col="target",
            n_valid_years=3,
        )

    def test_invalid_year_produces_missing_duration(self):
        frame = self.train.iloc[:2].copy()
        frame.loc[frame.index[0], "project_start_year"] = -1
        prepared, numeric, categorical = prepare_tabular_features(
            frame,
            numeric_cols=["project_start_year", "budget"],
            categorical_cols=["responsible_ministry"],
        )
        self.assertTrue(np.isnan(prepared.iloc[0]["project_start_year"]))
        self.assertTrue(np.isnan(prepared.iloc[0]["project_duration"]))
        self.assertIn("log1p_budget", numeric)
        self.assertEqual(categorical, ["responsible_ministry"])

    def test_embedding_alignment_rejects_reordered_metadata(self):
        reordered = self.metadata.iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "project_id order mismatch"):
            validate_embedding_input(
                self.train,
                self.embeddings,
                reordered,
                split="train",
            )

    def test_e1_preserves_nan_for_rows_never_in_validation(self):
        result = run_e1_embedding_lr(
            self.train,
            self.embeddings,
            self.metadata,
            self.folds,
        )
        validation_labels = pd.Index(
            np.concatenate([valid_idx.to_numpy() for _, valid_idx in self.folds])
        )
        old_labels = self.train.index.difference(validation_labels)
        self.assertTrue(result.oof.loc[old_labels].isna().all())
        self.assertTrue(result.oof.loc[validation_labels].notna().all())
        self.assertEqual(len(result.fold_metrics), 3)
        self.assertIn("seen_score", result.fold_metrics.columns)
        self.assertIn("unseen_score", result.fold_metrics.columns)

    def test_e1_e2_suite_saves_unified_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_all_experiments(
                train=self.train,
                folds=self.folds,
                train_embeddings=self.embeddings,
                train_embedding_metadata=self.metadata,
                numeric_cols=[
                    "project_start_year",
                    "project_end_year",
                    "project_fiscal_year",
                    "budget",
                ],
                categorical_cols=["responsible_ministry"],
                config={
                    "run_e3": False,
                    "run_e4": False,
                    "run_e5": False,
                    "run_e6": False,
                    "output_dir": directory,
                },
            )
            self.assertEqual(
                result.oof_predictions.columns.tolist(),
                ["E1_embedding_lr", "E2_embedding_tabular_lr"],
            )
            self.assertEqual(len(result.fold_metrics), 6)
            self.assertIn("latest", result.summary.columns)
            self.assertTrue((Path(directory) / "oof_predictions.parquet").exists())
            self.assertTrue((Path(directory) / "fold_metrics.parquet").exists())
            self.assertTrue((Path(directory) / "experiment_summary.csv").exists())

    def test_summary_uses_latest_validation_year(self):
        metrics = pd.DataFrame(
            {
                "experiment": ["x", "x"],
                "validation_year": [2022, 2020],
                "overall_score": [0.8, 0.6],
                "seen_score": [0.7, 0.5],
                "unseen_score": [0.9, 0.7],
                "fit_seconds": [2.0, 1.0],
                "predict_seconds": [0.2, 0.1],
            }
        )
        summary = build_experiment_summary(metrics)
        self.assertEqual(summary.loc["x", "latest"], 0.8)
        self.assertAlmostEqual(summary.loc["x", "mean"], 0.7)

    def test_e5_catboost_uses_native_categorical_features(self):
        fit_cat_features = []

        class FakeCatBoostClassifier:
            def __init__(self, **params):
                self.params = params

            def fit(self, x, y, cat_features, eval_set, use_best_model):
                del eval_set, use_best_model
                fit_cat_features.append(list(cat_features))
                self.probability = float(np.mean(y))
                self.columns = x.columns.tolist()
                return self

            def predict_proba(self, x):
                self.assert_columns = x.columns.tolist()
                positive = np.full(len(x), self.probability)
                return np.column_stack([1 - positive, positive])

            def get_best_iteration(self):
                return 7

        fake_module = types.ModuleType("catboost")
        fake_module.CatBoostClassifier = FakeCatBoostClassifier
        with patch.dict("sys.modules", {"catboost": fake_module}):
            result = run_e5_tabular_catboost(
                self.train,
                self.folds,
                numeric_cols=["project_start_year", "budget"],
                categorical_cols=["responsible_ministry"],
                catboost_config={"task_type": "CPU", "iterations": 10},
            )
        self.assertEqual(len(result.fold_metrics), 3)
        self.assertEqual(
            fit_cat_features,
            [["responsible_ministry"]] * 3,
        )
        self.assertTrue(result.oof.notna().sum() > 0)

    def test_e6_optional_pca_runs_inside_each_fold(self):
        class FakeXGBClassifier:
            def __init__(self, **params):
                self.params = params
                self.best_iteration = 4

            def fit(self, x, y, eval_set, verbose):
                del eval_set, verbose
                self.probability = float(np.mean(y))
                self.input_dim = x.shape[1]
                return self

            def predict_proba(self, x):
                positive = np.full(len(x), self.probability)
                return np.column_stack([1 - positive, positive])

        fake_module = types.ModuleType("xgboost")
        fake_module.XGBClassifier = FakeXGBClassifier
        with patch.dict("sys.modules", {"xgboost": fake_module}):
            result = run_e6_embedding_tabular_xgboost(
                self.train,
                self.embeddings,
                self.metadata,
                self.folds,
                numeric_cols=["project_start_year", "budget"],
                categorical_cols=["responsible_ministry"],
                pca_dim=2,
                xgboost_config={"device": "cpu", "n_estimators": 10},
            )
        self.assertEqual(result.name, "E6_embedding_tabular_xgb_pca2")
        self.assertTrue(
            result.fold_metrics["pca_explained_variance"].between(0, 1).all()
        )
        self.assertTrue((result.fold_metrics["input_dim"] > 2).all())

    def test_final_e1_does_not_require_tabular_columns(self):
        train = self.train[["project_id", "project_name", "target"]].copy()
        test = train.iloc[:3].drop(columns="target").copy()
        test.index = [900, 901, 902]
        test["project_id"] = ["t1", "t2", "t3"]
        test_embeddings = self.embeddings[:3]
        test_metadata = pd.DataFrame(
            {
                "split": "test",
                "original_index": test.index,
                "project_id": test["project_id"].to_numpy(),
            }
        )
        prediction = fit_full_and_predict_test(
            experiment="E1_embedding_lr",
            train=train,
            test=test,
            train_embeddings=self.embeddings,
            test_embeddings=test_embeddings,
            train_embedding_metadata=self.metadata,
            test_embedding_metadata=test_metadata,
            numeric_cols=[],
            categorical_cols=[],
        )
        self.assertEqual(prediction.shape, (3,))
        self.assertTrue(((prediction >= 0) & (prediction <= 1)).all())


if __name__ == "__main__":
    unittest.main()
