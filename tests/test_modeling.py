import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from modeling import (
    S1_TFIDF_SVD_MLP,
    S2_TFIDF_SVD_XGB,
    _mlp_monitor_improved,
    _mlp_validation_statistics,
    build_experiment_summary,
    default_modeling_config,
    fit_full_and_predict_test,
    fit_full_tfidf_svd_nonlinear_and_predict_test,
    prepare_tabular_features,
    run_all_experiments,
    run_e1_embedding_lr,
    run_e5_tabular_catboost,
    run_e6_embedding_tabular_xgboost,
    run_tfidf_lr_experiments,
    run_tfidf_svd_nonlinear_experiments,
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
                        "project_objective": f"目的{target}",
                        "project_summary": f"概要{year}。詳細{target}",
                        "current_issues": None if position == 2 else f"課題{target}",
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

    def test_text_statistics_are_deterministic_row_features(self):
        frame = pd.DataFrame(
            {
                "project_name": ["ＡI\n研究。次!", None],
                "project_objective": ["", "目的"],
                "project_summary": [None, "概要です。"],
                "current_issues": ["", "課題123"],
            },
            index=[10, 30],
        )
        prepared, numeric, categorical = prepare_tabular_features(
            frame,
            numeric_cols=[],
            categorical_cols=[],
            add_project_duration=False,
            add_log1p_budget=False,
            add_text_statistics=True,
            text_cols=[
                "project_name",
                "project_objective",
                "project_summary",
                "current_issues",
            ],
        )
        self.assertEqual(categorical, [])
        self.assertEqual(prepared.loc[10, "project_name__char_count"], 8.0)
        self.assertEqual(prepared.loc[10, "project_name__line_count"], 2.0)
        self.assertEqual(prepared.loc[10, "project_name__sentence_count"], 2.0)
        self.assertAlmostEqual(prepared.loc[10, "project_name__latin_ratio"], 0.25)
        self.assertAlmostEqual(prepared.loc[10, "project_name__kanji_ratio"], 0.375)
        self.assertAlmostEqual(
            prepared.loc[10, "project_name__punctuation_ratio"], 0.25
        )
        self.assertEqual(prepared.loc[10, "text__nonempty_count"], 1.0)
        self.assertAlmostEqual(prepared.loc[30, "current_issues__digit_ratio"], 0.6)
        self.assertTrue(np.isfinite(prepared[numeric].to_numpy()).all())

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
                    "run_t1": False,
                    "run_t2": False,
                    "run_t3": False,
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

    def test_auc_is_default_for_tree_and_mlp_early_stopping(self):
        config = default_modeling_config()
        self.assertEqual(config["catboost"]["loss_function"], "Logloss")
        self.assertEqual(config["catboost"]["eval_metric"], "AUC")
        self.assertEqual(config["xgboost"]["objective"], "binary:logistic")
        self.assertEqual(config["xgboost"]["eval_metric"], "auc")
        self.assertEqual(config["mlp"]["early_stop_metric"], "auc")

    def test_mlp_auc_monitor_maximizes_auc_while_loss_monitor_minimizes_loss(self):
        monitor, auc = _mlp_validation_statistics(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            0.7,
            "auc",
        )
        self.assertEqual(monitor, 1.0)
        self.assertEqual(auc, 1.0)
        self.assertTrue(_mlp_monitor_improved(0.8, 0.7, "auc", 1e-6))
        self.assertFalse(_mlp_monitor_improved(0.6, 0.7, "auc", 1e-6))
        self.assertTrue(_mlp_monitor_improved(0.6, 0.7, "loss", 1e-6))
        self.assertFalse(_mlp_monitor_improved(0.8, 0.7, "loss", 1e-6))

    def test_mlp_auc_monitor_rejects_one_class_validation(self):
        with self.assertRaisesRegex(ValueError, "both target classes"):
            _mlp_validation_statistics(
                np.array([1, 1]),
                np.array([0.2, 0.8]),
                0.7,
                "auc",
            )

    def test_e5_catboost_uses_native_categorical_features(self):
        fit_cat_features = []
        model_params = []

        class FakeCatBoostClassifier:
            def __init__(self, **params):
                self.params = params
                model_params.append(params)

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
        self.assertTrue(all(params["eval_metric"] == "AUC" for params in model_params))
        self.assertTrue(all(params["loss_function"] == "Logloss" for params in model_params))

    def test_e6_optional_pca_runs_inside_each_fold(self):
        model_params = []

        class FakeXGBClassifier:
            def __init__(self, **params):
                self.params = params
                model_params.append(params)
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
        self.assertTrue(all(params["eval_metric"] == "auc" for params in model_params))
        self.assertTrue(all(params["objective"] == "binary:logistic" for params in model_params))

    def test_t1_t2_t3_share_one_tfidf_fit_per_fold_and_stay_sparse(self):
        from text_features import fit_transform_tfidf_columns

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "modeling.fit_transform_tfidf_columns",
                wraps=fit_transform_tfidf_columns,
            ) as transform:
                results = run_tfidf_lr_experiments(
                    df=self.train,
                    folds=self.folds,
                    text_cols=["project_name"],
                    numeric_cols=["project_start_year", "budget"],
                    categorical_cols=["responsible_ministry"],
                    embeddings=self.embeddings,
                    embedding_metadata=self.metadata,
                    tfidf_config={
                        "ngram_range": (2, 3),
                        "min_df": 1,
                        "max_features": 1_000,
                    },
                    tfidf_lr_config={"dual": False, "tol": 1e-2},
                    tfidf_feature_output_dir=directory,
                )
            self.assertEqual(transform.call_count, len(self.folds))
            self.assertEqual(
                list(results),
                [
                    "T1_tfidf_lr",
                    "T2_tfidf_tabular_lr",
                    "T3_tfidf_embedding_tabular_lr",
                ],
            )
            validation_labels = pd.Index(
                np.concatenate([valid_idx.to_numpy() for _, valid_idx in self.folds])
            )
            old_labels = self.train.index.difference(validation_labels)
            for result in results.values():
                self.assertTrue(result.oof.loc[old_labels].isna().all())
                self.assertTrue(result.oof.loc[validation_labels].notna().all())
                self.assertEqual(result.fold_metrics["validation_year"].tolist(), [2020, 2021, 2022])
                self.assertTrue(result.metadata["features_are_sparse"])
                self.assertTrue((result.fold_metrics["sparse_memory_mib"] > 0).all())
            self.assertTrue(
                (Path(directory) / "fold_0_year_2020" / "train_features.csv.gz").exists()
            )
            self.assertTrue(results["T1_tfidf_lr"].fold_metrics["seen_score"].isna().all())

    def test_t3_rejects_embedding_metadata_misalignment(self):
        reordered = self.metadata.iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "project_id order mismatch"):
            run_tfidf_lr_experiments(
                df=self.train,
                folds=self.folds,
                text_cols=["project_name"],
                numeric_cols=["project_start_year", "budget"],
                categorical_cols=["responsible_ministry"],
                embeddings=self.embeddings,
                embedding_metadata=reordered,
                run_t1=False,
                run_t2=False,
                run_t3=True,
                tfidf_feature_output_dir=None,
            )

    def test_s1_s2_share_fold_tfidf_svd_and_preserve_oof_labels(self):
        from text_features import fit_transform_tfidf_columns

        class FakeXGBClassifier:
            def __init__(self, **params):
                self.params = params
                self.best_iteration = 2

            def fit(self, x, y, **kwargs):
                del x, kwargs
                self.probability = float(np.mean(y))
                return self

            def predict_proba(self, x):
                positive = np.full(len(x), self.probability)
                return np.column_stack([1 - positive, positive])

        def fake_mlp(x_train, y_train, x_valid, y_valid, config, random_state):
            del x_train, y_valid, config, random_state
            prediction = np.full(len(x_valid), float(np.mean(y_train)), dtype=np.float32)
            return prediction, {
                "fit_seconds": 0.01,
                "predict_seconds": 0.01,
                "input_dim": x_valid.shape[1],
                "device": "cpu",
                "best_iteration": 1,
                "early_stop_metric": "auc",
                "best_validation_loss": 0.7,
                "best_validation_auc": 0.5,
            }

        fake_module = types.ModuleType("xgboost")
        fake_module.XGBClassifier = FakeXGBClassifier
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.dict("sys.modules", {"xgboost": fake_module}),
                patch(
                    "modeling.fit_transform_tfidf_columns",
                    wraps=fit_transform_tfidf_columns,
                ) as tfidf_transform,
                patch(
                    "modeling._fit_transform_tfidf_svd",
                    wraps=__import__("modeling")._fit_transform_tfidf_svd,
                ) as svd_transform,
                patch("modeling._fit_predict_torch_mlp", side_effect=fake_mlp),
            ):
                results = run_tfidf_svd_nonlinear_experiments(
                    self.train,
                    self.folds,
                    ["project_name"],
                    self.embeddings,
                    self.metadata,
                    ["project_start_year", "budget"],
                    ["responsible_ministry"],
                    tfidf_config={"ngram_range": (2, 3), "min_df": 1},
                    svd_config={"n_components": 2, "n_iter": 3},
                    mlp_config={"device": "cpu"},
                    xgboost_config={"device": "cpu", "n_estimators": 5},
                    svd_feature_output_dir=directory,
                )
            self.assertEqual(tfidf_transform.call_count, len(self.folds))
            self.assertEqual(svd_transform.call_count, len(self.folds))
            self.assertTrue(
                (Path(directory) / "fold_0_year_2020/train_svd_features.csv.gz").exists()
            )
        self.assertEqual(set(results), {S1_TFIDF_SVD_MLP, S2_TFIDF_SVD_XGB})
        validation_labels = pd.Index(
            np.concatenate([valid_idx.to_numpy() for _, valid_idx in self.folds])
        )
        old_labels = self.train.index.difference(validation_labels)
        for result in results.values():
            self.assertTrue(result.oof.loc[old_labels].isna().all())
            self.assertTrue(result.oof.loc[validation_labels].notna().all())
            self.assertEqual(result.fold_metrics["svd_dim"].tolist(), [2, 2, 2])
            self.assertTrue(result.metadata["raw_tfidf_was_dense"] is False)

    def test_svd_experiment_rejects_dimension_above_fold_limit(self):
        with self.assertRaisesRegex(ValueError, "fold limit"):
            run_tfidf_svd_nonlinear_experiments(
                self.train,
                self.folds,
                ["project_name"],
                self.embeddings,
                self.metadata,
                ["project_start_year", "budget"],
                ["responsible_ministry"],
                run_s1=True,
                run_s2=False,
                tfidf_config={"ngram_range": (2, 3), "min_df": 1},
                svd_config={"n_components": 256},
                svd_feature_output_dir=None,
            )

    def test_full_s1_s2_fit_train_only_and_predict_unknown_category(self):
        from text_features import fit_transform_tfidf_columns

        test = self.train.iloc[:3].drop(columns="target").copy()
        test.index = [900, 901, 902]
        test["project_id"] = ["t1", "t2", "t3"]
        test["project_name"] = ["test only alpha", "test only beta", "test only gamma"]
        test["responsible_ministry"] = "UNKNOWN_MINISTRY"
        test_embeddings = self.embeddings[:3]
        test_metadata = pd.DataFrame(
            {
                "split": "test",
                "original_index": test.index,
                "project_id": test["project_id"].to_numpy(),
            }
        )

        class FakeXGBClassifier:
            def __init__(self, **params):
                self.params = params

            def fit(self, x, y, **kwargs):
                del x, kwargs
                self.probability = float(np.mean(y))
                return self

            def predict_proba(self, x):
                positive = np.full(len(x), self.probability)
                return np.column_stack([1 - positive, positive])

        def fake_mlp(x_train, y_train, x_valid, y_valid, config, random_state):
            del x_train, y_valid, config, random_state
            prediction = np.full(len(x_valid), float(np.mean(y_train)), dtype=np.float32)
            return prediction, {}

        fake_module = types.ModuleType("xgboost")
        fake_module.XGBClassifier = FakeXGBClassifier
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.dict("sys.modules", {"xgboost": fake_module}),
                patch(
                    "modeling.fit_transform_tfidf_columns",
                    wraps=fit_transform_tfidf_columns,
                ) as transform,
                patch("modeling._fit_predict_torch_mlp", side_effect=fake_mlp),
            ):
                predictions = fit_full_tfidf_svd_nonlinear_and_predict_test(
                    self.train,
                    test,
                    [S1_TFIDF_SVD_MLP, S2_TFIDF_SVD_XGB],
                    ["project_name"],
                    self.embeddings,
                    test_embeddings,
                    self.metadata,
                    test_metadata,
                    ["project_start_year", "budget"],
                    ["responsible_ministry"],
                    config={
                        "text_cols": ["project_name"],
                        "tfidf": {"ngram_range": (2, 3), "min_df": 1},
                        "tfidf_svd": {"n_components": 2, "n_iter": 3},
                        "mlp": {"device": "cpu", "full_epochs": 1},
                        "xgboost": {"device": "cpu", "n_estimators": 5},
                    },
                    svd_feature_output_dir=directory,
                )
            train_arg, test_arg = transform.call_args.args[:2]
            self.assertTrue(train_arg.index.equals(self.train.index))
            self.assertTrue(test_arg.index.equals(test.index))
            self.assertTrue((Path(directory) / "train_svd_features.csv.gz").exists())
            self.assertTrue((Path(directory) / "test_svd_features.csv.gz").exists())
        self.assertEqual(set(predictions), {S1_TFIDF_SVD_MLP, S2_TFIDF_SVD_XGB})
        for prediction in predictions.values():
            self.assertEqual(prediction.shape, (3,))
            self.assertTrue(((prediction >= 0) & (prediction <= 1)).all())

    def test_run_all_adds_t1_t2_t3_to_unified_oof(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = run_all_experiments(
                train=self.train,
                folds=self.folds,
                train_embeddings=self.embeddings,
                train_embedding_metadata=self.metadata,
                numeric_cols=["project_start_year", "budget"],
                categorical_cols=["responsible_ministry"],
                config={
                    "run_e1": False,
                    "run_e2": False,
                    "run_e3": False,
                    "run_e4": False,
                    "run_e5": False,
                    "run_e6": False,
                    "text_cols": ["project_name"],
                    "tfidf": {
                        "ngram_range": (2, 3),
                        "min_df": 1,
                        "max_features": 1_000,
                    },
                    "tfidf_feature_output_dir": None,
                    "tfidf_lr": {"dual": False, "tol": 1e-2},
                    "output_dir": directory,
                },
            )
        self.assertEqual(
            suite.oof_predictions.columns.tolist(),
            [
                "T1_tfidf_lr",
                "T2_tfidf_tabular_lr",
                "T3_tfidf_embedding_tabular_lr",
            ],
        )
        self.assertEqual(len(suite.fold_metrics), 9)

    def test_final_t3_fits_full_train_and_predicts_unknown_test_category(self):
        test = self.train.iloc[:3].drop(columns="target").copy()
        test.index = [900, 901, 902]
        test["project_id"] = ["t1", "t2", "t3"]
        test["project_name"] = ["テスト限定語A", "テスト限定語B", "テスト限定語C"]
        test["responsible_ministry"] = "UNKNOWN_MINISTRY"
        test_embeddings = self.embeddings[:3]
        test_metadata = pd.DataFrame(
            {
                "split": "test",
                "original_index": test.index,
                "project_id": test["project_id"].to_numpy(),
            }
        )
        prediction = fit_full_and_predict_test(
            experiment="T3_tfidf_embedding_tabular_lr",
            train=self.train,
            test=test,
            train_embeddings=self.embeddings,
            test_embeddings=test_embeddings,
            train_embedding_metadata=self.metadata,
            test_embedding_metadata=test_metadata,
            numeric_cols=["project_start_year", "budget"],
            categorical_cols=["responsible_ministry"],
            config={
                "text_cols": ["project_name"],
                "tfidf": {
                    "ngram_range": (2, 3),
                    "min_df": 1,
                    "max_features": 1_000,
                },
                "tfidf_lr": {"dual": False, "tol": 1e-2},
            },
        )
        self.assertEqual(prediction.shape, (3,))
        self.assertTrue(((prediction >= 0) & (prediction <= 1)).all())

    def test_final_t1_t2_do_not_require_embeddings(self):
        test = self.train.iloc[:3].drop(columns="target").copy()
        test.index = [900, 901, 902]
        test["project_id"] = ["t1", "t2", "t3"]
        test["responsible_ministry"] = "UNKNOWN_MINISTRY"
        config = {
            "text_cols": ["project_name"],
            "tfidf": {
                "ngram_range": (2, 3),
                "min_df": 1,
                "max_features": 1_000,
            },
            "tfidf_lr": {"dual": False, "tol": 1e-2},
        }
        for experiment in ("T1_tfidf_lr", "T2_tfidf_tabular_lr"):
            with self.subTest(experiment=experiment):
                prediction = fit_full_and_predict_test(
                    experiment=experiment,
                    train=self.train,
                    test=test,
                    train_embeddings=None,
                    test_embeddings=None,
                    train_embedding_metadata=None,
                    test_embedding_metadata=None,
                    numeric_cols=["project_start_year", "budget"],
                    categorical_cols=["responsible_ministry"],
                    config=config,
                )
                self.assertEqual(prediction.shape, (3,))
                self.assertTrue(((prediction >= 0) & (prediction <= 1)).all())

    def test_tfidf_experiments_forbid_target_as_text(self):
        with self.assertRaisesRegex(ValueError, "target_col"):
            run_tfidf_lr_experiments(
                df=self.train,
                folds=self.folds,
                text_cols=["project_name", "target"],
                numeric_cols=[],
                categorical_cols=[],
                run_t1=True,
                run_t2=False,
                run_t3=False,
                tfidf_feature_output_dir=None,
            )

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
