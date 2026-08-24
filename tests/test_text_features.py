import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from text_features import (
    compare_cv_results,
    cross_validate_text_columns,
    fit_full_text_model_predict,
    normalize_text,
)
from validation import make_time_series_cv


class TextFeaturesTest(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame(
            {
                "project_start_year": [2018, 2018, 2019, 2019, 2020, 2020, 2022, 2022],
                "project_name": ["量子研究", "観光支援", "量子開発", "給付事業", "半導体研究", "観光振興", "量子通信", "地域給付"],
                "project_objective": ["AI 技術", "地域 支援", "量子 技術", "生活 支援", "5G 技術", "観光 支援", "6G 技術", "給付 支援"],
                "project_summary": ["科学 開発", "旅行 事業", "研究 開発", "行政 事業", "科学 実証", "旅行 振興", "科学 通信", "行政 給付"],
                "target": [1, 0, 1, 0, 1, 0, 1, 0],
            },
            index=[10, 20, 31, 41, 52, 62, 73, 83],
        )
        self.folds, _ = make_time_series_cv(self.train, n_valid_years=2)
        self.tfidf_params = {
            "ngram_range": (2, 3),
            "min_df": 1,
            "max_features": 1_000,
        }

    def test_normalize_text(self):
        source = pd.Series(["ＡＩ\t研究\n  5G", np.nan])
        self.assertEqual(normalize_text(source).tolist(), ["AI 研究 5G", ""])

    def test_cv_saves_sparse_features_and_oof(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = cross_validate_text_columns(
                df=self.train,
                folds=self.folds,
                text_cols=["project_name", "project_summary"],
                tfidf_params=self.tfidf_params,
                feature_output_dir=temp_dir,
                model_name="combined",
            )

            self.assertEqual(result["oof"].notna().sum(), 4)
            self.assertTrue(result["oof"].loc[[10, 20, 31, 41]].isna().all())
            self.assertEqual(len(result["fold_scores"]), 2)

            output = Path(temp_dir) / "combined"
            self.assertTrue((output / "oof_predictions.csv.gz").exists())
            self.assertTrue((output / "fold_scores.csv").exists())
            self.assertTrue(
                (output / "fold_0_year_2020" / "train_features.csv.gz").exists()
            )
            feature_rows = pd.read_csv(
                output / "fold_0_year_2020" / "train_features.csv.gz"
            )
            self.assertEqual(
                feature_rows.columns.tolist(),
                ["row_index", "feature_index", "value"],
            )

    def test_full_fit_does_not_fit_on_test_text(self):
        test = pd.DataFrame(
            {
                "project_name": ["テスト限定語"],
                "project_objective": ["未知目的"],
                "project_summary": ["未知概要"],
            },
            index=[999],
        )
        result = fit_full_text_model_predict(
            train_df=self.train,
            test_df=test,
            text_cols=["project_name"],
            tfidf_params=self.tfidf_params,
            feature_output_dir=None,
        )

        vocabulary = set(result["vectorizers"]["project_name"].get_feature_names_out())
        self.assertNotIn("限定", vocabulary)
        self.assertEqual(result["test_pred"].index.tolist(), [999])

    def test_comparison_table(self):
        result = cross_validate_text_columns(
            df=self.train,
            folds=self.folds,
            text_cols=["project_name"],
            tfidf_params=self.tfidf_params,
            feature_output_dir=None,
            model_name="name_model",
        )
        comparison = compare_cv_results([result])
        self.assertIn("fold_2020", comparison.columns)
        self.assertIn("fold_2022", comparison.columns)
        self.assertIn("latest_fold_score", comparison.columns)


if __name__ == "__main__":
    unittest.main()
