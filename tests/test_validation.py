import unittest
import warnings

import pandas as pd

from validation import encode_binary_target, make_seen_project_mask, make_time_series_cv


class TimeSeriesCVTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "project_start_year": [-1, 2018, 2019, 2020, 2020, 2022, 2022, 2022],
                "project_name": ["ignored", "A", "A", "B", "A", "B", None, "C"],
                "target": [0, 0, 1, 1, 0, 1, 0, 1],
            },
            index=[101, 110, 125, 130, 145, 160, 175, 185],
        )

    def test_latest_observed_years_and_expanding_window(self):
        folds, diagnostics = make_time_series_cv(self.df, n_valid_years=3)

        self.assertEqual(diagnostics["validation_year"].tolist(), [2019, 2020, 2022])
        self.assertEqual(diagnostics["train_size"].tolist(), [1, 2, 4])
        self.assertEqual(diagnostics["validation_size"].tolist(), [1, 2, 3])
        self.assertEqual(diagnostics["seen_count"].tolist(), [1, 1, 1])
        self.assertEqual(diagnostics["unseen_count"].tolist(), [0, 1, 2])
        self.assertEqual(diagnostics["is_latest_fold"].tolist(), [False, False, True])

        for train_idx, valid_idx in folds:
            train_years = self.df.loc[train_idx, "project_start_year"]
            valid_years = self.df.loc[valid_idx, "project_start_year"]
            self.assertLess(train_years.max(), valid_years.iloc[0])
            self.assertNotIn(-1, train_years.tolist())
            self.assertNotIn(-1, valid_years.tolist())

    def test_seen_mask_uses_only_fold_training_rows(self):
        folds, _ = make_time_series_cv(self.df, n_valid_years=3)
        train_idx, valid_idx = folds[1]  # validation year 2020

        seen = make_seen_project_mask(self.df, train_idx, valid_idx)

        self.assertEqual(seen.index.tolist(), [130, 145])
        self.assertEqual(seen.tolist(), [False, True])

    def test_non_unique_index_is_rejected(self):
        duplicate_index_df = self.df.copy()
        duplicate_index_df.index = [1, 1, 2, 3, 4, 5, 6, 7]

        with self.assertRaisesRegex(ValueError, "index must be unique"):
            make_time_series_cv(duplicate_index_df)

    def test_fold_without_past_training_is_skipped_with_warning(self):
        one_year = pd.DataFrame(
            {
                "project_start_year": [2022, 2022],
                "project_name": ["A", "B"],
                "target": [0, 1],
            },
            index=[20, 40],
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            folds, diagnostics = make_time_series_cv(one_year, n_valid_years=1)

        self.assertEqual(folds, [])
        self.assertTrue(diagnostics.empty)
        self.assertTrue(any("training data is empty" in str(w.message) for w in caught))

    def test_encode_binary_target_maps_competition_labels_and_is_idempotent(self):
        raw = pd.Series(
            ["該当", "非該当", " 該当 ", 1, "0", True],
            index=[3, 7, 11, 20, 21, 30],
            name="science_tech_decision",
        )
        encoded = encode_binary_target(raw)
        self.assertEqual(encoded.tolist(), [1, 0, 1, 1, 0, 1])
        self.assertEqual(encoded.index.tolist(), raw.index.tolist())
        self.assertEqual(encoded.name, "science_tech_decision")
        self.assertEqual(str(encoded.dtype), "int8")

    def test_encode_binary_target_rejects_unknown_or_missing_labels(self):
        with self.assertRaisesRegex(ValueError, "unknown labels"):
            encode_binary_target(pd.Series(["該当", "対象外"], name="label"))
        with self.assertRaisesRegex(ValueError, "missing"):
            encode_binary_target(pd.Series(["該当", None], name="label"))


if __name__ == "__main__":
    unittest.main()
