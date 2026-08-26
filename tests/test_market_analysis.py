import unittest

import numpy as np
import pandas as pd

from market_analysis import (
    DEFAULT_WEIGHT_PROFILES,
    STRATEGIC_WEIGHTS,
    build_theme_long,
    compare_weight_profiles,
    compute_theme_metrics,
    compute_yearly_theme_metrics,
    concentration_hhi,
    merge_project_taxonomy,
    score_theme_metrics,
)


class MarketAnalysisTests(unittest.TestCase):
    def source_frames(self):
        train = pd.DataFrame(
            {
                "project_id": ["p1", "old", "shared"],
                "project_start_year": [2020, 2019, 2021],
                "responsible_ministry": ["M1", "M0", "M2"],
            },
            index=[10, 20, 30],
        )
        test = pd.DataFrame(
            {
                "project_id": ["shared", "p3"],
                "project_start_year": [2022, 2022],
                "responsible_ministry": ["M3", None],
            },
            index=[40, 50],
        )
        features = pd.DataFrame(
            {
                "source_split": ["train", "train", "test", "test"],
                "project_id": ["p1", "shared", "shared", "p3"],
                "project_start_year": [2020, 2021, 2022, 2022],
                "policy_domain": ["D01", "D01", "D02", "D02"],
                "admin_process": ["A01", "A01", "A03", "A09"],
                "ai_usecase": ["U04|U05", "U04", "U05", "U03"],
                "ai_applicability": [2, 1, 2, 0],
            }
        )
        return train, test, features

    def project_frame(self):
        return pd.DataFrame(
            {
                "project_key": ["train::p1", "train::p2", "test::p1", "test::p3", "test::p4"],
                "project_start_year": [2020, 2021, 2022, 2022, 2022],
                "responsible_ministry": ["M1", "M2", "M2", "M1", None],
                "policy_domain": ["D01", "D01", "D02", "D02", "D03"],
                "admin_process": ["A01", "A01", "A01", "A09", "A02"],
                "ai_usecase": ["U04|U05", "U04", "U04", "U03", "U99"],
                "ai_applicability": [2, 1, 2, 0, 0],
            }
        )

    def test_merge_uses_split_and_project_id_and_filters_old_year(self):
        train, test, features = self.source_frames()
        merged, diagnostics = merge_project_taxonomy(train, test, features)

        self.assertEqual(len(merged), 4)
        self.assertEqual(merged["project_key"].nunique(), 4)
        self.assertIn("train::shared", set(merged["project_key"]))
        self.assertIn("test::shared", set(merged["project_key"]))
        self.assertNotIn("train::old", set(merged["project_key"]))
        self.assertIn("responsible_ministry", merged.columns)
        self.assertEqual(diagnostics["train_rows_eligible"], 2)
        self.assertEqual(diagnostics["ministry_missing_rows"], 1)
        self.assertTrue(merged["project_start_year"].ge(2020).all())

    def test_merge_rejects_incomplete_taxonomy(self):
        train, test, features = self.source_frames()
        with self.assertRaisesRegex(ValueError, "source_only=1"):
            merge_project_taxonomy(train, test, features.iloc[:-1])

    def test_explode_creates_unique_project_theme_rows(self):
        theme_long = build_theme_long(self.project_frame())
        self.assertEqual(len(theme_long), 6)
        p1 = theme_long.loc[theme_long["project_key"].eq("train::p1")]
        self.assertEqual(set(p1["ai_usecase"]), {"U04", "U05"})
        self.assertFalse(
            theme_long.duplicated(["project_key", "admin_process", "ai_usecase"]).any()
        )

    def test_explode_rejects_duplicate_usecase_in_one_project(self):
        projects = self.project_frame().iloc[[0]].copy()
        projects["ai_usecase"] = "U04|U04"
        with self.assertRaisesRegex(ValueError, "same theme more than once"):
            build_theme_long(projects)

    def test_hhi_excludes_missing_and_matches_hand_calculation(self):
        values = pd.Series(["M1", "M1", "M2", None])
        self.assertAlmostEqual(concentration_hhi(values), (2 / 3) ** 2 + (1 / 3) ** 2)
        self.assertTrue(np.isnan(concentration_hhi(pd.Series([None, None]))))

    def test_metrics_match_hand_calculation_and_fit_rules(self):
        metrics = compute_theme_metrics(build_theme_long(self.project_frame()))
        a01_u04 = metrics.set_index("theme_code").loc["A01__U04"]
        self.assertEqual(a01_u04["project_count"], 3)
        self.assertEqual(a01_u04["high_app_project_count"], 2)
        self.assertAlmostEqual(a01_u04["high_app_rate"], 2 / 3)
        # Across non-U99 theme assignments the High-app prior is 3 / 5.
        self.assertAlmostEqual(a01_u04["high_app_rate_smoothed"], (2 + 20 * 0.6) / 23)
        self.assertEqual(a01_u04["ministry_breadth"], 2)
        self.assertEqual(a01_u04["domain_breadth"], 2)
        self.assertAlmostEqual(a01_u04["ministry_hhi"], 0.5)
        self.assertAlmostEqual(a01_u04["domain_hhi"], 0.5)
        self.assertAlmostEqual(a01_u04["concentration_hhi"], 0.5)
        self.assertEqual(a01_u04["aiu_fit"], 5)
        self.assertTrue(a01_u04["ranking_eligible"])

        u99 = metrics.set_index("theme_code").loc["A02__U99"]
        self.assertEqual(u99["high_app_project_count"], 0)
        self.assertEqual(u99["aiu_fit"], 2)
        self.assertFalse(u99["ranking_eligible"])

        zero_high = metrics.set_index("theme_code").loc["A09__U03"]
        self.assertFalse(zero_high["ranking_eligible"])

    def test_high_app_missing_ministry_is_excluded_from_breadth_and_hhi(self):
        projects = self.project_frame().iloc[[0]].copy()
        projects["responsible_ministry"] = None
        metrics = compute_theme_metrics(build_theme_long(projects)).set_index("theme_code")
        row = metrics.loc["A01__U04"]
        self.assertEqual(row["ministry_breadth"], 0)
        self.assertTrue(np.isnan(row["ministry_hhi"]))
        self.assertEqual(row["domain_breadth"], 1)
        self.assertEqual(row["domain_hhi"], 1)
        self.assertEqual(row["concentration_hhi"], 1)
        self.assertEqual(row["high_app_ministry_missing_rate"], 1)

    def test_scoring_excludes_zero_high_app_and_applies_weights(self):
        metrics = compute_theme_metrics(build_theme_long(self.project_frame()))
        scored = score_theme_metrics(metrics, weights=STRATEGIC_WEIGHTS)
        eligible = scored.loc[scored["ranking_eligible"]]
        ineligible = scored.loc[~scored["ranking_eligible"]]

        self.assertTrue(eligible["overall_score"].between(0, 100).all())
        self.assertEqual(eligible["overall_rank"].min(), 1)
        self.assertTrue(ineligible["overall_score"].isna().all())
        self.assertTrue(ineligible["overall_rank"].isna().all())

        row = eligible.iloc[0]
        expected = sum(
            row[f"{name}_score"] * weight
            for name, weight in STRATEGIC_WEIGHTS.items()
        )
        self.assertAlmostEqual(row["overall_score"], expected)

    def test_profile_comparison_has_score_and_rank_for_each_profile(self):
        metrics = compute_theme_metrics(build_theme_long(self.project_frame()))
        comparison = compare_weight_profiles(metrics)
        for profile in DEFAULT_WEIGHT_PROFILES:
            self.assertIn(f"{profile}_score", comparison)
            self.assertIn(f"{profile}_rank", comparison)

    def test_yearly_metrics_count_only_ai_applicability_two_as_high(self):
        yearly = compute_yearly_theme_metrics(build_theme_long(self.project_frame()))
        a01_u04_2021 = yearly.loc[
            yearly["theme_code"].eq("A01__U04")
            & yearly["project_start_year"].eq(2021)
        ].iloc[0]
        self.assertEqual(a01_u04_2021["project_count"], 1)
        self.assertEqual(a01_u04_2021["high_app_project_count"], 0)
        self.assertEqual(a01_u04_2021["high_app_rate"], 0)


if __name__ == "__main__":
    unittest.main()
