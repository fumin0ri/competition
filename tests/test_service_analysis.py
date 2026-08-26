import importlib.util
import unittest

import numpy as np
import pandas as pd

from market_analysis import build_theme_long
from service_analysis import (
    DOMAIN_OPPORTUNITY_WEIGHTS,
    build_service_opportunity_cards,
    compute_ministry_domain_opportunities,
    compute_theme_domain_metrics,
    compute_theme_domain_yearly_metrics,
    extract_distinctive_char_ngrams,
    score_theme_domain_metrics,
)


class ServiceAnalysisTests(unittest.TestCase):
    def theme_long(self):
        projects = pd.DataFrame(
            {
                "project_key": [f"train::p{i}" for i in range(1, 8)],
                "project_id": [f"p{i}" for i in range(1, 8)],
                "project_name": ["研究分類", "研究調査", "産業分類", "産業判定", "産業申請", "研究審査", "地域審査"],
                "project_objective": ["文書を分類する"] * 7,
                "project_summary": ["行政文書の判定を効率化する"] * 7,
                "current_issues": ["分類作業が多い", None, "産業データが多い", "判定に時間", "申請が大量", "審査負荷", "審査負荷"],
                "project_start_year": [2021, 2022, 2023, 2024, 2024, 2024, 2023],
                "responsible_ministry": ["M1", "M2", "M1", "M2", None, "M1", "M3"],
                "policy_domain": ["D01", "D01", "D02", "D02", "D02", "D01", "D13"],
                "admin_process": ["A01", "A01", "A01", "A01", "A01", "A03", "A03"],
                "ai_usecase": ["U04", "U04", "U04", "U04", "U04", "U05", "U05"],
                "ai_applicability": [2, 1, 2, 2, 2, 2, 2],
                "budget": [100, 200, 300, 100, -1, None, 50],
            }
        )
        return build_theme_long(projects)

    def test_domain_metrics_match_hand_calculation(self):
        metrics = compute_theme_domain_metrics(
            self.theme_long(), ["A01__U04"], recent_years=2, prior_strength=10
        ).set_index("policy_domain")
        d01 = metrics.loc["D01"]
        self.assertEqual(d01["project_count"], 2)
        self.assertEqual(d01["high_app_project_count"], 1)
        self.assertAlmostEqual(d01["high_app_rate"], 0.5)
        self.assertAlmostEqual(d01["theme_high_app_rate_prior"], 0.8)
        self.assertAlmostEqual(d01["high_app_rate_smoothed"], 9 / 12)
        self.assertEqual(d01["ministry_breadth"], 1)
        self.assertEqual(d01["ministry_hhi"], 1)
        self.assertEqual(d01["issue_coverage_rate"], 0.5)

        d02 = metrics.loc["D02"]
        self.assertEqual(d02["high_app_project_count"], 3)
        self.assertEqual(d02["ministry_breadth"], 2)
        self.assertAlmostEqual(d02["ministry_hhi"], 0.5)
        self.assertEqual(d02["high_app_budget_total"], 400)
        self.assertEqual(d02["high_app_budget_median"], 200)
        self.assertAlmostEqual(d02["high_app_budget_nonmissing_rate"], 2 / 3)
        self.assertAlmostEqual(d02["high_app_budget_hhi"], 0.625)
        self.assertEqual(d02["high_app_top3_budget_share"], 1)
        self.assertEqual(d02["recent_high_app_project_count"], 3)
        self.assertEqual(d02["recent_high_app_rate"], 1)

    def test_scoring_ranks_domains_within_each_theme(self):
        metrics = compute_theme_domain_metrics(
            self.theme_long(), ["A01__U04", "A03__U05"]
        )
        scored, effective = score_theme_domain_metrics(metrics)
        self.assertAlmostEqual(effective.sum(), 1)
        self.assertEqual(set(effective.index), set(DOMAIN_OPPORTUNITY_WEIGHTS))
        for _, group in scored.loc[scored["ranking_eligible"]].groupby("theme_code"):
            self.assertEqual(group["domain_rank_within_theme"].min(), 1)
        a01_top = scored.loc[
            scored["theme_code"].eq("A01__U04")
            & scored["domain_rank_within_theme"].eq(1),
            "policy_domain",
        ].iloc[0]
        self.assertEqual(a01_top, "D02")
        a03 = scored.loc[scored["theme_code"].eq("A03__U05")].set_index("policy_domain")
        self.assertEqual(a03.loc["D01", "high_app_budget_total_score"], 0)
        self.assertGreater(a03.loc["D13", "high_app_budget_total_score"], 0)

    def test_missing_budget_redistributes_weight(self):
        theme_long = self.theme_long().drop(columns="budget")
        metrics = compute_theme_domain_metrics(theme_long, ["A01__U04"])
        scored, effective = score_theme_domain_metrics(metrics)
        self.assertEqual(effective["high_app_budget_total"], 0)
        self.assertAlmostEqual(effective.sum(), 1)
        self.assertTrue(
            scored.loc[scored["ranking_eligible"], "domain_opportunity_score"].notna().all()
        )

    def test_yearly_and_ministry_outputs_keep_selected_scope(self):
        theme_long = self.theme_long()
        yearly = compute_theme_domain_yearly_metrics(theme_long, ["A01__U04"])
        self.assertEqual(set(yearly["theme_code"]), {"A01__U04"})
        d02_2024 = yearly.loc[
            yearly["policy_domain"].eq("D02")
            & yearly["project_start_year"].eq(2024)
        ].iloc[0]
        self.assertEqual(d02_2024["high_app_project_count"], 2)

        accounts = compute_ministry_domain_opportunities(theme_long, ["A01__U04"])
        self.assertFalse(accounts["responsible_ministry"].isna().any())
        self.assertEqual(set(accounts["theme_code"]), {"A01__U04"})
        self.assertIn("account_rank_within_theme_domain", accounts)

    def test_all_missing_ministries_return_an_empty_typed_account_table(self):
        theme_long = self.theme_long()
        theme_long["responsible_ministry"] = None
        accounts = compute_ministry_domain_opportunities(theme_long, ["A01__U04"])
        self.assertTrue(accounts.empty)
        self.assertIn("responsible_ministry", accounts.columns)
        self.assertIn("account_rank_within_theme_domain", accounts.columns)

    def test_service_cards_separate_evidence_and_hypothesis(self):
        metrics = compute_theme_domain_metrics(self.theme_long(), ["A01__U04"])
        scored, _ = score_theme_domain_metrics(metrics)
        cards = build_service_opportunity_cards(scored, top_domains_per_theme=1)
        self.assertEqual(len(cards), 1)
        card = cards.iloc[0]
        self.assertIn("分類・判定支援", card["recommended_offer"])
        self.assertIn("High-app=", card["observed_evidence"])
        self.assertEqual(card["catalog_rule_version"], "aiu_service_catalog_v1")

    def test_unknown_theme_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not observed"):
            compute_theme_domain_metrics(self.theme_long(), ["A99__U99"])

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn not installed")
    def test_distinctive_phrases_are_scoped_to_selected_pairs(self):
        theme_long = self.theme_long()
        pairs = pd.DataFrame(
            {"theme_code": ["A01__U04"], "policy_domain": ["D02"]}
        )
        phrases = extract_distinctive_char_ngrams(
            theme_long, pairs, top_n=5, min_df=1, ngram_range=(2, 3)
        )
        self.assertFalse(phrases.empty)
        self.assertEqual(set(phrases["theme_code"]), {"A01__U04"})
        self.assertEqual(set(phrases["policy_domain"]), {"D02"})


if __name__ == "__main__":
    unittest.main()
