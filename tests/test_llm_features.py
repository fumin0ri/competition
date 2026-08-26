import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from llm_features import (
    BudgetExceededError,
    FeatureValidationError,
    build_project_text,
    decode_ai_usecase,
    filter_projects_by_start_year,
    generate_llm_features,
    load_llm_features,
    parse_feature_payload,
    validate_feature_payload,
)


VALID_FEATURES = {
    "policy_domain": "D02",
    "admin_process": "A03",
    "ai_usecase": ["U02", "U05"],
    "ai_applicability": 2,
    "classification_reason": "申請文書の確認と採択判断が中心であるため。",
}


def bedrock_response(payload, *, input_tokens=100, output_tokens=40):
    return {
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tool-1",
                            "name": "classify_administrative_project",
                            "input": payload,
                        }
                    }
                ]
            }
        },
        "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens},
    }


class FakeBedrockClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected Bedrock API call")
        return self.responses.pop(0)


class LLMFeatureTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "project_id": ["p1", "p2"],
                "project_name": ["中小企業支援", "行政調査"],
                "project_objective": ["申請企業を支援する", None],
                "project_summary": ["申請を審査して補助する", "統計を作成する"],
                "current_issues": [None, "情報収集に時間がかかる"],
            },
            index=pd.Index([10, 30]),
        )

    def test_build_project_text_labels_fields_and_handles_missing_values(self):
        text = build_project_text(self.df.iloc[0])
        self.assertIn("事業名:\n中小企業支援", text)
        self.assertIn("事業目的:\n申請企業を支援する", text)
        self.assertIn("現状・課題:\n[欠損]", text)

    def test_filter_projects_keeps_only_numeric_years_from_2020(self):
        frame = pd.DataFrame(
            {
                "project_start_year": [2019, "2020", 2022, -1, None, "unknown"],
                "value": list("abcdef"),
            },
            index=[10, 20, 30, 40, 50, 60],
        )
        original = frame.copy(deep=True)
        filtered = filter_projects_by_start_year(frame, min_year=2020)
        self.assertEqual(filtered.index.tolist(), [20, 30])
        self.assertEqual(filtered["value"].tolist(), ["b", "c"])
        pd.testing.assert_frame_equal(frame, original)

    def test_parse_plain_and_fenced_json(self):
        plain = parse_feature_payload(json.dumps(VALID_FEATURES, ensure_ascii=False))
        fenced = parse_feature_payload(
            "説明の後にJSONです。\n```json\n"
            + json.dumps(VALID_FEATURES, ensure_ascii=False)
            + "\n```"
        )
        self.assertEqual(plain, fenced)

    def test_validation_rejects_invalid_taxonomy_and_consistency(self):
        invalid_cases = {
            "policy": {**VALID_FEATURES, "policy_domain": "D99"},
            "admin": {**VALID_FEATURES, "admin_process": "A99"},
            "usecase_type": {**VALID_FEATURES, "ai_usecase": "U02"},
            "too_many": {
                **VALID_FEATURES,
                "ai_usecase": ["U01", "U02", "U03", "U04"],
            },
            "duplicate": {**VALID_FEATURES, "ai_usecase": ["U02", "U02"]},
            "u99_mixed": {**VALID_FEATURES, "ai_usecase": ["U99", "U02"]},
            "applicability": {**VALID_FEATURES, "ai_applicability": 3},
            "zero_without_u99": {**VALID_FEATURES, "ai_applicability": 0},
            "u99_without_zero": {
                **VALID_FEATURES,
                "ai_usecase": ["U99"],
                "ai_applicability": 1,
            },
            "unexpected_score": {**VALID_FEATURES, "AIU_fit": 5},
        }
        for name, payload in invalid_cases.items():
            with self.subTest(name=name):
                with self.assertRaises(FeatureValidationError):
                    validate_feature_payload(payload)

    def test_decode_pipe_and_json_ai_usecase(self):
        self.assertEqual(decode_ai_usecase("U02|U05"), ["U02", "U05"])
        self.assertEqual(decode_ai_usecase('["U01", "U10"]'), ["U01", "U10"])

    def test_dry_run_does_not_call_api_or_write_cache(self):
        client = FakeBedrockClient([])
        with tempfile.TemporaryDirectory() as directory:
            result = generate_llm_features(
                self.df,
                split="all",
                model="amazon.nova-micro-v1:0",
                output_root=directory,
                dry_run=True,
                client=client,
                show_progress=False,
            )
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertIsNone(result["features"])
        self.assertEqual(result["report"]["selected_rows"], 2)
        self.assertEqual(client.calls, [])

    def test_generation_resume_cache_and_alignment(self):
        sample_client = FakeBedrockClient([bedrock_response(VALID_FEATURES)])
        with tempfile.TemporaryDirectory() as directory:
            first = generate_llm_features(
                self.df,
                split="all",
                model="amazon.nova-micro-v1:0",
                output_root=directory,
                n_rows=1,
                input_price_per_million=0.035,
                output_price_per_million=0.14,
                dry_run=False,
                client=sample_client,
                show_progress=False,
            )
            self.assertEqual(len(sample_client.calls), 1)
            self.assertEqual(first["features"]["original_index"].tolist(), [10])
            self.assertEqual(first["features"]["ai_usecase"].tolist(), ["U02|U05"])
            self.assertEqual(
                sample_client.calls[0]["toolConfig"]["toolChoice"]["tool"]["name"],
                "classify_administrative_project",
            )

            full_client = FakeBedrockClient([bedrock_response(VALID_FEATURES)])
            second = generate_llm_features(
                self.df,
                split="all",
                model="amazon.nova-micro-v1:0",
                output_root=directory,
                input_price_per_million=0.035,
                output_price_per_million=0.14,
                dry_run=False,
                client=full_client,
                show_progress=False,
            )
            loaded = load_llm_features(
                second["cache_dir"],
                split="all",
                expected_df=self.df,
            )
        self.assertEqual(len(full_client.calls), 1)
        self.assertEqual(second["from_cache_rows"], 1)
        self.assertEqual(len(loaded), 2)

    def test_invalid_response_is_retried_then_validated(self):
        invalid = {**VALID_FEATURES, "policy_domain": "D99"}
        client = FakeBedrockClient(
            [bedrock_response(invalid), bedrock_response(VALID_FEATURES)]
        )
        with tempfile.TemporaryDirectory() as directory:
            result = generate_llm_features(
                self.df.iloc[:1],
                split="sample",
                model="amazon.nova-micro-v1:0",
                output_root=directory,
                input_price_per_million=0.035,
                output_price_per_million=0.14,
                max_retries=1,
                initial_backoff_seconds=0,
                dry_run=False,
                client=client,
                show_progress=False,
            )
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(result["features"].loc[0, "validation_retry_count"], 1)
        self.assertTrue(result["errors"].empty)

    def test_budget_guard_runs_before_api(self):
        client = FakeBedrockClient([])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(BudgetExceededError):
                generate_llm_features(
                    self.df,
                    split="all",
                    model="amazon.nova-micro-v1:0",
                    output_root=directory,
                    input_price_per_million=1000,
                    output_price_per_million=1000,
                    max_budget_usd=0.01,
                    dry_run=False,
                    client=client,
                    show_progress=False,
                )
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
