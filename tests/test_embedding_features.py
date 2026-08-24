import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from embedding_features import (
    build_embedding_text,
    embed_sagemaker,
    generate_embeddings,
    l2_normalize_embeddings,
    load_embeddings,
    normalize_embedding_text,
)


class EmbeddingFeatureTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "project_id": ["p1", "p2", "p3", "p4", "p5"],
                "project_name": ["宇宙  開発", None, "量子", "海洋", "AI"],
                "project_objective": ["目的A", "", "目的C", "目的D", "目的E"],
                "project_summary": ["概要A", None, "概要C", "概要D", "概要E"],
                "science_tech_decision": [1, 0, 1, 0, 1],
            },
            index=[10, 20, 30, 40, 50],
        )

    @staticmethod
    def fake_embedder(texts, client, model, embedding_dim):
        del client, model
        dimension = embedding_dim or 4
        rows = []
        for text in texts:
            seed = [len(text), sum(ord(char) for char in text) % 997]
            rows.append((seed + [1.0] * dimension)[:dimension])
        return np.asarray(rows, dtype=np.float32), {
            "total_tokens": sum(len(text) for text in texts)
        }

    def test_normalization_and_empty_text(self):
        self.assertEqual(normalize_embedding_text("ＡＢＣ\n  １２３"), "ABC 123")
        empty = pd.Series(
            {"project_name": None, "project_objective": " ", "project_summary": np.nan}
        )
        self.assertEqual(build_embedding_text(empty), "[EMPTY]")

    def test_sagemaker_default_adapter_invokes_realtime_endpoint(self):
        class RuntimeClient:
            def __init__(self):
                self.calls = []

            def invoke_endpoint(self, **kwargs):
                self.calls.append(kwargs)
                request = json.loads(kwargs["Body"].decode("utf-8"))
                vectors = [[float(len(text)), 1.0, 2.0] for text in request["inputs"]]
                return {"Body": io.BytesIO(json.dumps({"embeddings": vectors}).encode())}

        runtime = RuntimeClient()
        matrix, usage = embed_sagemaker(
            ["a", "日本語"],
            runtime,
            "jp-embedding-v1",
            3,
            endpoint_name="competition-embedding-endpoint",
        )
        self.assertEqual(matrix.shape, (2, 3))
        self.assertEqual(usage, {})
        self.assertEqual(len(runtime.calls), 1)
        self.assertEqual(
            runtime.calls[0]["EndpointName"], "competition-embedding-endpoint"
        )
        self.assertEqual(runtime.calls[0]["ContentType"], "application/json")

    def test_sagemaker_custom_request_and_response_adapters(self):
        class RuntimeClient:
            def invoke_endpoint(self, **kwargs):
                request = json.loads(kwargs["Body"].decode("utf-8"))
                self.request = request
                self.kwargs = kwargs
                response = {"result": [[3.0, 4.0] for _ in request["sentences"]]}
                return {"Body": io.BytesIO(json.dumps(response).encode())}

        runtime = RuntimeClient()

        def request_builder(texts, model, embedding_dim):
            return {
                "sentences": list(texts),
                "parameters": {"model": model, "dimension": embedding_dim},
            }

        def response_parser(body, expected_rows):
            payload = json.loads(body.decode("utf-8"))
            self.assertEqual(len(payload["result"]), expected_rows)
            return np.asarray(payload["result"], dtype=np.float32), {"total_tokens": 7}

        matrix, usage = embed_sagemaker(
            ["one", "two"],
            runtime,
            "custom-model-contract",
            2,
            endpoint_name="custom-endpoint",
            request_builder=request_builder,
            response_parser=response_parser,
            invoke_endpoint_kwargs={"TargetVariant": "variant-b"},
        )
        np.testing.assert_allclose(matrix, [[3.0, 4.0], [3.0, 4.0]])
        self.assertEqual(usage["total_tokens"], 7)
        self.assertEqual(runtime.request["parameters"]["dimension"], 2)
        self.assertEqual(runtime.kwargs["TargetVariant"], "variant-b")

    def test_sagemaker_generate_uses_endpoint_and_adapter_cache_identity(self):
        class RuntimeClient:
            def invoke_endpoint(self, **kwargs):
                request = json.loads(kwargs["Body"].decode("utf-8"))
                vectors = [[1.0, 2.0, 3.0] for _ in request["inputs"]]
                return {"Body": io.BytesIO(json.dumps(vectors).encode())}

        with tempfile.TemporaryDirectory() as directory:
            result = generate_embeddings(
                self.frame,
                split="train",
                provider="sagemaker",
                model="jp-embedding-v1",
                endpoint_name="competition-embedding-endpoint",
                region_name="ap-northeast-1",
                adapter_id="json-inputs-v1",
                output_root=directory,
                embedding_dim=3,
                batch_size=2,
                dry_run=False,
                client=RuntimeClient(),
                show_progress=False,
            )
            self.assertEqual(result["embeddings"].shape, (5, 3))
            config = json.loads(
                (result["cache_dir"] / "config.json").read_text(encoding="utf-8")
            )
            provider_config = config["provider_config"]
            self.assertEqual(
                provider_config["endpoint_name"], "competition-embedding-endpoint"
            )
            self.assertEqual(provider_config["adapter_id"], "json-inputs-v1")

    def test_sagemaker_invoke_options_require_explicit_cache_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "sagemaker_cache_identity"):
                generate_embeddings(
                    self.frame,
                    split="train",
                    provider="sagemaker",
                    model="jp-embedding-v1",
                    output_root=directory,
                    dry_run=True,
                    invoke_endpoint_kwargs={"TargetVariant": "variant-b"},
                    show_progress=False,
                )

    def test_dry_run_does_not_call_api_or_write_cache(self):
        def must_not_run(*args, **kwargs):
            raise AssertionError("API embedder was called during dry-run")

        with tempfile.TemporaryDirectory() as directory:
            result = generate_embeddings(
                self.frame,
                split="train",
                provider="sagemaker",
                model="jp-embedding-v1",
                output_root=directory,
                embedding_dim=3,
                dry_run=True,
                embedder=must_not_run,
                show_progress=False,
            )
            self.assertIsNone(result["embeddings"])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_generate_resume_load_and_alignment(self):
        calls = []

        def recording_embedder(texts, client, model, embedding_dim):
            calls.append(list(texts))
            return self.fake_embedder(texts, client, model, embedding_dim)

        with tempfile.TemporaryDirectory() as directory:
            result = generate_embeddings(
                self.frame,
                split="train",
                provider="sagemaker",
                model="jp-embedding-v1",
                output_root=directory,
                embedding_dim=3,
                batch_size=2,
                max_budget_usd=1.0,
                dry_run=False,
                client=object(),
                embedder=recording_embedder,
                show_progress=False,
            )
            self.assertEqual(len(calls), 3)
            self.assertEqual(result["embeddings"].shape, (5, 3))
            self.assertEqual(result["embeddings"].dtype, np.float32)
            self.assertEqual(
                result["metadata"]["original_index"].tolist(),
                self.frame.index.tolist(),
            )
            self.assertEqual(
                result["metadata"]["project_id"].tolist(),
                self.frame["project_id"].tolist(),
            )
            for required in (
                "split",
                "original_index",
                "project_id",
                "text_hash",
                "provider",
                "model",
                "embedding_dim",
            ):
                self.assertIn(required, result["metadata"].columns)

            cache_dir = result["cache_dir"]
            self.assertTrue((cache_dir / "train_embeddings.npy").exists())
            self.assertTrue((cache_dir / "train_metadata.parquet").exists())
            self.assertTrue((cache_dir / "config.json").exists())
            self.assertTrue((cache_dir / "train_progress.json").exists())

            def must_not_run(*args, **kwargs):
                raise AssertionError("Completed cache should prevent an API call")

            cached = generate_embeddings(
                self.frame,
                split="train",
                provider="sagemaker",
                model="jp-embedding-v1",
                output_root=directory,
                embedding_dim=3,
                batch_size=2,
                max_budget_usd=0.0,
                dry_run=False,
                embedder=must_not_run,
                show_progress=False,
            )
            self.assertTrue(cached["from_cache"])
            loaded, metadata = load_embeddings(
                cache_dir,
                split="train",
                expected_df=self.frame,
            )
            np.testing.assert_allclose(loaded, result["embeddings"])
            self.assertEqual(len(metadata), len(self.frame))

    def test_changed_text_rejects_completed_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            first = generate_embeddings(
                self.frame,
                split="train",
                provider="sagemaker",
                model="jp-embedding-v1",
                output_root=directory,
                embedding_dim=3,
                batch_size=5,
                dry_run=False,
                client=object(),
                embedder=self.fake_embedder,
                show_progress=False,
            )
            changed = self.frame.copy()
            changed.loc[10, "project_summary"] = "変更後"
            with self.assertRaisesRegex(ValueError, "Cache configuration mismatch"):
                generate_embeddings(
                    changed,
                    split="train",
                    provider="sagemaker",
                    model="jp-embedding-v1",
                    output_root=directory,
                    embedding_dim=3,
                    batch_size=5,
                    dry_run=False,
                    embedder=self.fake_embedder,
                    show_progress=False,
                )
            self.assertTrue(first["cache_dir"].exists())

    def test_interrupted_run_resumes_from_completed_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            call_count = 0

            def fail_on_second_batch(texts, client, model, embedding_dim):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise RuntimeError("simulated interruption")
                return self.fake_embedder(texts, client, model, embedding_dim)

            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                generate_embeddings(
                    self.frame,
                    split="train",
                    provider="sagemaker",
                    model="jp-embedding-v1",
                    output_root=directory,
                    embedding_dim=3,
                    batch_size=2,
                    max_retries=0,
                    dry_run=False,
                    client=object(),
                    embedder=fail_on_second_batch,
                    show_progress=False,
                )

            resumed_calls = []

            def recording_embedder(texts, client, model, embedding_dim):
                resumed_calls.append(list(texts))
                return self.fake_embedder(texts, client, model, embedding_dim)

            resumed = generate_embeddings(
                self.frame,
                split="train",
                provider="sagemaker",
                model="jp-embedding-v1",
                output_root=directory,
                embedding_dim=3,
                batch_size=2,
                max_retries=0,
                dry_run=False,
                client=object(),
                embedder=recording_embedder,
                show_progress=False,
            )
            self.assertEqual(len(resumed_calls), 2)
            self.assertEqual(resumed["embeddings"].shape, (5, 3))
            self.assertFalse(resumed["from_cache"])

    def test_target_cannot_be_sent(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "target_col"):
                generate_embeddings(
                    self.frame,
                    split="train",
                    provider="sagemaker",
                    model="jp-embedding-v1",
                    output_root=directory,
                    text_cols=["project_name", "science_tech_decision"],
                    dry_run=True,
                    show_progress=False,
                )

    def test_l2_normalization_keeps_zero_rows_finite(self):
        array = np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
        normalized = l2_normalize_embeddings(array)
        np.testing.assert_allclose(normalized[0], [0.6, 0.8])
        np.testing.assert_allclose(normalized[1], [0.0, 0.0])
        self.assertTrue(np.isfinite(normalized).all())


if __name__ == "__main__":
    unittest.main()
