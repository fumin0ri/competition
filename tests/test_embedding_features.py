import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from embedding_features import (
    build_embedding_text,
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
                "target": [1, 0, 1, 0, 1],
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

    def test_dry_run_does_not_call_api_or_write_cache(self):
        def must_not_run(*args, **kwargs):
            raise AssertionError("API embedder was called during dry-run")

        with tempfile.TemporaryDirectory() as directory:
            result = generate_embeddings(
                self.frame,
                split="train",
                provider="gemini",
                model="gemini-embedding-2",
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
                provider="gemini",
                model="gemini-embedding-2",
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
                provider="gemini",
                model="gemini-embedding-2",
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
                provider="gemini",
                model="gemini-embedding-2",
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
                    provider="gemini",
                    model="gemini-embedding-2",
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
                    provider="gemini",
                    model="gemini-embedding-2",
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
                provider="gemini",
                model="gemini-embedding-2",
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
                    provider="gemini",
                    model="gemini-embedding-2",
                    output_root=directory,
                    text_cols=["project_name", "target"],
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
