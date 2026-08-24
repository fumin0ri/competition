"""Leakage-safe CV baselines for embeddings, tabular, and TF-IDF features."""

from __future__ import annotations

import logging
import math
import random
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize

from embedding_features import verify_embedding_alignment
from text_features import fit_transform_tfidf_columns, save_sparse_features_csv
from validation import make_seen_project_mask


LOGGER = logging.getLogger(__name__)
SUPPORTED_METRICS = {"roc_auc", "log_loss"}


@dataclass
class ExperimentResult:
    """OOF predictions and fold-level diagnostics for one experiment."""

    name: str
    oof: pd.Series
    fold_metrics: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentSuiteResult:
    """Combined artifacts produced by ``run_all_experiments``."""

    results: dict[str, ExperimentResult]
    oof_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    summary: pd.DataFrame
    test_predictions: dict[str, np.ndarray] = field(default_factory=dict)


def default_modeling_config() -> dict[str, Any]:
    """Return an editable baseline config, tuned for one NVIDIA T4."""
    return {
        "run_e1": True,
        "run_e2": True,
        "run_e3": True,
        "run_e4": True,
        "run_e5": True,
        "run_e6": True,
        "run_t1": True,
        "run_t2": True,
        "run_t3": True,
        "e6_pca_dims": [None],
        "metric": "roc_auc",
        "random_state": 42,
        "embedding_scaling": "l2",
        "text_cols": [
            "project_name",
            "project_objective",
            "project_summary",
        ],
        "tfidf": {
            "analyzer": "char",
            "ngram_range": (2, 5),
            "min_df": 2,
            "max_features": 300_000,
            "sublinear_tf": True,
            "dtype": np.float32,
        },
        "tfidf_lr": {
            "C": 1.0,
            "max_iter": 3000,
            "solver": "liblinear",
            "dual": True,
        },
        "tfidf_feature_output_dir": "data/csv/tfidf_shared",
        "lr": {
            "C": 1.0,
            "max_iter": 3000,
            "solver": "lbfgs",
        },
        "mlp": {
            "hidden_dims": [256, 64],
            "dropout": 0.2,
            "batch_size": 256,
            "max_epochs": 60,
            "full_epochs": 30,
            "patience": 8,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "device": "cuda",
            "allow_cpu_fallback": True,
            "use_amp": True,
        },
        "catboost": {
            "iterations": 1000,
            "learning_rate": 0.05,
            "depth": 6,
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "early_stopping_rounds": 80,
            "task_type": "GPU",
            "devices": "0",
            "verbose": False,
            "allow_writing_files": False,
        },
        "xgboost": {
            "n_estimators": 1000,
            "learning_rate": 0.03,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "early_stopping_rounds": 80,
            "tree_method": "hist",
            "device": "cuda",
            "n_jobs": -1,
        },
        "feature_engineering": {
            "add_project_duration": True,
            "add_log1p_budget": True,
            "budget_col": "budget",
            "invalid_year_value": -1,
        },
        "run_final_test_prediction": False,
        "final_experiments": [],
        "output_dir": "outputs",
    }


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when it is installed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _merge_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = default_modeling_config()
    if config is None:
        return merged
    for key, value in config.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def _validate_metric(metric: str) -> str:
    metric = metric.lower()
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"metric must be one of {sorted(SUPPORTED_METRICS)}")
    return metric


def _safe_score(y_true: Sequence[int], prediction: Sequence[float], metric: str) -> float:
    """Score a subset, returning NaN when ROC-AUC is undefined."""
    metric = _validate_metric(metric)
    y = np.asarray(y_true)
    probability = np.clip(np.asarray(prediction, dtype=float), 1e-7, 1 - 1e-7)
    if len(y) == 0:
        return float("nan")
    if metric == "roc_auc":
        if np.unique(y).size < 2:
            return float("nan")
        return float(roc_auc_score(y, probability))
    return float(log_loss(y, probability, labels=[0, 1]))


def _validate_binary_target(target: pd.Series, target_col: str) -> None:
    if target.isna().any():
        raise ValueError(f"{target_col!r} contains missing values.")
    values = set(pd.unique(target))
    if not values.issubset({0, 1, False, True}):
        raise ValueError(f"{target_col!r} must be binary 0/1; found {sorted(values)}")


def _positions_from_labels(df: pd.DataFrame, labels: pd.Index) -> np.ndarray:
    if not df.index.is_unique:
        raise ValueError("df.index must be unique because folds contain label indices.")
    positions = df.index.get_indexer(pd.Index(labels))
    if (positions < 0).any():
        missing = pd.Index(labels)[positions < 0].tolist()[:5]
        raise KeyError(f"Fold contains labels not present in df.index: {missing}")
    return positions


def validate_embedding_input(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    project_id_col: str = "project_id",
    split: str | None = None,
) -> np.ndarray:
    """Validate explicit metadata alignment and return float32 embeddings."""
    matrix = np.asarray(embeddings)
    if matrix.ndim != 2 or matrix.shape[0] != len(df):
        raise ValueError("Embedding shape does not match the source DataFrame.")
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings contain non-finite values.")
    verify_embedding_alignment(
        df=df,
        metadata=metadata,
        project_id_col=project_id_col,
        split=split,
    )
    return np.asarray(matrix, dtype=np.float32)


def prepare_tabular_features(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    *,
    add_project_duration: bool = True,
    add_log1p_budget: bool = True,
    budget_col: str = "budget",
    invalid_year_value: int | float = -1,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Create deterministic tabular features without learning from data."""
    numeric = list(dict.fromkeys(numeric_cols))
    categorical = list(dict.fromkeys(categorical_cols))
    overlap = sorted(set(numeric) & set(categorical))
    if overlap:
        raise ValueError(f"Columns cannot be both numeric and categorical: {overlap}")
    required = set(numeric) | set(categorical)
    if add_project_duration:
        required |= {"project_start_year", "project_end_year"}
    if add_log1p_budget:
        required.add(budget_col)
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing tabular columns: {missing}")

    result = pd.DataFrame(index=df.index)
    year_columns = {
        "project_start_year",
        "project_end_year",
        "project_fiscal_year",
    }
    for column in numeric:
        values = pd.to_numeric(df[column], errors="coerce")
        if column in year_columns:
            values = values.mask(values.eq(invalid_year_value))
        result[column] = values.astype(float)

    for column in categorical:
        values = df[column].astype(object)
        result[column] = values.where(pd.notna(values), np.nan)

    if add_project_duration:
        start = pd.to_numeric(df["project_start_year"], errors="coerce")
        end = pd.to_numeric(df["project_end_year"], errors="coerce")
        valid = start.ne(invalid_year_value) & end.ne(invalid_year_value)
        duration = (end - start).where(valid & end.ge(start))
        result["project_duration"] = duration.astype(float)
        if "project_duration" not in numeric:
            numeric.append("project_duration")

    if add_log1p_budget:
        budget = pd.to_numeric(df[budget_col], errors="coerce")
        result["log1p_budget"] = np.log1p(budget.clip(lower=0))
        if "log1p_budget" not in numeric:
            numeric.append("log1p_budget")

    return result, numeric, categorical


def _make_tabular_preprocessor(
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers: list[tuple[str, Any, Sequence[str]]] = []
    if numeric_cols:
        transformers.append(("numeric", Pipeline(numeric_steps), list(numeric_cols)))
    if categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="__MISSING__",
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                dtype=np.float32,
                            ),
                        ),
                    ]
                ),
                list(categorical_cols),
            )
        )
    if not transformers:
        raise ValueError("At least one numeric or categorical column is required.")
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def _make_sparse_tabular_preprocessor(
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> ColumnTransformer:
    """Build fold-fitted tabular preprocessing that preserves sparse output."""
    transformers: list[tuple[str, Any, Sequence[str]]] = []
    if numeric_cols:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                list(numeric_cols),
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="__MISSING__",
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=True,
                                dtype=np.float32,
                            ),
                        ),
                    ]
                ),
                list(categorical_cols),
            )
        )
    if not transformers:
        raise ValueError("At least one numeric or categorical column is required.")
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )


def _as_float32_csr(matrix: Any) -> sparse.csr_matrix:
    result = sparse.csr_matrix(matrix, dtype=np.float32)
    if result.ndim != 2 or not np.isfinite(result.data).all():
        raise ValueError("Sparse feature matrix is invalid or non-finite.")
    return result


def _sparse_hstack(*matrices: Any) -> sparse.csr_matrix:
    result = sparse.hstack(
        [_as_float32_csr(matrix) for matrix in matrices],
        format="csr",
        dtype=np.float32,
    )
    if not sparse.isspmatrix_csr(result):
        raise AssertionError("Combined features must remain CSR sparse.")
    return result


def _csr_memory_mib(matrix: sparse.csr_matrix) -> float:
    total_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    return float(total_bytes / 1024**2)


def _fit_transform_embeddings(
    train_embedding: np.ndarray,
    valid_embedding: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray, Any | None]:
    method = method.lower()
    if method == "l2":
        return (
            np.asarray(normalize(train_embedding), dtype=np.float32),
            np.asarray(normalize(valid_embedding), dtype=np.float32),
            None,
        )
    if method == "standard":
        scaler = StandardScaler()
        return (
            np.asarray(scaler.fit_transform(train_embedding), dtype=np.float32),
            np.asarray(scaler.transform(valid_embedding), dtype=np.float32),
            scaler,
        )
    if method == "none":
        return (
            np.asarray(train_embedding, dtype=np.float32),
            np.asarray(valid_embedding, dtype=np.float32),
            None,
        )
    raise ValueError("embedding_scaling must be 'l2', 'standard', or 'none'.")


def _transform_embeddings_with_fitted(
    train_embedding: np.ndarray,
    test_embedding: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    train_transformed, test_transformed, _ = _fit_transform_embeddings(
        train_embedding,
        test_embedding,
        method,
    )
    return train_transformed, test_transformed


def _combine_dense(*arrays: np.ndarray) -> np.ndarray:
    matrix = np.hstack([np.asarray(array, dtype=np.float32) for array in arrays])
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Combined model input is invalid or non-finite.")
    estimated_gib = matrix.nbytes / 1024**3
    if estimated_gib >= 1.0:
        warnings.warn(
            f"Dense model matrix uses approximately {estimated_gib:.2f} GiB.",
            UserWarning,
            stacklevel=2,
        )
    return matrix


def _fold_year(df: pd.DataFrame, valid_idx: pd.Index, year_col: str) -> Any:
    years = pd.unique(df.loc[valid_idx, year_col])
    if len(years) != 1:
        raise ValueError("Each validation fold must contain exactly one year.")
    return years[0].item() if isinstance(years[0], np.generic) else years[0]


def _run_cv(
    *,
    experiment: str,
    df: pd.DataFrame,
    folds: Sequence[tuple[pd.Index, pd.Index]],
    fold_runner: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, Mapping[str, Any]]],
    target_col: str,
    project_col: str,
    year_col: str,
    metric: str,
    random_state: int,
    metadata: Mapping[str, Any] | None = None,
) -> ExperimentResult:
    metric = _validate_metric(metric)
    _validate_binary_target(df[target_col], target_col)
    set_global_seed(random_state)
    oof = pd.Series(np.nan, index=df.index, dtype=float, name=experiment)
    records: list[dict[str, Any]] = []

    for fold_number, (train_idx, valid_idx) in enumerate(folds):
        train_idx = pd.Index(train_idx)
        valid_idx = pd.Index(valid_idx)
        train_positions = _positions_from_labels(df, train_idx)
        valid_positions = _positions_from_labels(df, valid_idx)
        if np.intersect1d(train_positions, valid_positions).size:
            raise ValueError("Training and validation indices overlap.")
        y_train = df.loc[train_idx, target_col].to_numpy(dtype=np.float32)
        y_valid = df.loc[valid_idx, target_col].to_numpy(dtype=np.float32)
        if np.unique(y_train).size < 2:
            raise ValueError(f"Fold {fold_number} training target has only one class.")

        prediction, details = fold_runner(
            train_positions,
            valid_positions,
            y_train,
            y_valid,
        )
        prediction = np.asarray(prediction, dtype=float).reshape(-1)
        if len(prediction) != len(valid_idx) or not np.isfinite(prediction).all():
            raise ValueError("Fold prediction length/values are invalid.")
        if ((prediction < 0) | (prediction > 1)).any():
            raise ValueError("Validation predictions must be probabilities in [0, 1].")
        oof.loc[valid_idx] = prediction

        seen = make_seen_project_mask(
            df=df,
            train_idx=train_idx,
            valid_idx=valid_idx,
            project_col=project_col,
        ).to_numpy(dtype=bool)
        record = {
            "experiment": experiment,
            "fold": fold_number,
            "validation_year": _fold_year(df, valid_idx, year_col),
            "train_rows": len(train_idx),
            "valid_rows": len(valid_idx),
            "metric": metric,
            "overall_score": _safe_score(y_valid, prediction, metric),
            "seen_score": _safe_score(y_valid[seen], prediction[seen], metric),
            "unseen_score": _safe_score(y_valid[~seen], prediction[~seen], metric),
            "seen_ratio": float(seen.mean()) if len(seen) else float("nan"),
            "seen_rows": int(seen.sum()),
            "unseen_rows": int((~seen).sum()),
            **dict(details),
        }
        records.append(record)
        LOGGER.info(
            "%s fold=%s year=%s score=%.6f fit=%.1fs predict=%.1fs",
            experiment,
            fold_number,
            record["validation_year"],
            record["overall_score"],
            record.get("fit_seconds", float("nan")),
            record.get("predict_seconds", float("nan")),
        )

    return ExperimentResult(
        name=experiment,
        oof=oof,
        fold_metrics=pd.DataFrame(records),
        metadata=dict(metadata or {}),
    )


def _lr_from_config(config: Mapping[str, Any], random_state: int) -> LogisticRegression:
    return LogisticRegression(
        C=float(config.get("C", 1.0)),
        max_iter=int(config.get("max_iter", 3000)),
        solver=str(config.get("solver", "lbfgs")),
        random_state=random_state,
    )


def run_e1_embedding_lr(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_metadata: pd.DataFrame,
    folds: Sequence[tuple[pd.Index, pd.Index]],
    *,
    target_col: str = "target",
    project_col: str = "project_name",
    project_id_col: str = "project_id",
    year_col: str = "project_start_year",
    metric: str = "roc_auc",
    embedding_scaling: str = "l2",
    lr_config: Mapping[str, Any] | None = None,
    random_state: int = 42,
) -> ExperimentResult:
    """E1: embedding -> Logistic Regression."""
    matrix = validate_embedding_input(
        df, embeddings, embedding_metadata, project_id_col, split="train"
    )
    lr_config = lr_config or {}

    def fold_runner(train_pos, valid_pos, y_train, y_valid):
        del y_valid
        started = time.perf_counter()
        x_train, x_valid, _ = _fit_transform_embeddings(
            matrix[train_pos], matrix[valid_pos], embedding_scaling
        )
        model = _lr_from_config(lr_config, random_state)
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        prediction = model.predict_proba(x_valid)[:, 1]
        predict_seconds = time.perf_counter() - started
        return prediction, {
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "input_dim": x_train.shape[1],
            "device": "cpu",
            "best_iteration": np.nan,
            "pca_explained_variance": np.nan,
        }

    return _run_cv(
        experiment="E1_embedding_lr",
        df=df,
        folds=folds,
        fold_runner=fold_runner,
        target_col=target_col,
        project_col=project_col,
        year_col=year_col,
        metric=metric,
        random_state=random_state,
    )


def run_e2_embedding_tabular_lr(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_metadata: pd.DataFrame,
    folds: Sequence[tuple[pd.Index, pd.Index]],
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    *,
    target_col: str = "target",
    project_col: str = "project_name",
    project_id_col: str = "project_id",
    year_col: str = "project_start_year",
    metric: str = "roc_auc",
    embedding_scaling: str = "l2",
    lr_config: Mapping[str, Any] | None = None,
    feature_config: Mapping[str, Any] | None = None,
    random_state: int = 42,
) -> ExperimentResult:
    """E2: embedding + imputed/scaled/one-hot tabular -> LR."""
    matrix = validate_embedding_input(
        df, embeddings, embedding_metadata, project_id_col, split="train"
    )
    table, numeric, categorical = prepare_tabular_features(
        df, numeric_cols, categorical_cols, **dict(feature_config or {})
    )
    lr_config = lr_config or {}

    def fold_runner(train_pos, valid_pos, y_train, y_valid):
        del y_valid
        train_labels = df.index[train_pos]
        valid_labels = df.index[valid_pos]
        preprocessor = _make_tabular_preprocessor(
            numeric, categorical, scale_numeric=True
        )
        started = time.perf_counter()
        tab_train = np.asarray(
            preprocessor.fit_transform(table.loc[train_labels]), dtype=np.float32
        )
        tab_valid = np.asarray(
            preprocessor.transform(table.loc[valid_labels]), dtype=np.float32
        )
        emb_train, emb_valid, _ = _fit_transform_embeddings(
            matrix[train_pos], matrix[valid_pos], embedding_scaling
        )
        x_train = _combine_dense(emb_train, tab_train)
        x_valid = _combine_dense(emb_valid, tab_valid)
        model = _lr_from_config(lr_config, random_state)
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        prediction = model.predict_proba(x_valid)[:, 1]
        predict_seconds = time.perf_counter() - started
        return prediction, {
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "input_dim": x_train.shape[1],
            "device": "cpu",
            "best_iteration": np.nan,
            "pca_explained_variance": np.nan,
        }

    return _run_cv(
        experiment="E2_embedding_tabular_lr",
        df=df,
        folds=folds,
        fold_runner=fold_runner,
        target_col=target_col,
        project_col=project_col,
        year_col=year_col,
        metric=metric,
        random_state=random_state,
    )


def _resolve_torch_device(requested: str, allow_cpu_fallback: bool) -> Any:
    try:
        import torch
    except ImportError as error:
        raise ImportError("Install torch to run E3/E4 MLP experiments.") from error
    requested = requested.lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        if not allow_cpu_fallback:
            raise RuntimeError("CUDA was requested for MLP, but no CUDA device is available.")
        warnings.warn("CUDA is unavailable; E3/E4 will run on CPU.", stacklevel=2)
        return torch.device("cpu")
    return torch.device(requested)


def _fit_predict_torch_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray | None,
    config: Mapping[str, Any],
    random_state: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a compact binary MLP with validation-loss early stopping."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise ImportError("Install torch to run E3/E4 MLP experiments.") from error

    set_global_seed(random_state)
    device = _resolve_torch_device(
        str(config.get("device", "cuda")),
        bool(config.get("allow_cpu_fallback", True)),
    )
    hidden_dims = [int(value) for value in config.get("hidden_dims", [256, 64])]
    if not hidden_dims or any(value <= 0 for value in hidden_dims):
        raise ValueError("mlp.hidden_dims must contain positive integers.")
    dropout = float(config.get("dropout", 0.2))
    if not 0 <= dropout < 1:
        raise ValueError("mlp.dropout must be in [0, 1).")

    layers: list[Any] = []
    previous = x_train.shape[1]
    for hidden in hidden_dims:
        layers.extend([nn.Linear(previous, hidden), nn.ReLU(), nn.Dropout(dropout)])
        previous = hidden
    layers.append(nn.Linear(previous, 1))
    model = nn.Sequential(*layers).to(device)

    train_x = torch.from_numpy(np.asarray(x_train, dtype=np.float32))
    train_y = torch.from_numpy(np.asarray(y_train, dtype=np.float32).reshape(-1, 1))
    generator = torch.Generator()
    generator.manual_seed(random_state)
    pin_memory = device.type == "cuda"
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=int(config.get("batch_size", 256)),
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        generator=generator,
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )
    amp_enabled = bool(config.get("use_amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    max_epochs = int(config.get("max_epochs", 60))
    patience = int(config.get("patience", 8))
    if max_epochs <= 0 or patience < 0:
        raise ValueError("mlp.max_epochs must be positive and patience non-negative.")

    valid_loader = None
    if y_valid is not None:
        valid_dataset = TensorDataset(
            torch.from_numpy(np.asarray(x_valid, dtype=np.float32)),
            torch.from_numpy(np.asarray(y_valid, dtype=np.float32).reshape(-1, 1)),
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=max(512, int(config.get("batch_size", 256))),
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )

    best_state: dict[str, Any] | None = None
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=pin_memory)
            batch_y = batch_y.to(device, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if y_valid is None:
            best_epoch = epoch
            continue

        model.eval()
        valid_loss_sum = 0.0
        valid_rows = 0
        assert valid_loader is not None
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x = batch_x.to(device, non_blocking=pin_memory)
                batch_y = batch_y.to(device, non_blocking=pin_memory)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    loss = criterion(model(batch_x), batch_y)
                valid_loss_sum += float(loss.detach().cpu()) * len(batch_x)
                valid_rows += len(batch_x)
        mean_valid_loss = valid_loss_sum / valid_rows
        if mean_valid_loss < best_loss - 1e-6:
            best_loss = mean_valid_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    model.eval()
    predictions: list[np.ndarray] = []
    prediction_loader = DataLoader(
        TensorDataset(torch.from_numpy(np.asarray(x_valid, dtype=np.float32))),
        batch_size=max(512, int(config.get("batch_size", 256))),
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    with torch.no_grad():
        for (batch_x,) in prediction_loader:
            batch_x = batch_x.to(device, non_blocking=pin_memory)
            logits = model(batch_x)
            predictions.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
    prediction = np.concatenate(predictions).astype(np.float32, copy=False)
    predict_seconds = time.perf_counter() - started
    details = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "input_dim": x_train.shape[1],
        "device": str(device),
        "best_iteration": best_epoch,
        "best_validation_loss": best_loss if y_valid is not None else np.nan,
        "pca_explained_variance": np.nan,
    }
    if device.type == "cuda":
        model.to("cpu")
        del model, optimizer, scaler, loader, valid_loader, prediction_loader
        del batch_x, batch_y
        torch.cuda.empty_cache()
    return prediction, details


def run_e3_embedding_mlp(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_metadata: pd.DataFrame,
    folds: Sequence[tuple[pd.Index, pd.Index]],
    *,
    target_col: str = "target",
    project_col: str = "project_name",
    project_id_col: str = "project_id",
    year_col: str = "project_start_year",
    metric: str = "roc_auc",
    embedding_scaling: str = "l2",
    mlp_config: Mapping[str, Any] | None = None,
    random_state: int = 42,
) -> ExperimentResult:
    """E3: embedding -> small PyTorch MLP."""
    matrix = validate_embedding_input(
        df, embeddings, embedding_metadata, project_id_col, split="train"
    )
    mlp_config = mlp_config or {}

    def fold_runner(train_pos, valid_pos, y_train, y_valid):
        started = time.perf_counter()
        x_train, x_valid, _ = _fit_transform_embeddings(
            matrix[train_pos], matrix[valid_pos], embedding_scaling
        )
        preprocessing_seconds = time.perf_counter() - started
        prediction, details = _fit_predict_torch_mlp(
            x_train, y_train, x_valid, y_valid, mlp_config, random_state
        )
        details["fit_seconds"] += preprocessing_seconds
        return prediction, details

    return _run_cv(
        experiment="E3_embedding_mlp",
        df=df,
        folds=folds,
        fold_runner=fold_runner,
        target_col=target_col,
        project_col=project_col,
        year_col=year_col,
        metric=metric,
        random_state=random_state,
    )


def run_e4_embedding_tabular_mlp(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_metadata: pd.DataFrame,
    folds: Sequence[tuple[pd.Index, pd.Index]],
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    *,
    target_col: str = "target",
    project_col: str = "project_name",
    project_id_col: str = "project_id",
    year_col: str = "project_start_year",
    metric: str = "roc_auc",
    embedding_scaling: str = "l2",
    mlp_config: Mapping[str, Any] | None = None,
    feature_config: Mapping[str, Any] | None = None,
    random_state: int = 42,
) -> ExperimentResult:
    """E4: embedding + imputed/scaled/one-hot tabular -> small MLP."""
    matrix = validate_embedding_input(
        df, embeddings, embedding_metadata, project_id_col, split="train"
    )
    table, numeric, categorical = prepare_tabular_features(
        df, numeric_cols, categorical_cols, **dict(feature_config or {})
    )
    mlp_config = mlp_config or {}

    def fold_runner(train_pos, valid_pos, y_train, y_valid):
        started = time.perf_counter()
        train_labels = df.index[train_pos]
        valid_labels = df.index[valid_pos]
        preprocessor = _make_tabular_preprocessor(
            numeric, categorical, scale_numeric=True
        )
        tab_train = np.asarray(
            preprocessor.fit_transform(table.loc[train_labels]), dtype=np.float32
        )
        tab_valid = np.asarray(
            preprocessor.transform(table.loc[valid_labels]), dtype=np.float32
        )
        emb_train, emb_valid, _ = _fit_transform_embeddings(
            matrix[train_pos], matrix[valid_pos], embedding_scaling
        )
        x_train = _combine_dense(emb_train, tab_train)
        x_valid = _combine_dense(emb_valid, tab_valid)
        preprocessing_seconds = time.perf_counter() - started
        if x_train.shape[1] > 5000:
            warnings.warn(
                f"E4 MLP input has {x_train.shape[1]} features; monitor T4 memory.",
                UserWarning,
                stacklevel=2,
            )
        prediction, details = _fit_predict_torch_mlp(
            x_train, y_train, x_valid, y_valid, mlp_config, random_state
        )
        details["fit_seconds"] += preprocessing_seconds
        return prediction, details

    return _run_cv(
        experiment="E4_embedding_tabular_mlp",
        df=df,
        folds=folds,
        fold_runner=fold_runner,
        target_col=target_col,
        project_col=project_col,
        year_col=year_col,
        metric=metric,
        random_state=random_state,
    )


def _catboost_frame(
    table: pd.DataFrame,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> pd.DataFrame:
    result = table[[*numeric_cols, *categorical_cols]].copy()
    for column in numeric_cols:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(float)
    for column in categorical_cols:
        result[column] = result[column].fillna("__MISSING__").astype(str)
    return result


def run_e5_tabular_catboost(
    df: pd.DataFrame,
    folds: Sequence[tuple[pd.Index, pd.Index]],
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    *,
    target_col: str = "target",
    project_col: str = "project_name",
    year_col: str = "project_start_year",
    metric: str = "roc_auc",
    catboost_config: Mapping[str, Any] | None = None,
    feature_config: Mapping[str, Any] | None = None,
    random_state: int = 42,
) -> ExperimentResult:
    """E5: native numerical/categorical tabular features -> CatBoost."""
    try:
        from catboost import CatBoostClassifier
    except ImportError as error:
        raise ImportError("Install catboost to run E5.") from error

    table, numeric, categorical = prepare_tabular_features(
        df, numeric_cols, categorical_cols, **dict(feature_config or {})
    )
    table = _catboost_frame(table, numeric, categorical)
    params = dict(catboost_config or {})
    params.setdefault("iterations", 1000)
    params.setdefault("learning_rate", 0.05)
    params.setdefault("depth", 6)
    params.setdefault("loss_function", "Logloss")
    params.setdefault("eval_metric", "Logloss")
    params.setdefault("early_stopping_rounds", 80)
    params.setdefault("verbose", False)
    params.setdefault("allow_writing_files", False)
    params["random_seed"] = random_state

    def fold_runner(train_pos, valid_pos, y_train, y_valid):
        train_labels = df.index[train_pos]
        valid_labels = df.index[valid_pos]
        model = CatBoostClassifier(**params)
        started = time.perf_counter()
        model.fit(
            table.loc[train_labels],
            y_train,
            cat_features=categorical,
            eval_set=(table.loc[valid_labels], y_valid),
            use_best_model=True,
        )
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        prediction = model.predict_proba(table.loc[valid_labels])[:, 1]
        predict_seconds = time.perf_counter() - started
        task_type = str(params.get("task_type", "CPU")).upper()
        return prediction, {
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "input_dim": table.shape[1],
            "device": task_type.lower(),
            "best_iteration": model.get_best_iteration(),
            "pca_explained_variance": np.nan,
            "fully_deterministic": task_type != "GPU",
        }

    return _run_cv(
        experiment="E5_tabular_catboost",
        df=df,
        folds=folds,
        fold_runner=fold_runner,
        target_col=target_col,
        project_col=project_col,
        year_col=year_col,
        metric=metric,
        random_state=random_state,
        metadata={
            "note": "CatBoost GPU training is not bitwise deterministic.",
        },
    )


def _fit_pca(
    train_embedding: np.ndarray,
    valid_embedding: np.ndarray,
    n_components: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, PCA | None, float]:
    if n_components is None:
        return (
            np.asarray(train_embedding, dtype=np.float32),
            np.asarray(valid_embedding, dtype=np.float32),
            None,
            float("nan"),
        )
    n_components = int(n_components)
    maximum = min(train_embedding.shape[0], train_embedding.shape[1])
    if n_components <= 0 or n_components > maximum:
        raise ValueError(
            f"PCA n_components={n_components} exceeds fold limit {maximum}."
        )
    solver = "randomized" if n_components < maximum else "full"
    pca = PCA(
        n_components=n_components,
        svd_solver=solver,
        random_state=random_state,
    )
    train_reduced = pca.fit_transform(train_embedding)
    valid_reduced = pca.transform(valid_embedding)
    explained = float(pca.explained_variance_ratio_.sum())
    return (
        np.asarray(train_reduced, dtype=np.float32),
        np.asarray(valid_reduced, dtype=np.float32),
        pca,
        explained,
    )


def run_e6_embedding_tabular_xgboost(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_metadata: pd.DataFrame,
    folds: Sequence[tuple[pd.Index, pd.Index]],
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    *,
    pca_dim: int | None = None,
    target_col: str = "target",
    project_col: str = "project_name",
    project_id_col: str = "project_id",
    year_col: str = "project_start_year",
    metric: str = "roc_auc",
    xgboost_config: Mapping[str, Any] | None = None,
    feature_config: Mapping[str, Any] | None = None,
    random_state: int = 42,
) -> ExperimentResult:
    """E6: raw or fold-fitted-PCA embedding + tabular -> XGBoost."""
    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise ImportError("Install xgboost to run E6.") from error

    matrix = validate_embedding_input(
        df, embeddings, embedding_metadata, project_id_col, split="train"
    )
    table, numeric, categorical = prepare_tabular_features(
        df, numeric_cols, categorical_cols, **dict(feature_config or {})
    )
    params = dict(xgboost_config or {})
    params.setdefault("n_estimators", 1000)
    params.setdefault("learning_rate", 0.03)
    params.setdefault("max_depth", 6)
    params.setdefault("subsample", 0.8)
    params.setdefault("colsample_bytree", 0.8)
    params.setdefault("objective", "binary:logistic")
    params.setdefault("eval_metric", "logloss")
    params.setdefault("early_stopping_rounds", 80)
    params.setdefault("tree_method", "hist")
    params.setdefault("device", "cpu")
    params.setdefault("n_jobs", -1)
    params["random_state"] = random_state
    experiment = (
        "E6_embedding_tabular_xgb"
        if pca_dim is None
        else f"E6_embedding_tabular_xgb_pca{int(pca_dim)}"
    )

    def fold_runner(train_pos, valid_pos, y_train, y_valid):
        train_labels = df.index[train_pos]
        valid_labels = df.index[valid_pos]
        preprocessor = _make_tabular_preprocessor(
            numeric, categorical, scale_numeric=False
        )
        started = time.perf_counter()
        tab_train = np.asarray(
            preprocessor.fit_transform(table.loc[train_labels]), dtype=np.float32
        )
        tab_valid = np.asarray(
            preprocessor.transform(table.loc[valid_labels]), dtype=np.float32
        )
        emb_train, emb_valid, _, explained = _fit_pca(
            matrix[train_pos], matrix[valid_pos], pca_dim, random_state
        )
        x_train = _combine_dense(emb_train, tab_train)
        x_valid = _combine_dense(emb_valid, tab_valid)
        model = XGBClassifier(**params)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            verbose=False,
        )
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        prediction = model.predict_proba(x_valid)[:, 1]
        predict_seconds = time.perf_counter() - started
        return prediction, {
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "input_dim": x_train.shape[1],
            "device": str(params.get("device", "cpu")),
            "best_iteration": getattr(model, "best_iteration", np.nan),
            "pca_explained_variance": explained,
        }

    return _run_cv(
        experiment=experiment,
        df=df,
        folds=folds,
        fold_runner=fold_runner,
        target_col=target_col,
        project_col=project_col,
        year_col=year_col,
        metric=metric,
        random_state=random_state,
        metadata={"pca_dim": pca_dim},
    )


def _tfidf_lr_from_config(
    config: Mapping[str, Any],
    random_state: int,
) -> LogisticRegression:
    params = {
        "C": 1.0,
        "max_iter": 3000,
        "solver": "liblinear",
        "dual": True,
        **dict(config),
    }
    params["random_state"] = random_state
    return LogisticRegression(**params)


def run_tfidf_lr_experiments(
    df: pd.DataFrame,
    folds: Sequence[tuple[pd.Index, pd.Index]],
    text_cols: Sequence[str],
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    *,
    embeddings: np.ndarray | None = None,
    embedding_metadata: pd.DataFrame | None = None,
    run_t1: bool = True,
    run_t2: bool = True,
    run_t3: bool = True,
    target_col: str = "target",
    project_col: str = "project_name",
    project_id_col: str = "project_id",
    year_col: str = "project_start_year",
    metric: str = "roc_auc",
    embedding_scaling: str = "l2",
    tfidf_config: Mapping[str, Any] | None = None,
    tfidf_lr_config: Mapping[str, Any] | None = None,
    feature_config: Mapping[str, Any] | None = None,
    tfidf_feature_output_dir: str | Path | None = "data/csv/tfidf_shared",
    random_state: int = 42,
) -> dict[str, ExperimentResult]:
    """Run T1-T3 while fitting each fold's TF-IDF vectorizers only once."""
    enabled = {
        "T1_tfidf_lr": bool(run_t1),
        "T2_tfidf_tabular_lr": bool(run_t2),
        "T3_tfidf_embedding_tabular_lr": bool(run_t3),
    }
    enabled_names = [name for name, should_run in enabled.items() if should_run]
    if not enabled_names:
        return {}
    if not text_cols or len(set(text_cols)) != len(text_cols):
        raise ValueError("text_cols must be non-empty and contain no duplicates.")
    if target_col in text_cols:
        raise ValueError("target_col must never be included in TF-IDF text_cols.")
    missing = [
        column
        for column in [*text_cols, target_col, project_col, year_col]
        if column not in df.columns
    ]
    if missing:
        raise KeyError(f"Missing TF-IDF experiment columns: {missing}")
    metric = _validate_metric(metric)
    _validate_binary_target(df[target_col], target_col)
    set_global_seed(random_state)

    embedding_matrix: np.ndarray | None = None
    if run_t3:
        if embeddings is None or embedding_metadata is None:
            raise ValueError("T3 requires embeddings and embedding_metadata.")
        embedding_matrix = validate_embedding_input(
            df,
            embeddings,
            embedding_metadata,
            project_id_col,
            split="train",
        )

    table: pd.DataFrame | None = None
    numeric: list[str] = []
    categorical: list[str] = []
    if run_t2 or run_t3:
        table, numeric, categorical = prepare_tabular_features(
            df,
            numeric_cols,
            categorical_cols,
            **dict(feature_config or {}),
        )

    oof = {
        name: pd.Series(np.nan, index=df.index, dtype=float, name=name)
        for name in enabled_names
    }
    fold_records: dict[str, list[dict[str, Any]]] = {
        name: [] for name in enabled_names
    }
    output_root = (
        Path(tfidf_feature_output_dir)
        if tfidf_feature_output_dir is not None
        else None
    )

    for fold_number, (train_idx, valid_idx) in enumerate(folds):
        train_idx = pd.Index(train_idx)
        valid_idx = pd.Index(valid_idx)
        train_positions = _positions_from_labels(df, train_idx)
        valid_positions = _positions_from_labels(df, valid_idx)
        if np.intersect1d(train_positions, valid_positions).size:
            raise ValueError("Training and validation indices overlap.")
        y_train = df.loc[train_idx, target_col].to_numpy(dtype=np.float32)
        y_valid = df.loc[valid_idx, target_col].to_numpy(dtype=np.float32)
        if np.unique(y_train).size < 2:
            raise ValueError(f"Fold {fold_number} training target has only one class.")

        started = time.perf_counter()
        tfidf_train, tfidf_valid, _, feature_names = fit_transform_tfidf_columns(
            train_df=df.loc[train_idx],
            transform_df=df.loc[valid_idx],
            text_cols=text_cols,
            tfidf_params=tfidf_config,
        )
        tfidf_train = _as_float32_csr(tfidf_train)
        tfidf_valid = _as_float32_csr(tfidf_valid)
        tfidf_seconds = time.perf_counter() - started

        tab_train: sparse.csr_matrix | None = None
        tab_valid: sparse.csr_matrix | None = None
        tabular_seconds = 0.0
        if run_t2 or run_t3:
            assert table is not None
            started = time.perf_counter()
            preprocessor = _make_sparse_tabular_preprocessor(numeric, categorical)
            tab_train = _as_float32_csr(
                preprocessor.fit_transform(table.loc[train_idx])
            )
            tab_valid = _as_float32_csr(
                preprocessor.transform(table.loc[valid_idx])
            )
            tabular_seconds = time.perf_counter() - started

        emb_train: sparse.csr_matrix | None = None
        emb_valid: sparse.csr_matrix | None = None
        embedding_seconds = 0.0
        if run_t3:
            assert embedding_matrix is not None
            started = time.perf_counter()
            dense_train, dense_valid, _ = _fit_transform_embeddings(
                embedding_matrix[train_positions],
                embedding_matrix[valid_positions],
                embedding_scaling,
            )
            emb_train = _as_float32_csr(dense_train)
            emb_valid = _as_float32_csr(dense_valid)
            embedding_seconds = time.perf_counter() - started

        fold_features: dict[str, tuple[sparse.csr_matrix, sparse.csr_matrix, float]] = {}
        if run_t1:
            fold_features["T1_tfidf_lr"] = (
                tfidf_train,
                tfidf_valid,
                tfidf_seconds,
            )
        if run_t2:
            assert tab_train is not None and tab_valid is not None
            fold_features["T2_tfidf_tabular_lr"] = (
                _sparse_hstack(tfidf_train, tab_train),
                _sparse_hstack(tfidf_valid, tab_valid),
                tfidf_seconds + tabular_seconds,
            )
        if run_t3:
            assert tab_train is not None and tab_valid is not None
            assert emb_train is not None and emb_valid is not None
            fold_features["T3_tfidf_embedding_tabular_lr"] = (
                _sparse_hstack(tfidf_train, emb_train, tab_train),
                _sparse_hstack(tfidf_valid, emb_valid, tab_valid),
                tfidf_seconds + tabular_seconds + embedding_seconds,
            )

        seen = make_seen_project_mask(
            df=df,
            train_idx=train_idx,
            valid_idx=valid_idx,
            project_col=project_col,
        ).to_numpy(dtype=bool)
        validation_year = _fold_year(df, valid_idx, year_col)

        for experiment, (x_train, x_valid, preprocessing_seconds) in fold_features.items():
            if not sparse.isspmatrix_csr(x_train) or not sparse.isspmatrix_csr(x_valid):
                raise AssertionError("T1-T3 features must remain CSR sparse.")
            model = _tfidf_lr_from_config(tfidf_lr_config or {}, random_state)
            started = time.perf_counter()
            model.fit(x_train, y_train)
            model_fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            prediction = np.asarray(model.predict_proba(x_valid)[:, 1], dtype=float)
            predict_seconds = time.perf_counter() - started
            if len(prediction) != len(valid_idx) or not np.isfinite(prediction).all():
                raise ValueError("TF-IDF validation predictions are invalid.")
            oof[experiment].loc[valid_idx] = prediction
            memory_mib = _csr_memory_mib(x_train)
            if (
                experiment == "T3_tfidf_embedding_tabular_lr"
                and memory_mib >= 1024
            ):
                warnings.warn(
                    f"T3 fold {fold_number} CSR matrix uses {memory_mib:.1f} MiB.",
                    UserWarning,
                    stacklevel=2,
                )
            fold_records[experiment].append(
                {
                    "experiment": experiment,
                    "fold": fold_number,
                    "validation_year": validation_year,
                    "train_rows": len(train_idx),
                    "valid_rows": len(valid_idx),
                    "metric": metric,
                    "overall_score": _safe_score(y_valid, prediction, metric),
                    "seen_score": _safe_score(
                        y_valid[seen], prediction[seen], metric
                    ),
                    "unseen_score": _safe_score(
                        y_valid[~seen], prediction[~seen], metric
                    ),
                    "seen_ratio": float(seen.mean()) if len(seen) else float("nan"),
                    "seen_rows": int(seen.sum()),
                    "unseen_rows": int((~seen).sum()),
                    "fit_seconds": preprocessing_seconds + model_fit_seconds,
                    "predict_seconds": predict_seconds,
                    "shared_tfidf_seconds": tfidf_seconds,
                    "model_fit_seconds": model_fit_seconds,
                    "input_dim": x_train.shape[1],
                    "n_nonzero": int(x_train.nnz),
                    "sparse_memory_mib": memory_mib,
                    "device": "cpu",
                    "best_iteration": np.nan,
                    "pca_explained_variance": np.nan,
                }
            )
            LOGGER.info(
                "%s fold=%s year=%s dim=%s nnz=%s memory=%.1fMiB",
                experiment,
                fold_number,
                validation_year,
                x_train.shape[1],
                x_train.nnz,
                memory_mib,
            )

        if output_root is not None:
            fold_dir = output_root / f"fold_{fold_number}_year_{validation_year}"
            save_sparse_features_csv(
                tfidf_train,
                train_idx,
                feature_names,
                fold_dir / "train_features.csv.gz",
                feature_names_path=fold_dir / "feature_names.csv.gz",
            )
            save_sparse_features_csv(
                tfidf_valid,
                valid_idx,
                feature_names,
                fold_dir / "valid_features.csv.gz",
            )

    return {
        name: ExperimentResult(
            name=name,
            oof=oof[name],
            fold_metrics=pd.DataFrame(fold_records[name]),
            metadata={
                "text_cols": list(text_cols),
                "tfidf_config": dict(tfidf_config or {}),
                "features_are_sparse": True,
            },
        )
        for name in enabled_names
    }


def build_oof_predictions(
    results: Sequence[ExperimentResult],
    index: pd.Index,
) -> pd.DataFrame:
    """Combine every experiment OOF without filling non-validation rows."""
    oof = pd.DataFrame(index=index.copy())
    for result in results:
        if not result.oof.index.equals(index):
            raise ValueError(f"OOF index mismatch for {result.name}")
        if result.name in oof.columns:
            raise ValueError(f"Duplicate experiment name: {result.name}")
        oof[result.name] = result.oof
    return oof


def build_experiment_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Create fold-year columns plus mean/latest/seen/unseen/time summaries."""
    required = {
        "experiment",
        "validation_year",
        "overall_score",
        "seen_score",
        "unseen_score",
        "fit_seconds",
        "predict_seconds",
    }
    missing = sorted(required - set(fold_metrics.columns))
    if missing:
        raise KeyError(f"fold_metrics is missing columns: {missing}")
    if fold_metrics.empty:
        return pd.DataFrame()

    pivot = fold_metrics.pivot(
        index="experiment",
        columns="validation_year",
        values="overall_score",
    )
    pivot.columns = [f"fold_{year}" for year in pivot.columns]
    summary = pivot.copy()
    grouped = fold_metrics.groupby("experiment", sort=False)
    summary["mean"] = grouped["overall_score"].mean()
    latest = (
        fold_metrics.sort_values("validation_year")
        .groupby("experiment", sort=False)
        .tail(1)
        .set_index("experiment")["overall_score"]
    )
    summary["latest"] = latest
    summary["seen_mean"] = grouped["seen_score"].mean()
    summary["unseen_mean"] = grouped["unseen_score"].mean()
    summary["fit_seconds_total"] = grouped["fit_seconds"].sum()
    summary["predict_seconds_total"] = grouped["predict_seconds"].sum()
    summary.index.name = "experiment"
    return summary.sort_index()


def _atomic_save_numpy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.save(file, np.asarray(values, dtype=np.float32), allow_pickle=False)
    temporary.replace(path)


def _atomic_save_parquet(path: Path, frame: pd.DataFrame, *, index: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=index)
    temporary.replace(path)


def save_experiment_outputs(
    oof_predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str | Path = "outputs",
) -> dict[str, Path]:
    """Persist unified OOF, long fold diagnostics, and comparison tables."""
    directory = Path(output_dir)
    paths = {
        "oof": directory / "oof_predictions.parquet",
        "fold_metrics": directory / "fold_metrics.parquet",
        "summary_parquet": directory / "experiment_summary.parquet",
        "summary_csv": directory / "experiment_summary.csv",
    }
    _atomic_save_parquet(paths["oof"], oof_predictions, index=True)
    _atomic_save_parquet(paths["fold_metrics"], fold_metrics, index=False)
    _atomic_save_parquet(paths["summary_parquet"], summary, index=True)
    temporary_csv = paths["summary_csv"].with_suffix(".csv.tmp")
    summary.to_csv(temporary_csv, index=True)
    temporary_csv.replace(paths["summary_csv"])
    return paths


def _full_tabular_transform(
    train_table: pd.DataFrame,
    test_table: pd.DataFrame,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    *,
    scale_numeric: bool,
) -> tuple[np.ndarray, np.ndarray]:
    preprocessor = _make_tabular_preprocessor(
        numeric_cols,
        categorical_cols,
        scale_numeric=scale_numeric,
    )
    train_values = np.asarray(
        preprocessor.fit_transform(train_table), dtype=np.float32
    )
    test_values = np.asarray(preprocessor.transform(test_table), dtype=np.float32)
    return train_values, test_values


def _full_sparse_tabular_transform(
    train_table: pd.DataFrame,
    test_table: pd.DataFrame,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    preprocessor = _make_sparse_tabular_preprocessor(numeric_cols, categorical_cols)
    train_values = _as_float32_csr(preprocessor.fit_transform(train_table))
    test_values = _as_float32_csr(preprocessor.transform(test_table))
    return train_values, test_values


def fit_full_and_predict_test(
    experiment: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_embeddings: np.ndarray | None,
    test_embeddings: np.ndarray | None,
    train_embedding_metadata: pd.DataFrame | None,
    test_embedding_metadata: pd.DataFrame | None,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    *,
    config: Mapping[str, Any] | None = None,
    pca_dim: int | None = None,
    target_col: str = "target",
    project_id_col: str = "project_id",
) -> np.ndarray:
    """Fit one selected experiment on all train rows and predict test."""
    cfg = _merge_config(config)
    seed = int(cfg["random_state"])
    set_global_seed(seed)
    _validate_binary_target(train[target_col], target_col)
    y_train = train[target_col].to_numpy(dtype=np.float32)
    uses_embedding = experiment in {
        "E1_embedding_lr",
        "E2_embedding_tabular_lr",
        "E3_embedding_mlp",
        "E4_embedding_tabular_mlp",
        "T3_tfidf_embedding_tabular_lr",
    } or experiment.startswith("E6_embedding_tabular_xgb")
    if uses_embedding:
        if (
            train_embeddings is None
            or test_embeddings is None
            or train_embedding_metadata is None
            or test_embedding_metadata is None
        ):
            raise ValueError(f"{experiment} requires train/test embeddings and metadata.")
        train_matrix = validate_embedding_input(
            train,
            train_embeddings,
            train_embedding_metadata,
            project_id_col,
            split="train",
        )
        test_matrix = validate_embedding_input(
            test,
            test_embeddings,
            test_embedding_metadata,
            project_id_col,
            split="test",
        )
    else:
        train_matrix = np.empty((len(train), 0), dtype=np.float32)
        test_matrix = np.empty((len(test), 0), dtype=np.float32)

    uses_tabular = experiment in {
        "E2_embedding_tabular_lr",
        "E4_embedding_tabular_mlp",
        "E5_tabular_catboost",
        "T2_tfidf_tabular_lr",
        "T3_tfidf_embedding_tabular_lr",
    } or experiment.startswith("E6_embedding_tabular_xgb")
    if uses_tabular:
        feature_config = dict(cfg["feature_engineering"])
        train_table, numeric, categorical = prepare_tabular_features(
            train, numeric_cols, categorical_cols, **feature_config
        )
        test_table, test_numeric, test_categorical = prepare_tabular_features(
            test, numeric_cols, categorical_cols, **feature_config
        )
        if numeric != test_numeric or categorical != test_categorical:
            raise ValueError("Train/test engineered tabular feature definitions differ.")
    else:
        train_table = pd.DataFrame(index=train.index)
        test_table = pd.DataFrame(index=test.index)
        numeric, categorical = [], []

    if experiment in {
        "T1_tfidf_lr",
        "T2_tfidf_tabular_lr",
        "T3_tfidf_embedding_tabular_lr",
    }:
        if target_col in cfg["text_cols"]:
            raise ValueError("target_col must never be included in TF-IDF text_cols.")
        tfidf_train, tfidf_test, _, _ = fit_transform_tfidf_columns(
            train_df=train,
            transform_df=test,
            text_cols=cfg["text_cols"],
            tfidf_params=cfg["tfidf"],
        )
        tfidf_train = _as_float32_csr(tfidf_train)
        tfidf_test = _as_float32_csr(tfidf_test)
        if experiment == "T1_tfidf_lr":
            x_train, x_test = tfidf_train, tfidf_test
        else:
            tab_train, tab_test = _full_sparse_tabular_transform(
                train_table,
                test_table,
                numeric,
                categorical,
            )
            if experiment == "T2_tfidf_tabular_lr":
                x_train = _sparse_hstack(tfidf_train, tab_train)
                x_test = _sparse_hstack(tfidf_test, tab_test)
            else:
                emb_train, emb_test = _transform_embeddings_with_fitted(
                    train_matrix,
                    test_matrix,
                    str(cfg["embedding_scaling"]),
                )
                x_train = _sparse_hstack(tfidf_train, emb_train, tab_train)
                x_test = _sparse_hstack(tfidf_test, emb_test, tab_test)
        model = _tfidf_lr_from_config(cfg["tfidf_lr"], seed)
        model.fit(x_train, y_train)
        prediction = model.predict_proba(x_test)[:, 1]
    elif experiment == "E1_embedding_lr":
        x_train, x_test = _transform_embeddings_with_fitted(
            train_matrix, test_matrix, str(cfg["embedding_scaling"])
        )
        model = _lr_from_config(cfg["lr"], seed)
        model.fit(x_train, y_train)
        prediction = model.predict_proba(x_test)[:, 1]
    elif experiment == "E2_embedding_tabular_lr":
        tab_train, tab_test = _full_tabular_transform(
            train_table, test_table, numeric, categorical, scale_numeric=True
        )
        emb_train, emb_test = _transform_embeddings_with_fitted(
            train_matrix, test_matrix, str(cfg["embedding_scaling"])
        )
        model = _lr_from_config(cfg["lr"], seed)
        model.fit(_combine_dense(emb_train, tab_train), y_train)
        prediction = model.predict_proba(_combine_dense(emb_test, tab_test))[:, 1]
    elif experiment in {"E3_embedding_mlp", "E4_embedding_tabular_mlp"}:
        emb_train, emb_test = _transform_embeddings_with_fitted(
            train_matrix, test_matrix, str(cfg["embedding_scaling"])
        )
        if experiment == "E4_embedding_tabular_mlp":
            tab_train, tab_test = _full_tabular_transform(
                train_table, test_table, numeric, categorical, scale_numeric=True
            )
            x_train = _combine_dense(emb_train, tab_train)
            x_test = _combine_dense(emb_test, tab_test)
        else:
            x_train, x_test = emb_train, emb_test
        mlp_config = dict(cfg["mlp"])
        mlp_config["max_epochs"] = int(mlp_config.get("full_epochs", 30))
        prediction, _ = _fit_predict_torch_mlp(
            x_train, y_train, x_test, None, mlp_config, seed
        )
    elif experiment == "E5_tabular_catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as error:
            raise ImportError("Install catboost to run E5.") from error
        train_cat = _catboost_frame(train_table, numeric, categorical)
        test_cat = _catboost_frame(test_table, numeric, categorical)
        params = dict(cfg["catboost"])
        params.pop("early_stopping_rounds", None)
        params["random_seed"] = seed
        model = CatBoostClassifier(**params)
        model.fit(train_cat, y_train, cat_features=categorical)
        prediction = model.predict_proba(test_cat)[:, 1]
    elif experiment.startswith("E6_embedding_tabular_xgb"):
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError("Install xgboost to run E6.") from error
        tab_train, tab_test = _full_tabular_transform(
            train_table, test_table, numeric, categorical, scale_numeric=False
        )
        emb_train, emb_test, _, _ = _fit_pca(
            train_matrix, test_matrix, pca_dim, seed
        )
        params = dict(cfg["xgboost"])
        params.pop("early_stopping_rounds", None)
        params["random_state"] = seed
        model = XGBClassifier(**params)
        model.fit(_combine_dense(emb_train, tab_train), y_train, verbose=False)
        prediction = model.predict_proba(_combine_dense(emb_test, tab_test))[:, 1]
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    prediction = np.asarray(prediction, dtype=np.float32).reshape(-1)
    if len(prediction) != len(test) or not np.isfinite(prediction).all():
        raise ValueError("Final test predictions are invalid.")
    if ((prediction < 0) | (prediction > 1)).any():
        raise ValueError("Final test predictions must be probabilities in [0, 1].")
    return prediction


def run_all_experiments(
    train: pd.DataFrame,
    folds: Sequence[tuple[pd.Index, pd.Index]],
    train_embeddings: np.ndarray | None,
    train_embedding_metadata: pd.DataFrame | None,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    *,
    config: Mapping[str, Any] | None = None,
    target_col: str = "target",
    project_col: str = "project_name",
    project_id_col: str = "project_id",
    year_col: str = "project_start_year",
    test: pd.DataFrame | None = None,
    test_embeddings: np.ndarray | None = None,
    test_embedding_metadata: pd.DataFrame | None = None,
) -> ExperimentSuiteResult:
    """Run enabled E1-E6/T1-T3 with exactly the supplied time folds."""
    cfg = _merge_config(config)
    metric = _validate_metric(str(cfg["metric"]))
    seed = int(cfg["random_state"])
    common = {
        "target_col": target_col,
        "project_col": project_col,
        "year_col": year_col,
        "metric": metric,
        "random_state": seed,
    }
    embedding_common = {
        **common,
        "project_id_col": project_id_col,
    }
    results: list[ExperimentResult] = []
    needs_train_embeddings = any(
        bool(cfg[key])
        for key in ("run_e1", "run_e2", "run_e3", "run_e4", "run_e6", "run_t3")
    )
    if needs_train_embeddings and (
        train_embeddings is None or train_embedding_metadata is None
    ):
        raise ValueError("Enabled experiments require train embeddings and metadata.")

    if cfg["run_e1"]:
        results.append(
            run_e1_embedding_lr(
                train,
                train_embeddings,
                train_embedding_metadata,
                folds,
                embedding_scaling=str(cfg["embedding_scaling"]),
                lr_config=cfg["lr"],
                **embedding_common,
            )
        )
    if cfg["run_e2"]:
        results.append(
            run_e2_embedding_tabular_lr(
                train,
                train_embeddings,
                train_embedding_metadata,
                folds,
                numeric_cols,
                categorical_cols,
                embedding_scaling=str(cfg["embedding_scaling"]),
                lr_config=cfg["lr"],
                feature_config=cfg["feature_engineering"],
                **embedding_common,
            )
        )
    if cfg["run_e3"]:
        results.append(
            run_e3_embedding_mlp(
                train,
                train_embeddings,
                train_embedding_metadata,
                folds,
                embedding_scaling=str(cfg["embedding_scaling"]),
                mlp_config=cfg["mlp"],
                **embedding_common,
            )
        )
    if cfg["run_e4"]:
        results.append(
            run_e4_embedding_tabular_mlp(
                train,
                train_embeddings,
                train_embedding_metadata,
                folds,
                numeric_cols,
                categorical_cols,
                embedding_scaling=str(cfg["embedding_scaling"]),
                mlp_config=cfg["mlp"],
                feature_config=cfg["feature_engineering"],
                **embedding_common,
            )
        )
    if cfg["run_e5"]:
        results.append(
            run_e5_tabular_catboost(
                train,
                folds,
                numeric_cols,
                categorical_cols,
                catboost_config=cfg["catboost"],
                feature_config=cfg["feature_engineering"],
                **common,
            )
        )
    if cfg["run_e6"]:
        pca_dims = list(dict.fromkeys(cfg.get("e6_pca_dims", [None])))
        if not pca_dims:
            raise ValueError("e6_pca_dims must contain at least one value when E6 runs.")
        for pca_dim in pca_dims:
            results.append(
                run_e6_embedding_tabular_xgboost(
                    train,
                    train_embeddings,
                    train_embedding_metadata,
                    folds,
                    numeric_cols,
                    categorical_cols,
                    pca_dim=pca_dim,
                    xgboost_config=cfg["xgboost"],
                    feature_config=cfg["feature_engineering"],
                    **embedding_common,
                )
            )
    tfidf_results = run_tfidf_lr_experiments(
        df=train,
        folds=folds,
        text_cols=cfg["text_cols"],
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        embeddings=train_embeddings,
        embedding_metadata=train_embedding_metadata,
        run_t1=bool(cfg["run_t1"]),
        run_t2=bool(cfg["run_t2"]),
        run_t3=bool(cfg["run_t3"]),
        target_col=target_col,
        project_col=project_col,
        project_id_col=project_id_col,
        year_col=year_col,
        metric=metric,
        embedding_scaling=str(cfg["embedding_scaling"]),
        tfidf_config=cfg["tfidf"],
        tfidf_lr_config=cfg["tfidf_lr"],
        feature_config=cfg["feature_engineering"],
        tfidf_feature_output_dir=cfg["tfidf_feature_output_dir"],
        random_state=seed,
    )
    results.extend(tfidf_results.values())
    if not results:
        raise ValueError("No experiment is enabled in config.")

    result_map = {result.name: result for result in results}
    oof_predictions = build_oof_predictions(results, train.index)
    fold_metrics = pd.concat(
        [result.fold_metrics for result in results], ignore_index=True
    )
    summary = build_experiment_summary(fold_metrics)
    output_dir = Path(cfg["output_dir"])
    save_experiment_outputs(
        oof_predictions,
        fold_metrics,
        summary,
        output_dir=output_dir,
    )

    test_predictions: dict[str, np.ndarray] = {}
    if cfg["run_final_test_prediction"]:
        if test is None:
            raise ValueError("test is required when run_final_test_prediction=True.")
        selected = list(cfg.get("final_experiments", []))
        if not selected:
            raise ValueError(
                "Set final_experiments after reviewing CV results before final refit."
            )
        unknown = sorted(set(selected) - set(result_map))
        if unknown:
            raise ValueError(f"final_experiments were not run in CV: {unknown}")
        embedding_free = {
            "E5_tabular_catboost",
            "T1_tfidf_lr",
            "T2_tfidf_tabular_lr",
        }
        needs_embeddings = any(name not in embedding_free for name in selected)
        if needs_embeddings and (
            test_embeddings is None or test_embedding_metadata is None
        ):
            raise ValueError(
                "Selected final experiments require test embeddings and metadata."
            )
        prediction_dir = output_dir / "test_predictions"
        for experiment in selected:
            pca_dim = result_map[experiment].metadata.get("pca_dim")
            prediction = fit_full_and_predict_test(
                experiment=experiment,
                train=train,
                test=test,
                train_embeddings=train_embeddings,
                test_embeddings=test_embeddings,
                train_embedding_metadata=train_embedding_metadata,
                test_embedding_metadata=test_embedding_metadata,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                config=cfg,
                pca_dim=pca_dim,
                target_col=target_col,
                project_id_col=project_id_col,
            )
            test_predictions[experiment] = prediction
            _atomic_save_numpy(prediction_dir / f"{experiment}.npy", prediction)

    return ExperimentSuiteResult(
        results=result_map,
        oof_predictions=oof_predictions,
        fold_metrics=fold_metrics,
        summary=summary,
        test_predictions=test_predictions,
    )


__all__ = [
    "ExperimentResult",
    "ExperimentSuiteResult",
    "build_experiment_summary",
    "build_oof_predictions",
    "default_modeling_config",
    "fit_full_and_predict_test",
    "prepare_tabular_features",
    "run_all_experiments",
    "run_e1_embedding_lr",
    "run_e2_embedding_tabular_lr",
    "run_e3_embedding_mlp",
    "run_e4_embedding_tabular_mlp",
    "run_e5_tabular_catboost",
    "run_e6_embedding_tabular_xgboost",
    "run_tfidf_lr_experiments",
    "save_experiment_outputs",
    "set_global_seed",
    "validate_embedding_input",
]
