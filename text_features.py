"""Character TF-IDF experiments with leakage-safe time-series CV."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from validation import Fold, make_seen_project_mask


Metric: TypeAlias = str | Callable[[pd.Series, np.ndarray], float]

DEFAULT_TFIDF_PARAMS: dict[str, Any] = {
    "analyzer": "char",
    "ngram_range": (2, 5),
    "min_df": 2,
    "max_features": 300_000,
    "sublinear_tf": True,
    "dtype": np.float32,
}

# For high-dimensional sparse text, liblinear with the dual formulation is a
# strong, deterministic binary-classification baseline when features outnumber
# rows. Override solver/dual through logreg_params for unusually large datasets.
DEFAULT_LOGREG_PARAMS: dict[str, Any] = {
    "C": 1.0,
    "max_iter": 3000,
    "solver": "liblinear",
    "dual": True,
    "random_state": 42,
}


def normalize_text(series: pd.Series) -> pd.Series:
    """Apply minimal normalization without deleting numbers or symbols."""

    def normalize_one(value: object) -> str:
        if pd.isna(value):
            return ""
        text = unicodedata.normalize("NFKC", str(value))
        return re.sub(r"\s+", " ", text).strip()

    return series.map(normalize_one)


def make_tfidf_vectorizer(
    tfidf_params: Mapping[str, Any] | None = None,
) -> TfidfVectorizer:
    """Build a configurable character n-gram vectorizer."""
    params = {**DEFAULT_TFIDF_PARAMS, **dict(tfidf_params or {})}
    return TfidfVectorizer(**params)


def make_logistic_regression(
    logreg_params: Mapping[str, Any] | None = None,
) -> LogisticRegression:
    """Build the sparse-text binary classifier used by the experiments."""
    params = {**DEFAULT_LOGREG_PARAMS, **dict(logreg_params or {})}
    return LogisticRegression(**params)


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _safe_path_name(value: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("_.")
    return safe or "experiment"


def _validate_folds(df: pd.DataFrame, folds: Sequence[Fold]) -> None:
    if not df.index.is_unique:
        raise ValueError("df.index must be unique because fold labels are used with .loc.")
    for fold, (train_idx, valid_idx) in enumerate(folds):
        if len(train_idx) == 0 or len(valid_idx) == 0:
            raise ValueError(f"Fold {fold} contains an empty train or validation split.")
        if not train_idx.intersection(valid_idx).empty:
            raise ValueError(f"Fold {fold} has overlapping train and validation labels.")
        if (df.index.get_indexer(train_idx) < 0).any():
            raise KeyError(f"Fold {fold} contains unknown training labels.")
        if (df.index.get_indexer(valid_idx) < 0).any():
            raise KeyError(f"Fold {fold} contains unknown validation labels.")


def _validate_binary_target(target: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(target, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{name!r} contains missing or non-numeric values.")
    values = set(numeric.unique().tolist())
    if not values.issubset({0, 1}) or len(values) < 2:
        raise ValueError(f"{name!r} must contain both binary classes 0 and 1.")
    return numeric.astype(np.int8)


def _score_predictions(
    y_true: pd.Series,
    prediction: pd.Series | np.ndarray,
    metric: Metric,
) -> float:
    if len(y_true) == 0:
        return float("nan")
    probability = np.asarray(prediction, dtype=float)
    if isinstance(metric, str):
        metric_name = metric.lower().replace("-", "_")
        if metric_name in {"roc_auc", "auc"}:
            if y_true.nunique() < 2:
                return float("nan")
            return float(roc_auc_score(y_true, probability))
        if metric_name in {"log_loss", "logloss"}:
            return float(log_loss(y_true, probability, labels=[0, 1]))
        raise ValueError("metric must be 'roc_auc', 'log_loss', or a callable.")
    return float(metric(y_true, probability))


def _prefixed_feature_names(
    vectorizers: Mapping[str, TfidfVectorizer],
    text_cols: Sequence[str],
) -> np.ndarray:
    names: list[np.ndarray] = []
    for column in text_cols:
        column_names = vectorizers[column].get_feature_names_out()
        names.append(np.char.add(f"{column}__", column_names.astype(str)))
    return np.concatenate(names) if names else np.array([], dtype=str)


def _fit_transform_columns(
    train_df: pd.DataFrame,
    transform_df: pd.DataFrame,
    text_cols: Sequence[str],
    tfidf_params: Mapping[str, Any] | None,
) -> tuple[
    sparse.csr_matrix,
    sparse.csr_matrix,
    dict[str, TfidfVectorizer],
    np.ndarray,
]:
    """Fit each vectorizer on train only and transform a second dataset."""
    train_matrices: list[sparse.spmatrix] = []
    transform_matrices: list[sparse.spmatrix] = []
    vectorizers: dict[str, TfidfVectorizer] = {}

    for column in text_cols:
        vectorizer = make_tfidf_vectorizer(tfidf_params)
        train_text = normalize_text(train_df[column])
        transform_text = normalize_text(transform_df[column])

        # Leakage boundary: transform_df is never passed to fit or fit_transform.
        train_matrix = vectorizer.fit_transform(train_text)
        transform_matrix = vectorizer.transform(transform_text)

        vectorizers[column] = vectorizer
        train_matrices.append(train_matrix)
        transform_matrices.append(transform_matrix)

    train_features = sparse.hstack(train_matrices, format="csr", dtype=np.float32)
    transform_features = sparse.hstack(
        transform_matrices, format="csr", dtype=np.float32
    )
    feature_names = _prefixed_feature_names(vectorizers, text_cols)

    assert train_features.shape[1] == transform_features.shape[1]
    assert train_features.shape[1] == len(feature_names)
    return train_features, transform_features, vectorizers, feature_names


def save_sparse_features_csv(
    matrix: sparse.spmatrix,
    row_index: pd.Index,
    feature_names: Sequence[str],
    output_path: str | Path,
    feature_names_path: str | Path | None = None,
    row_chunk_size: int = 2_000,
) -> None:
    """Save a sparse matrix as compressed long-form CSV without densifying it.

    The feature file contains ``row_index, feature_index, value``. A separate
    feature-name file maps each integer feature index to its prefixed n-gram.
    Paths ending in ``.gz`` are written as gzip-compressed CSV.
    """
    if row_chunk_size <= 0:
        raise ValueError("row_chunk_size must be greater than 0.")

    csr = sparse.csr_matrix(matrix)
    row_labels = pd.Index(row_index)
    names = np.asarray(feature_names, dtype=str)
    if csr.shape != (len(row_labels), len(names)):
        raise ValueError(
            "matrix shape must match the lengths of row_index and feature_names."
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    wrote_rows = False
    for start in range(0, csr.shape[0], row_chunk_size):
        stop = min(start + row_chunk_size, csr.shape[0])
        local_rows, columns, values = sparse.find(csr[start:stop])
        if len(values) == 0:
            continue
        chunk = pd.DataFrame(
            {
                "row_index": row_labels.to_numpy()[start + local_rows],
                "feature_index": columns.astype(np.int32),
                "value": values.astype(np.float32),
            }
        )
        chunk.to_csv(
            output,
            mode="a" if wrote_rows else "w",
            header=not wrote_rows,
            index=False,
            compression="infer",
        )
        wrote_rows = True

    if not wrote_rows:
        pd.DataFrame(columns=["row_index", "feature_index", "value"]).to_csv(
            output, index=False, compression="infer"
        )

    if feature_names_path is not None:
        names_output = Path(feature_names_path)
        names_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "feature_index": np.arange(len(names), dtype=np.int32),
                "feature": names,
            }
        ).to_csv(names_output, index=False, compression="infer")


def _save_fold_outputs(
    output_dir: Path,
    train_features: sparse.spmatrix,
    valid_features: sparse.spmatrix,
    train_idx: pd.Index,
    valid_idx: pd.Index,
    feature_names: np.ndarray,
    validation_output: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names_path = output_dir / "feature_names.csv.gz"
    save_sparse_features_csv(
        train_features,
        train_idx,
        feature_names,
        output_dir / "train_features.csv.gz",
        feature_names_path=feature_names_path,
    )
    save_sparse_features_csv(
        valid_features,
        valid_idx,
        feature_names,
        output_dir / "valid_features.csv.gz",
    )
    validation_output.to_csv(
        output_dir / "validation_predictions.csv.gz",
        index_label="row_index",
        compression="gzip",
    )


def cross_validate_text_columns(
    df: pd.DataFrame,
    folds: Sequence[Fold],
    text_cols: Sequence[str],
    target_col: str = "target",
    project_col: str = "project_name",
    year_col: str = "project_start_year",
    model_name: str = "all_columns_char_tfidf",
    metric: Metric = "roc_auc",
    tfidf_params: Mapping[str, Any] | None = None,
    logreg_params: Mapping[str, Any] | None = None,
    feature_output_dir: str | Path | None = "data/csv",
) -> dict[str, Any]:
    """Evaluate independent per-column TF-IDF matrices joined with hstack.

    Every vectorizer is recreated and fit only on that fold's training rows.
    Generated sparse features, OOF predictions, and fold metrics are saved as
    CSV when ``feature_output_dir`` is not ``None``.
    """
    if not text_cols:
        raise ValueError("text_cols must contain at least one column.")
    if len(set(text_cols)) != len(text_cols):
        raise ValueError("text_cols must not contain duplicates.")
    _require_columns(df, [*text_cols, target_col, project_col, year_col])
    _validate_folds(df, folds)
    target = _validate_binary_target(df[target_col], target_col)

    oof = pd.Series(np.nan, index=df.index, dtype=float, name="prediction")
    fold_rows: list[dict[str, Any]] = []
    experiment_dir = (
        Path(feature_output_dir) / _safe_path_name(model_name)
        if feature_output_dir is not None
        else None
    )

    for fold, (train_idx, valid_idx) in enumerate(folds):
        train_df = df.loc[train_idx]
        valid_df = df.loc[valid_idx]
        train_target = target.loc[train_idx]
        valid_target = target.loc[valid_idx]
        if train_target.nunique() < 2:
            raise ValueError(f"Fold {fold} training target contains only one class.")

        valid_years = pd.to_numeric(valid_df[year_col], errors="raise").unique()
        if len(valid_years) != 1:
            raise ValueError(f"Fold {fold} validation rows span multiple years.")
        validation_year = int(valid_years[0])

        train_features, valid_features, _, feature_names = _fit_transform_columns(
            train_df=train_df,
            transform_df=valid_df,
            text_cols=text_cols,
            tfidf_params=tfidf_params,
        )
        model = make_logistic_regression(logreg_params)
        model.fit(train_features, train_target)
        valid_prediction = model.predict_proba(valid_features)[:, 1]
        oof.loc[valid_idx] = valid_prediction

        prediction = pd.Series(valid_prediction, index=valid_idx, name="prediction")
        seen_project = make_seen_project_mask(
            df=df,
            train_idx=train_idx,
            valid_idx=valid_idx,
            project_col=project_col,
        )
        overall_score = _score_predictions(valid_target, prediction, metric)
        seen_score = _score_predictions(
            valid_target.loc[seen_project], prediction.loc[seen_project], metric
        )
        unseen_score = _score_predictions(
            valid_target.loc[~seen_project], prediction.loc[~seen_project], metric
        )

        fold_rows.append(
            {
                "model": model_name,
                "fold": fold,
                "validation_year": validation_year,
                "metric": metric if isinstance(metric, str) else metric.__name__,
                "score": overall_score,
                "seen_score": seen_score,
                "unseen_score": unseen_score,
                "seen_ratio": float(seen_project.mean()),
                "train_size": len(train_idx),
                "validation_size": len(valid_idx),
                "n_features": train_features.shape[1],
                "is_latest_fold": fold == len(folds) - 1,
            }
        )

        if experiment_dir is not None:
            validation_output = pd.DataFrame(
                {
                    target_col: valid_target,
                    "prediction": prediction,
                    "seen_project": seen_project,
                    year_col: valid_df[year_col],
                },
                index=valid_idx,
            )
            fold_dir = experiment_dir / f"fold_{fold}_year_{validation_year}"
            _save_fold_outputs(
                output_dir=fold_dir,
                train_features=train_features,
                valid_features=valid_features,
                train_idx=train_idx,
                valid_idx=valid_idx,
                feature_names=feature_names,
                validation_output=validation_output,
            )

    fold_scores = pd.DataFrame(fold_rows)
    if experiment_dir is not None:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                target_col: target,
                "prediction": oof,
                year_col: df[year_col],
                "evaluated": oof.notna(),
            },
            index=df.index,
        ).to_csv(
            experiment_dir / "oof_predictions.csv.gz",
            index_label="row_index",
            compression="gzip",
        )
        fold_scores.to_csv(experiment_dir / "fold_scores.csv", index=False)

    return {
        "model_name": model_name,
        "text_cols": list(text_cols),
        "metric": metric if isinstance(metric, str) else metric.__name__,
        "oof": oof,
        "fold_scores": fold_scores,
    }


def cross_validate_single_text_column(
    df: pd.DataFrame,
    folds: Sequence[Fold],
    text_col: str,
    target_col: str = "target",
    project_col: str = "project_name",
    year_col: str = "project_start_year",
    model_name: str | None = None,
    metric: Metric = "roc_auc",
    tfidf_params: Mapping[str, Any] | None = None,
    logreg_params: Mapping[str, Any] | None = None,
    feature_output_dir: str | Path | None = "data/csv",
) -> dict[str, Any]:
    """Evaluate one text column using the same leakage-safe implementation."""
    return cross_validate_text_columns(
        df=df,
        folds=folds,
        text_cols=[text_col],
        target_col=target_col,
        project_col=project_col,
        year_col=year_col,
        model_name=model_name or f"{text_col}_char_tfidf",
        metric=metric,
        tfidf_params=tfidf_params,
        logreg_params=logreg_params,
        feature_output_dir=feature_output_dir,
    )


def show_top_coefficients(
    vectorizers: Mapping[str, TfidfVectorizer],
    model: LogisticRegression,
    top_n: int = 50,
) -> dict[str, pd.DataFrame]:
    """Print and return the strongest positive and negative text features."""
    if top_n <= 0:
        raise ValueError("top_n must be greater than 0.")
    text_cols = list(vectorizers)
    feature_names = _prefixed_feature_names(vectorizers, text_cols)
    coefficients = np.asarray(model.coef_).ravel()
    if len(coefficients) != len(feature_names):
        raise ValueError("Model coefficients do not match vectorizer features.")

    table = pd.DataFrame(
        {"feature": feature_names, "coefficient": coefficients}
    )
    positive = table.nlargest(top_n, "coefficient").reset_index(drop=True)
    negative = table.nsmallest(top_n, "coefficient").reset_index(drop=True)

    print("target=1 direction")
    print(positive.to_string(index=False))
    print("\ntarget=0 direction")
    print(negative.to_string(index=False))
    return {"positive": positive, "negative": negative}


def fit_full_text_model_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_cols: Sequence[str],
    target_col: str = "target",
    model_name: str = "all_columns_char_tfidf_final",
    tfidf_params: Mapping[str, Any] | None = None,
    logreg_params: Mapping[str, Any] | None = None,
    feature_output_dir: str | Path | None = "data/csv",
) -> dict[str, Any]:
    """Fit vectorizers/model on all train rows and predict test probabilities."""
    if not text_cols:
        raise ValueError("text_cols must contain at least one column.")
    _require_columns(train_df, [*text_cols, target_col])
    _require_columns(test_df, list(text_cols))
    if not train_df.index.is_unique or not test_df.index.is_unique:
        raise ValueError("train_df and test_df indices must be unique.")

    target = _validate_binary_target(train_df[target_col], target_col)
    train_features, test_features, vectorizers, feature_names = (
        _fit_transform_columns(
            train_df=train_df,
            transform_df=test_df,
            text_cols=text_cols,
            tfidf_params=tfidf_params,
        )
    )
    model = make_logistic_regression(logreg_params)
    model.fit(train_features, target)
    test_prediction = pd.Series(
        model.predict_proba(test_features)[:, 1],
        index=test_df.index,
        name="prediction",
    )

    if feature_output_dir is not None:
        output_dir = Path(feature_output_dir) / _safe_path_name(model_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_sparse_features_csv(
            train_features,
            train_df.index,
            feature_names,
            output_dir / "train_features.csv.gz",
            feature_names_path=output_dir / "feature_names.csv.gz",
        )
        save_sparse_features_csv(
            test_features,
            test_df.index,
            feature_names,
            output_dir / "test_features.csv.gz",
        )
        test_prediction.to_csv(
            output_dir / "test_predictions.csv.gz",
            index_label="row_index",
            compression="gzip",
        )

    return {
        "vectorizers": vectorizers,
        "model": model,
        "test_pred": test_prediction,
        "feature_names": feature_names,
    }


def compare_cv_results(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Create a compact model-by-year score table with latest-fold diagnostics."""
    rows: list[dict[str, Any]] = []
    for result in results:
        scores = pd.DataFrame(result["fold_scores"])
        if scores.empty:
            continue
        scores = scores.sort_values("fold")
        row: dict[str, Any] = {"model": result["model_name"]}
        for record in scores.itertuples(index=False):
            row[f"fold_{record.validation_year}"] = record.score
        latest = scores.iloc[-1]
        row.update(
            {
                "mean": float(scores["score"].mean()),
                "latest_fold_score": float(latest["score"]),
                "seen_score": float(scores["seen_score"].mean()),
                "unseen_score": float(scores["unseen_score"].mean()),
                "latest_seen_score": float(latest["seen_score"]),
                "latest_unseen_score": float(latest["unseen_score"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()


__all__ = [
    "compare_cv_results",
    "cross_validate_single_text_column",
    "cross_validate_text_columns",
    "fit_full_text_model_predict",
    "make_logistic_regression",
    "make_tfidf_vectorizer",
    "normalize_text",
    "save_sparse_features_csv",
    "show_top_coefficients",
]
