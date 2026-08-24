"""ROC-AUC hill-climbing ensemble and submission helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class HillClimbingEnsembleResult:
    """Artifacts from greedy weighted blending on aligned OOF predictions."""

    weights: pd.Series
    oof_prediction: pd.Series
    score: float
    individual_scores: pd.Series
    history: pd.DataFrame
    n_scored_rows: int


def _resolve_target(target: pd.Series | Sequence[int], index: pd.Index) -> pd.Series:
    if isinstance(target, pd.Series):
        if not target.index.equals(index):
            raise ValueError("target index must exactly match oof_predictions.index.")
        values = target.copy()
    else:
        values = pd.Series(np.asarray(target), index=index)
    if values.isna().any():
        raise ValueError("target contains missing values.")
    if not set(pd.unique(values)).issubset({0, 1, False, True}):
        raise ValueError("target must be binary 0/1.")
    return values.astype(int)


def _validate_oof_candidates(
    oof_predictions: pd.DataFrame,
    target: pd.Series | Sequence[int],
    candidate_models: Sequence[str] | None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if not oof_predictions.index.is_unique:
        raise ValueError("oof_predictions.index must be unique.")
    if candidate_models is None:
        candidates = list(oof_predictions.columns)
    else:
        candidates = list(candidate_models)
    if not candidates:
        raise ValueError("At least one candidate model is required.")
    if len(candidates) != len(set(candidates)):
        raise ValueError("candidate_models must not contain duplicates.")
    missing = [name for name in candidates if name not in oof_predictions.columns]
    if missing:
        raise KeyError(f"OOF predictions are missing candidate models: {missing}")

    predictions = oof_predictions.loc[:, candidates].apply(
        pd.to_numeric, errors="raise"
    )
    reference_mask = predictions.iloc[:, 0].notna()
    for name in predictions.columns[1:]:
        if not predictions[name].notna().equals(reference_mask):
            raise ValueError(
                "All ensemble candidates must use exactly the same OOF rows; "
                f"{name!r} differs from {predictions.columns[0]!r}."
            )
    if not reference_mask.any():
        raise ValueError("No rows contain OOF predictions.")
    scored = predictions.loc[reference_mask]
    if not np.isfinite(scored.to_numpy(dtype=float)).all():
        raise ValueError("OOF predictions contain non-finite values.")
    if ((scored < 0) | (scored > 1)).any(axis=None):
        raise ValueError("OOF predictions must be probabilities in [0, 1].")

    target_series = _resolve_target(target, oof_predictions.index)
    scored_target = target_series.loc[reference_mask]
    if scored_target.nunique() < 2:
        raise ValueError("ROC-AUC requires both target classes in scored OOF rows.")
    return scored, scored_target, reference_mask


def hill_climb_auc(
    oof_predictions: pd.DataFrame,
    target: pd.Series | Sequence[int],
    *,
    candidate_models: Sequence[str] | None = None,
    max_steps: int = 50,
    weight_grid: Sequence[float] | None = None,
    min_improvement: float = 1e-6,
) -> HillClimbingEnsembleResult:
    """Greedily maximize pooled OOF ROC-AUC with convex prediction blends.

    The best individual model initializes the ensemble. At each step the function
    searches every candidate and mixing coefficient ``alpha`` using
    ``(1 - alpha) * current + alpha * candidate``. The search stops when no trial
    improves ROC-AUC by at least ``min_improvement``.
    """
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative.")
    if min_improvement < 0:
        raise ValueError("min_improvement must be non-negative.")
    if weight_grid is None:
        grid = np.arange(0.05, 0.55, 0.05, dtype=float)
    else:
        grid = np.asarray(list(weight_grid), dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("weight_grid must contain at least one value.")
    if not np.isfinite(grid).all() or ((grid <= 0) | (grid >= 1)).any():
        raise ValueError("Every weight_grid value must be finite and between 0 and 1.")
    grid = np.unique(grid)

    scored, scored_target, scored_mask = _validate_oof_candidates(
        oof_predictions, target, candidate_models
    )
    individual_scores = pd.Series(
        {
            name: float(roc_auc_score(scored_target, scored[name]))
            for name in scored.columns
        },
        name="oof_auc",
        dtype=float,
    )
    # idxmax is deterministic and preserves the caller's model order on ties.
    initial_model = str(individual_scores.idxmax())
    current = scored[initial_model].to_numpy(dtype=float, copy=True)
    current_score = float(individual_scores.loc[initial_model])
    weights = pd.Series(0.0, index=scored.columns, name="weight")
    weights.loc[initial_model] = 1.0
    history: list[dict[str, float | int | str]] = [
        {
            "step": 0,
            "added_model": initial_model,
            "alpha": 1.0,
            "oof_auc": current_score,
            "improvement": np.nan,
        }
    ]

    for step in range(1, max_steps + 1):
        best_score = current_score
        best_model: str | None = None
        best_alpha: float | None = None
        best_prediction: np.ndarray | None = None
        for name in scored.columns:
            candidate = scored[name].to_numpy(dtype=float, copy=False)
            for alpha in grid:
                trial = (1.0 - alpha) * current + alpha * candidate
                trial_score = float(roc_auc_score(scored_target, trial))
                if trial_score > best_score + min_improvement:
                    best_score = trial_score
                    best_model = str(name)
                    best_alpha = float(alpha)
                    best_prediction = trial
        if best_model is None or best_alpha is None or best_prediction is None:
            break

        improvement = best_score - current_score
        weights *= 1.0 - best_alpha
        weights.loc[best_model] += best_alpha
        current = best_prediction
        current_score = best_score
        history.append(
            {
                "step": step,
                "added_model": best_model,
                "alpha": best_alpha,
                "oof_auc": current_score,
                "improvement": improvement,
            }
        )

    weights[weights.abs() < 1e-12] = 0.0
    weights /= weights.sum()
    full_oof = pd.Series(
        np.nan,
        index=oof_predictions.index.copy(),
        name="hill_climbing_ensemble",
        dtype=float,
    )
    full_oof.loc[scored_mask] = current
    return HillClimbingEnsembleResult(
        weights=weights,
        oof_prediction=full_oof,
        score=current_score,
        individual_scores=individual_scores.sort_values(ascending=False),
        history=pd.DataFrame(history),
        n_scored_rows=int(scored_mask.sum()),
    )


def blend_test_predictions(
    test_predictions: Mapping[str, Sequence[float]],
    weights: Mapping[str, float] | pd.Series,
) -> np.ndarray:
    """Apply fitted ensemble weights to aligned test probabilities."""
    weight_series = pd.Series(dict(weights), dtype=float)
    weight_series = weight_series[weight_series > 0]
    if weight_series.empty or not np.isfinite(weight_series).all():
        raise ValueError("weights must contain at least one finite positive value.")
    missing = [name for name in weight_series.index if name not in test_predictions]
    if missing:
        raise KeyError(f"Test predictions are missing weighted models: {missing}")

    arrays: list[np.ndarray] = []
    expected_length: int | None = None
    for name in weight_series.index:
        values = np.asarray(test_predictions[name], dtype=float)
        if values.ndim != 1:
            raise ValueError(f"Test prediction for {name!r} must be one-dimensional.")
        if expected_length is None:
            expected_length = len(values)
        elif len(values) != expected_length:
            raise ValueError("All test prediction arrays must have the same length.")
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"Test prediction for {name!r} is not valid probability data.")
        arrays.append(values)

    normalized_weights = weight_series.to_numpy() / weight_series.sum()
    matrix = np.column_stack(arrays)
    return np.asarray(matrix @ normalized_weights, dtype=np.float64)


def evaluate_fold_auc(
    prediction: pd.Series,
    target: pd.Series | Sequence[int],
    folds: Sequence[tuple[pd.Index, pd.Index]],
    *,
    years: pd.Series | Sequence[int] | None = None,
) -> pd.DataFrame:
    """Report ensemble ROC-AUC on each original validation fold."""
    if not isinstance(prediction, pd.Series):
        raise TypeError("prediction must be a pandas Series with label index.")
    if not prediction.index.is_unique:
        raise ValueError("prediction.index must be unique.")
    target_series = _resolve_target(target, prediction.index)
    if years is None:
        year_series = None
    elif isinstance(years, pd.Series):
        if not years.index.equals(prediction.index):
            raise ValueError("years index must exactly match prediction.index.")
        year_series = years
    else:
        year_series = pd.Series(np.asarray(years), index=prediction.index)

    records: list[dict[str, float | int]] = []
    for fold_number, (_, valid_idx) in enumerate(folds):
        labels = pd.Index(valid_idx)
        positions = prediction.index.get_indexer(labels)
        if (positions < 0).any():
            raise KeyError("A validation fold contains labels missing from prediction.index.")
        fold_prediction = prediction.loc[labels]
        if fold_prediction.isna().any() or not np.isfinite(fold_prediction).all():
            raise ValueError(f"Fold {fold_number} contains missing/non-finite predictions.")
        fold_target = target_series.loc[labels]
        auc = (
            float(roc_auc_score(fold_target, fold_prediction))
            if fold_target.nunique() >= 2
            else float("nan")
        )
        record: dict[str, float | int] = {
            "fold": fold_number,
            "n_valid": len(labels),
            "roc_auc": auc,
        }
        if year_series is not None:
            unique_years = pd.unique(year_series.loc[labels].dropna())
            if len(unique_years) != 1:
                raise ValueError(
                    f"Fold {fold_number} must contain exactly one validation year."
                )
            record["validation_year"] = int(unique_years[0])
        records.append(record)
    columns = ["fold", "validation_year", "n_valid", "roc_auc"]
    if year_series is None:
        columns.remove("validation_year")
    return pd.DataFrame(records, columns=columns)


def make_submission(
    test: pd.DataFrame,
    prediction: Sequence[float],
    *,
    id_col: str,
    prediction_col: str = "target",
    output_path: str | Path = "submission.csv",
) -> pd.DataFrame:
    """Validate IDs/predictions and atomically write a two-column submission CSV."""
    if id_col not in test.columns:
        raise KeyError(f"test is missing ID_COL {id_col!r}.")
    if prediction_col == id_col:
        raise ValueError("prediction_col must differ from id_col.")
    if test[id_col].isna().any():
        raise ValueError(f"{id_col!r} contains missing values.")
    if not test[id_col].is_unique:
        raise ValueError(f"{id_col!r} must be unique in test.")
    values = np.asarray(prediction, dtype=float)
    if values.ndim != 1 or len(values) != len(test):
        raise ValueError("prediction must be one-dimensional and match len(test).")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("prediction must contain finite probabilities in [0, 1].")

    submission = pd.DataFrame(
        {
            id_col: test[id_col].to_numpy(copy=True),
            prediction_col: values,
        }
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    submission.to_csv(temporary, index=False)
    temporary.replace(path)
    return submission


def save_ensemble_outputs(
    result: HillClimbingEnsembleResult,
    output_dir: str | Path = "outputs",
) -> dict[str, Path]:
    """Save reusable ensemble diagnostics without filling unscored OOF rows."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "weights": directory / "ensemble_weights.csv",
        "history": directory / "ensemble_history.csv",
        "oof": directory / "ensemble_oof.parquet",
    }
    result.weights.rename("weight").to_csv(paths["weights"], index_label="experiment")
    result.history.to_csv(paths["history"], index=False)
    result.oof_prediction.to_frame().to_parquet(paths["oof"], index=True)
    return paths


__all__ = [
    "HillClimbingEnsembleResult",
    "blend_test_predictions",
    "evaluate_fold_auc",
    "hill_climb_auc",
    "make_submission",
    "save_ensemble_outputs",
]
