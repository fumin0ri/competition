"""Reusable validation utilities for project-level competition data."""

from __future__ import annotations

import warnings
from typing import TypeAlias

import numpy as np
import pandas as pd


Fold: TypeAlias = tuple[pd.Index, pd.Index]

DIAGNOSTIC_COLUMNS = [
    "fold",
    "validation_year",
    "train_size",
    "validation_size",
    "validation_target_mean",
    "train_target_mean",
    "seen_project_rate",
    "seen_count",
    "unseen_count",
    "is_latest_fold",
]


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _coerce_years(df: pd.DataFrame, year_col: str) -> pd.Series:
    """Return numeric, integer-like years without changing the source frame."""
    years = pd.to_numeric(df[year_col], errors="coerce")
    non_integer = years.notna() & ~np.isclose(years, np.round(years))
    if non_integer.any():
        examples = df.loc[non_integer, year_col].head(5).tolist()
        raise ValueError(
            f"{year_col!r} must contain integer-like years; examples: {examples}"
        )
    return years


def _validate_binary_target(
    df: pd.DataFrame,
    target_col: str,
    eligible_mask: pd.Series,
) -> pd.Series:
    """Validate the target used for fold-level means and return it as numeric."""
    target = pd.to_numeric(df[target_col], errors="coerce")
    eligible_target = target.loc[eligible_mask]

    if eligible_target.isna().any():
        raise ValueError(
            f"{target_col!r} contains missing or non-numeric values in eligible rows."
        )

    values = set(eligible_target.unique().tolist())
    if not values.issubset({0, 1}):
        raise ValueError(
            f"{target_col!r} must be binary (0/1); found values: {sorted(values)}"
        )
    return target


def make_seen_project_mask(
    df: pd.DataFrame,
    train_idx: pd.Index,
    valid_idx: pd.Index,
    project_col: str = "project_name",
) -> pd.Series:
    """Mark validation rows whose project name appeared in this fold's training data.

    Matching is exact. Missing project names are always treated as unseen. The
    returned Boolean Series is indexed by ``valid_idx`` and can be aligned with
    validation labels or predictions selected using ``.loc``.
    """
    _require_columns(df, [project_col])
    if not df.index.is_unique:
        raise ValueError("df.index must be unique when label indices are used with .loc.")

    train_names = df.loc[train_idx, project_col]
    valid_names = df.loc[valid_idx, project_col]
    known_names = pd.Index(train_names.dropna().unique())

    seen = valid_names.notna() & valid_names.isin(known_names)
    return seen.astype(bool).rename("seen_project")


def make_time_series_cv(
    df: pd.DataFrame,
    year_col: str = "project_start_year",
    project_col: str = "project_name",
    target_col: str = "target",
    n_valid_years: int = 3,
    invalid_year: int = -1,
) -> tuple[list[Fold], pd.DataFrame]:
    """Create expanding-window folds using the latest observed valid years.

    Each fold uses rows with ``year < validation_year`` for training and rows
    with ``year == validation_year`` for validation. Rows whose year equals
    ``invalid_year`` or cannot be interpreted as a year are excluded only from
    the returned fold indices; ``df`` itself is never modified.

    The returned indices are *labels* from ``df.index``. Use them with ``.loc``.
    A unique DataFrame index is required so that selecting a returned label does
    not unexpectedly select multiple rows.
    """
    _require_columns(df, [year_col, project_col, target_col])

    if not df.index.is_unique:
        raise ValueError("df.index must be unique when label indices are used with .loc.")
    if isinstance(n_valid_years, bool) or not isinstance(n_valid_years, int):
        raise TypeError("n_valid_years must be a positive integer.")
    if n_valid_years <= 0:
        raise ValueError("n_valid_years must be greater than 0.")

    years = _coerce_years(df, year_col)
    eligible_mask = years.notna() & years.ne(invalid_year)
    target = _validate_binary_target(df, target_col, eligible_mask)

    available_years = np.sort(years.loc[eligible_mask].astype(np.int64).unique())
    if len(available_years) == 0:
        raise ValueError(f"No valid years remain in {year_col!r}.")

    if len(available_years) < n_valid_years:
        warnings.warn(
            f"Only {len(available_years)} valid year(s) exist; using all of them "
            f"instead of the requested {n_valid_years}.",
            UserWarning,
            stacklevel=2,
        )

    validation_years = available_years[-n_valid_years:]
    latest_validation_year = int(validation_years[-1])

    folds: list[Fold] = []
    diagnostic_rows: list[dict[str, object]] = []

    for validation_year_value in validation_years:
        validation_year = int(validation_year_value)
        train_mask = eligible_mask & years.lt(validation_year)
        valid_mask = eligible_mask & years.eq(validation_year)

        train_idx = pd.Index(df.index[train_mask], name=df.index.name)
        valid_idx = pd.Index(df.index[valid_mask], name=df.index.name)

        if len(train_idx) == 0:
            warnings.warn(
                f"Skipping validation year {validation_year}: training data is empty.",
                UserWarning,
                stacklevel=2,
            )
            continue

        # Leakage-prevention assertions. These are intentionally kept close to
        # fold construction so future edits cannot silently weaken the split.
        train_years = years.loc[train_idx]
        valid_year_values = years.loc[valid_idx]
        assert train_years.max() < validation_year
        assert valid_year_values.eq(validation_year).all()
        assert train_idx.intersection(valid_idx).empty
        assert not train_years.eq(invalid_year).any()
        assert not valid_year_values.eq(invalid_year).any()
        assert train_years.notna().all()
        assert valid_year_values.notna().all()

        seen_project = make_seen_project_mask(
            df=df,
            train_idx=train_idx,
            valid_idx=valid_idx,
            project_col=project_col,
        )
        seen_count = int(seen_project.sum())
        unseen_count = int((~seen_project).sum())

        folds.append((train_idx, valid_idx))
        diagnostic_rows.append(
            {
                "fold": len(folds) - 1,
                "validation_year": validation_year,
                "train_size": len(train_idx),
                "validation_size": len(valid_idx),
                "validation_target_mean": float(target.loc[valid_idx].mean()),
                "train_target_mean": float(target.loc[train_idx].mean()),
                "seen_project_rate": float(seen_project.mean()),
                "seen_count": seen_count,
                "unseen_count": unseen_count,
                "is_latest_fold": validation_year == latest_validation_year,
            }
        )

    diagnostics = pd.DataFrame(diagnostic_rows, columns=DIAGNOSTIC_COLUMNS)
    return folds, diagnostics


__all__ = ["Fold", "make_seen_project_mask", "make_time_series_cv"]
