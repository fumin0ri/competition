"""Deterministic theme metrics for the government AI market analysis.

The LLM is responsible only for taxonomy classification.  This module joins
those classifications back to the source tables and computes auditable market
metrics for each ``admin_process x ai_usecase`` theme.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from llm_features import (
    ADMIN_PROCESSES,
    AI_USECASES,
    POLICY_DOMAINS,
    decode_ai_usecase,
)


DEFAULT_ADMIN_FIT = {
    **{code: 2 for code in ["A01", "A02", "A06", "A12", "A13"]},
    **{
        code: 1
        for code in ["A03", "A04", "A05", "A07", "A08", "A10", "A11"]
    },
    "A09": 0,
}
DEFAULT_USECASE_FIT = {
    **{code: 3 for code in ["U04", "U05", "U06", "U07", "U08", "U09", "U11"]},
    **{code: 2 for code in ["U01", "U02", "U03", "U10"]},
    "U99": 0,
}

STRATEGIC_WEIGHTS = {
    "high_app_project_count": 0.20,
    "high_app_rate_smoothed": 0.15,
    "ministry_breadth": 0.15,
    "domain_breadth": 0.10,
    "low_concentration": 0.10,
    "aiu_fit": 0.30,
}
EQUAL_WEIGHTS = {key: 1 / 6 for key in STRATEGIC_WEIGHTS}
MARKET_SIZE_WEIGHTS = {
    "high_app_project_count": 0.30,
    "high_app_rate_smoothed": 0.20,
    "ministry_breadth": 0.15,
    "domain_breadth": 0.10,
    "low_concentration": 0.10,
    "aiu_fit": 0.15,
}
DEFAULT_WEIGHT_PROFILES = {
    "strategic": STRATEGIC_WEIGHTS,
    "equal": EQUAL_WEIGHTS,
    "market_size": MARKET_SIZE_WEIGHTS,
}


def _require_columns(df: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def _normalize_key(value: object) -> str:
    if pd.isna(value):
        raise ValueError("project_id must not be missing.")
    return str(value)


def _prepare_source(
    df: pd.DataFrame,
    *,
    split: str,
    id_col: str,
    year_col: str,
    ministry_col: str,
    min_year: int,
) -> tuple[pd.DataFrame, int]:
    _require_columns(df, [id_col, year_col, ministry_col], split)
    reserved = {"source_split", "source_index", "_project_id_key", "_numeric_year"}
    conflicts = sorted(reserved.intersection(df.columns))
    if conflicts:
        raise ValueError(f"{split} contains reserved analysis columns: {conflicts}")
    # Preserve all source columns so the notebook can inspect the projects that
    # make up a selected theme.  Only the required structured columns affect
    # the metric calculations.
    prepared = df.copy()
    prepared["source_split"] = split
    prepared["source_index"] = df.index
    prepared["_project_id_key"] = prepared[id_col].map(_normalize_key)
    prepared["_numeric_year"] = pd.to_numeric(prepared[year_col], errors="coerce")
    original_count = len(prepared)
    prepared = prepared.loc[prepared["_numeric_year"].ge(min_year)].copy()

    duplicate = prepared.duplicated(["source_split", "_project_id_key"], keep=False)
    if duplicate.any():
        examples = prepared.loc[duplicate, ["source_split", id_col]].head().to_dict("records")
        raise ValueError(f"source_split + {id_col} must be unique; examples: {examples}")
    return prepared, original_count


def merge_project_taxonomy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: pd.DataFrame,
    *,
    id_col: str = "project_id",
    year_col: str = "project_start_year",
    ministry_col: str = "responsible_ministry",
    min_year: int = 2020,
) -> tuple[pd.DataFrame, pd.Series]:
    """Join saved LLM taxonomy to eligible train/test projects.

    The join key is ``source_split + project_id``.  Both missing taxonomy rows
    and taxonomy rows without a matching source project are treated as errors,
    because either condition would make the downstream ranking incomplete.
    """

    if isinstance(min_year, bool) or not isinstance(min_year, int):
        raise ValueError("min_year must be an integer year.")

    train_source, train_original = _prepare_source(
        train,
        split="train",
        id_col=id_col,
        year_col=year_col,
        ministry_col=ministry_col,
        min_year=min_year,
    )
    test_source, test_original = _prepare_source(
        test,
        split="test",
        id_col=id_col,
        year_col=year_col,
        ministry_col=ministry_col,
        min_year=min_year,
    )
    source = pd.concat([train_source, test_source], ignore_index=True)
    source = source.rename(
        columns={year_col: "source_project_start_year", ministry_col: "responsible_ministry"}
    )

    required_features = [
        "source_split",
        id_col,
        year_col,
        "policy_domain",
        "admin_process",
        "ai_usecase",
        "ai_applicability",
    ]
    _require_columns(features, required_features, "features")
    taxonomy = features.copy()
    taxonomy["_project_id_key"] = taxonomy[id_col].map(_normalize_key)
    taxonomy["_numeric_year"] = pd.to_numeric(taxonomy[year_col], errors="coerce")
    taxonomy = taxonomy.loc[taxonomy["_numeric_year"].ge(min_year)].copy()
    duplicate = taxonomy.duplicated(["source_split", "_project_id_key"], keep=False)
    if duplicate.any():
        examples = taxonomy.loc[duplicate, ["source_split", id_col]].head().to_dict("records")
        raise ValueError(f"taxonomy keys must be unique; examples: {examples}")

    taxonomy = taxonomy.rename(columns={year_col: "llm_project_start_year"})
    merged = source.merge(
        taxonomy,
        on=["source_split", "_project_id_key"],
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_source", "_llm"),
    )
    source_only = int(merged["_merge"].eq("left_only").sum())
    taxonomy_only = int(merged["_merge"].eq("right_only").sum())
    if source_only or taxonomy_only:
        raise ValueError(
            "source and taxonomy must match one-to-one after the year filter: "
            f"source_only={source_only}, taxonomy_only={taxonomy_only}."
        )

    source_year = pd.to_numeric(merged["source_project_start_year"], errors="coerce")
    llm_year = pd.to_numeric(merged["llm_project_start_year"], errors="coerce")
    if not source_year.eq(llm_year).all():
        raise ValueError("project_start_year differs between source and taxonomy data.")

    source_id = f"{id_col}_source"
    llm_id = f"{id_col}_llm"
    if source_id in merged.columns and llm_id in merged.columns:
        merged[id_col] = merged[source_id]
        merged = merged.drop(columns=[source_id, llm_id])
    merged[year_col] = source_year.astype("Int64")
    merged["project_key"] = (
        merged["source_split"].astype(str) + "::" + merged["_project_id_key"]
    )
    merged = merged.drop(
        columns=[
            "_project_id_key",
            "_numeric_year_source",
            "_numeric_year_llm",
            "source_project_start_year",
            "llm_project_start_year",
            "_merge",
        ],
        errors="ignore",
    )

    diagnostics = pd.Series(
        {
            "train_rows_original": train_original,
            "test_rows_original": test_original,
            "train_rows_eligible": len(train_source),
            "test_rows_eligible": len(test_source),
            "taxonomy_rows_eligible": len(taxonomy),
            "merged_rows": len(merged),
            "ministry_missing_rows": int(merged["responsible_ministry"].isna().sum()),
            "ministry_missing_rate": float(merged["responsible_ministry"].isna().mean()),
        },
        name="value",
    )
    return merged, diagnostics


def build_theme_long(projects: pd.DataFrame) -> pd.DataFrame:
    """Explode multi-label use cases into one auditable row per project/theme."""

    _require_columns(
        projects,
        [
            "project_key",
            "project_start_year",
            "responsible_ministry",
            "policy_domain",
            "admin_process",
            "ai_usecase",
            "ai_applicability",
        ],
        "projects",
    )
    result = projects.copy()
    if result["project_key"].duplicated().any():
        raise ValueError("project_key must be unique before exploding ai_usecase.")
    if not result["admin_process"].isin(ADMIN_PROCESSES).all():
        raise ValueError("admin_process contains an unknown category.")
    if not result["policy_domain"].isin(POLICY_DOMAINS).all():
        raise ValueError("policy_domain contains an unknown category.")
    applicability = pd.to_numeric(result["ai_applicability"], errors="coerce")
    if not applicability.isin([0, 1, 2]).all():
        raise ValueError("ai_applicability must contain only 0, 1, or 2.")
    result["ai_applicability"] = applicability.astype(int)
    result["ai_usecase"] = result["ai_usecase"].map(decode_ai_usecase)
    result = result.explode("ai_usecase", ignore_index=True)
    if not result["ai_usecase"].isin(AI_USECASES).all():
        raise ValueError("ai_usecase contains an unknown category.")
    if result.duplicated(["project_key", "admin_process", "ai_usecase"]).any():
        raise ValueError("a project must not contain the same theme more than once.")

    result["theme_code"] = result["admin_process"] + "__" + result["ai_usecase"]
    result["admin_process_label"] = result["admin_process"].map(
        lambda code: f"{code} {ADMIN_PROCESSES[code]}"
    )
    result["ai_usecase_label"] = result["ai_usecase"].map(
        lambda code: f"{code} {AI_USECASES[code]}"
    )
    result["theme_label"] = (
        result["admin_process_label"] + " × " + result["ai_usecase_label"]
    )
    result["is_high_app"] = result["ai_applicability"].eq(2)
    result["is_u99"] = result["ai_usecase"].eq("U99")
    return result


def concentration_hhi(values: pd.Series) -> float:
    """Return the Herfindahl-Hirschman index after excluding missing values."""

    valid = values.dropna()
    if valid.empty:
        return float("nan")
    shares = valid.value_counts(normalize=True)
    return float(np.square(shares).sum())


def _validate_fit_mapping(mapping: Mapping[str, int], expected: set[str], name: str) -> None:
    if set(mapping) != expected:
        missing = sorted(expected.difference(mapping))
        extra = sorted(set(mapping).difference(expected))
        raise ValueError(f"{name} keys differ from taxonomy; missing={missing}, extra={extra}.")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in mapping.values()):
        raise ValueError(f"{name} values must be integers.")


def compute_theme_metrics(
    theme_long: pd.DataFrame,
    *,
    prior_strength: float = 20.0,
    admin_fit: Mapping[str, int] = DEFAULT_ADMIN_FIT,
    usecase_fit: Mapping[str, int] = DEFAULT_USECASE_FIT,
) -> pd.DataFrame:
    """Compute counts, breadth, concentration, and deterministic AIU fit."""

    if prior_strength < 0:
        raise ValueError("prior_strength must be non-negative.")
    _validate_fit_mapping(admin_fit, set(ADMIN_PROCESSES), "admin_fit")
    _validate_fit_mapping(usecase_fit, set(AI_USECASES), "usecase_fit")
    _require_columns(
        theme_long,
        [
            "project_key",
            "admin_process",
            "ai_usecase",
            "theme_code",
            "theme_label",
            "is_high_app",
            "responsible_ministry",
            "policy_domain",
        ],
        "theme_long",
    )

    non_u99 = theme_long.loc[theme_long["ai_usecase"].ne("U99")]
    global_high_app_rate = (
        float(non_u99["is_high_app"].mean()) if len(non_u99) else 0.0
    )
    rows: list[dict[str, Any]] = []
    for (admin_process, ai_usecase), group in theme_long.groupby(
        ["admin_process", "ai_usecase"], sort=True, observed=True
    ):
        if group["project_key"].duplicated().any():
            raise ValueError("theme_long has duplicate project/theme rows.")
        high = group.loc[group["is_high_app"]]
        total_count = int(group["project_key"].nunique())
        high_count = int(high["project_key"].nunique())
        ministry_hhi = concentration_hhi(high["responsible_ministry"])
        domain_hhi = concentration_hhi(high["policy_domain"])
        hhi_values = [value for value in [ministry_hhi, domain_hhi] if not np.isnan(value)]
        combined_hhi = float(np.mean(hhi_values)) if hhi_values else float("nan")
        rows.append(
            {
                "theme_code": group["theme_code"].iloc[0],
                "theme_label": group["theme_label"].iloc[0],
                "admin_process": admin_process,
                "ai_usecase": ai_usecase,
                "project_count": total_count,
                "high_app_project_count": high_count,
                "high_app_rate": high_count / total_count,
                "high_app_rate_smoothed": (
                    high_count + prior_strength * global_high_app_rate
                )
                / (total_count + prior_strength),
                "ministry_breadth": int(high["responsible_ministry"].nunique(dropna=True)),
                "domain_breadth": int(high["policy_domain"].nunique(dropna=True)),
                "ministry_hhi": ministry_hhi,
                "domain_hhi": domain_hhi,
                "concentration_hhi": combined_hhi,
                "high_app_ministry_missing_rate": (
                    float(high["responsible_ministry"].isna().mean()) if high_count else float("nan")
                ),
                "admin_fit": int(admin_fit[admin_process]),
                "usecase_fit": int(usecase_fit[ai_usecase]),
                "aiu_fit": int(admin_fit[admin_process] + usecase_fit[ai_usecase]),
                "ranking_eligible": bool(ai_usecase != "U99" and high_count > 0),
                "global_high_app_rate_prior": global_high_app_rate,
                "prior_strength": float(prior_strength),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["admin_process", "ai_usecase"], ignore_index=True
    )


def _validate_weights(weights: Mapping[str, float]) -> None:
    expected = set(STRATEGIC_WEIGHTS)
    if set(weights) != expected:
        raise ValueError(f"weights must contain exactly {sorted(expected)}.")
    if any(not np.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("weights must be finite and non-negative.")
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("weights must sum to 1.0.")


def score_theme_metrics(
    metrics: pd.DataFrame,
    *,
    weights: Mapping[str, float] = STRATEGIC_WEIGHTS,
) -> pd.DataFrame:
    """Add percentile component scores, a weighted score, and rank."""

    _validate_weights(weights)
    required = [
        "ranking_eligible",
        "high_app_project_count",
        "high_app_rate_smoothed",
        "ministry_breadth",
        "domain_breadth",
        "concentration_hhi",
        "aiu_fit",
    ]
    _require_columns(metrics, required, "metrics")
    scored = metrics.copy()
    eligible = scored["ranking_eligible"].astype(bool)

    inputs = {
        "high_app_project_count": scored["high_app_project_count"],
        "high_app_rate_smoothed": scored["high_app_rate_smoothed"],
        "ministry_breadth": scored["ministry_breadth"],
        "domain_breadth": scored["domain_breadth"],
        "low_concentration": 1.0 - scored["concentration_hhi"],
    }
    score_columns: dict[str, str] = {}
    for name, values in inputs.items():
        column = f"{name}_score"
        scored[column] = np.nan
        scored.loc[eligible, column] = values.loc[eligible].rank(
            method="average", pct=True, na_option="bottom"
        ) * 100.0
        score_columns[name] = column
    scored["aiu_fit_score"] = np.where(eligible, scored["aiu_fit"] / 5.0 * 100.0, np.nan)
    score_columns["aiu_fit"] = "aiu_fit_score"

    scored["overall_score"] = np.nan
    weighted = sum(scored[score_columns[name]] * weight for name, weight in weights.items())
    scored.loc[eligible, "overall_score"] = weighted.loc[eligible]
    scored["overall_rank"] = np.nan
    scored.loc[eligible, "overall_rank"] = scored.loc[eligible, "overall_score"].rank(
        method="min", ascending=False
    )
    scored["overall_rank"] = scored["overall_rank"].astype("Int64")
    return scored.sort_values(
        ["ranking_eligible", "overall_score", "theme_code"],
        ascending=[False, False, True],
        na_position="last",
        ignore_index=True,
    )


def compare_weight_profiles(
    metrics: pd.DataFrame,
    *,
    profiles: Mapping[str, Mapping[str, float]] = DEFAULT_WEIGHT_PROFILES,
) -> pd.DataFrame:
    """Return one row per theme with score/rank columns for each profile."""

    if not profiles:
        raise ValueError("profiles must not be empty.")
    result = metrics[["theme_code", "theme_label", "ranking_eligible"]].copy()
    for profile, weights in profiles.items():
        scored = score_theme_metrics(metrics, weights=weights).set_index("theme_code")
        result[f"{profile}_score"] = result["theme_code"].map(scored["overall_score"])
        result[f"{profile}_rank"] = result["theme_code"].map(scored["overall_rank"]).astype("Int64")
    return result.sort_values(
        ["ranking_eligible", "strategic_rank" if "strategic" in profiles else f"{next(iter(profiles))}_rank"],
        ascending=[False, True],
        na_position="last",
        ignore_index=True,
    )


def compute_yearly_theme_metrics(theme_long: pd.DataFrame) -> pd.DataFrame:
    """Compute yearly project count and High-app rate for each observed theme."""

    _require_columns(
        theme_long,
        [
            "project_start_year",
            "project_key",
            "admin_process",
            "ai_usecase",
            "theme_code",
            "theme_label",
            "is_high_app",
        ],
        "theme_long",
    )
    rows: list[dict[str, Any]] = []
    for (year, admin_process, ai_usecase), group in theme_long.groupby(
        ["project_start_year", "admin_process", "ai_usecase"],
        sort=True,
        observed=True,
    ):
        total = int(group["project_key"].nunique())
        high = int(group.loc[group["is_high_app"], "project_key"].nunique())
        rows.append(
            {
                "project_start_year": int(year),
                "theme_code": group["theme_code"].iloc[0],
                "theme_label": group["theme_label"].iloc[0],
                "admin_process": admin_process,
                "ai_usecase": ai_usecase,
                "project_count": total,
                "high_app_project_count": high,
                "high_app_rate": high / total,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["project_start_year", "admin_process", "ai_usecase"], ignore_index=True
    )


__all__ = [
    "DEFAULT_ADMIN_FIT",
    "DEFAULT_USECASE_FIT",
    "DEFAULT_WEIGHT_PROFILES",
    "EQUAL_WEIGHTS",
    "MARKET_SIZE_WEIGHTS",
    "STRATEGIC_WEIGHTS",
    "build_theme_long",
    "compare_weight_profiles",
    "compute_theme_metrics",
    "compute_yearly_theme_metrics",
    "concentration_hhi",
    "merge_project_taxonomy",
    "score_theme_metrics",
]
