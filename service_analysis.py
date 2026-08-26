"""Domain and service-design analysis for selected AI market themes.

All opportunity scores are deterministic summaries of the classified projects.
The service catalogue is an editable hypothesis library; it is never presented
as an LLM-generated or data-observed fact.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from llm_features import ADMIN_PROCESSES, AI_USECASES, POLICY_DOMAINS
from market_analysis import concentration_hhi


DOMAIN_OPPORTUNITY_WEIGHTS = {
    "high_app_project_count": 0.25,
    "high_app_rate_smoothed": 0.15,
    "ministry_breadth": 0.15,
    "low_ministry_concentration": 0.10,
    "high_app_budget_total": 0.15,
    "recent_high_app_project_count": 0.10,
    "issue_coverage_rate": 0.10,
}

ADMIN_CONSULTING_CATALOG = {
    "A01": ("調査・分析高度化", "データ棚卸し、分析テーマ・KPI設計", "調査期間、分析工数、示唆採用率"),
    "A02": ("政策形成支援", "論点整理、エビデンス設計、政策シナリオ比較", "政策立案期間、根拠資料作成工数"),
    "A03": ("申請・審査改革", "審査フロー可視化、判断基準・Human-in-the-loop設計", "審査期間、処理件数、再確認率"),
    "A04": ("給付・資金配分高度化", "配分ルール、需要・執行データ、説明責任の整理", "執行率、配分期間、未執行額"),
    "A05": ("規制・監督高度化", "リスク分類、検査優先順位、監査証跡設計", "検査的中率、検査工数、見逃し率"),
    "A06": ("モニタリング・EBPM", "ロジックモデル、KPI、評価データ基盤設計", "評価期間、KPI更新頻度、改善施策数"),
    "A07": ("窓口・相談改革", "問い合わせ分類、顧客導線、エスカレーション設計", "一次解決率、応答時間、転送率"),
    "A08": ("広報・情報提供高度化", "対象者セグメント、コンテンツ運用、効果測定設計", "到達率、理解度、制作工数"),
    "A09": ("調達・委託管理改革", "調達プロセス、仕様・評価基準、契約管理の整理", "調達期間、契約管理工数、再調達率"),
    "A10": ("公共サービス・現場DX", "現場業務・データフロー、運用制約、安全要件の整理", "処理時間、稼働率、現場負荷"),
    "A11": ("庁内業務改革", "業務量調査、標準化、ナレッジ・権限設計", "作業時間、手戻り率、自動化率"),
    "A12": ("デジタル人材育成", "スキル定義、研修体系、実務適用・定着設計", "受講完了率、スキル向上、実務適用率"),
    "A13": ("AI実証・社会実装", "ユースケース選定、PoC評価、実装ロードマップ策定", "PoC達成率、本番移行率、導入効果"),
}

USECASE_SERVICE_CATALOG = {
    "U01": ("行政知識検索・RAG", "文書・権限・検索品質の棚卸し", "根拠付き検索/RAGプロトタイプ", "検索基盤、権限制御、評価・更新運用", "誤引用、アクセス権、根拠追跡"),
    "U02": ("文書読解・構造化", "帳票種類、項目、正解データの整理", "抽出・要約精度検証", "文書処理パイプライン、品質監視、人手確認", "個人情報、抽出誤り、版管理"),
    "U03": ("行政文書生成支援", "文書テンプレート、承認フロー、表現ルール整理", "下書き生成・編集支援PoC", "テンプレート連携、レビュー、利用ログ", "誤情報、著作権、最終承認責任"),
    "U04": ("分類・判定支援", "ラベル・判定基準、誤判定コストの定義", "分類モデルと説明画面PoC", "MLOps、閾値管理、Human-in-the-loop", "バイアス、精度劣化、異議申立て"),
    "U05": ("審査・スコアリング支援", "審査基準、裁量、過去判断の棚卸し", "優先順位・スコア提示PoC", "審査画面連携、説明可能性、監査証跡", "公平性、説明責任、自動決定回避"),
    "U06": ("予測・政策シミュレーション", "目的変数、予測期間、意思決定接続の設計", "予測モデル・シナリオ比較PoC", "再学習、誤差監視、政策シナリオ運用", "ドリフト、不確実性、因果との混同"),
    "U07": ("異常・不正検知", "リスク事象、調査結果、誤検知コストの整理", "異常スコア・調査優先度PoC", "アラート運用、ケース管理、継続学習", "誤検知、監視過剰、調査責任"),
    "U08": ("資源配分・最適化", "制約条件、目的関数、現行計画の整理", "最適化・シミュレーションPoC", "計画業務連携、再計算、例外処理", "制約漏れ、局所最適、説明可能性"),
    "U09": ("画像・センサ解析", "撮像条件、ラベル、現場通信・機器制約の整理", "認識・検知モデルPoC", "撮像運用、エッジ/クラウド連携、精度監視", "環境変化、見逃し、安全責任"),
    "U10": ("対話・問い合わせ支援", "問い合わせ分類、回答根拠、引継ぎ条件の整理", "対話/RAG・オペレータ支援PoC", "チャネル連携、有人転送、会話品質監視", "誤回答、本人確認、緊急時対応"),
    "U11": ("行政AIエージェント", "業務分解、利用ツール、権限・停止条件の整理", "限定業務のエージェントPoC", "ツール連携、承認ゲート、実行監視・監査", "権限逸脱、連鎖誤動作、責任分界"),
    "U99": ("AI適用見送り・再診断", "AI以外の業務改善余地を確認", "原則PoCなし", "必要時に業務標準化を先行", "無理なAI導入、費用対効果"),
}


def _require_columns(df: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def _selected_rows(theme_long: pd.DataFrame, selected_theme_codes: Sequence[str]) -> pd.DataFrame:
    if not selected_theme_codes:
        raise ValueError("selected_theme_codes must contain at least one theme.")
    requested = list(dict.fromkeys(selected_theme_codes))
    observed = set(theme_long["theme_code"])
    unknown = sorted(set(requested).difference(observed))
    if unknown:
        raise ValueError(f"selected themes are not observed: {unknown}")
    return theme_long.loc[theme_long["theme_code"].isin(requested)].copy()


def _nonempty_text_rate(values: pd.Series) -> float:
    cleaned = values.fillna("").astype(str).str.strip().str.lower()
    return float((~cleaned.isin(["", "nan", "none", "null", "[欠損]"])).mean())


def _valid_budget(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(numeric.ge(0))


def _budget_hhi(values: pd.Series) -> float:
    valid = _valid_budget(values).dropna()
    total = valid.sum()
    if valid.empty or total <= 0:
        return float("nan")
    return float(np.square(valid / total).sum())


def _top_n_budget_share(values: pd.Series, n: int = 3) -> float:
    valid = _valid_budget(values).dropna()
    total = valid.sum()
    if valid.empty or total <= 0:
        return float("nan")
    return float(valid.nlargest(n).sum() / total)


def compute_theme_domain_metrics(
    theme_long: pd.DataFrame,
    selected_theme_codes: Sequence[str],
    *,
    budget_col: str = "budget",
    issue_col: str = "current_issues",
    recent_years: int = 2,
    prior_strength: float = 10.0,
) -> pd.DataFrame:
    """Summarize the policy-domain opportunities inside selected themes."""

    if recent_years <= 0:
        raise ValueError("recent_years must be positive.")
    if prior_strength < 0:
        raise ValueError("prior_strength must be non-negative.")
    _require_columns(
        theme_long,
        [
            "project_key",
            "project_start_year",
            "theme_code",
            "theme_label",
            "admin_process",
            "ai_usecase",
            "policy_domain",
            "responsible_ministry",
            "is_high_app",
        ],
        "theme_long",
    )
    selected = _selected_rows(theme_long, selected_theme_codes)
    if selected.duplicated(["project_key", "theme_code"]).any():
        raise ValueError("theme_long has duplicate project/theme rows.")
    if not selected["policy_domain"].isin(POLICY_DOMAINS).all():
        raise ValueError("policy_domain contains an unknown category.")

    year = pd.to_numeric(selected["project_start_year"], errors="coerce")
    if year.isna().any():
        raise ValueError("project_start_year must be numeric for domain analysis.")
    selected["project_start_year"] = year.astype(int)
    max_year = int(year.max())
    recent_start = max_year - recent_years + 1
    previous_start = recent_start - recent_years
    has_budget = budget_col in selected.columns
    has_issues = issue_col in selected.columns
    theme_priors = selected.groupby("theme_code", observed=True)["is_high_app"].mean()

    rows: list[dict[str, Any]] = []
    for (theme_code, domain), group in selected.groupby(
        ["theme_code", "policy_domain"], sort=True, observed=True
    ):
        high = group.loc[group["is_high_app"]]
        recent = group.loc[group["project_start_year"].ge(recent_start)]
        previous = group.loc[
            group["project_start_year"].between(previous_start, recent_start - 1)
        ]
        total_count = int(group["project_key"].nunique())
        high_count = int(high["project_key"].nunique())
        recent_count = int(recent["project_key"].nunique())
        recent_high = int(recent.loc[recent["is_high_app"], "project_key"].nunique())
        previous_count = int(previous["project_key"].nunique())
        previous_high = int(previous.loc[previous["is_high_app"], "project_key"].nunique())
        prior = float(theme_priors[theme_code])
        high_budget = _valid_budget(high[budget_col]) if has_budget else pd.Series(dtype=float)

        recent_rate = recent_high / recent_count if recent_count else float("nan")
        previous_rate = previous_high / previous_count if previous_count else float("nan")
        rows.append(
            {
                "theme_code": theme_code,
                "theme_label": group["theme_label"].iloc[0],
                "admin_process": group["admin_process"].iloc[0],
                "ai_usecase": group["ai_usecase"].iloc[0],
                "policy_domain": domain,
                "policy_domain_label": f"{domain} {POLICY_DOMAINS[domain]}",
                "project_count": total_count,
                "high_app_project_count": high_count,
                "high_app_rate": high_count / total_count,
                "high_app_rate_smoothed": (high_count + prior_strength * prior)
                / (total_count + prior_strength),
                "theme_high_app_rate_prior": prior,
                "ministry_breadth": int(high["responsible_ministry"].nunique(dropna=True)),
                "ministry_hhi": concentration_hhi(high["responsible_ministry"]),
                "high_app_ministry_missing_rate": (
                    float(high["responsible_ministry"].isna().mean()) if high_count else float("nan")
                ),
                "high_app_budget_total": float(high_budget.sum(min_count=1)) if has_budget else float("nan"),
                "high_app_budget_median": float(high_budget.median()) if has_budget else float("nan"),
                "high_app_budget_nonmissing_rate": (
                    float(high_budget.notna().mean()) if has_budget and high_count else float("nan")
                ),
                "high_app_budget_hhi": _budget_hhi(high[budget_col]) if has_budget else float("nan"),
                "high_app_top3_budget_share": (
                    _top_n_budget_share(high[budget_col]) if has_budget else float("nan")
                ),
                "issue_coverage_rate": _nonempty_text_rate(group[issue_col]) if has_issues else float("nan"),
                "recent_start_year": recent_start,
                "recent_project_count": recent_count,
                "recent_high_app_project_count": recent_high,
                "recent_high_app_rate": recent_rate,
                "previous_start_year": previous_start,
                "previous_project_count": previous_count,
                "previous_high_app_project_count": previous_high,
                "previous_high_app_rate": previous_rate,
                "high_app_rate_momentum": recent_rate - previous_rate,
                "latest_observed_year": int(group["project_start_year"].max()),
                "ranking_eligible": bool(high_count > 0),
                "prior_strength": float(prior_strength),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["theme_code", "policy_domain"], ignore_index=True
    )


def _validate_weights(weights: Mapping[str, float]) -> None:
    if set(weights) != set(DOMAIN_OPPORTUNITY_WEIGHTS):
        raise ValueError(
            f"weights must contain exactly {sorted(DOMAIN_OPPORTUNITY_WEIGHTS)}."
        )
    if any(not np.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("weights must be finite and non-negative.")
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("weights must sum to 1.0.")


def score_theme_domain_metrics(
    metrics: pd.DataFrame,
    *,
    weights: Mapping[str, float] = DOMAIN_OPPORTUNITY_WEIGHTS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Rank domains within each theme and redistribute unavailable weights."""

    _validate_weights(weights)
    _require_columns(
        metrics,
        [
            "theme_code",
            "ranking_eligible",
            "high_app_project_count",
            "high_app_rate_smoothed",
            "ministry_breadth",
            "ministry_hhi",
            "high_app_budget_total",
            "recent_high_app_project_count",
            "issue_coverage_rate",
        ],
        "metrics",
    )
    scored = metrics.copy()
    eligible = scored["ranking_eligible"].astype(bool)
    inputs = {
        "high_app_project_count": scored["high_app_project_count"],
        "high_app_rate_smoothed": scored["high_app_rate_smoothed"],
        "ministry_breadth": scored["ministry_breadth"],
        "low_ministry_concentration": 1.0 - scored["ministry_hhi"],
        "high_app_budget_total": scored["high_app_budget_total"],
        "recent_high_app_project_count": scored["recent_high_app_project_count"],
        "issue_coverage_rate": scored["issue_coverage_rate"],
    }
    available = {
        name: bool(values.loc[eligible].notna().any()) for name, values in inputs.items()
    }
    kept_weight = sum(weight for name, weight in weights.items() if available[name])
    if kept_weight <= 0:
        raise ValueError("no scoring component is available.")
    effective_weights = pd.Series(
        {
            name: (weight / kept_weight if available[name] else 0.0)
            for name, weight in weights.items()
        },
        name="effective_weight",
    )

    score_columns: dict[str, str] = {}
    for name, values in inputs.items():
        column = f"{name}_score"
        scored[column] = np.nan
        if available[name]:
            scored.loc[eligible, column] = 0.0
            valid = eligible & values.notna()
            scored.loc[valid, column] = values.loc[valid].groupby(
                scored.loc[valid, "theme_code"]
            ).rank(method="average", pct=True) * 100.0
        score_columns[name] = column

    scored["domain_opportunity_score"] = np.nan
    weighted = sum(
        scored[score_columns[name]].fillna(0) * effective_weights[name]
        for name in weights
    )
    scored.loc[eligible, "domain_opportunity_score"] = weighted.loc[eligible]
    scored["domain_rank_within_theme"] = pd.Series(pd.NA, index=scored.index, dtype="Int64")
    scored.loc[eligible, "domain_rank_within_theme"] = (
        scored.loc[eligible, "domain_opportunity_score"]
        .groupby(scored.loc[eligible, "theme_code"])
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    scored["portfolio_rank"] = pd.Series(pd.NA, index=scored.index, dtype="Int64")
    scored.loc[eligible, "portfolio_rank"] = scored.loc[
        eligible, "domain_opportunity_score"
    ].rank(method="min", ascending=False).astype("Int64")
    return (
        scored.sort_values(
            ["theme_code", "ranking_eligible", "domain_rank_within_theme", "policy_domain"],
            ascending=[True, False, True, True],
            na_position="last",
            ignore_index=True,
        ),
        effective_weights,
    )


def compute_theme_domain_yearly_metrics(
    theme_long: pd.DataFrame,
    selected_theme_codes: Sequence[str],
) -> pd.DataFrame:
    """Compute yearly evidence for theme/domain trend charts."""

    selected = _selected_rows(theme_long, selected_theme_codes)
    rows: list[dict[str, Any]] = []
    for (year, theme_code, domain), group in selected.groupby(
        ["project_start_year", "theme_code", "policy_domain"],
        sort=True,
        observed=True,
    ):
        total = int(group["project_key"].nunique())
        high = int(group.loc[group["is_high_app"], "project_key"].nunique())
        rows.append(
            {
                "project_start_year": int(year),
                "theme_code": theme_code,
                "policy_domain": domain,
                "policy_domain_label": f"{domain} {POLICY_DOMAINS[domain]}",
                "project_count": total,
                "high_app_project_count": high,
                "high_app_rate": high / total,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["theme_code", "policy_domain", "project_start_year"], ignore_index=True
    )


def compute_ministry_domain_opportunities(
    theme_long: pd.DataFrame,
    selected_theme_codes: Sequence[str],
    *,
    budget_col: str = "budget",
) -> pd.DataFrame:
    """Create an account map by theme, policy domain, and ministry."""

    selected = _selected_rows(theme_long, selected_theme_codes)
    selected = selected.loc[selected["responsible_ministry"].notna()].copy()
    has_budget = budget_col in selected.columns
    rows: list[dict[str, Any]] = []
    for (theme_code, domain, ministry), group in selected.groupby(
        ["theme_code", "policy_domain", "responsible_ministry"],
        sort=True,
        observed=True,
    ):
        high = group.loc[group["is_high_app"]]
        total = int(group["project_key"].nunique())
        high_count = int(high["project_key"].nunique())
        budget = _valid_budget(high[budget_col]) if has_budget else pd.Series(dtype=float)
        rows.append(
            {
                "theme_code": theme_code,
                "policy_domain": domain,
                "policy_domain_label": f"{domain} {POLICY_DOMAINS[domain]}",
                "responsible_ministry": ministry,
                "project_count": total,
                "high_app_project_count": high_count,
                "high_app_rate": high_count / total,
                "high_app_budget_total": float(budget.sum(min_count=1)) if has_budget else float("nan"),
                "latest_observed_year": int(group["project_start_year"].max()),
            }
        )
    columns = [
        "theme_code",
        "policy_domain",
        "policy_domain_label",
        "responsible_ministry",
        "project_count",
        "high_app_project_count",
        "high_app_rate",
        "high_app_budget_total",
        "latest_observed_year",
    ]
    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        result["account_rank_within_theme_domain"] = pd.Series(dtype="Int64")
        return result
    result["account_rank_within_theme_domain"] = (
        result.groupby(["theme_code", "policy_domain"])["high_app_project_count"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    return result.sort_values(
        ["theme_code", "policy_domain", "account_rank_within_theme_domain", "responsible_ministry"],
        ignore_index=True,
    )


def build_service_opportunity_cards(
    scored_domain_metrics: pd.DataFrame,
    *,
    top_domains_per_theme: int = 3,
    admin_catalog: Mapping[str, tuple[str, str, str]] = ADMIN_CONSULTING_CATALOG,
    usecase_catalog: Mapping[str, tuple[str, str, str, str, str]] = USECASE_SERVICE_CATALOG,
) -> pd.DataFrame:
    """Combine observed evidence with editable service-design hypotheses."""

    if top_domains_per_theme <= 0:
        raise ValueError("top_domains_per_theme must be positive.")
    if set(admin_catalog) != set(ADMIN_PROCESSES):
        raise ValueError("admin_catalog must cover every admin_process.")
    if set(usecase_catalog) != set(AI_USECASES):
        raise ValueError("usecase_catalog must cover every ai_usecase.")
    _require_columns(
        scored_domain_metrics,
        [
            "theme_code",
            "admin_process",
            "ai_usecase",
            "policy_domain",
            "policy_domain_label",
            "domain_rank_within_theme",
            "domain_opportunity_score",
            "high_app_project_count",
            "high_app_rate",
            "ministry_breadth",
            "high_app_budget_total",
        ],
        "scored_domain_metrics",
    )
    selected = scored_domain_metrics.loc[
        scored_domain_metrics["domain_rank_within_theme"].notna()
        & scored_domain_metrics["domain_rank_within_theme"].le(top_domains_per_theme)
    ].copy()
    rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        admin_name, consulting_entry, process_kpi = admin_catalog[row.admin_process]
        service_name, discovery, poc, implementation, governance = usecase_catalog[row.ai_usecase]
        budget_text = (
            f"、High-app予算合計={row.high_app_budget_total:,.0f}"
            if pd.notna(row.high_app_budget_total)
            else ""
        )
        rows.append(
            {
                "theme_code": row.theme_code,
                "policy_domain": row.policy_domain,
                "policy_domain_label": row.policy_domain_label,
                "domain_rank_within_theme": row.domain_rank_within_theme,
                "domain_opportunity_score": row.domain_opportunity_score,
                "recommended_offer": f"{row.policy_domain_label}向け {admin_name} × {service_name}",
                "observed_evidence": (
                    f"High-app={row.high_app_project_count}件、率={row.high_app_rate:.1%}、"
                    f"省庁数={row.ministry_breadth}{budget_text}"
                ),
                "consulting_entry": consulting_entry,
                "discovery_and_assessment": discovery,
                "poc_deliverable": poc,
                "implementation_support": implementation,
                "governance_and_risks": governance,
                "kpi_hypotheses": process_kpi,
                "catalog_rule_version": "aiu_service_catalog_v1",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["theme_code", "domain_rank_within_theme"], ignore_index=True
    )


def extract_distinctive_char_ngrams(
    theme_long: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    *,
    text_cols: Sequence[str] = (
        "project_name",
        "project_objective",
        "project_summary",
        "current_issues",
    ),
    top_n: int = 15,
    min_df: int = 2,
    ngram_range: tuple[int, int] = (2, 5),
    max_features: int = 30_000,
) -> pd.DataFrame:
    """Extract descriptive Japanese char n-grams for selected theme/domains.

    This is exploratory text evidence, not a causal importance score.  The
    sklearn import is intentionally lazy so non-text analyses remain usable in
    a minimal pandas environment.
    """

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:  # pragma: no cover - depends on runtime packages
        raise ImportError(
            "scikit-learn is required for phrase extraction; install requirements.txt."
        ) from exc
    if top_n <= 0 or min_df <= 0:
        raise ValueError("top_n and min_df must be positive.")
    available_text = [column for column in text_cols if column in theme_long.columns]
    if not available_text:
        raise KeyError("none of text_cols exist in theme_long.")
    _require_columns(selected_pairs, ["theme_code", "policy_domain"], "selected_pairs")
    pairs = selected_pairs[["theme_code", "policy_domain"]].drop_duplicates()
    selected = theme_long.merge(
        pairs, on=["theme_code", "policy_domain"], how="inner", validate="many_to_one"
    ).copy()
    selected = selected.drop_duplicates(["project_key", "theme_code", "policy_domain"])
    if selected.empty:
        return pd.DataFrame(
            columns=["theme_code", "policy_domain", "phrase", "mean_tfidf", "distinctiveness"]
        )
    documents = selected[available_text].fillna("").astype(str).agg("。".join, axis=1)
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        min_df=min(min_df, len(documents)),
        max_df=0.95 if len(documents) >= 5 else 1.0,
        max_features=max_features,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(documents)
    names = vectorizer.get_feature_names_out()
    readable = np.array(
        [bool(re.fullmatch(r"[0-9A-Za-zぁ-んァ-ヶ一-龥ー]+", phrase)) for phrase in names]
    )
    rows: list[dict[str, Any]] = []
    for (theme_code, domain), indices in selected.groupby(
        ["theme_code", "policy_domain"], observed=True
    ).groups.items():
        positions = selected.index.get_indexer(indices)
        group_mean = np.asarray(matrix[positions].mean(axis=0)).ravel()
        other_positions = np.setdiff1d(np.arange(len(selected)), positions)
        other_mean = (
            np.asarray(matrix[other_positions].mean(axis=0)).ravel()
            if len(other_positions)
            else np.zeros_like(group_mean)
        )
        distinctiveness = group_mean - other_mean
        candidate_indices = np.flatnonzero(readable & (distinctiveness > 0))
        top_indices = candidate_indices[
            np.argsort(distinctiveness[candidate_indices])[::-1][:top_n]
        ]
        for rank, feature_index in enumerate(top_indices, start=1):
            rows.append(
                {
                    "theme_code": theme_code,
                    "policy_domain": domain,
                    "phrase_rank": rank,
                    "phrase": names[feature_index],
                    "mean_tfidf": float(group_mean[feature_index]),
                    "distinctiveness": float(distinctiveness[feature_index]),
                    "document_count": len(positions),
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "ADMIN_CONSULTING_CATALOG",
    "DOMAIN_OPPORTUNITY_WEIGHTS",
    "USECASE_SERVICE_CATALOG",
    "build_service_opportunity_cards",
    "compute_ministry_domain_opportunities",
    "compute_theme_domain_metrics",
    "compute_theme_domain_yearly_metrics",
    "extract_distinctive_char_ngrams",
    "score_theme_domain_metrics",
]
