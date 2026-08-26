"""Generate validated administrative-project taxonomy features with Bedrock.

This module is independent from the science/technology classification pipeline.
It uses the Bedrock Converse API, forced tool use, local schema validation, and
one-row checkpoints so a paid run can be resumed without calling completed rows.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeAlias

import pandas as pd

from embedding_features import create_bedrock_runtime_client


PROMPT_VERSION = "ai_market_taxonomy_v1"
TOOL_NAME = "classify_administrative_project"
DEFAULT_TEXT_COLS = [
    "project_name",
    "project_objective",
    "project_summary",
    "current_issues",
]
FIELD_LABELS = {
    "project_name": "事業名",
    "project_objective": "事業目的",
    "project_summary": "事業概要",
    "current_issues": "現状・課題",
}

POLICY_DOMAINS = {
    "D01": "科学技術・研究開発",
    "D02": "産業・中小企業・スタートアップ",
    "D03": "雇用・労働・人材",
    "D04": "教育・文化・スポーツ",
    "D05": "医療・健康",
    "D06": "福祉・社会保障",
    "D07": "少子化・こども・男女共同参画",
    "D08": "農林水産・食料",
    "D09": "環境・エネルギー・脱炭素",
    "D10": "国土・インフラ・交通",
    "D11": "防災・復興・国土強靱化",
    "D12": "デジタル・通信・行政DX",
    "D13": "地域振興・地方創生",
    "D14": "外交・安全保障・治安",
    "D15": "行政運営・制度・財政",
}
ADMIN_PROCESSES = {
    "A01": "調査・情報収集・分析",
    "A02": "政策立案・制度設計",
    "A03": "申請受付・審査・認定",
    "A04": "補助・給付・資金配分",
    "A05": "規制・監督・検査",
    "A06": "モニタリング・評価・EBPM",
    "A07": "相談・問い合わせ・窓口対応",
    "A08": "広報・情報提供・普及啓発",
    "A09": "調達・契約・委託管理",
    "A10": "公共サービス・現場業務",
    "A11": "庁内管理・業務運営",
    "A12": "人材育成・研修提供",
    "A13": "研究開発・実証・社会実装",
}
AI_USECASES = {
    "U01": "検索・RAG・知識参照",
    "U02": "要約・情報抽出・構造化",
    "U03": "文書生成・編集",
    "U04": "分類・判定",
    "U05": "審査・スコアリング・優先順位付け",
    "U06": "予測・将来推計",
    "U07": "異常検知・不正検知",
    "U08": "最適化・資源配分",
    "U09": "画像・映像・センサ解析",
    "U10": "対話・問い合わせ対応",
    "U11": "AIエージェント・業務自動化",
    "U99": "明確なAIユースケースなし",
}


def _category_lines(categories: Mapping[str, str]) -> str:
    return "\n".join(f"- {code}: {label}" for code, label in categories.items())


SYSTEM_PROMPT = f"""あなたは行政事業の説明文を、後段の市場分析用に分類する担当者です。
市場性やAIUへの適合度を評価せず、入力文から客観的に読み取れる分類だけを返してください。

policy_domainは主目的に最も近い1件を選ぶ:
{_category_lines(POLICY_DOMAINS)}
境界: 研究/R&DはD01、教育・学生支援はD04、民間企業DXはD02、行政・自治体DXはD12、
医療・健康はD05、介護・障害・生活保障はD06、こども・保育はD07、平時インフラはD10、災害対応はD11。

admin_processは事業目的を達成する中心的業務を1件選ぶ:
{_category_lines(ADMIN_PROCESSES)}
個々の作業ではなく主要成果物で判断する。調査結果が成果ならA01、政策策定が成果ならA02、
新規情報収集はA01、既存施策の追跡評価はA06、申請確認・採否はA03、採択後の資金配分はA04、
法令遵守・不正確認はA05、効果・KPI評価はA06、行政への申請審査はA03、行政の委託先選定はA09、
既存サービス運営はA10、新技術の開発・実証はA13。

ai_usecaseは入力から合理的に想定できるものを1〜3件選ぶ:
{_category_lines(AI_USECASES)}
U99は単独でのみ使う。単純分類はU04、複数基準の評価はU05、検索はU01、対話はU10。
存在しないデータや業務を仮定しない。

ai_applicabilityは0/1/2の1値。0=具体的活用が困難、1=周辺業務への部分的活用、
2=中核業務に明確な活用余地。0ならai_usecaseはU99のみ、U99なら0にする。

classification_reasonは、主業務とAIが担える情報処理を日本語で簡潔に説明する。
AIU_fit、expected_impact、scalability、market_size、business_attractiveness、
opportunity_scoreその他の評価スコアは絶対に生成しない。"""


TOOL_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "policy_domain": {
            "type": "string",
            "description": "D01からD15のいずれか1つ",
        },
        "admin_process": {
            "type": "string",
            "description": "A01からA13のいずれか1つ",
        },
        "ai_usecase": {
            "type": "array",
            "items": {"type": "string"},
            "description": "U01からU11またはU99。重複なしで1〜3件。U99は単独",
        },
        "ai_applicability": {
            "type": "integer",
            "description": "0、1、2のいずれか",
        },
        "classification_reason": {
            "type": "string",
            "description": "判定根拠を簡潔な日本語で記述",
        },
    },
    "required": [
        "policy_domain",
        "admin_process",
        "ai_usecase",
        "ai_applicability",
        "classification_reason",
    ],
}


class FeatureValidationError(ValueError):
    """Raised when the LLM response does not satisfy the local schema."""


class BudgetExceededError(RuntimeError):
    """Raised before a request that could exceed the configured budget."""


RawCaller: TypeAlias = Callable[..., Mapping[str, Any]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value)).strip("_.")
    return safe or "value"


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_save_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip")
    os.replace(temporary, path)


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_project_text(
    row: pd.Series,
    text_cols: Sequence[str] = DEFAULT_TEXT_COLS,
    *,
    max_characters_per_field: int | None = None,
) -> str:
    """Build the labeled four-field text sent to the LLM."""
    missing = [column for column in text_cols if column not in row.index]
    if missing:
        raise KeyError(f"Missing text columns: {missing}")
    if max_characters_per_field is not None and max_characters_per_field <= 0:
        raise ValueError("max_characters_per_field must be positive or None.")

    sections: list[str] = []
    for column in text_cols:
        value = _normalize_text(row[column])
        if max_characters_per_field is not None:
            value = value[:max_characters_per_field]
        label = FIELD_LABELS.get(column, column)
        sections.append(f"{label}:\n{value or '[欠損]'}")
    return "\n\n".join(sections)


def filter_projects_by_start_year(
    df: pd.DataFrame,
    *,
    year_col: str = "project_start_year",
    min_year: int = 2020,
) -> pd.DataFrame:
    """Return a copy of rows on/after ``min_year`` without mutating ``df``."""
    if year_col not in df.columns:
        raise KeyError(f"Missing year column: {year_col}")
    if isinstance(min_year, bool) or not isinstance(min_year, int):
        raise ValueError("min_year must be an integer.")
    numeric_year = pd.to_numeric(df[year_col], errors="coerce")
    return df.loc[numeric_year.ge(min_year)].copy()


def build_classification_prompt(project_text: str) -> str:
    """Build one user message without asking the model for market evaluation."""
    return (
        "以下の行政事業を指定taxonomyで分類してください。"
        "必ずclassify_administrative_project toolを1回呼び出してください。\n\n"
        "<administrative_project>\n"
        f"{project_text}\n"
        "</administrative_project>"
    )


def build_tool_config() -> dict[str, Any]:
    """Return a Converse tool definition compatible with Amazon Nova tool use."""
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": TOOL_NAME,
                    "description": "行政事業を固定taxonomyへ分類する",
                    "inputSchema": {"json": TOOL_INPUT_SCHEMA},
                }
            }
        ],
        "toolChoice": {"tool": {"name": TOOL_NAME}},
    }


def _extract_json_text(value: str) -> Mapping[str, Any]:
    text = value.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.I)
    if fenced:
        text = fenced.group(1).strip()
    decoder = json.JSONDecoder()
    for position, character in enumerate(text):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            return parsed
    raise FeatureValidationError("LLM response does not contain a JSON object.")


def parse_feature_payload(payload: Mapping[str, Any] | str) -> dict[str, Any]:
    """Parse a mapping, plain JSON, or fenced JSON and validate all fields."""
    if isinstance(payload, str):
        parsed = _extract_json_text(payload)
    elif isinstance(payload, Mapping):
        parsed = payload
    else:
        raise FeatureValidationError("Feature payload must be a mapping or JSON text.")
    return validate_feature_payload(parsed)


def validate_feature_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the five permitted LLM output fields."""
    required = {
        "policy_domain",
        "admin_process",
        "ai_usecase",
        "ai_applicability",
        "classification_reason",
    }
    keys = set(payload)
    missing = sorted(required - keys)
    extra = sorted(keys - required)
    if missing:
        raise FeatureValidationError(f"Missing response fields: {missing}")
    if extra:
        raise FeatureValidationError(f"Unexpected response fields: {extra}")

    policy_domain = payload["policy_domain"]
    if policy_domain not in POLICY_DOMAINS:
        raise FeatureValidationError("policy_domain must be one of D01-D15.")
    admin_process = payload["admin_process"]
    if admin_process not in ADMIN_PROCESSES:
        raise FeatureValidationError("admin_process must be one of A01-A13.")

    usecases = payload["ai_usecase"]
    if not isinstance(usecases, list):
        raise FeatureValidationError("ai_usecase must be a list.")
    if not 1 <= len(usecases) <= 3:
        raise FeatureValidationError("ai_usecase must contain 1 to 3 items.")
    if any(not isinstance(value, str) or value not in AI_USECASES for value in usecases):
        raise FeatureValidationError("ai_usecase contains an unknown category.")
    if len(usecases) != len(set(usecases)):
        raise FeatureValidationError("ai_usecase must not contain duplicates.")
    if "U99" in usecases and usecases != ["U99"]:
        raise FeatureValidationError("U99 must not be combined with other use cases.")

    applicability = payload["ai_applicability"]
    if isinstance(applicability, bool) or not isinstance(applicability, int):
        raise FeatureValidationError("ai_applicability must be an integer 0, 1, or 2.")
    if applicability not in {0, 1, 2}:
        raise FeatureValidationError("ai_applicability must be 0, 1, or 2.")
    if (applicability == 0) != (usecases == ["U99"]):
        raise FeatureValidationError("U99 and ai_applicability=0 must be consistent.")

    reason = payload["classification_reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise FeatureValidationError("classification_reason must be non-empty text.")
    reason = _normalize_text(reason)
    if len(reason) > 1000:
        raise FeatureValidationError("classification_reason must be at most 1000 characters.")

    return {
        "policy_domain": policy_domain,
        "admin_process": admin_process,
        "ai_usecase": list(usecases),
        "ai_applicability": applicability,
        "classification_reason": reason,
    }


def parse_bedrock_converse_response(
    response: Mapping[str, Any],
) -> tuple[dict[str, Any], str, int, int]:
    """Extract a forced tool input, with text JSON as a compatibility fallback."""
    try:
        content = response["output"]["message"]["content"]
    except (KeyError, TypeError) as error:
        raise FeatureValidationError("Bedrock response is missing message content.") from error
    if not isinstance(content, list):
        raise FeatureValidationError("Bedrock message content must be a list.")

    payload: Mapping[str, Any] | str | None = None
    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, Mapping):
            continue
        tool_use = block.get("toolUse")
        if isinstance(tool_use, Mapping) and tool_use.get("name") == TOOL_NAME:
            payload = tool_use.get("input")
            break
        if isinstance(block.get("text"), str):
            text_parts.append(str(block["text"]))
    if payload is None and text_parts:
        payload = "\n".join(text_parts)
    if payload is None:
        raise FeatureValidationError("Bedrock response did not call the classification tool.")

    usage = response.get("usage", {})
    if not isinstance(usage, Mapping):
        usage = {}
    input_tokens = int(usage.get("inputTokens", 0) or 0)
    output_tokens = int(usage.get("outputTokens", 0) or 0)
    raw = json.dumps(content, ensure_ascii=False, default=str)
    return parse_feature_payload(payload), raw, input_tokens, output_tokens


def call_bedrock_converse(
    project_text: str,
    client: Any,
    model: str,
    *,
    max_output_tokens: int = 400,
    temperature: float = 0.0,
    top_p: float = 1.0,
    additional_model_request_fields: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Call Bedrock Converse once; credentials are owned by the boto3 client."""
    kwargs: dict[str, Any] = {
        "modelId": model,
        "system": [{"text": SYSTEM_PROMPT}],
        "messages": [
            {
                "role": "user",
                "content": [{"text": build_classification_prompt(project_text)}],
            }
        ],
        "inferenceConfig": {
            "maxTokens": int(max_output_tokens),
            "temperature": float(temperature),
            "topP": float(top_p),
        },
        "toolConfig": build_tool_config(),
    }
    if additional_model_request_fields:
        kwargs["additionalModelRequestFields"] = dict(additional_model_request_fields)
    return client.converse(**kwargs)


def _exception_status(error: Exception) -> int | None:
    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return None
    metadata = response.get("ResponseMetadata", {})
    if isinstance(metadata, Mapping) and metadata.get("HTTPStatusCode") is not None:
        return int(metadata["HTTPStatusCode"])
    return None


def _is_retryable(error: Exception) -> bool:
    status = _exception_status(error)
    if status in {408, 409, 429} or (status is not None and status >= 500):
        return True
    name = type(error).__name__.lower()
    message = str(error).lower()
    markers = (
        "throttl",
        "timeout",
        "temporar",
        "connection",
        "serviceunavailable",
        "internalserver",
    )
    return any(marker in name or marker in message for marker in markers)


def _fingerprint(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _cache_directory(output_root: str | Path, config: Mapping[str, Any]) -> Path:
    return Path(output_root) / (
        f"bedrock_{_safe_name(str(config['model']))}_"
        f"{_safe_name(str(config['prompt_version']))}_{_fingerprint(config)}"
    )


def _append_failure(
    cache_dir: Path,
    *,
    split: str,
    row_position: int,
    row_key: str,
    error: Exception,
    input_tokens: int = 0,
    output_tokens: int = 0,
    raw_response: str = "",
) -> None:
    record = {
        "timestamp": _utc_now(),
        "split": split,
        "row_position": row_position,
        "row_key": row_key,
        "error_type": type(error).__name__,
        "status_code": _exception_status(error),
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "raw_response": raw_response,
    }
    path = cache_dir / "failures.jsonl"
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _cached_usage(cache_dir: Path) -> tuple[int, int]:
    input_tokens = 0
    output_tokens = 0
    shard_root = cache_dir / "shards"
    if shard_root.exists():
        for path in shard_root.rglob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            input_tokens += int(payload.get("input_tokens", 0) or 0)
            output_tokens += int(payload.get("output_tokens", 0) or 0)
    failure_path = cache_dir / "failures.jsonl"
    if failure_path.exists():
        for line in failure_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            input_tokens += int(payload.get("input_tokens", 0) or 0)
            output_tokens += int(payload.get("output_tokens", 0) or 0)
    return input_tokens, output_tokens


def _cost_usd(
    input_tokens: int,
    output_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float,
) -> float:
    return (
        input_tokens / 1_000_000 * input_price_per_million
        + output_tokens / 1_000_000 * output_price_per_million
    )


def _estimate_input_tokens(project_text: str, characters_per_token: float) -> int:
    total_characters = len(SYSTEM_PROMPT) + len(build_classification_prompt(project_text))
    return max(1, math.ceil(total_characters / characters_per_token))


def _validate_or_write_config(cache_dir: Path, config: Mapping[str, Any]) -> None:
    path = cache_dir / "config.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if {key: saved.get(key) for key in config} != dict(config):
            raise ValueError("LLM feature cache configuration mismatch.")
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, {**dict(config), "config_fingerprint": _fingerprint(config)})


def _load_checkpoint(path: Path, expected: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for field in ("row_key", "text_hash", "config_fingerprint"):
        if str(payload.get(field)) != str(expected[field]):
            raise ValueError("LLM feature checkpoint does not match the source row.")
    parsed = validate_feature_payload(
        {
            "policy_domain": payload.get("policy_domain"),
            "admin_process": payload.get("admin_process"),
            "ai_usecase": decode_ai_usecase(payload.get("ai_usecase")),
            "ai_applicability": payload.get("ai_applicability"),
            "classification_reason": payload.get("classification_reason"),
        }
    )
    return {**payload, **parsed, "ai_usecase": "|".join(parsed["ai_usecase"])}


def decode_ai_usecase(value: object) -> list[str]:
    """Read the canonical pipe-separated representation (or a JSON list)."""
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str):
        raise FeatureValidationError("Saved ai_usecase must be text or a list.")
    text = value.strip()
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise FeatureValidationError("JSON ai_usecase must decode to a list.")
        return [str(item) for item in parsed]
    return [item for item in text.split("|") if item]


def generate_llm_features(
    df: pd.DataFrame,
    *,
    split: str,
    model: str,
    output_root: str | Path = "data/llm_features",
    text_cols: Sequence[str] = DEFAULT_TEXT_COLS,
    project_id_col: str = "project_id",
    prompt_version: str = PROMPT_VERSION,
    region_name: str | None = None,
    aws_profile_name: str | None = None,
    n_rows: int | None = None,
    max_characters_per_field: int | None = 4000,
    max_output_tokens: int = 400,
    temperature: float = 0.0,
    top_p: float = 1.0,
    characters_per_token: float = 1.0,
    input_price_per_million: float = 0.0,
    output_price_per_million: float = 0.0,
    max_budget_usd: float = 20.0,
    max_retries: int = 3,
    initial_backoff_seconds: float = 1.0,
    dry_run: bool = True,
    save_raw_response: bool = False,
    continue_on_error: bool = True,
    client: Any | None = None,
    raw_caller: RawCaller = call_bedrock_converse,
    additional_model_request_fields: Mapping[str, Any] | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Generate, validate, checkpoint, and save one feature row per project."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", split):
        raise ValueError("split may contain only letters, numbers, _, ., and -.")
    if not isinstance(prompt_version, str) or not prompt_version.strip():
        raise ValueError("prompt_version must be a non-empty string.")
    required = [*text_cols, project_id_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    if not df.index.is_unique:
        raise ValueError("df.index must be unique.")
    ids = df[project_id_col]
    if ids.isna().any() or not ids.map(str).is_unique:
        raise ValueError(f"{project_id_col!r} must be non-null and unique.")
    if n_rows is not None and n_rows <= 0:
        raise ValueError("n_rows must be positive or None.")
    if max_output_tokens <= 0 or characters_per_token <= 0:
        raise ValueError("Token and character settings must be positive.")
    if input_price_per_million < 0 or output_price_per_million < 0:
        raise ValueError("Token prices must be non-negative.")
    if max_budget_usd <= 0:
        raise ValueError("max_budget_usd must be positive.")
    if max_retries < 0 or initial_backoff_seconds < 0:
        raise ValueError("Retry settings must be non-negative.")

    texts = [
        build_project_text(
            row,
            text_cols=text_cols,
            max_characters_per_field=max_characters_per_field,
        )
        for _, row in df.iterrows()
    ]
    selected_count = len(df) if n_rows is None else min(n_rows, len(df))
    positions = list(range(selected_count))
    estimated_input_tokens_by_row = [
        _estimate_input_tokens(texts[position], characters_per_token)
        for position in positions
    ]
    estimated_input_tokens = int(sum(estimated_input_tokens_by_row))
    maximum_output_tokens = int(selected_count * max_output_tokens)
    estimated_max_cost = _cost_usd(
        estimated_input_tokens,
        maximum_output_tokens,
        input_price_per_million,
        output_price_per_million,
    )
    report = {
        "total_source_rows": len(df),
        "selected_rows": selected_count,
        "estimated_api_calls": selected_count,
        "estimated_input_tokens": estimated_input_tokens,
        "maximum_output_tokens": maximum_output_tokens,
        "input_price_per_million": input_price_per_million,
        "output_price_per_million": output_price_per_million,
        "estimated_max_cost_usd": estimated_max_cost,
        "max_budget_usd": max_budget_usd,
        "model": model,
        "region_name": region_name,
        "prompt_version": prompt_version,
        "sample_prompt": build_classification_prompt(texts[0]) if texts else "",
    }
    if dry_run:
        print("LLM feature dry run (API is not called)")
        for key, value in report.items():
            if key != "sample_prompt":
                print(f"{key}: {value}")
        return {
            "features": None,
            "errors": pd.DataFrame(),
            "cache_dir": None,
            "report": report,
            "from_cache_rows": 0,
        }

    if input_price_per_million <= 0 or output_price_per_million <= 0:
        raise ValueError("Set current positive input/output token prices before API use.")
    if estimated_max_cost > max_budget_usd:
        raise BudgetExceededError(
            "Estimated maximum cost exceeds max_budget_usd; no API request was sent."
        )

    inference_identity = {
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "additional_model_request_fields": dict(additional_model_request_fields or {}),
    }
    config = {
        "model": model,
        "region_name": region_name,
        "prompt_version": prompt_version,
        "system_prompt_sha256": _text_hash(SYSTEM_PROMPT),
        "tool_schema_sha256": _text_hash(
            json.dumps(TOOL_INPUT_SCHEMA, ensure_ascii=False, sort_keys=True)
        ),
        "text_cols": list(text_cols),
        "project_id_col": project_id_col,
        "max_characters_per_field": max_characters_per_field,
        "inference": inference_identity,
        "save_raw_response": bool(save_raw_response),
    }
    fingerprint = _fingerprint(config)
    cache_dir = _cache_directory(output_root, config)
    _validate_or_write_config(cache_dir, config)
    shard_dir = cache_dir / "shards" / split
    shard_dir.mkdir(parents=True, exist_ok=True)

    prior_input_tokens, prior_output_tokens = _cached_usage(cache_dir)
    current_input_tokens = prior_input_tokens
    current_output_tokens = prior_output_tokens
    if _cost_usd(
        current_input_tokens,
        current_output_tokens,
        input_price_per_million,
        output_price_per_million,
    ) >= max_budget_usd:
        raise BudgetExceededError("Cached spend has already reached max_budget_usd.")

    if client is None:
        client = create_bedrock_runtime_client(
            region_name=region_name,
            profile_name=aws_profile_name,
        )

    iterator: Any = positions
    progress_bar = None
    if show_progress:
        try:
            from tqdm.auto import tqdm

            progress_bar = tqdm(positions, total=len(positions), desc=f"bedrock {model}")
            iterator = progress_bar
        except ImportError:
            pass

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    cached_rows = 0
    for position in iterator:
        source_index = df.index[position]
        project_id = str(ids.iloc[position])
        row_key = f"{split}::{project_id}"
        expected = {
            "row_key": row_key,
            "text_hash": _text_hash(texts[position]),
            "config_fingerprint": fingerprint,
        }
        checkpoint_path = shard_dir / f"{position:08d}.json"
        checkpoint = _load_checkpoint(checkpoint_path, expected)
        if checkpoint is not None:
            rows.append(checkpoint)
            cached_rows += 1
            continue

        api_retry_count = 0
        validation_retry_count = 0
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            next_input = estimated_input_tokens_by_row[position]
            projected_cost = _cost_usd(
                current_input_tokens + next_input,
                current_output_tokens + max_output_tokens,
                input_price_per_million,
                output_price_per_million,
            )
            if projected_cost > max_budget_usd:
                raise BudgetExceededError(
                    f"Budget guard stopped before row {position}; completed checkpoints are reusable."
                )
            response_input_tokens = 0
            response_output_tokens = 0
            response: Mapping[str, Any] | None = None
            try:
                response = raw_caller(
                    texts[position],
                    client,
                    model,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    additional_model_request_fields=additional_model_request_fields,
                )
                parsed, raw, response_input_tokens, response_output_tokens = (
                    parse_bedrock_converse_response(response)
                )
                current_input_tokens += response_input_tokens or next_input
                current_output_tokens += response_output_tokens
                record = {
                    "row_position": position,
                    "original_index": _json_scalar(source_index),
                    "project_id": project_id,
                    "row_key": row_key,
                    "text_hash": expected["text_hash"],
                    "policy_domain": parsed["policy_domain"],
                    "admin_process": parsed["admin_process"],
                    "ai_usecase": "|".join(parsed["ai_usecase"]),
                    "ai_applicability": parsed["ai_applicability"],
                    "classification_reason": parsed["classification_reason"],
                    "model": model,
                    "prompt_version": prompt_version,
                    "input_tokens": response_input_tokens or next_input,
                    "output_tokens": response_output_tokens,
                    "api_retry_count": api_retry_count,
                    "validation_retry_count": validation_retry_count,
                    "config_fingerprint": fingerprint,
                    "raw_response": raw if save_raw_response else "",
                }
                _atomic_write_json(checkpoint_path, record)
                rows.append(record)
                last_error = None
                break
            except FeatureValidationError as error:
                last_error = error
                validation_retry_count += 1
                usage = response.get("usage", {}) if isinstance(response, Mapping) else {}
                if isinstance(usage, Mapping):
                    response_input_tokens = int(usage.get("inputTokens", 0) or 0)
                    response_output_tokens = int(usage.get("outputTokens", 0) or 0)
                invalid_raw = ""
                if save_raw_response and isinstance(response, Mapping):
                    invalid_raw = json.dumps(
                        response.get("output", {}), ensure_ascii=False, default=str
                    )
                current_input_tokens += response_input_tokens or next_input
                current_output_tokens += response_output_tokens
                _append_failure(
                    cache_dir,
                    split=split,
                    row_position=position,
                    row_key=row_key,
                    error=error,
                    input_tokens=response_input_tokens or next_input,
                    output_tokens=response_output_tokens,
                    raw_response=invalid_raw,
                )
            except Exception as error:
                last_error = error
                api_retry_count += 1
                _append_failure(
                    cache_dir,
                    split=split,
                    row_position=position,
                    row_key=row_key,
                    error=error,
                )
                if not _is_retryable(error):
                    break
            if attempt < max_retries and initial_backoff_seconds > 0:
                delay = initial_backoff_seconds * (2**attempt) + random.random() * 0.25
                time.sleep(delay)

        if last_error is not None:
            error_record = {
                "row_position": position,
                "original_index": _json_scalar(source_index),
                "project_id": project_id,
                "row_key": row_key,
                "error_type": type(last_error).__name__,
                "api_retry_count": api_retry_count,
                "validation_retry_count": validation_retry_count,
            }
            errors.append(error_record)
            if not continue_on_error:
                raise last_error

        if progress_bar is not None:
            spent = _cost_usd(
                current_input_tokens,
                current_output_tokens,
                input_price_per_million,
                output_price_per_million,
            )
            progress_bar.set_postfix(rows=len(rows), errors=len(errors), spend=f"${spent:.4f}")

    columns = [
        "row_position",
        "original_index",
        "project_id",
        "row_key",
        "text_hash",
        "policy_domain",
        "admin_process",
        "ai_usecase",
        "ai_applicability",
        "classification_reason",
        "model",
        "prompt_version",
        "input_tokens",
        "output_tokens",
        "api_retry_count",
        "validation_retry_count",
        "config_fingerprint",
        "raw_response",
    ]
    features = pd.DataFrame(rows).reindex(columns=columns)
    if not features.empty:
        features = features.sort_values("row_position").reset_index(drop=True)
    error_frame = pd.DataFrame(errors)
    feature_path = cache_dir / f"{split}_features.csv.gz"
    error_path = cache_dir / f"{split}_errors.csv.gz"
    _atomic_save_csv(feature_path, features)
    _atomic_save_csv(error_path, error_frame)

    actual_cost = _cost_usd(
        current_input_tokens,
        current_output_tokens,
        input_price_per_million,
        output_price_per_million,
    )
    progress = {
        "split": split,
        "selected_rows": selected_count,
        "successful_rows": len(features),
        "error_rows": len(error_frame),
        "cached_rows": cached_rows,
        "charged_or_estimated_input_tokens": current_input_tokens,
        "charged_output_tokens": current_output_tokens,
        "estimated_spend_usd": actual_cost,
        "max_budget_usd": max_budget_usd,
        "updated_at": _utc_now(),
    }
    _atomic_write_json(cache_dir / f"{split}_progress.json", progress)
    return {
        "features": features,
        "errors": error_frame,
        "cache_dir": cache_dir,
        "feature_path": feature_path,
        "error_path": error_path,
        "report": {**report, **progress},
        "from_cache_rows": cached_rows,
    }


def load_llm_features(
    cache_dir: str | Path,
    *,
    split: str,
    expected_df: pd.DataFrame | None = None,
    project_id_col: str = "project_id",
) -> pd.DataFrame:
    """Load the canonical CSV and optionally verify row order against a DataFrame."""
    path = Path(cache_dir) / f"{split}_features.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Saved LLM features do not exist: {path}")
    features = pd.read_csv(path, compression="gzip")
    if features["row_position"].duplicated().any() or features["row_key"].duplicated().any():
        raise ValueError("Saved LLM features contain duplicate rows.")
    if expected_df is not None:
        expected_ids = expected_df[project_id_col].map(str).tolist()
        if features["project_id"].map(str).tolist() != expected_ids:
            raise ValueError("Saved LLM feature order does not match expected_df.")
    for _, row in features.iterrows():
        validate_feature_payload(
            {
                "policy_domain": row["policy_domain"],
                "admin_process": row["admin_process"],
                "ai_usecase": decode_ai_usecase(row["ai_usecase"]),
                "ai_applicability": int(row["ai_applicability"]),
                "classification_reason": row["classification_reason"],
            }
        )
    return features


__all__ = [
    "ADMIN_PROCESSES",
    "AI_USECASES",
    "BudgetExceededError",
    "DEFAULT_TEXT_COLS",
    "FeatureValidationError",
    "POLICY_DOMAINS",
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "build_classification_prompt",
    "build_project_text",
    "build_tool_config",
    "call_bedrock_converse",
    "decode_ai_usecase",
    "filter_projects_by_start_year",
    "generate_llm_features",
    "load_llm_features",
    "parse_bedrock_converse_response",
    "parse_feature_payload",
    "validate_feature_payload",
]
