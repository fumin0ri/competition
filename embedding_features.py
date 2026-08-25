"""Generate, checkpoint, validate, and load text embeddings.

The primary runtime is Amazon Bedrock ``InvokeModel``. Request and response
adapters are injectable because Bedrock payloads differ by model provider.
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

import numpy as np
import pandas as pd


Embedder: TypeAlias = Callable[
    [Sequence[str], Any, str, int | None],
    tuple[np.ndarray, Mapping[str, int]],
]
BedrockRequestBuilder: TypeAlias = Callable[
    [str, str, int | None],
    bytes | str | Mapping[str, Any],
]
BedrockResponseParser: TypeAlias = Callable[
    [bytes],
    np.ndarray | tuple[np.ndarray, Mapping[str, int]],
]

NORMALIZATION_VERSION = "nfkc_whitespace_v1"
GEMINI_2_CLASSIFICATION_PREFIX = "task: classification | query: "
DEFAULT_TEXT_COLS = [
    "project_name",
    "project_objective",
    "project_summary",
    "current_issues",
]
DEFAULT_TEXT_TEMPLATE = (
    "プロジェクト名:\n{project_name}\n\n"
    "目的:\n{project_objective}\n\n"
    "概要:\n{project_summary}\n\n"
    "現状の課題:\n{current_issues}"
)

# Prices are configurable and must be checked against the official pricing
# pages immediately before a paid run. Defaults reflect standard paid text
# input pricing checked on 2026-08-24.
DEFAULT_PRICE_PER_MILLION_TOKENS = {
    # Bedrock price depends on the selected model and can change. Keep zero as
    # the safe default and require an explicit current price for cost estimates.
    "bedrock": 0.0,
    "openai": 0.13,  # text-embedding-3-large
    "gemini": 0.20,  # gemini-embedding-2
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


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


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.save(file, np.asarray(array, dtype=np.float32), allow_pickle=False)
    os.replace(temporary, path)


def _atomic_save_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def normalize_embedding_text(value: object) -> str:
    """Normalize one value without deleting numbers, Latin text, or symbols."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def build_embedding_text(
    row: pd.Series,
    text_cols: Sequence[str] = DEFAULT_TEXT_COLS,
    text_template: str | None = DEFAULT_TEXT_TEMPLATE,
) -> str:
    """Build one labeled API input from selected text columns only."""
    missing = [column for column in text_cols if column not in row.index]
    if missing:
        raise KeyError(f"Missing text columns: {missing}")
    normalized = {column: normalize_embedding_text(row[column]) for column in text_cols}
    if not any(normalized.values()):
        return "[EMPTY]"

    if text_template is not None:
        try:
            return normalize_embedding_text(text_template.format(**normalized))
        except KeyError as error:
            raise KeyError(
                f"text_template references an unavailable column: {error.args[0]}"
            ) from error

    return normalize_embedding_text(
        "\n\n".join(f"{column}:\n{normalized[column]}" for column in text_cols)
    )


def text_sha256(text: str) -> str:
    """Hash the exact normalized and truncated text sent to the provider."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prepare_gemini_instruction(text: str, model: str) -> str:
    # Current Gemini Embedding 2 does not support the task_type field. Google
    # recommends expressing classification as a task prefix in the text.
    if model.startswith("gemini-embedding-2"):
        return GEMINI_2_CLASSIFICATION_PREFIX + text
    return text


def _openai_encoding(model: str) -> Any:
    try:
        import tiktoken
    except ImportError as error:
        raise ImportError("Install tiktoken to count and truncate OpenAI inputs.") from error
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _prepare_inputs(
    df: pd.DataFrame,
    provider: str,
    model: str,
    text_cols: Sequence[str],
    text_template: str | None,
    max_input_tokens: int,
    gemini_chars_per_token: float,
) -> tuple[list[str], pd.DataFrame, dict[str, Any]]:
    if max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be greater than 0.")
    if gemini_chars_per_token <= 0:
        raise ValueError("gemini_chars_per_token must be greater than 0.")

    raw_texts = [
        build_embedding_text(row, text_cols=text_cols, text_template=text_template)
        for _, row in df.iterrows()
    ]
    empty_count = sum(text == "[EMPTY]" for text in raw_texts)
    provider_texts = [
        _prepare_gemini_instruction(text, model) if provider == "gemini" else text
        for text in raw_texts
    ]

    prepared: list[str] = []
    token_counts: list[int] = []
    original_token_counts: list[int] = []
    truncated: list[bool] = []

    if provider == "openai":
        encoding = _openai_encoding(model)
        for text in provider_texts:
            tokens = encoding.encode(text)
            original_token_counts.append(len(tokens))
            was_truncated = len(tokens) > max_input_tokens
            if was_truncated:
                text = encoding.decode(tokens[:max_input_tokens])
                tokens = encoding.encode(text)
            prepared.append(text)
            token_counts.append(len(tokens))
            truncated.append(was_truncated)
    elif provider in {"gemini", "bedrock"}:
        # These providers do not expose a model-independent local tokenizer.
        # Use a configurable character estimate and truncate locally; a custom
        # Bedrock request builder may add a model-specific truncation option.
        character_limit = max(1, int(max_input_tokens * gemini_chars_per_token))
        for text in provider_texts:
            estimated = max(1, math.ceil(len(text) / gemini_chars_per_token))
            original_token_counts.append(estimated)
            was_truncated = len(text) > character_limit
            if was_truncated:
                text = text[:character_limit]
            prepared.append(text)
            token_counts.append(
                max(1, math.ceil(len(text) / gemini_chars_per_token))
            )
            truncated.append(was_truncated)
    else:
        raise ValueError("provider must be 'bedrock', 'openai', or 'gemini'.")

    details = pd.DataFrame(
        {
            "text_hash": [text_sha256(text) for text in prepared],
            "character_count": [len(text) for text in prepared],
            "estimated_tokens": token_counts,
            "original_estimated_tokens": original_token_counts,
            "truncated": truncated,
        },
        index=df.index,
    )
    report = {
        "total_rows": len(df),
        "empty_text_rows": empty_count,
        "truncated_rows": int(sum(truncated)),
        "total_estimated_tokens": int(sum(token_counts)),
        "original_estimated_tokens": int(sum(original_token_counts)),
        "min_text_characters": int(details["character_count"].min()) if len(df) else 0,
        "median_text_characters": float(details["character_count"].median()) if len(df) else 0.0,
        "max_text_characters": int(details["character_count"].max()) if len(df) else 0,
    }
    return prepared, details, report


def create_openai_client(timeout_seconds: float = 60.0) -> Any:
    """Create the official OpenAI client from OPENAI_API_KEY."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    try:
        from openai import OpenAI
    except ImportError as error:
        raise ImportError("Install the openai package.") from error
    return OpenAI(api_key=api_key, timeout=timeout_seconds)


def create_gemini_client() -> Any:
    """Create the current Google GenAI client from GEMINI_API_KEY."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")
    try:
        from google import genai
    except ImportError as error:
        raise ImportError("Install the google-genai package.") from error
    return genai.Client(api_key=api_key)


def create_bedrock_runtime_client(
    region_name: str | None = None,
    profile_name: str | None = None,
) -> Any:
    """Create a Bedrock Runtime client using the standard AWS credential chain.

    Credentials are never accepted as arguments. Locally, boto3 can use an AWS
    profile; on an AWS managed notebook it normally obtains temporary
    credentials from the execution role automatically.
    """
    try:
        import boto3
    except ImportError as error:
        raise ImportError("Install the boto3 package to invoke Amazon Bedrock.") from error
    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    return session.client("bedrock-runtime")


def build_bedrock_titan_request(
    text: str,
    model: str,
    embedding_dim: int | None,
) -> Mapping[str, Any]:
    """Build an Amazon Titan Text Embeddings V2 request.

    Titan accepts one non-empty ``inputText`` per ``InvokeModel`` call and
    supports 1024, 512, or 256 output dimensions. A different Bedrock model
    should supply its own request builder and response parser.
    """
    del model
    if embedding_dim not in {None, 1024, 512, 256}:
        raise ValueError("Titan Text Embeddings V2 dimensions must be 1024, 512, or 256.")
    request: dict[str, Any] = {
        "inputText": text,
        "normalize": True,
        "embeddingTypes": ["float"],
    }
    if embedding_dim is not None:
        request["dimensions"] = embedding_dim
    return request


def parse_bedrock_titan_response(
    body: bytes,
) -> tuple[np.ndarray, Mapping[str, int]]:
    """Parse an Amazon Titan Text Embeddings V2 response."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Bedrock response is not valid UTF-8 JSON. Provide a custom "
            "response_parser for the selected model."
        ) from error
    if not isinstance(payload, Mapping):
        raise ValueError("Titan response must be a JSON object.")
    embedding = payload.get("embedding")
    if embedding is None:
        by_type = payload.get("embeddingsByType")
        if isinstance(by_type, Mapping):
            embedding = by_type.get("float")
    vector = np.asarray(embedding, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(
            "Titan response does not contain a one-dimensional float embedding."
        )
    usage = {"total_tokens": int(payload.get("inputTextTokenCount", 0) or 0)}
    return vector, usage


def build_bedrock_cohere_classification_request(
    text: str,
    model: str,
    embedding_dim: int | None,
) -> Mapping[str, Any]:
    """Build a Cohere Embed v3 request for downstream classification.

    Cohere Embed v3 on Bedrock has a fixed 1024-dimensional float output. The
    shared Bedrock runner invokes the model once per text, so ``texts`` contains
    exactly one item. ``truncate='END'`` retains the beginning of long project
    descriptions and lets the service enforce its token boundary.
    """
    del model
    if embedding_dim not in {None, 1024}:
        raise ValueError("Cohere Embed v3 produces exactly 1024 dimensions.")
    return {
        "texts": [text],
        "input_type": "classification",
        "truncate": "END",
    }


def parse_bedrock_cohere_response(
    body: bytes,
) -> tuple[np.ndarray, Mapping[str, int]]:
    """Parse one float embedding from a Cohere Embed v3 Bedrock response."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Cohere response is not valid UTF-8 JSON.") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Cohere response must be a JSON object.")
    embeddings = payload.get("embeddings")
    if isinstance(embeddings, Mapping):
        embeddings = embeddings.get("float")
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != 1 or matrix.shape[1] == 0:
        raise ValueError(
            "Cohere response must contain exactly one non-empty float embedding."
        )
    return matrix[0], {"total_tokens": 0}


def embed_bedrock_cohere(
    texts: Sequence[str],
    client: Any,
    model: str,
    embedding_dim: int | None,
) -> tuple[np.ndarray, Mapping[str, int]]:
    """Embed up to 96 classification texts in one Cohere Bedrock invocation."""
    if not model:
        raise ValueError("model must be a Cohere Bedrock model ID or ARN.")
    if embedding_dim not in {None, 1024}:
        raise ValueError("Cohere Embed v3 produces exactly 1024 dimensions.")
    batch = list(texts)
    if not 1 <= len(batch) <= 96:
        raise ValueError("Cohere Embed v3 accepts between 1 and 96 texts per call.")
    response = client.invoke_model(
        modelId=model,
        body=json.dumps(
            {
                "texts": batch,
                "input_type": "classification",
                "truncate": "END",
            },
            ensure_ascii=False,
        ).encode("utf-8"),
        contentType="application/json",
        accept="application/json",
    )
    response_body = response.get("body") if isinstance(response, Mapping) else None
    if response_body is None:
        raise ValueError("Bedrock response does not contain body.")
    raw = response_body.read() if hasattr(response_body, "read") else response_body
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if not isinstance(raw, bytes):
        raise TypeError("Cohere response body must be bytes, str, or a readable stream.")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Cohere response is not valid UTF-8 JSON.") from error
    embeddings = payload.get("embeddings") if isinstance(payload, Mapping) else None
    if isinstance(embeddings, Mapping):
        embeddings = embeddings.get("float")
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(batch) or matrix.shape[1] == 0:
        raise ValueError("Cohere response embedding matrix does not match the request.")
    if embedding_dim is not None and matrix.shape[1] != embedding_dim:
        raise ValueError(
            f"Cohere returned dimension {matrix.shape[1]}, expected {embedding_dim}."
        )
    return matrix, {"total_tokens": 0}


def embed_bedrock(
    texts: Sequence[str],
    client: Any,
    model: str,
    embedding_dim: int | None,
    *,
    content_type: str = "application/json",
    accept: str = "application/json",
    request_builder: BedrockRequestBuilder = build_bedrock_titan_request,
    response_parser: BedrockResponseParser = parse_bedrock_titan_response,
    invoke_model_kwargs: Mapping[str, Any] | None = None,
    max_retries: int = 6,
    initial_backoff_seconds: float = 1.0,
) -> tuple[np.ndarray, Mapping[str, int]]:
    """Create a batch by invoking a Bedrock embedding model once per text.

    Bedrock embedding request schemas are model-specific. The default adapter
    targets Titan Text Embeddings V2, whose native request contains one text.
    Each invocation is retried independently so a transient failure does not
    repeat already successful billable requests in the same checkpoint batch.
    """
    if not model:
        raise ValueError("model must be a Bedrock model ID, inference profile, or ARN.")
    extra = dict(invoke_model_kwargs or {})
    reserved = {"modelId", "body", "contentType", "accept"}.intersection(extra)
    if reserved:
        raise ValueError(
            f"invoke_model_kwargs may not override reserved keys: {sorted(reserved)}"
        )

    vectors: list[np.ndarray] = []
    total_tokens = 0
    for text in texts:
        request = request_builder(text, model, embedding_dim)
        if isinstance(request, Mapping):
            body: bytes | str = json.dumps(request, ensure_ascii=False).encode("utf-8")
        elif isinstance(request, (bytes, str)):
            body = request
        else:
            raise TypeError("request_builder must return bytes, str, or a mapping.")

        def invoke_once() -> tuple[np.ndarray, Mapping[str, int]]:
            response = client.invoke_model(
                modelId=model,
                body=body,
                contentType=content_type,
                accept=accept,
                **extra,
            )
            response_body = response.get("body") if isinstance(response, Mapping) else None
            if response_body is None:
                raise ValueError("Bedrock response does not contain body.")
            raw = response_body.read() if hasattr(response_body, "read") else response_body
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
            if not isinstance(raw, bytes):
                raise TypeError(
                    "Bedrock response body must be bytes, str, or a readable stream."
                )
            parsed = response_parser(raw)
            if isinstance(parsed, tuple):
                vector, usage = parsed
            else:
                vector, usage = parsed, {}
            array = np.asarray(vector, dtype=np.float32)
            if array.ndim == 2 and array.shape[0] == 1:
                array = array[0]
            if array.ndim != 1 or array.size == 0:
                raise ValueError(
                    "Bedrock response parser must return one one-dimensional embedding."
                )
            return array, usage

        vector, usage = _call_with_retry(
            invoke_once,
            max_retries=max_retries,
            initial_backoff_seconds=initial_backoff_seconds,
        )
        if vectors and vector.shape != vectors[0].shape:
            raise ValueError("Embedding dimension changed within a Bedrock batch.")
        vectors.append(vector)
        total_tokens += int(usage.get("total_tokens", 0))

    if not vectors:
        return np.empty((0, embedding_dim or 0), dtype=np.float32), {"total_tokens": 0}
    return np.vstack(vectors).astype(np.float32, copy=False), {
        "total_tokens": total_tokens
    }


def list_embedding_models(provider: str, client: Any | None = None) -> list[str]:
    """List embedding-like model IDs visible to the configured account."""
    provider = provider.lower()
    if provider == "openai":
        client = client or create_openai_client()
        model_ids = [model.id for model in client.models.list().data]
    elif provider == "gemini":
        client = client or create_gemini_client()
        model_ids = [
            getattr(model, "name", "")
            for model in client.models.list()
        ]
    elif provider == "bedrock":
        raise ValueError(
            "Bedrock model availability is region/account specific. Configure the "
            "model ID or inference profile explicitly."
        )
    else:
        raise ValueError("provider must be 'bedrock', 'openai', or 'gemini'.")
    return sorted(model_id for model_id in model_ids if "embed" in model_id.lower())


def embed_openai(
    texts: Sequence[str],
    client: Any,
    model: str,
    embedding_dim: int | None,
) -> tuple[np.ndarray, Mapping[str, int]]:
    """Embed a batch using the official OpenAI Python SDK."""
    kwargs: dict[str, Any] = {
        "model": model,
        "input": list(texts),
        "encoding_format": "float",
    }
    if embedding_dim is not None:
        kwargs["dimensions"] = embedding_dim
    response = client.embeddings.create(**kwargs)
    ordered = sorted(response.data, key=lambda item: item.index)
    matrix = np.asarray([item.embedding for item in ordered], dtype=np.float32)
    usage = getattr(response, "usage", None)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    return matrix, {"total_tokens": total_tokens}


def embed_gemini(
    texts: Sequence[str],
    client: Any,
    model: str,
    embedding_dim: int | None,
) -> tuple[np.ndarray, Mapping[str, int]]:
    """Embed separate text inputs using the current Google GenAI SDK."""
    try:
        from google.genai import types
    except ImportError as error:
        raise ImportError("Install the google-genai package.") from error

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=text)],
        )
        for text in texts
    ]
    config_kwargs: dict[str, Any] = {"auto_truncate": True}
    if embedding_dim is not None:
        config_kwargs["output_dimensionality"] = embedding_dim
    if not model.startswith("gemini-embedding-2"):
        config_kwargs["task_type"] = "CLASSIFICATION"

    response = client.models.embed_content(
        model=model,
        contents=contents,
        config=types.EmbedContentConfig(**config_kwargs),
    )
    embeddings = getattr(response, "embeddings", None) or []
    if len(embeddings) != len(texts):
        raise RuntimeError(
            "Gemini returned an unexpected number of embeddings. Each input must "
            "be wrapped in a separate Content object; no silent fallback is used."
        )
    matrix = np.asarray([item.values for item in embeddings], dtype=np.float32)
    metadata = getattr(response, "metadata", None)
    billable_characters = int(
        getattr(metadata, "billable_character_count", 0) or 0
    )
    return matrix, {"billable_characters": billable_characters}


def _exception_status(error: Exception) -> int | None:
    for attribute in ("status_code", "code"):
        value = getattr(error, attribute, None)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    response = getattr(error, "response", None)
    if isinstance(response, Mapping):
        metadata = response.get("ResponseMetadata", {})
        if isinstance(metadata, Mapping):
            try:
                return int(metadata.get("HTTPStatusCode"))
            except (TypeError, ValueError):
                pass
    return None


def _is_retryable(error: Exception) -> bool:
    status = _exception_status(error)
    if status is not None:
        return status in {408, 409, 429, 500, 502, 503, 504}
    name = type(error).__name__.lower()
    return any(
        word in name
        for word in (
            "timeout",
            "connection",
            "temporar",
            "throttl",
            "modelnotready",
            "serviceunavailable",
            "internalserver",
        )
    )


def _call_with_retry(
    operation: Callable[[], tuple[np.ndarray, Mapping[str, int]]],
    max_retries: int,
    initial_backoff_seconds: float,
) -> tuple[np.ndarray, Mapping[str, int]]:
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative.")
    for attempt in range(max_retries + 1):
        try:
            return operation()
        except Exception as error:
            if attempt >= max_retries or not _is_retryable(error):
                raise
            delay = initial_backoff_seconds * (2**attempt)
            delay += random.uniform(0.0, max(0.1, delay * 0.25))
            time.sleep(delay)
    raise AssertionError("unreachable")


def _embedding_config(
    provider: str,
    model: str,
    embedding_dim: int | None,
    text_cols: Sequence[str],
    text_template: str | None,
    project_id_col: str,
    target_col: str,
    max_input_tokens: int,
    gemini_chars_per_token: float,
    batch_size: int,
    provider_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = {
        "provider": provider,
        "model": model,
        "embedding_dim": embedding_dim,
        "text_cols": list(text_cols),
        "text_template": text_template,
        "normalization_version": NORMALIZATION_VERSION,
        "project_id_col": project_id_col,
        "target_col": target_col,
        "max_input_tokens": max_input_tokens,
        "gemini_chars_per_token": gemini_chars_per_token,
        "batch_size": batch_size,
        "gemini_2_task_format": GEMINI_2_CLASSIFICATION_PREFIX,
    }
    if provider_config:
        config["provider_config"] = dict(provider_config)
    return config


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _resolve_cache_dir(
    output_root: str | Path,
    config: Mapping[str, Any],
) -> Path:
    dimension = config["embedding_dim"]
    dimension_label = f"dim{dimension}" if dimension is not None else "dimdefault"
    directory = (
        f"{_safe_name(str(config['provider']))}_"
        f"{_safe_name(str(config['model']))}_{dimension_label}_"
        f"{_config_fingerprint(config)}"
    )
    return Path(output_root) / directory


def _validate_or_write_config(cache_dir: Path, config: Mapping[str, Any]) -> None:
    config_path = cache_dir / "config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        saved_core = {key: saved.get(key) for key in config}
        if saved_core != dict(config):
            raise ValueError("Cache configuration mismatch")
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(
            config_path,
            {**dict(config), "config_fingerprint": _config_fingerprint(config)},
        )


def _build_metadata(
    df: pd.DataFrame,
    split: str,
    project_id_col: str,
    details: pd.DataFrame,
    provider: str,
    model: str,
    embedding_dim: int | None,
    config_fingerprint: str,
) -> pd.DataFrame:
    project_ids = df[project_id_col]
    if project_ids.isna().any() or not project_ids.is_unique:
        raise ValueError(f"{project_id_col!r} must be non-null and unique per split.")
    metadata = pd.DataFrame(
        {
            "row_position": np.arange(len(df), dtype=np.int64),
            "split": split,
            "original_index": [_json_scalar(value) for value in df.index],
            "project_id": project_ids.map(str).to_numpy(),
            "row_key": [f"{split}::{value}" for value in project_ids.map(str)],
            "text_hash": details.loc[df.index, "text_hash"].to_numpy(),
            "provider": provider,
            "model": model,
            "embedding_dim": embedding_dim,
            "estimated_tokens": details.loc[df.index, "estimated_tokens"].to_numpy(),
            "character_count": details.loc[df.index, "character_count"].to_numpy(),
            "truncated": details.loc[df.index, "truncated"].to_numpy(),
            "config_fingerprint": config_fingerprint,
        }
    )
    assert metadata["row_key"].is_unique
    return metadata


def _batch_ranges(
    token_counts: Sequence[int],
    batch_size: int,
    provider: str,
) -> list[tuple[int, int]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(token_counts):
        stop = min(start + batch_size, len(token_counts))
        if provider == "openai":
            # Official limit: 300,000 total tokens across all inputs/request.
            while stop > start + 1 and sum(token_counts[start:stop]) > 300_000:
                stop -= 1
            if sum(token_counts[start:stop]) > 300_000:
                raise ValueError("A single OpenAI batch exceeds 300,000 tokens.")
        ranges.append((start, stop))
        start = stop
    return ranges


def _validate_checkpoint(
    embedding_path: Path,
    metadata_path: Path,
    expected_metadata: pd.DataFrame,
) -> np.ndarray | None:
    if not embedding_path.exists() and not metadata_path.exists():
        return None
    if embedding_path.exists() != metadata_path.exists():
        raise ValueError("Incomplete checkpoint: embedding/metadata shard mismatch")
    matrix = np.load(embedding_path, allow_pickle=False)
    metadata = pd.read_parquet(metadata_path)
    expected = expected_metadata.reset_index(drop=True)
    for column in ("row_key", "text_hash", "config_fingerprint"):
        if metadata[column].astype(str).tolist() != expected[column].astype(str).tolist():
            raise ValueError("Cache configuration mismatch")
    if matrix.shape[0] != len(metadata) or not np.isfinite(matrix).all():
        raise ValueError("Invalid embedding checkpoint")
    return np.asarray(matrix, dtype=np.float32)


def _append_failure_log(cache_dir: Path, split: str, start: int, stop: int, error: Exception) -> None:
    # Do not persist request text or full provider errors, which could echo input.
    record = {
        "timestamp": _utc_now(),
        "split": split,
        "start": start,
        "stop": stop,
        "error_type": type(error).__name__,
        "status_code": _exception_status(error),
        "message": "API request failed; inspect the exception in the active session.",
    }
    path = cache_dir / "failures.jsonl"
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _print_dry_run(report: Mapping[str, Any]) -> None:
    print("Embedding dry run (API is not called)")
    for key, value in report.items():
        print(f"{key}: {value}")
    if report.get("provider") == "bedrock":
        print("Set a current model-specific token price to enable a cost estimate.")
    else:
        print("Verify price_per_million_tokens against the official pricing page before running.")


def generate_embeddings(
    df: pd.DataFrame,
    split: str,
    provider: str,
    model: str,
    output_root: str | Path = "data/embeddings",
    text_cols: Sequence[str] = DEFAULT_TEXT_COLS,
    text_template: str | None = DEFAULT_TEXT_TEMPLATE,
    project_id_col: str = "project_id",
    target_col: str = "science_tech_decision",
    embedding_dim: int | None = 1024,
    batch_size: int = 32,
    max_retries: int = 6,
    initial_backoff_seconds: float = 1.0,
    max_input_tokens: int = 8000,
    gemini_chars_per_token: float = 1.0,
    price_per_million_tokens: float | None = None,
    max_budget_usd: float = 20.0,
    dry_run: bool = True,
    client: Any | None = None,
    embedder: Embedder | None = None,
    region_name: str | None = None,
    aws_profile_name: str | None = None,
    content_type: str = "application/json",
    accept: str = "application/json",
    request_builder: BedrockRequestBuilder = build_bedrock_titan_request,
    response_parser: BedrockResponseParser = parse_bedrock_titan_response,
    invoke_model_kwargs: Mapping[str, Any] | None = None,
    adapter_id: str = "titan-text-embeddings-v2-v1",
    bedrock_cache_identity: Mapping[str, Any] | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Generate resumable embeddings and save all artifacts below output_root.

    ``target_col`` is explicitly forbidden from ``text_cols`` and is never sent
    to an API. In dry-run mode no client is initialized and no API is called.
    """
    provider = provider.lower()
    if provider not in {"bedrock", "openai", "gemini"}:
        raise ValueError("provider must be 'bedrock', 'openai', or 'gemini'.")
    if provider == "bedrock":
        if not adapter_id.strip():
            raise ValueError("adapter_id must be a non-empty cache version string.")
        if invoke_model_kwargs and not bedrock_cache_identity:
            raise ValueError(
                "bedrock_cache_identity is required when invoke_model_kwargs may "
                "change how the embeddings are produced."
            )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", split):
        raise ValueError("split may contain only letters, numbers, _, ., and -.")
    if target_col in text_cols:
        raise ValueError("target_col must never be included in text_cols.")
    required = [*text_cols, project_id_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    if not df.index.is_unique:
        raise ValueError("df.index must be unique.")
    if embedding_dim is not None and embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive or None.")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative.")
    if initial_backoff_seconds < 0:
        raise ValueError("initial_backoff_seconds must be non-negative.")
    if max_budget_usd < 0:
        raise ValueError("max_budget_usd must be non-negative.")

    texts, details, base_report = _prepare_inputs(
        df=df,
        provider=provider,
        model=model,
        text_cols=text_cols,
        text_template=text_template,
        max_input_tokens=max_input_tokens,
        gemini_chars_per_token=gemini_chars_per_token,
    )
    price = (
        DEFAULT_PRICE_PER_MILLION_TOKENS[provider]
        if price_per_million_tokens is None
        else float(price_per_million_tokens)
    )
    if price < 0:
        raise ValueError("price_per_million_tokens must be non-negative.")
    estimated_cost = base_report["total_estimated_tokens"] / 1_000_000 * price
    report = {
        **base_report,
        "split": split,
        "provider": provider,
        "model": model,
        "embedding_dim": embedding_dim,
        "price_per_million_tokens": price,
        "estimated_cost_usd": estimated_cost,
        "max_budget_usd": max_budget_usd,
    }
    if provider == "bedrock":
        report.update(
            {
                "region_name": region_name,
                "adapter_id": adapter_id,
            }
        )
    if dry_run:
        _print_dry_run(report)
        return {
            "embeddings": None,
            "metadata": None,
            "cache_dir": None,
            "report": report,
            "from_cache": False,
        }
    provider_config: dict[str, Any] | None = None
    if provider == "bedrock":
        provider_config = {
            "region_name": region_name,
            "content_type": content_type,
            "accept": accept,
            "adapter_id": adapter_id,
            "cache_identity": dict(bedrock_cache_identity or {}),
        }
    config = _embedding_config(
        provider=provider,
        model=model,
        embedding_dim=embedding_dim,
        text_cols=text_cols,
        text_template=text_template,
        project_id_col=project_id_col,
        target_col=target_col,
        max_input_tokens=max_input_tokens,
        gemini_chars_per_token=gemini_chars_per_token,
        batch_size=batch_size,
        provider_config=provider_config,
    )
    fingerprint = _config_fingerprint(config)
    cache_dir = _resolve_cache_dir(output_root, config)
    _validate_or_write_config(cache_dir, config)
    expected_metadata = _build_metadata(
        df=df,
        split=split,
        project_id_col=project_id_col,
        details=details,
        provider=provider,
        model=model,
        embedding_dim=embedding_dim,
        config_fingerprint=fingerprint,
    )

    final_embedding_path = cache_dir / f"{split}_embeddings.npy"
    final_metadata_path = cache_dir / f"{split}_metadata.parquet"
    if final_embedding_path.exists() and final_metadata_path.exists():
        embeddings, metadata = load_embeddings(
            cache_dir=cache_dir,
            split=split,
            expected_df=df,
            project_id_col=project_id_col,
            expected_text_hashes=expected_metadata["text_hash"],
        )
        return {
            "embeddings": embeddings,
            "metadata": metadata,
            "cache_dir": cache_dir,
            "report": {**report, "cache_hit": True},
            "from_cache": True,
        }
    if final_embedding_path.exists() != final_metadata_path.exists():
        raise ValueError("Incomplete final cache: embedding/metadata mismatch")

    # A complete, validated cache is returned above without requiring an API
    # key. The budget guard applies only when new paid requests may be made.
    if estimated_cost > max_budget_usd:
        raise RuntimeError(
            "Estimated cost exceeds configured budget. API processing was not started."
        )

    embedder_handles_retries = False
    if client is None:
        if provider == "bedrock":
            client = create_bedrock_runtime_client(
                region_name=region_name,
                profile_name=aws_profile_name,
            )
        elif provider == "openai":
            client = create_openai_client()
        else:
            client = create_gemini_client()
    if embedder is None:
        if provider == "bedrock":
            def bedrock_embedder(
                batch_texts: Sequence[str],
                runtime_client: Any,
                selected_model: str,
                selected_dimension: int | None,
            ) -> tuple[np.ndarray, Mapping[str, int]]:
                return embed_bedrock(
                    batch_texts,
                    runtime_client,
                    selected_model,
                    selected_dimension,
                    content_type=content_type,
                    accept=accept,
                    request_builder=request_builder,
                    response_parser=response_parser,
                    invoke_model_kwargs=invoke_model_kwargs,
                    max_retries=max_retries,
                    initial_backoff_seconds=initial_backoff_seconds,
                )

            embedder = bedrock_embedder
            embedder_handles_retries = True
        elif provider == "openai":
            embedder = embed_openai
        else:
            embedder = embed_gemini

    ranges = _batch_ranges(details["estimated_tokens"].tolist(), batch_size, provider)
    shard_dir = cache_dir / "shards" / split
    shard_dir.mkdir(parents=True, exist_ok=True)
    progress_path = cache_dir / f"{split}_progress.json"
    completed_rows = 0
    provider_tokens = 0
    resolved_dimension: int | None = None

    iterator: Any = ranges
    progress_bar = None
    if show_progress:
        try:
            from tqdm.auto import tqdm

            progress_bar = tqdm(ranges, total=len(ranges), desc=f"{provider} {model}")
            iterator = progress_bar
        except ImportError:
            pass

    for start, stop in iterator:
        stem = f"{start:08d}_{stop:08d}"
        shard_embedding_path = shard_dir / f"{stem}.npy"
        shard_metadata_path = shard_dir / f"{stem}.parquet"
        expected_batch_metadata = expected_metadata.iloc[start:stop].copy()
        matrix = _validate_checkpoint(
            shard_embedding_path,
            shard_metadata_path,
            expected_batch_metadata,
        )
        usage: Mapping[str, int] = {}
        if matrix is None:
            try:
                if embedder_handles_retries:
                    matrix, usage = embedder(
                        texts[start:stop], client, model, embedding_dim
                    )
                else:
                    matrix, usage = _call_with_retry(
                        lambda: embedder(
                            texts[start:stop], client, model, embedding_dim
                        ),
                        max_retries=max_retries,
                        initial_backoff_seconds=initial_backoff_seconds,
                    )
            except Exception as error:
                _append_failure_log(cache_dir, split, start, stop, error)
                raise
            matrix = np.asarray(matrix, dtype=np.float32)
            if matrix.ndim != 2 or matrix.shape[0] != stop - start:
                raise ValueError("Provider returned an invalid embedding matrix shape.")
            if not np.isfinite(matrix).all():
                raise ValueError("Provider returned non-finite embedding values.")
            expected_batch_metadata["embedding_dim"] = matrix.shape[1]
            _atomic_save_npy(shard_embedding_path, matrix)
            _atomic_save_parquet(shard_metadata_path, expected_batch_metadata)

        if resolved_dimension is None:
            resolved_dimension = int(matrix.shape[1])
        elif matrix.shape[1] != resolved_dimension:
            raise ValueError("Embedding dimension changed between batches.")
        if embedding_dim is not None and matrix.shape[1] != embedding_dim:
            raise ValueError(
                f"Provider returned dimension {matrix.shape[1]}, expected {embedding_dim}."
            )

        completed_rows += stop - start
        reported_tokens = int(usage.get("total_tokens", 0) or 0)
        provider_tokens += (
            reported_tokens
            if reported_tokens > 0
            else int(details["estimated_tokens"].iloc[start:stop].sum())
        )
        progress = {
            "split": split,
            "provider": provider,
            "model": model,
            "completed_rows": completed_rows,
            "total_rows": len(df),
            "provider_or_estimated_tokens": provider_tokens,
            "estimated_spend_usd": provider_tokens / 1_000_000 * price,
            "truncated_rows": int(details["truncated"].iloc[:stop].sum()),
            "updated_at": _utc_now(),
        }
        _atomic_write_json(progress_path, progress)
        if progress_bar is not None:
            progress_bar.set_postfix(
                rows=f"{completed_rows}/{len(df)}",
                spend=f"${progress['estimated_spend_usd']:.4f}",
            )

    if resolved_dimension is None:
        raise ValueError("No embeddings were generated.")

    temporary_final = final_embedding_path.with_suffix(".npy.tmp")
    final_memmap = np.lib.format.open_memmap(
        temporary_final,
        mode="w+",
        dtype=np.float32,
        shape=(len(df), resolved_dimension),
    )
    metadata_parts: list[pd.DataFrame] = []
    for start, stop in ranges:
        stem = f"{start:08d}_{stop:08d}"
        final_memmap[start:stop] = np.load(
            shard_dir / f"{stem}.npy", allow_pickle=False
        )
        metadata_parts.append(pd.read_parquet(shard_dir / f"{stem}.parquet"))
    final_memmap.flush()
    del final_memmap
    os.replace(temporary_final, final_embedding_path)

    final_metadata = pd.concat(metadata_parts, ignore_index=True)
    if final_metadata["row_key"].tolist() != expected_metadata["row_key"].tolist():
        raise ValueError("Final metadata order does not match the source rows.")
    _atomic_save_parquet(final_metadata_path, final_metadata)

    saved_config = json.loads((cache_dir / "config.json").read_text(encoding="utf-8"))
    saved_config["resolved_embedding_dim"] = resolved_dimension
    saved_config["completed_at"] = _utc_now()
    _atomic_write_json(cache_dir / "config.json", saved_config)

    embeddings, metadata = load_embeddings(
        cache_dir=cache_dir,
        split=split,
        expected_df=df,
        project_id_col=project_id_col,
        expected_text_hashes=expected_metadata["text_hash"],
    )
    return {
        "embeddings": embeddings,
        "metadata": metadata,
        "cache_dir": cache_dir,
        "report": {**report, "cache_hit": False},
        "from_cache": False,
    }


def verify_embedding_alignment(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    project_id_col: str = "project_id",
    split: str | None = None,
) -> None:
    """Assert explicit project-ID and original-index alignment."""
    if len(df) != len(metadata):
        raise ValueError("Embedding metadata length does not match the source data.")
    if split is not None and not metadata["split"].eq(split).all():
        raise ValueError("Embedding metadata split mismatch.")
    expected_ids = df[project_id_col].map(str).tolist()
    if metadata["project_id"].map(str).tolist() != expected_ids:
        raise ValueError("Embedding metadata project_id order mismatch.")
    if metadata["original_index"].map(str).tolist() != df.index.map(str).tolist():
        raise ValueError("Embedding metadata original_index order mismatch.")
    if not metadata["project_id"].is_unique:
        raise ValueError("Embedding metadata project_id is not unique.")


def load_embeddings(
    cache_dir: str | Path,
    split: str,
    expected_df: pd.DataFrame | None = None,
    project_id_col: str = "project_id",
    expected_text_hashes: Sequence[str] | None = None,
    mmap_mode: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load and validate embeddings; optionally return a NumPy memory map."""
    directory = Path(cache_dir)
    embedding_path = directory / f"{split}_embeddings.npy"
    metadata_path = directory / f"{split}_metadata.parquet"
    if not embedding_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Saved embeddings for split {split!r} are incomplete.")
    embeddings = np.load(
        embedding_path,
        mmap_mode=mmap_mode,
        allow_pickle=False,
    )
    metadata = pd.read_parquet(metadata_path)
    if embeddings.ndim != 2 or len(embeddings) != len(metadata):
        raise ValueError("Embedding array and metadata length/shape mismatch.")
    if not np.isfinite(embeddings).all():
        raise ValueError("Saved embeddings contain non-finite values.")
    if metadata["row_position"].tolist() != list(range(len(metadata))):
        raise ValueError("Embedding metadata row positions are not contiguous.")
    if expected_text_hashes is not None:
        expected = pd.Series(expected_text_hashes).map(str).tolist()
        if metadata["text_hash"].map(str).tolist() != expected:
            raise ValueError("Cache configuration mismatch")
    if expected_df is not None:
        verify_embedding_alignment(
            expected_df,
            metadata,
            project_id_col=project_id_col,
            split=split,
        )
    return embeddings, metadata


def l2_normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Return a normalized copy; never overwrite the saved raw embeddings."""
    array = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return array / norms


__all__ = [
    "DEFAULT_PRICE_PER_MILLION_TOKENS",
    "DEFAULT_TEXT_COLS",
    "DEFAULT_TEXT_TEMPLATE",
    "build_embedding_text",
    "build_bedrock_cohere_classification_request",
    "build_bedrock_titan_request",
    "create_gemini_client",
    "create_openai_client",
    "create_bedrock_runtime_client",
    "embed_gemini",
    "embed_openai",
    "embed_bedrock",
    "embed_bedrock_cohere",
    "generate_embeddings",
    "l2_normalize_embeddings",
    "list_embedding_models",
    "load_embeddings",
    "normalize_embedding_text",
    "parse_bedrock_titan_response",
    "parse_bedrock_cohere_response",
    "text_sha256",
    "verify_embedding_alignment",
]
