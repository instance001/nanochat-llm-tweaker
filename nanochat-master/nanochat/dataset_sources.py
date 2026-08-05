"""
External corpus source import helpers.

These helpers intentionally import bounded samples into local corpus files.
Training still reads local_corpus through nanochat.dataset.
"""

from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

from nanochat.dataset import _extract_text_fragments

HF_OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
MAX_IMPORT_DOCS = 1_000_000


SOURCE_RECOMMENDATIONS = [
    {
        "match_terms": ["code", "coding", "programming", "python", "javascript", "typescript", "software", "developer"],
        "dataset_id": "bigcode/the-stack-dedup",
        "config": "",
        "split": "train",
        "text_column": "content",
        "why": "Code-heavy corpus for programming language patterns. Check access rules and license metadata before use.",
    },
    {
        "match_terms": ["science", "education", "tutor", "teacher", "explain", "academic", "study", "lesson"],
        "dataset_id": "HuggingFaceFW/fineweb-edu",
        "config": "",
        "split": "train",
        "text_column": "text",
        "why": "Education-filtered web text is a reasonable starting corpus for explanatory prose and tutoring behavior.",
    },
    {
        "match_terms": ["assistant", "chat", "conversation", "support", "helpdesk", "customer", "dialogue"],
        "dataset_id": "OpenAssistant/oasst1",
        "config": "",
        "split": "train",
        "text_column": "text",
        "why": "Conversation data can help shape assistant-style turns. Use as SFT inspiration carefully, not as a replacement for curated local behavior data.",
    },
    {
        "match_terms": ["medical", "health", "clinical", "doctor", "patient"],
        "dataset_id": "HuggingFaceFW/fineweb-edu",
        "config": "",
        "split": "train",
        "text_column": "text",
        "why": "Use only as a broad language source. Medical models need carefully licensed, reviewed domain data and strong evaluation before any real use.",
    },
    {
        "match_terms": ["legal", "law", "contract", "policy", "compliance"],
        "dataset_id": "HuggingFaceFW/fineweb",
        "config": "",
        "split": "train",
        "text_column": "text",
        "why": "Use as broad language background only. Legal specialization needs jurisdiction-specific, licensed source material and review.",
    },
    {
        "match_terms": ["general", "web", "knowledge", "writer", "blog", "research", "summarize", "summary"],
        "dataset_id": "HuggingFaceFW/fineweb",
        "config": "",
        "split": "train",
        "text_column": "text",
        "why": "General web text is a broad baseline for tokenizer/base-model language patterns.",
    },
]


@contextmanager
def huggingface_network_enabled() -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in HF_OFFLINE_ENV_VARS}
    for name in HF_OFFLINE_ENV_VARS:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _default_hf_loader(
    dataset_id: str,
    *,
    config: str = "",
    revision: str = "",
    split: str = "train",
    streaming: bool = True,
    trust_remote_code: bool = False,
):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Hugging Face imports require the `datasets` package.") from exc
    kwargs: dict[str, Any] = {
        "split": split,
        "streaming": streaming,
        "trust_remote_code": trust_remote_code,
    }
    if config:
        kwargs["name"] = config
    if revision:
        kwargs["revision"] = revision
    with huggingface_network_enabled():
        return load_dataset(dataset_id, **kwargs)


def _nested_value(record: dict[str, Any], key_path: str) -> Any:
    value: Any = record
    for key in key_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _text_from_record(record: Any, text_column: str) -> str | None:
    if isinstance(record, dict) and text_column:
        value = _nested_value(record, text_column)
        if value is not None:
            fragments = list(_extract_text_fragments(value))
            if fragments:
                return "\n".join(fragments).strip() or None
    fragments = list(_extract_text_fragments(record))
    if fragments:
        return "\n".join(fragments).strip() or None
    return None


def _validate_import_bounds(limit_docs: int, sample_size: int, max_chars: int) -> None:
    if limit_docs < 1 or limit_docs > MAX_IMPORT_DOCS:
        raise ValueError(f"limit_docs must be between 1 and {MAX_IMPORT_DOCS}.")
    if sample_size < 1 or sample_size > 100:
        raise ValueError("sample_size must be between 1 and 100.")
    if max_chars < 0:
        raise ValueError("max_chars must be 0 or greater.")


def collect_huggingface_corpus_records(
    dataset_id: str,
    *,
    config: str = "",
    revision: str = "",
    split: str = "train",
    text_column: str = "text",
    limit_docs: int = 1000,
    max_chars: int = 0,
    sample_size: int = 12,
    source_label: str = "",
    trust_remote_code: bool = False,
    loader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    dataset_id = dataset_id.strip()
    config = config.strip()
    revision = revision.strip()
    split = split.strip() or "train"
    text_column = text_column.strip()
    source = source_label.strip() or f"hf:{dataset_id}:{split}"
    if not dataset_id:
        raise ValueError("A Hugging Face dataset id is required.")
    _validate_import_bounds(limit_docs, sample_size, max_chars)

    load = loader or _default_hf_loader
    iterable = load(
        dataset_id,
        config=config,
        revision=revision,
        split=split,
        streaming=True,
        trust_remote_code=trust_remote_code,
    )

    records: list[dict[str, Any]] = []
    skipped_empty = 0
    seen_rows = 0
    total_chars = 0
    stopped_reason = "limit_docs"
    columns: list[str] = []
    for source_row_index, raw_record in enumerate(iterable):
        seen_rows += 1
        if not columns and isinstance(raw_record, dict):
            columns = sorted(str(key) for key in raw_record.keys())
        text = _text_from_record(raw_record, text_column)
        if not text:
            skipped_empty += 1
            continue
        if max_chars and total_chars + len(text) > max_chars and records:
            stopped_reason = "max_chars"
            break
        records.append(
            {
                "text": text,
                "source": source,
                "row_index": len(records),
                "source_row_index": source_row_index,
                "hf_dataset": dataset_id,
                "hf_config": config,
                "hf_revision": revision,
                "hf_split": split,
            }
        )
        total_chars += len(text)
        if len(records) >= limit_docs:
            stopped_reason = "limit_docs"
            break
        if max_chars and total_chars >= max_chars:
            stopped_reason = "max_chars"
            break

    if not records:
        raise ValueError("No usable text records were found in the selected dataset slice.")
    return {
        "source_type": "huggingface",
        "dataset_id": dataset_id,
        "config": config,
        "dataset_revision": revision,
        "split": split,
        "text_column": text_column,
        "source": source,
        "streaming": True,
        "limit_docs": limit_docs,
        "max_chars": max_chars,
        "seen_rows": seen_rows,
        "row_count": len(records),
        "character_count": total_chars,
        "skipped_empty_rows": skipped_empty,
        "stopped_reason": stopped_reason,
        "source_columns": columns,
        "sample_rows": records[:sample_size],
        "records": records,
        "notes": [
            "Imported records are local corpus rows after write; training remains local/offline.",
            "Review the dataset card and license before using external corpora for redistribution.",
        ],
    }


def preview_huggingface_corpus_records(dataset_id: str, **kwargs: Any) -> dict[str, Any]:
    payload = collect_huggingface_corpus_records(dataset_id, **kwargs)
    payload.pop("records", None)
    return payload


def records_to_jsonl(records: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n"


def recommend_huggingface_corpus_sources(goal: str, *, max_results: int = 4) -> dict[str, Any]:
    description = " ".join(str(goal or "").lower().split())
    max_results = max(1, min(max_results, 8))
    scored = []
    for item in SOURCE_RECOMMENDATIONS:
        score = sum(1 for term in item["match_terms"] if term in description)
        scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    selected = [item for score, item in scored if score > 0][:max_results]
    if not selected:
        selected = [item for item in SOURCE_RECOMMENDATIONS if item["dataset_id"] == "HuggingFaceFW/fineweb"][:1]
    recommendations = []
    for item in selected:
        recommendations.append(
            {
                "source_type": "huggingface",
                "dataset_id": item["dataset_id"],
                "config": item["config"],
                "split": item["split"],
                "text_column": item["text_column"],
                "suggested_limit_docs": 1000,
                "suggested_train_val_ratio": 0.1,
                "suggested_shard_size_docs": 10000,
                "license_note": "Review the dataset card, license, and redistribution terms before training or sharing outputs.",
                "why": item["why"],
            }
        )
    return {
        "goal": goal,
        "recommendations": recommendations,
        "notes": [
            "Use preview_hf_corpus_import before importing.",
            "Use import_hf_corpus only after the user approves the source, bounds, and license note.",
            "Imported shards include a manifest receipt and training remains offline afterward.",
        ],
    }


def slugify_import_name(dataset_id: str) -> str:
    leaf = dataset_id.strip().rstrip("/").split("/")[-1] or "hf_import"
    slug = re.sub(r"[^A-Za-z0-9]+", "_", leaf).strip("_").lower()
    return slug or "hf_import"


def split_records_for_import(records: list[dict[str, Any]], train_val_ratio: float = 0.0) -> dict[str, list[dict[str, Any]]]:
    if train_val_ratio < 0 or train_val_ratio >= 1:
        raise ValueError("train_val_ratio must be at least 0 and less than 1.")
    if not records:
        raise ValueError("Cannot split an empty import.")
    val_count = 0
    if train_val_ratio > 0 and len(records) > 1:
        val_count = max(1, round(len(records) * train_val_ratio))
        val_count = min(val_count, len(records) - 1)
    train_count = len(records) - val_count
    return {
        "train": records[:train_count],
        "val": records[train_count:] if val_count else [],
    }


def shard_import_records(
    records_by_split: dict[str, list[dict[str, Any]]],
    *,
    dataset_id: str,
    source_split: str,
    shard_size_docs: int = 10000,
    output_format: str = "parquet",
) -> list[dict[str, Any]]:
    if shard_size_docs < 1:
        raise ValueError("shard_size_docs must be 1 or greater.")
    suffix = "jsonl" if output_format == "jsonl" else "parquet"
    base_name = slugify_import_name(dataset_id)
    shards: list[dict[str, Any]] = []
    for split_name in ("train", "val"):
        split_records = records_by_split.get(split_name, [])
        for start in range(0, len(split_records), shard_size_docs):
            shard_records = split_records[start:start + shard_size_docs]
            shard_index = (start // shard_size_docs) + 1
            path = f"{split_name}/{base_name}_{split_name}_{shard_index:05d}.{suffix}"
            shards.append(
                {
                    "split": split_name,
                    "path": path,
                    "row_count": len(shard_records),
                    "character_count": sum(len(str(record.get("text", ""))) for record in shard_records),
                    "records": shard_records,
                }
            )
    return shards


def build_import_manifest(
    import_payload: dict[str, Any],
    written_shards: list[dict[str, Any]],
    *,
    train_val_ratio: float,
    shard_size_docs: int,
    output_format: str,
    license_note: str = "",
    tool_version: str = "nanochat.builder corpus import-hf v1",
) -> dict[str, Any]:
    content_hashes = {shard["path"]: shard.get("sha256", "") for shard in written_shards}
    shard_paths = [shard["path"] for shard in written_shards]
    return {
        "manifest_schema": "nanochat.external_corpus_import.v1",
        "source_type": import_payload.get("source_type", "huggingface"),
        "dataset_id": import_payload.get("dataset_id", ""),
        "dataset_revision": import_payload.get("dataset_revision", ""),
        "dataset_config": import_payload.get("config", ""),
        "source_split": import_payload.get("split", ""),
        "text_column": import_payload.get("text_column", ""),
        "filters": {
            "max_chars": import_payload.get("max_chars", 0),
            "skipped_empty_rows": import_payload.get("skipped_empty_rows", 0),
            "stopped_reason": import_payload.get("stopped_reason", ""),
            "source_columns": import_payload.get("source_columns", []),
        },
        "requested_limit": import_payload.get("limit_docs", 0),
        "actual_documents": import_payload.get("row_count", 0),
        "actual_characters": import_payload.get("character_count", 0),
        "train_val_ratio": train_val_ratio,
        "shard_size_docs": shard_size_docs,
        "output_format": output_format,
        "shard_paths": shard_paths,
        "content_hashes": content_hashes,
        "license_note": license_note,
        "import_timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_version": tool_version,
        "notes": import_payload.get("notes", []),
    }
