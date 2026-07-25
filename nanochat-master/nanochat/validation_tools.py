"""
Validation helpers for local builder datasets.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from nanochat.dataset import TEXT_SUFFIXES, iter_documents_from_path, list_local_data_files
from nanochat.sft_dataset_tools import normalize_conversation


def _empty_report(kind: str, path: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": path,
        "ok": False,
        "errors": [],
        "warnings": [],
        "file_count": 0,
        "record_count": 0,
        "empty_count": 0,
        "character_count": 0,
        "estimated_tokens": 0,
        "active_token_count": None,
        "active_tokenizer": {
            "available": False,
            "path": "",
            "error": "",
        },
        "files": [],
        "unsupported_file_count": 0,
        "unsupported_files": [],
        "duplicate_document_count": 0,
        "duplicate_documents": [],
        "schema_hints": [],
    }


def _estimate_tokens(character_count: int) -> int:
    return max(0, int((character_count + 3) // 4))


def _is_supported_corpus_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    return suffix == ".parquet" or suffix in TEXT_SUFFIXES


def _unsupported_corpus_files(target: Path) -> list[Path]:
    if target.is_file():
        return [] if _is_supported_corpus_file(target) else [target]
    if not target.is_dir():
        return []
    return sorted(path for path in target.rglob("*") if path.is_file() and not _is_supported_corpus_file(path))


def _document_digest(text: str) -> str:
    return hashlib.sha1(" ".join(text.split()).encode("utf-8")).hexdigest()


def _normalize_document_preview(text: str, limit: int = 160) -> str:
    return " ".join(text.split())[:limit]


def _sft_schema_hints() -> list[str]:
    return [
        "Use JSONL with one conversation per line.",
        "Each line must be a JSON array of message objects.",
        "Roles must alternate user, assistant, user, assistant.",
        "Each message needs string fields named role and content.",
    ]


def _corpus_schema_hints() -> list[str]:
    return [
        "Corpus files can be plain text/code, JSON, JSONL, or parquet.",
        "For JSON/JSONL, text is extracted from keys such as text, content, prompt, response, question, and answer.",
        "For parquet, install pyarrow and include text-like columns or nested message content.",
        "Put training files under local_corpus/train and validation files under local_corpus/val.",
    ]


def _load_active_tokenizer(tokenizer_dir: str | Path | None):
    if not tokenizer_dir:
        return None, {"available": False, "path": "", "error": ""}
    target = Path(tokenizer_dir)
    status = {"available": False, "path": str(target), "error": ""}
    if not target.exists():
        status["error"] = "Tokenizer directory not found."
        return None, status
    try:
        from nanochat.tokenizer import RustBPETokenizer

        tokenizer = RustBPETokenizer.from_directory(str(target))
    except Exception as exc:
        status["error"] = str(exc)
        return None, status
    status["available"] = True
    return tokenizer, status


def _count_active_tokens(tokenizer: Any, text: str) -> int:
    encoded = tokenizer.encode(text)
    if isinstance(encoded, list):
        return len(encoded)
    return len(list(encoded))


def validate_sft_jsonl_content(content: str, path: str = "") -> dict[str, Any]:
    report = _empty_report("sft_jsonl", path)
    report["schema_hints"] = _sft_schema_hints()
    report["file_count"] = 1
    lines = str(content or "").splitlines()
    if not lines:
        report["errors"].append("SFT file is empty.")
        return report

    role_counts = {"user": 0, "assistant": 0}
    max_turns = 0
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            report["empty_count"] += 1
            continue
        try:
            payload = json.loads(line)
            conversation = normalize_conversation(payload)
        except (json.JSONDecodeError, ValueError) as exc:
            report["errors"].append(f"Line {line_number}: {exc}")
            continue
        report["record_count"] += 1
        max_turns = max(max_turns, len(conversation))
        for message in conversation:
            role_counts[message["role"]] = role_counts.get(message["role"], 0) + 1
            report["character_count"] += len(message["content"])

    if report["record_count"] == 0:
        report["errors"].append("No valid SFT conversations were found.")
    if report["empty_count"]:
        report["warnings"].append(f"Skipped {report['empty_count']} empty line(s).")
    if report["record_count"] > 0 and max_turns <= 2:
        report["warnings"].append("All valid conversations are single-turn pairs. That is okay, but multi-turn examples may improve behavior.")

    report["role_counts"] = role_counts
    report["max_turns"] = max_turns
    report["estimated_tokens"] = _estimate_tokens(report["character_count"])
    report["ok"] = not report["errors"]
    return report


def validate_sft_jsonl_file(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists() or not target.is_file():
        report = _empty_report("sft_jsonl", str(target))
        report["errors"].append(f"SFT file not found: {target}")
        return report
    content = target.read_text(encoding="utf-8", errors="replace")
    return validate_sft_jsonl_content(content, path=str(target))


def validate_corpus_path(path: str | Path, tokenizer_dir: str | Path | None = None) -> dict[str, Any]:
    target = Path(path)
    report = _empty_report("corpus", str(target))
    report["schema_hints"] = _corpus_schema_hints()
    tokenizer, tokenizer_status = _load_active_tokenizer(tokenizer_dir)
    report["active_tokenizer"] = tokenizer_status
    if not target.exists():
        report["errors"].append(f"Corpus path not found: {target}")
        return report

    unsupported_files = _unsupported_corpus_files(target)
    report["unsupported_file_count"] = len(unsupported_files)
    report["unsupported_files"] = [str(path) for path in unsupported_files[:20]]
    if unsupported_files:
        report["warnings"].append(
            f"Ignored {len(unsupported_files)} unsupported corpus file(s). Supported text extensions are text/code/JSON/JSONL plus parquet."
        )

    files = [target] if target.is_file() and _is_supported_corpus_file(target) else list_local_data_files(split="train", data_dir=target)
    report["file_count"] = len(files)
    if not files:
        report["errors"].append("No supported corpus files were found.")
        return report

    seen_documents: dict[str, dict[str, Any]] = {}
    for file_path in files:
        file_entry = {
            "path": str(file_path),
            "document_count": 0,
            "character_count": 0,
            "errors": [],
        }
        try:
            for document in iter_documents_from_path(file_path):
                text = str(document or "").strip()
                if not text:
                    report["empty_count"] += 1
                    continue
                digest = _document_digest(text)
                if digest in seen_documents:
                    report["duplicate_document_count"] += 1
                    if len(report["duplicate_documents"]) < 20:
                        report["duplicate_documents"].append(
                            {
                                "path": str(file_path),
                                "first_path": seen_documents[digest]["path"],
                                "character_count": len(text),
                                "preview": _normalize_document_preview(text),
                            }
                        )
                else:
                    seen_documents[digest] = {"path": str(file_path)}
                if tokenizer is not None:
                    try:
                        report["active_token_count"] = (report["active_token_count"] or 0) + _count_active_tokens(tokenizer, text)
                    except Exception as exc:
                        report["active_tokenizer"]["available"] = False
                        report["active_tokenizer"]["error"] = str(exc)
                        tokenizer = None
                file_entry["document_count"] += 1
                file_entry["character_count"] += len(text)
        except Exception as exc:
            file_entry["errors"].append(str(exc))
            report["errors"].append(f"{file_path.name}: {exc}")
        report["record_count"] += file_entry["document_count"]
        report["character_count"] += file_entry["character_count"]
        report["files"].append(file_entry)

    if report["record_count"] == 0:
        report["errors"].append("No readable corpus documents were found.")
    if target.is_dir():
        train_dir = target / "train"
        val_dir = target / "val"
        if not train_dir.exists():
            report["warnings"].append("No train/ folder exists; train and val reads will both fall back to the corpus root.")
        if not val_dir.exists():
            report["warnings"].append("No val/ folder exists; validation reads will fall back to the corpus root.")
    if report["duplicate_document_count"]:
        report["warnings"].append(f"Found {report['duplicate_document_count']} exact duplicate corpus document(s).")

    report["estimated_tokens"] = _estimate_tokens(report["character_count"])
    report["ok"] = not report["errors"]
    return report
