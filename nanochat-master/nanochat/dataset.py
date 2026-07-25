"""
Local-only corpus helpers for tokenizer and base-model training.

The dashboard and training scripts use this module to read user-provided local
data only. No download path lives here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCAL_CORPUS_DIR = REPO_ROOT / "local_corpus"
LOCAL_CORPUS_ENV = "NANOCHAT_LOCAL_CORPUS_DIR"

TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".json",
    ".jsonl",
    ".jsx",
    ".md",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".rst",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
TEXT_KEYS = (
    "text",
    "content",
    "body",
    "instruction",
    "input",
    "output",
    "prompt",
    "response",
    "question",
    "answer",
)


def get_local_corpus_dir(data_dir: str | os.PathLike[str] | None = None) -> Path:
    raw = data_dir or os.environ.get(LOCAL_CORPUS_ENV) or str(DEFAULT_LOCAL_CORPUS_DIR)
    return Path(raw).expanduser().resolve()


def _resolve_split_dir(corpus_dir: Path, split: str) -> Path:
    split_dir = corpus_dir / split
    return split_dir if split_dir.exists() else corpus_dir


def list_local_data_files(split: str = "train", data_dir: str | os.PathLike[str] | None = None) -> list[Path]:
    corpus_dir = get_local_corpus_dir(data_dir)
    if not corpus_dir.exists():
        return []
    search_root = _resolve_split_dir(corpus_dir, split)
    if not search_root.exists():
        return []
    paths = []
    for path in search_root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".parquet" or suffix in TEXT_SUFFIXES:
            paths.append(path)
    return sorted(paths)


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    text = text.strip()
    return text or None


def _conversation_to_text(payload: Any) -> str | None:
    if isinstance(payload, dict) and isinstance(payload.get("messages"), list):
        payload = payload["messages"]
    if isinstance(payload, list) and payload and all(isinstance(item, dict) and "content" in item for item in payload):
        parts = [item.get("content", "").strip() for item in payload if item.get("content")]
        joined = "\n".join(part for part in parts if part)
        return joined or None
    return None


def _extract_text_fragments(payload: Any) -> Iterator[str]:
    conversation_text = _conversation_to_text(payload)
    if conversation_text is not None:
        yield conversation_text
        return

    if isinstance(payload, str):
        normalized = _normalize_text(payload)
        if normalized is not None:
            yield normalized
        return

    if isinstance(payload, dict):
        preferred_keys = [key for key in TEXT_KEYS if key in payload]
        keys = preferred_keys or list(payload.keys())
        for key in keys:
            yield from _extract_text_fragments(payload[key])
        return

    if isinstance(payload, list):
        for item in payload:
            yield from _extract_text_fragments(item)


def _iter_json_documents(path: Path) -> Iterator[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                normalized = _normalize_text(line)
                if normalized is not None:
                    yield normalized
                continue
            yielded = False
            for fragment in _extract_text_fragments(payload):
                yield fragment
                yielded = True
            if not yielded:
                normalized = _normalize_text(line)
                if normalized is not None:
                    yield normalized


def _iter_json_file_documents(path: Path) -> Iterator[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        raw = handle.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        normalized = _normalize_text(raw)
        if normalized is not None:
            yield normalized
        return
    for fragment in _extract_text_fragments(payload):
        yield fragment


def _iter_text_file_documents(path: Path) -> Iterator[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    normalized = _normalize_text(text)
    if normalized is not None:
        yield normalized


def _iter_parquet_documents(path: Path) -> Iterator[str]:
    if pq is None:
        raise RuntimeError(
            f"Parquet support requires pyarrow, but it is not installed. "
            f"Install it with `uv sync --extra parquet` from nanochat-master, "
            f"or convert {path.name} to text/JSONL."
        )
    parquet_file = pq.ParquetFile(path)
    for row_group_index in range(parquet_file.num_row_groups):
        row_group = parquet_file.read_row_group(row_group_index)
        for record in row_group.to_pylist():
            for fragment in _extract_text_fragments(record):
                yield fragment


def iter_documents_from_path(path: Path) -> Iterator[str]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from _iter_json_documents(path)
    elif suffix == ".json":
        yield from _iter_json_file_documents(path)
    elif suffix == ".parquet":
        yield from _iter_parquet_documents(path)
    else:
        yield from _iter_text_file_documents(path)


def iter_indexed_local_documents(split: str = "train", data_dir: str | os.PathLike[str] | None = None) -> Iterator[tuple[int, str]]:
    doc_index = 0
    for path in list_local_data_files(split=split, data_dir=data_dir):
        for document in iter_documents_from_path(path):
            yield doc_index, document
            doc_index += 1


def iter_local_documents(split: str = "train", data_dir: str | os.PathLike[str] | None = None) -> Iterator[str]:
    for _, document in iter_indexed_local_documents(split=split, data_dir=data_dir):
        yield document


def parquets_iter_batched(
    split: str,
    start: int = 0,
    step: int = 1,
    data_dir: str | os.PathLike[str] | None = None,
    batch_size: int = 128,
):
    """
    Backward-compatible name used by tokenizer scripts.

    The iterator now reads batches of local text documents instead of remote parquet row groups.
    """
    assert split in {"train", "val"}, "split must be 'train' or 'val'"
    batch: list[str] = []
    for doc_index, document in iter_indexed_local_documents(split=split, data_dir=data_dir):
        if doc_index < start:
            continue
        if (doc_index - start) % step != 0:
            continue
        batch.append(document)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def corpus_summary(data_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    corpus_dir = get_local_corpus_dir(data_dir)
    splits = {}
    for split in ("train", "val"):
        files = list_local_data_files(split=split, data_dir=corpus_dir)
        split_dir = _resolve_split_dir(corpus_dir, split)
        splits[split] = {
            "path": str(split_dir),
            "file_count": len(files),
            "sample_files": [str(path) for path in files[:5]],
        }
    return {
        "corpus_dir": str(corpus_dir),
        "exists": corpus_dir.exists(),
        "splits": splits,
    }


def inspect_local_corpus(
    split: str = "train",
    data_dir: str | os.PathLike[str] | None = None,
    show_docs: int = 3,
    max_chars: int = 240,
    long_doc_chars: int = 8000,
) -> dict[str, Any]:
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'.")
    if show_docs < 0:
        raise ValueError("show_docs must be 0 or greater.")
    if max_chars < 1:
        raise ValueError("max_chars must be 1 or greater.")
    if long_doc_chars < 1:
        raise ValueError("long_doc_chars must be 1 or greater.")

    corpus_dir = get_local_corpus_dir(data_dir)
    summary = corpus_summary(data_dir=corpus_dir)
    files = list_local_data_files(split=split, data_dir=corpus_dir)
    documents = []
    document_count = 0
    character_count = 0
    file_type_counts: dict[str, int] = {}
    per_file = []
    seen_hashes: dict[str, int] = {}
    duplicate_count = 0
    very_long_document_count = 0
    very_long_documents = []
    longest_documents = []
    for path in files:
        suffix = path.suffix.lower() or "<none>"
        file_type_counts[suffix] = file_type_counts.get(suffix, 0) + 1
        file_document_count = 0
        file_character_count = 0
        try:
            for document in iter_documents_from_path(path):
                text = str(document or "").strip()
                if not text:
                    continue
                doc_chars = len(text)
                doc_index = document_count
                document_count += 1
                character_count += doc_chars
                file_document_count += 1
                file_character_count += doc_chars
                digest = hashlib.sha1(" ".join(text.split()).encode("utf-8")).hexdigest()
                if digest in seen_hashes:
                    duplicate_count += 1
                else:
                    seen_hashes[digest] = doc_index
                doc_info = {
                    "index": doc_index,
                    "path": str(path),
                    "character_count": doc_chars,
                    "text": text[:max_chars],
                    "truncated": doc_chars > max_chars,
                }
                if len(documents) < show_docs:
                    documents.append(doc_info)
                longest_documents.append(doc_info)
                longest_documents = sorted(longest_documents, key=lambda item: item["character_count"], reverse=True)[:5]
                if doc_chars >= long_doc_chars:
                    very_long_document_count += 1
                    if len(very_long_documents) < 10:
                        very_long_documents.append(doc_info)
        except Exception as exc:
            per_file.append(
                {
                    "path": str(path),
                    "suffix": suffix,
                    "document_count": file_document_count,
                    "character_count": file_character_count,
                    "error": str(exc),
                }
            )
            continue
        per_file.append(
            {
                "path": str(path),
                "suffix": suffix,
                "document_count": file_document_count,
                "character_count": file_character_count,
                "error": "",
            }
        )
    split_summary = summary["splits"][split]
    estimated_tokens = max(0, int((character_count + 3) // 4))
    return {
        "corpus_dir": str(corpus_dir),
        "split": split,
        "split_path": split_summary["path"],
        "exists": summary["exists"],
        "file_count": split_summary["file_count"],
        "sample_files": split_summary["sample_files"],
        "document_count": document_count,
        "character_count": character_count,
        "estimated_tokens": estimated_tokens,
        "average_document_chars": round(character_count / document_count, 2) if document_count else 0,
        "max_document_chars": longest_documents[0]["character_count"] if longest_documents else 0,
        "duplicate_document_count": duplicate_count,
        "unique_document_count": document_count - duplicate_count,
        "file_type_counts": file_type_counts,
        "per_file": per_file,
        "long_doc_chars": long_doc_chars,
        "very_long_document_count": very_long_document_count,
        "very_long_documents": very_long_documents,
        "longest_documents": longest_documents,
        "shown_document_count": len(documents),
        "max_chars": max_chars,
        "documents": documents,
        "summary": summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the local training corpus")
    parser.add_argument("--data-dir", type=str, default="", help="Local corpus directory (default: local_corpus or NANOCHAT_LOCAL_CORPUS_DIR)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Split to inspect")
    parser.add_argument("--show-docs", type=int, default=3, help="Show the first N extracted documents")
    args = parser.parse_args()

    summary = corpus_summary(data_dir=args.data_dir or None)
    print(json.dumps(summary, indent=2))
    print()
    for index, (_, document) in enumerate(iter_indexed_local_documents(split=args.split, data_dir=args.data_dir or None)):
        print(f"[doc {index}] {document[:240]}")
        if index + 1 >= args.show_docs:
            break
