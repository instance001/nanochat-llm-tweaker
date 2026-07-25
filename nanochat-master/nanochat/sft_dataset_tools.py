"""
Helpers for drafting local SFT JSONL conversation files.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None

PARQUET_INSTALL_HINT = "Install parquet support with `uv sync --extra parquet` from nanochat-master."


def _normalize_message(message: dict[str, Any], index: int) -> dict[str, str]:
    if not isinstance(message, dict):
        raise ValueError(f"Conversation message {index} must be an object.")
    role = str(message.get("role", "")).strip()
    content = str(message.get("content", "")).strip()
    if role not in {"user", "assistant"}:
        raise ValueError(f"Conversation message {index} has invalid role: {role}")
    if not content:
        raise ValueError(f"Conversation message {index} is missing content.")
    return {"role": role, "content": content}


def normalize_conversation(conversation: Any) -> list[dict[str, str]]:
    if isinstance(conversation, dict) and "messages" in conversation:
        conversation = conversation["messages"]
    if not isinstance(conversation, list):
        raise ValueError("A conversation must be a list of messages.")
    if len(conversation) < 2:
        raise ValueError("A conversation must have at least two messages.")

    normalized = [_normalize_message(message, index) for index, message in enumerate(conversation)]
    for index, message in enumerate(normalized):
        expected_role = "user" if index % 2 == 0 else "assistant"
        if message["role"] != expected_role:
            raise ValueError(
                f"Conversation message {index} has role {message['role']} but should be {expected_role}."
            )
    return normalized


def normalize_conversations(payload: dict[str, Any]) -> list[list[dict[str, str]]]:
    if "conversations" in payload:
        raw = payload["conversations"]
        if not isinstance(raw, list):
            raise ValueError("'conversations' must be a list.")
        return [normalize_conversation(conversation) for conversation in raw]

    if "conversation" in payload:
        return [normalize_conversation(payload["conversation"])]

    if "pairs" in payload:
        raw_pairs = payload["pairs"]
        if not isinstance(raw_pairs, list):
            raise ValueError("'pairs' must be a list.")
        conversations = []
        for index, pair in enumerate(raw_pairs):
            if not isinstance(pair, dict):
                raise ValueError(f"Pair {index} must be an object.")
            user = str(pair.get("user", "")).strip()
            assistant = str(pair.get("assistant", "")).strip()
            if not user or not assistant:
                raise ValueError(f"Pair {index} must include both 'user' and 'assistant'.")
            conversations.append(
                [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
            )
        return conversations

    raise ValueError("No SFT conversation data provided. Use 'conversation', 'conversations', or 'pairs'.")


def conversations_to_jsonl(conversations: list[list[dict[str, str]]]) -> str:
    return "\n".join(json.dumps(conversation, ensure_ascii=True) for conversation in conversations) + "\n"


def split_conversations(
    conversations: list[list[dict[str, str]]],
    val_ratio: float = 0.1,
    seed: int = 1337,
) -> dict[str, Any]:
    if not conversations:
        raise ValueError("No SFT conversations were provided for splitting.")
    if val_ratio <= 0 or val_ratio >= 1:
        raise ValueError("val_ratio must be greater than 0 and less than 1.")

    normalized = [normalize_conversation(conversation) for conversation in conversations]
    indexed = list(enumerate(normalized))
    rng = random.Random(seed)
    rng.shuffle(indexed)
    if len(indexed) == 1:
        val_count = 0
    else:
        val_count = max(1, round(len(indexed) * val_ratio))
        val_count = min(val_count, len(indexed) - 1)
    val_items = indexed[:val_count]
    train_items = indexed[val_count:]

    return {
        "train": [conversation for _, conversation in train_items],
        "val": [conversation for _, conversation in val_items],
        "train_indices": [index for index, _ in train_items],
        "val_indices": [index for index, _ in val_items],
        "conversation_count": len(normalized),
        "train_count": len(train_items),
        "val_count": len(val_items),
        "val_ratio": val_ratio,
        "seed": seed,
    }


def conversations_from_jsonl(content: str) -> list[list[dict[str, str]]]:
    conversations = []
    for line_number, raw_line in enumerate(str(content or "").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {line_number}: {exc}") from exc
        try:
            conversations.append(normalize_conversation(payload))
        except ValueError as exc:
            raise ValueError(f"Line {line_number}: {exc}") from exc
    if not conversations:
        raise ValueError("No valid SFT conversations were found.")
    return conversations


def conversations_to_parquet_records(conversations: list[list[dict[str, str]]], source: str = "sft_jsonl") -> list[dict[str, Any]]:
    records = []
    for index, conversation in enumerate(conversations):
        normalized = normalize_conversation(conversation)
        user_turns = sum(1 for message in normalized if message["role"] == "user")
        assistant_turns = sum(1 for message in normalized if message["role"] == "assistant")
        character_count = sum(len(message["content"]) for message in normalized)
        records.append(
            {
                "messages": json.dumps(normalized, ensure_ascii=True),
                "turn_count": len(normalized),
                "user_turn_count": user_turns,
                "assistant_turn_count": assistant_turns,
                "character_count": character_count,
                "source": source,
                "conversation_index": index,
            }
        )
    return records


def preview_sft_parquet_export(content: str, source: str = "sft_jsonl") -> dict[str, Any]:
    conversations = conversations_from_jsonl(content)
    records = conversations_to_parquet_records(conversations, source=source)
    columns = sorted({key for record in records for key in record.keys()})
    return {
        "format": "parquet",
        "conversation_count": len(conversations),
        "row_count": len(records),
        "column_count": len(columns),
        "columns": columns,
        "sample_rows": records[:5],
    }


def preview_sft_jsonl_export(content: str) -> dict[str, Any]:
    conversations = conversations_from_jsonl(content)
    sample_jsonl = conversations_to_jsonl(conversations[:5])
    return {
        "format": "jsonl",
        "conversation_count": len(conversations),
        "row_count": len(conversations),
        "sample_jsonl": sample_jsonl,
        "sample_conversations": conversations[:5],
    }


def preview_sft_split_export(content: str, val_ratio: float = 0.1, seed: int = 1337) -> dict[str, Any]:
    conversations = conversations_from_jsonl(content)
    split = split_conversations(conversations, val_ratio=val_ratio, seed=seed)
    return {
        "format": "split_jsonl",
        "conversation_count": split["conversation_count"],
        "row_count": split["conversation_count"],
        "train_count": split["train_count"],
        "val_count": split["val_count"],
        "train_indices": split["train_indices"],
        "val_indices": split["val_indices"],
        "val_ratio": split["val_ratio"],
        "seed": split["seed"],
        "sample_train_jsonl": conversations_to_jsonl(split["train"][:5]),
        "sample_val_jsonl": conversations_to_jsonl(split["val"][:5]) if split["val"] else "",
    }


def write_sft_parquet_file(path: str | Path, content: str, source: str = "sft_jsonl") -> dict[str, Any]:
    if pa is None or pq is None:
        raise RuntimeError(f"SFT parquet export requires pyarrow, but it is not installed locally. {PARQUET_INSTALL_HINT}")
    target = Path(path)
    if target.suffix.lower() != ".parquet":
        raise ValueError("SFT parquet export targets must end with .parquet.")
    target.parent.mkdir(parents=True, exist_ok=True)
    conversations = conversations_from_jsonl(content)
    records = conversations_to_parquet_records(conversations, source=source)
    columns = sorted({key for record in records for key in record.keys()})
    table = pa.Table.from_pylist(records)
    pq.write_table(table, target)
    return {
        "path": str(target),
        "format": "parquet",
        "size": target.stat().st_size,
        "updated_at": target.stat().st_mtime,
        "conversation_count": len(conversations),
        "row_count": len(records),
        "column_count": len(columns),
        "columns": columns,
        "sample_rows": records[:5],
    }


def merge_jsonl(existing_text: str, new_jsonl: str, mode: str = "append") -> str:
    normalized_mode = mode.strip().lower() if mode else "append"
    if normalized_mode not in {"append", "overwrite"}:
        raise ValueError("mode must be 'append' or 'overwrite'.")
    if normalized_mode == "overwrite" or not existing_text.strip():
        return new_jsonl
    if existing_text.endswith("\n"):
        return existing_text + new_jsonl
    return existing_text + "\n" + new_jsonl


def sft_schema_payload() -> dict[str, Any]:
    example = [
        {"role": "user", "content": "What should you do when you are uncertain?"},
        {"role": "assistant", "content": "Say that clearly, separate facts from guesses, and suggest a way to verify the answer."},
    ]
    return {
        "format": "JSONL, one conversation per line",
        "requirements": [
            "Each line must be a JSON array of message objects.",
            "Roles must alternate user, assistant, user, assistant.",
            "Each conversation must start with a user message.",
            "Each message must contain string fields named role and content.",
        ],
        "example_conversation": example,
        "example_jsonl_line": json.dumps(example, ensure_ascii=True),
    }
