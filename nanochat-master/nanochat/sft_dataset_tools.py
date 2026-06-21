"""
Helpers for drafting local SFT JSONL conversation files.
"""

from __future__ import annotations

import json
from typing import Any


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
