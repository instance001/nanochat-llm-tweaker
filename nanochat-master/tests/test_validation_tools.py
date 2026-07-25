from __future__ import annotations

import json

from nanochat.sft_dataset_tools import conversations_from_jsonl, conversations_to_parquet_records, split_conversations
from nanochat.validation_tools import validate_corpus_path, validate_sft_jsonl_content


def test_validate_sft_jsonl_content_accepts_valid_conversations():
    content = json.dumps(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi."},
        ]
    )

    report = validate_sft_jsonl_content(content, path="chat_train.jsonl")

    assert report["ok"] is True
    assert report["record_count"] == 1
    assert report["role_counts"] == {"user": 1, "assistant": 1}
    assert report["estimated_tokens"] > 0
    assert any("one conversation per line" in hint for hint in report["schema_hints"])


def test_validate_sft_jsonl_content_reports_bad_line_number():
    report = validate_sft_jsonl_content(
        json.dumps(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi."},
            ]
        )
        + "\n"
        + "not json",
    )

    assert report["ok"] is False
    assert report["record_count"] == 1
    assert "Line 2" in report["errors"][0]


def test_validate_sft_jsonl_content_rejects_bad_role_order():
    content = json.dumps(
        [
            {"role": "assistant", "content": "Hi."},
            {"role": "user", "content": "Hello"},
        ]
    )

    report = validate_sft_jsonl_content(content)

    assert report["ok"] is False
    assert "should be user" in report["errors"][0]


def test_sft_jsonl_to_parquet_records_preserves_messages():
    content = json.dumps(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi."},
        ]
    )

    conversations = conversations_from_jsonl(content)
    records = conversations_to_parquet_records(conversations, source="chat_train.jsonl")

    assert len(records) == 1
    assert records[0]["turn_count"] == 2
    assert records[0]["source"] == "chat_train.jsonl"
    assert '"role": "user"' in records[0]["messages"]


def test_split_conversations_is_deterministic_and_preserves_all_rows():
    content = "\n".join(
        [
            '[{"role":"user","content":"u0"},{"role":"assistant","content":"a0"}]',
            '[{"role":"user","content":"u1"},{"role":"assistant","content":"a1"}]',
            '[{"role":"user","content":"u2"},{"role":"assistant","content":"a2"}]',
            '[{"role":"user","content":"u3"},{"role":"assistant","content":"a3"}]',
        ]
    )
    conversations = conversations_from_jsonl(content)

    first = split_conversations(conversations, val_ratio=0.25, seed=7)
    second = split_conversations(conversations, val_ratio=0.25, seed=7)

    assert first["train_count"] == 3
    assert first["val_count"] == 1
    assert first["train_indices"] == second["train_indices"]
    assert first["val_indices"] == second["val_indices"]
    assert sorted(first["train_indices"] + first["val_indices"]) == [0, 1, 2, 3]


def test_validate_corpus_path_counts_text_documents(tmp_path):
    corpus = tmp_path / "local_corpus"
    train_dir = corpus / "train"
    val_dir = corpus / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir()
    (train_dir / "one.txt").write_text("alpha", encoding="utf-8")
    (train_dir / "two.jsonl").write_text('{"text":"beta"}\n{"text":"gamma"}', encoding="utf-8")

    report = validate_corpus_path(corpus)

    assert report["ok"] is True
    assert report["file_count"] == 2
    assert report["record_count"] == 3
    assert report["character_count"] == len("alpha") + len("beta") + len("gamma")
    assert any("plain text/code" in hint for hint in report["schema_hints"])


def test_validate_corpus_path_counts_active_tokenizer_tokens(tmp_path, monkeypatch):
    class FakeTokenizer:
        def encode(self, text):
            return text.split()

    corpus = tmp_path / "local_corpus"
    train_dir = corpus / "train"
    tokenizer_dir = tmp_path / "tokenizer"
    train_dir.mkdir(parents=True)
    tokenizer_dir.mkdir()
    (train_dir / "one.txt").write_text("alpha beta", encoding="utf-8")

    monkeypatch.setattr(
        "nanochat.validation_tools._load_active_tokenizer",
        lambda path: (FakeTokenizer(), {"available": True, "path": str(path), "error": ""}),
    )

    report = validate_corpus_path(corpus, tokenizer_dir=tokenizer_dir)

    assert report["active_tokenizer"]["available"] is True
    assert report["active_token_count"] == 2


def test_validate_corpus_path_reports_unsupported_files(tmp_path):
    corpus = tmp_path / "local_corpus"
    train_dir = corpus / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "one.txt").write_text("alpha", encoding="utf-8")
    (train_dir / "ignored.bin").write_bytes(b"\x00\x01")

    report = validate_corpus_path(corpus)

    assert report["ok"] is True
    assert report["file_count"] == 1
    assert report["record_count"] == 1
    assert report["unsupported_file_count"] == 1
    assert any("ignored.bin" in path for path in report["unsupported_files"])
    assert any("Ignored 1 unsupported" in warning for warning in report["warnings"])


def test_validate_corpus_path_reports_exact_duplicate_documents(tmp_path):
    corpus = tmp_path / "local_corpus"
    train_dir = corpus / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "one.txt").write_text("same text", encoding="utf-8")
    (train_dir / "two.txt").write_text("same   text\n", encoding="utf-8")

    report = validate_corpus_path(corpus)

    assert report["ok"] is True
    assert report["record_count"] == 2
    assert report["duplicate_document_count"] == 1
    assert report["duplicate_documents"][0]["preview"] == "same text"
    assert any("exact duplicate" in warning for warning in report["warnings"])


def test_validate_corpus_path_rejects_unsupported_single_file(tmp_path):
    target = tmp_path / "ignored.bin"
    target.write_bytes(b"\x00\x01")

    report = validate_corpus_path(target)

    assert report["ok"] is False
    assert report["file_count"] == 0
    assert report["unsupported_file_count"] == 1
    assert "No supported corpus files" in report["errors"][0]


def test_validate_corpus_path_warns_when_split_folders_are_missing(tmp_path):
    corpus = tmp_path / "local_corpus"
    corpus.mkdir()
    (corpus / "one.txt").write_text("alpha", encoding="utf-8")

    report = validate_corpus_path(corpus)

    assert report["ok"] is True
    assert any("train/" in warning for warning in report["warnings"])
    assert any("val/" in warning for warning in report["warnings"])


def test_validate_corpus_path_reports_empty_corpus(tmp_path):
    corpus = tmp_path / "empty"
    corpus.mkdir()

    report = validate_corpus_path(corpus)

    assert report["ok"] is False
    assert "No supported corpus files" in report["errors"][0]
