from __future__ import annotations

import json

from nanochat.preflight import run_stage_preflight


def base_context(tmp_path):
    corpus = tmp_path / "local_corpus"
    sandbox = tmp_path / "assistant_sandbox"
    base = tmp_path / "cache"
    corpus.mkdir()
    sandbox.mkdir()
    base.mkdir()
    return {
        "base_dir": base,
        "corpus_dir": corpus,
        "sandbox_dir": sandbox,
        "tokenizer_ready": False,
        "identity_exists": False,
        "checkpoint_sets": {"base": {"tags": []}, "sft": {"tags": []}, "rl": {"tags": []}},
    }


def test_preflight_tokenizer_train_blocks_empty_corpus(tmp_path):
    result = run_stage_preflight("tokenizer_train", {}, **base_context(tmp_path))

    assert result["ok"] is False
    assert any(check["code"] == "corpus_empty" for check in result["checks"])


def test_preflight_base_train_requires_tokenizer(tmp_path):
    context = base_context(tmp_path)
    train_dir = context["corpus_dir"] / "train"
    train_dir.mkdir()
    (train_dir / "reference.txt").write_text("alpha", encoding="utf-8")

    result = run_stage_preflight("base_train", {"save_every": 50}, **context)

    assert result["ok"] is False
    assert any(check["code"] == "tokenizer_missing" for check in result["checks"])
    assert any(check["code"] == "corpus_present" for check in result["checks"])


def test_preflight_chat_sft_accepts_valid_sandbox_jsonl(tmp_path):
    context = base_context(tmp_path)
    context["identity_exists"] = True
    context["checkpoint_sets"] = {"base": {"tags": [{"tag": "d4"}]}, "sft": {"tags": []}, "rl": {"tags": []}}
    (context["sandbox_dir"] / "chat_train.jsonl").write_text(
        json.dumps(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_stage_preflight(
        "chat_sft",
        {"train_files": "chat_train.jsonl", "include_identity": 1, "save_every": 50},
        **context,
    )

    assert result["ok"] is True
    assert any(check["code"] == "sft_training_file_valid" for check in result["checks"])


def test_preflight_chat_sft_rejects_parquet_training_file(tmp_path):
    context = base_context(tmp_path)
    context["checkpoint_sets"] = {"base": {"tags": [{"tag": "d4"}]}, "sft": {"tags": []}, "rl": {"tags": []}}

    result = run_stage_preflight("chat_sft", {"train_files": "train/data.parquet"}, **context)

    assert result["ok"] is False
    assert any(check["code"] == "sft_file_is_parquet" for check in result["checks"])


def test_preflight_chat_rl_requires_sft_checkpoint(tmp_path):
    result = run_stage_preflight("chat_rl", {}, **base_context(tmp_path))

    assert result["ok"] is False
    assert any(check["code"] == "sft_checkpoint_missing" for check in result["checks"])
