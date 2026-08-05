from __future__ import annotations

import json

import pytest

from nanochat import builder
from nanochat.sandbox_tools import pa, pq


def _last_json(stdout: str) -> dict:
    return json.loads(stdout)


def test_builder_cli_status_prints_json(tmp_path, capsys):
    exit_code = builder.main(["status", "--corpus-dir", str(tmp_path / "corpus"), "--sandbox-dir", str(tmp_path / "sandbox")])

    assert exit_code == 0
    payload = _last_json(capsys.readouterr().out)
    assert "repo_root" in payload
    assert payload["corpus"]["corpus_dir"] == str((tmp_path / "corpus").resolve())


def test_builder_cli_corpus_validate_returns_nonzero_for_empty_corpus(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    exit_code = builder.main(["corpus", "validate", str(corpus)])

    assert exit_code == 1
    payload = _last_json(capsys.readouterr().out)
    assert payload["ok"] is False


def test_builder_cli_sft_validate_returns_zero_for_valid_file(tmp_path, capsys):
    sft_file = tmp_path / "chat_train.jsonl"
    sft_file.write_text('[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]\n', encoding="utf-8")

    exit_code = builder.main(["sft", "validate", str(sft_file)])

    assert exit_code == 0
    payload = _last_json(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["record_count"] == 1


def test_builder_cli_job_preflight_returns_nonzero_for_missing_corpus(tmp_path, capsys):
    exit_code = builder.main(
        [
            "job",
            "preflight",
            "--job-type",
            "tokenizer_train",
            "--corpus-dir",
            str(tmp_path / "missing_corpus"),
            "--base-dir",
            str(tmp_path / "base"),
            "--sandbox-dir",
            str(tmp_path / "sandbox"),
        ]
    )

    assert exit_code == 1
    payload = _last_json(capsys.readouterr().out)
    assert payload["ok"] is False
    assert any(check["code"] == "corpus_empty" for check in payload["checks"])


def test_builder_cli_job_preview_returns_command_for_valid_tokenizer_job(tmp_path, capsys):
    corpus = tmp_path / "corpus with space"
    train = corpus / "train"
    train.mkdir(parents=True)
    (train / "reference.txt").write_text("alpha", encoding="utf-8")

    exit_code = builder.main(
        [
            "job",
            "preview",
            "--job-type",
            "tokenizer_train",
            "--params-json",
            json.dumps({"vocab_size": 16384, "corpus_dir": str(corpus)}),
            "--corpus-dir",
            str(corpus),
            "--base-dir",
            str(tmp_path / "base"),
            "--sandbox-dir",
            str(tmp_path / "sandbox"),
        ]
    )

    assert exit_code == 0
    payload = _last_json(capsys.readouterr().out)
    assert payload["command"][1:3] == ["-m", "scripts.tok_train"]
    assert "--vocab-size" in payload["command"]
    assert f'"{corpus}"' in payload["display_command"]
    assert payload["preflight"]["ok"] is True


def test_builder_cli_job_preview_reads_params_file(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    train = corpus / "train"
    train.mkdir(parents=True)
    (train / "reference.txt").write_text("alpha", encoding="utf-8")
    params_file = tmp_path / "params.json"
    params_file.write_text('{"max_corpus_docs": 4}', encoding="utf-8")

    exit_code = builder.main(
        [
            "job",
            "preview",
            "--job-type",
            "tokenizer_eval",
            "--params-file",
            str(params_file),
            "--corpus-dir",
            str(corpus),
        ]
    )

    assert exit_code == 0
    payload = _last_json(capsys.readouterr().out)
    assert "--max-corpus-docs" in payload["command"]
    assert "4" in payload["command"]


def test_builder_cli_runtime_list_models_finds_gguf(tmp_path, capsys):
    model_dir = tmp_path / "assistant_models"
    model_dir.mkdir()
    model = model_dir / "demo.gguf"
    model.write_bytes(b"demo")

    exit_code = builder.main(["runtime", "list-models", "--repo-root", str(tmp_path)])

    assert exit_code == 0
    payload = _last_json(capsys.readouterr().out)
    assert payload["repo_root"] == str(tmp_path.resolve())
    assert payload["models"][0]["name"] == "demo.gguf"


def test_builder_cli_runtime_recommend_prints_bundle_status(tmp_path, capsys):
    runtime_dir = tmp_path / "runtime" / "windows"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "llama-server.exe").write_bytes(b"demo")
    model_dir = tmp_path / "assistant_models"
    model_dir.mkdir()
    (model_dir / "demo.gguf").write_bytes(b"demo")

    exit_code = builder.main(["runtime", "recommend", "--repo-root", str(tmp_path)])

    assert exit_code == 0
    payload = _last_json(capsys.readouterr().out)
    assert payload["bundle"]["files"]["server_exists"] is True
    assert payload["bundle"]["recommended_model"]["name"] == "demo.gguf"
    assert payload["recommended_config"]["model_path"].endswith("demo.gguf")


def test_builder_cli_corpus_import_hf_preview_uses_streaming_helper(monkeypatch, capsys):
    def fake_preview(**kwargs):
        assert kwargs["dataset_id"] == "demo/corpus"
        assert kwargs["limit_docs"] == 5
        return {"source_type": "huggingface", "dataset_id": "demo/corpus", "row_count": 2}

    monkeypatch.setattr(builder, "preview_huggingface_corpus_records", fake_preview)

    exit_code = builder.main(["corpus", "import-hf", "--dataset", "demo/corpus", "--limit-docs", "5", "--preview"])

    assert exit_code == 0
    payload = _last_json(capsys.readouterr().out)
    assert payload["row_count"] == 2


def test_builder_cli_corpus_import_hf_writes_jsonl(monkeypatch, tmp_path, capsys):
    def fake_collect(**kwargs):
        return {
            "source_type": "huggingface",
            "dataset_id": kwargs["dataset_id"],
            "split": kwargs["split"],
            "row_count": 1,
            "character_count": 5,
            "limit_docs": 1,
            "max_chars": 0,
            "skipped_empty_rows": 0,
            "stopped_reason": "limit_docs",
            "source_columns": ["text"],
            "records": [{"text": "alpha", "source": "hf:demo/corpus:train", "row_index": 0}],
        }

    monkeypatch.setattr(builder, "collect_huggingface_corpus_records", fake_collect)
    corpus = tmp_path / "corpus"

    exit_code = builder.main(
        [
            "corpus",
            "import-hf",
            "--dataset",
            "demo/corpus",
            "--corpus-dir",
            str(corpus),
            "--output-format",
            "jsonl",
        ]
    )

    assert exit_code == 0
    payload = _last_json(capsys.readouterr().out)
    assert payload["manifest_path"] == "corpus_import_manifest.json"
    assert payload["import"]["row_count"] == 1
    assert payload["shard_paths"] == ["train/corpus_train_00001.jsonl"]
    assert json.loads((corpus / "train" / "corpus_train_00001.jsonl").read_text(encoding="utf-8").splitlines()[0])["text"] == "alpha"
    manifest = json.loads((corpus / "corpus_import_manifest.json").read_text(encoding="utf-8"))
    assert manifest["actual_documents"] == 1
    assert manifest["shard_paths"] == ["train/corpus_train_00001.jsonl"]


@pytest.mark.skipif(pa is None or pq is None, reason="pyarrow is required for parquet corpus CLI tests")
def test_builder_cli_corpus_convert_writes_parquet(tmp_path, capsys):
    input_file = tmp_path / "notes.md"
    input_file.write_text("# Intro\nAlpha\n\n## Next\nBeta", encoding="utf-8")
    corpus = tmp_path / "corpus"

    exit_code = builder.main(
        [
            "corpus",
            "convert",
            "--input",
            str(input_file),
            "--output",
            "train/notes.parquet",
            "--corpus-dir",
            str(corpus),
            "--mode",
            "markdown_sections",
        ]
    )

    assert exit_code == 0
    payload = _last_json(capsys.readouterr().out)
    assert payload["path"] == "train/notes.parquet"
    assert payload["row_count"] == 2
    assert (corpus / "train" / "notes.parquet").exists()
