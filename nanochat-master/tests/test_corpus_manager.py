from __future__ import annotations

import json

import pytest

from nanochat.dataset import inspect_local_corpus
from nanochat.sandbox_tools import CorpusManager, SandboxManager, pa, pq


def test_corpus_manager_reads_text_files(tmp_path):
    manager = CorpusManager(tmp_path)
    written = manager.write_file("train/reference.txt", "alpha\nbeta")

    assert written["path"] == "train/reference.txt"
    assert written["kind"] == "text"

    loaded = manager.read_file("train/reference.txt")
    assert loaded["content"] == "alpha\nbeta"
    assert loaded["editable_as_text"] is True


def test_inspect_local_corpus_matches_training_document_extraction(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "records.jsonl").write_text(
        '{"prompt":"first question","response":"first answer"}\n'
        '{"messages":[{"role":"user","content":"second question"},{"role":"assistant","content":"second answer"}]}\n',
        encoding="utf-8",
    )

    report = inspect_local_corpus(split="train", data_dir=tmp_path, show_docs=3, max_chars=80)

    assert report["file_count"] == 1
    assert report["document_count"] == 3
    assert report["shown_document_count"] == 3
    assert report["documents"][0]["text"] == "first question"
    assert report["documents"][1]["text"] == "first answer"
    assert report["documents"][2]["text"] == "second question\nsecond answer"


def test_inspect_local_corpus_reports_statistics(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "a.txt").write_text("same document", encoding="utf-8")
    (train_dir / "b.txt").write_text("same document", encoding="utf-8")
    (train_dir / "c.jsonl").write_text('{"text":"' + ("x" * 30) + '"}\n', encoding="utf-8")

    report = inspect_local_corpus(split="train", data_dir=tmp_path, show_docs=2, max_chars=10, long_doc_chars=20)

    assert report["document_count"] == 3
    assert report["duplicate_document_count"] == 1
    assert report["unique_document_count"] == 2
    assert report["file_type_counts"] == {".jsonl": 1, ".txt": 2}
    assert report["very_long_document_count"] == 1
    assert report["estimated_tokens"] > 0
    assert report["documents"][0]["truncated"] is True


@pytest.mark.skipif(pa is None or pq is None, reason="pyarrow is required for parquet sandbox tests")
def test_sandbox_manager_previews_parquet_files(tmp_path):
    manager = SandboxManager(tmp_path)
    target = tmp_path / "exports" / "chat_train.parquet"
    target.parent.mkdir(parents=True)
    table = pa.Table.from_pylist([{"messages": "[]", "turn_count": 0}])
    pq.write_table(table, target)

    listed = manager.list_files()
    loaded = manager.read_file("exports/chat_train.parquet")

    assert listed[0]["kind"] == "parquet"
    assert listed[0]["row_count"] == 1
    assert loaded["editable_as_text"] is False
    assert loaded["row_count"] == 1
    assert loaded["sample_rows"] == [{"messages": "[]", "turn_count": 0}]


@pytest.mark.skipif(pa is None or pq is None, reason="pyarrow is required for parquet corpus tests")
def test_corpus_manager_writes_and_reads_parquet_files(tmp_path):
    manager = CorpusManager(tmp_path)
    records = [
        {"text": "first row", "source": "a"},
        {"text": "second row", "source": "b"},
    ]

    written = manager.write_parquet_file("train/reference.parquet", records)

    assert written["path"] == "train/reference.parquet"
    assert written["kind"] == "parquet"
    assert written["row_count"] == 2
    assert written["editable_as_text"] is False
    assert written["sample_rows"] == records
    assert "text" in written["columns"]

    loaded = manager.read_file("train/reference.parquet")
    assert loaded["row_count"] == 2
    assert loaded["sample_rows"] == records
    assert json.loads(loaded["content"]) == records


@pytest.mark.skipif(pa is None or pq is None, reason="pyarrow is required for parquet corpus tests")
def test_corpus_manager_appends_parquet_records(tmp_path):
    manager = CorpusManager(tmp_path)
    manager.write_parquet_file("train/reference.parquet", [{"text": "first"}], mode="overwrite")

    loaded = manager.write_parquet_file("train/reference.parquet", [{"text": "second"}], mode="append")

    assert loaded["row_count"] == 2
    assert loaded["sample_rows"] == [{"text": "first"}, {"text": "second"}]


@pytest.mark.skipif(pa is None or pq is None, reason="pyarrow is required for parquet corpus tests")
def test_corpus_manager_rejects_plain_text_write_for_parquet(tmp_path):
    manager = CorpusManager(tmp_path)

    with pytest.raises(ValueError, match="structured records"):
        manager.write_file("train/reference.parquet", "not parquet")


@pytest.mark.skipif(pa is None or pq is None, reason="pyarrow is required for parquet corpus tests")
def test_corpus_manager_write_from_content_routes_jsonl_to_parquet(tmp_path):
    manager = CorpusManager(tmp_path)
    content = '\n'.join([
        '{"text": "first row", "source": "a"}',
        '{"text": "second row", "source": "b"}',
    ])

    written = manager.write_from_content("train/reference.parquet", content)

    assert written["kind"] == "parquet"
    assert written["row_count"] == 2
    assert written["sample_rows"] == [
        {"text": "first row", "source": "a"},
        {"text": "second row", "source": "b"},
    ]


def test_corpus_manager_converts_paragraphs_to_parquet_records(tmp_path):
    manager = CorpusManager(tmp_path)

    preview = manager.preview_parquet_conversion(
        "first paragraph\n\nsecond paragraph\n\n",
        conversion_mode="paragraphs",
        source="unit-test",
    )

    assert preview["row_count"] == 2
    assert preview["rejected_empty_rows"] == 1
    assert preview["sample_rows"] == [
        {"text": "first paragraph", "source": "unit-test", "row_index": 0},
        {"text": "second paragraph", "source": "unit-test", "row_index": 1},
    ]


def test_corpus_manager_converts_lines_to_parquet_records(tmp_path):
    manager = CorpusManager(tmp_path)

    preview = manager.preview_parquet_conversion("alpha\n\nbeta", conversion_mode="lines")

    assert preview["row_count"] == 2
    assert preview["rejected_empty_rows"] == 1
    assert preview["sample_rows"][0]["text"] == "alpha"
    assert preview["sample_rows"][1]["text"] == "beta"


def test_corpus_manager_splits_paragraphs_deterministically(tmp_path):
    manager = CorpusManager(tmp_path)
    content = "one\n\ntwo\n\nthree\n\nfour"

    first = manager.preview_split_content(content, split_mode="paragraphs", val_ratio=0.25, seed=7)
    second = manager.preview_split_content(content, split_mode="paragraphs", val_ratio=0.25, seed=7)

    assert first["train_count"] == 3
    assert first["val_count"] == 1
    assert first["train_indices"] == second["train_indices"]
    assert first["val_indices"] == second["val_indices"]
    assert sorted(first["train_indices"] + first["val_indices"]) == [0, 1, 2, 3]


def test_corpus_manager_writes_split_files(tmp_path):
    manager = CorpusManager(tmp_path)

    result = manager.write_split_files(
        "train/split.txt",
        "val/split.txt",
        "one\n\ntwo\n\nthree\n\nfour",
        split_mode="paragraphs",
        val_ratio=0.25,
        seed=7,
    )

    assert result["train_path"] == "train/split.txt"
    assert result["val_path"] == "val/split.txt"
    assert (tmp_path / "train" / "split.txt").exists()
    assert (tmp_path / "val" / "split.txt").exists()
    assert len((tmp_path / "train" / "split.txt").read_text(encoding="utf-8").strip().split("\n\n")) == 3
    assert len((tmp_path / "val" / "split.txt").read_text(encoding="utf-8").strip().split("\n\n")) == 1


def test_corpus_manager_converts_markdown_sections_to_parquet_records(tmp_path):
    manager = CorpusManager(tmp_path)

    preview = manager.preview_parquet_conversion(
        "# Intro\nFirst section.\n\n## Details\nSecond section.",
        conversion_mode="markdown_sections",
    )

    assert preview["row_count"] == 2
    assert preview["sample_rows"][0]["heading"] == "Intro"
    assert preview["sample_rows"][0]["text"] == "Intro\n\nFirst section."
    assert preview["sample_rows"][1]["heading"] == "Details"


def test_corpus_manager_converts_json_array_to_parquet_records(tmp_path):
    manager = CorpusManager(tmp_path)

    preview = manager.preview_parquet_conversion(
        '[{"text": "one", "source": "json"}, {"text": "two", "source": "json"}]',
        conversion_mode="json_array",
    )

    assert preview["row_count"] == 2
    assert preview["columns"] == ["source", "text"]


def test_corpus_manager_jsonl_conversion_reports_line_number(tmp_path):
    manager = CorpusManager(tmp_path)

    with pytest.raises(ValueError, match="line 2"):
        manager.preview_parquet_conversion('{"text": "ok"}\nnot-json', conversion_mode="jsonl")


@pytest.mark.skipif(pa is None or pq is None, reason="pyarrow is required for parquet corpus tests")
def test_corpus_manager_writes_converted_parquet_file(tmp_path):
    manager = CorpusManager(tmp_path)

    written = manager.write_converted_parquet_file(
        "train/converted.parquet",
        "alpha\n\nbeta",
        conversion_mode="paragraphs",
    )

    assert written["path"] == "train/converted.parquet"
    assert written["row_count"] == 2
    assert written["conversion"]["conversion_mode"] == "paragraphs"
    assert written["sample_rows"][0]["text"] == "alpha"
