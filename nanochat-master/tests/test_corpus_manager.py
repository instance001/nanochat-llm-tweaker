from __future__ import annotations

import json

import pytest

from nanochat.sandbox_tools import CorpusManager, pa, pq


def test_corpus_manager_reads_text_files(tmp_path):
    manager = CorpusManager(tmp_path)
    written = manager.write_file("train/reference.txt", "alpha\nbeta")

    assert written["path"] == "train/reference.txt"
    assert written["kind"] == "text"

    loaded = manager.read_file("train/reference.txt")
    assert loaded["content"] == "alpha\nbeta"
    assert loaded["editable_as_text"] is True


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
