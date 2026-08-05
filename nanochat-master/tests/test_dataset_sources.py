from __future__ import annotations

import pytest

from nanochat.dataset_sources import (
    build_import_manifest,
    collect_huggingface_corpus_records,
    preview_huggingface_corpus_records,
    recommend_huggingface_corpus_sources,
    shard_import_records,
    split_records_for_import,
)


def fake_loader(dataset_id, *, config="", revision="", split="train", streaming=True, trust_remote_code=False):
    assert dataset_id == "demo/corpus"
    assert config == "plain"
    assert revision == ""
    assert split == "train"
    assert streaming is True
    assert trust_remote_code is False
    return iter(
        [
            {"text": "alpha", "meta": "one"},
            {"text": "", "meta": "empty"},
            {"text": "beta", "meta": "two"},
            {"body": "fallback body", "meta": "three"},
        ]
    )


def test_collect_huggingface_corpus_records_from_fake_stream():
    payload = collect_huggingface_corpus_records(
        "demo/corpus",
        config="plain",
        split="train",
        text_column="text",
        limit_docs=3,
        loader=fake_loader,
    )

    assert payload["source_type"] == "huggingface"
    assert payload["dataset_id"] == "demo/corpus"
    assert payload["dataset_revision"] == ""
    assert payload["row_count"] == 3
    assert payload["skipped_empty_rows"] == 1
    assert payload["records"][0]["text"] == "alpha"
    assert payload["records"][1]["text"] == "beta"
    assert payload["records"][2]["text"] == "fallback body"
    assert payload["records"][2]["source_row_index"] == 3
    assert payload["source_columns"] == ["meta", "text"]


def test_preview_huggingface_corpus_records_omits_full_records():
    preview = preview_huggingface_corpus_records(
        "demo/corpus",
        config="plain",
        split="train",
        limit_docs=2,
        sample_size=1,
        loader=fake_loader,
    )

    assert "records" not in preview
    assert preview["row_count"] == 2
    assert len(preview["sample_rows"]) == 1


def test_huggingface_corpus_import_requires_bounded_limit():
    with pytest.raises(ValueError, match="limit_docs"):
        collect_huggingface_corpus_records("demo/corpus", limit_docs=0, loader=fake_loader)


def test_import_records_split_and_shard_deterministically():
    records = [{"text": f"doc {index}", "row_index": index} for index in range(5)]

    split = split_records_for_import(records, train_val_ratio=0.4)
    shards = shard_import_records(
        split,
        dataset_id="HuggingFaceFW/fineweb",
        source_split="train",
        shard_size_docs=2,
        output_format="parquet",
    )

    assert [shard["path"] for shard in shards] == [
        "train/fineweb_train_00001.parquet",
        "train/fineweb_train_00002.parquet",
        "val/fineweb_val_00001.parquet",
    ]
    assert [shard["row_count"] for shard in shards] == [2, 1, 2]


def test_build_import_manifest_records_receipt_fields():
    payload = {
        "source_type": "huggingface",
        "dataset_id": "HuggingFaceFW/fineweb",
        "dataset_revision": "abc123",
        "config": "sample-10BT",
        "split": "train",
        "text_column": "text",
        "max_chars": 1000,
        "skipped_empty_rows": 2,
        "stopped_reason": "limit_docs",
        "source_columns": ["text"],
        "limit_docs": 10,
        "row_count": 8,
        "character_count": 400,
    }
    manifest = build_import_manifest(
        payload,
        [{"path": "train/fineweb_train_00001.parquet", "sha256": "deadbeef"}],
        train_val_ratio=0.1,
        shard_size_docs=10000,
        output_format="parquet",
        license_note="Check dataset card.",
    )

    assert manifest["dataset_id"] == "HuggingFaceFW/fineweb"
    assert manifest["dataset_revision"] == "abc123"
    assert manifest["source_split"] == "train"
    assert manifest["requested_limit"] == 10
    assert manifest["actual_documents"] == 8
    assert manifest["actual_characters"] == 400
    assert manifest["train_val_ratio"] == 0.1
    assert manifest["shard_paths"] == ["train/fineweb_train_00001.parquet"]
    assert manifest["content_hashes"]["train/fineweb_train_00001.parquet"] == "deadbeef"
    assert manifest["license_note"] == "Check dataset card."
    assert manifest["tool_version"]


def test_recommend_huggingface_sources_from_goal():
    payload = recommend_huggingface_corpus_sources("Let's build a model that explains Python errors to beginners")

    assert payload["recommendations"]
    dataset_ids = [item["dataset_id"] for item in payload["recommendations"]]
    assert "bigcode/the-stack-dedup" in dataset_ids
    assert payload["recommendations"][0]["suggested_train_val_ratio"] == 0.1
    assert "preview_hf_corpus_import" in payload["notes"][0]
