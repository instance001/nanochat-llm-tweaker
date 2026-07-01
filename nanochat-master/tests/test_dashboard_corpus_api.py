from __future__ import annotations

import importlib
import sys

import pytest
from fastapi.testclient import TestClient

from nanochat.sandbox_tools import pa, pq


@pytest.fixture
def dashboard_client(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["chat_web_test"])
    chat_web = importlib.import_module("scripts.chat_web")
    chat_web = importlib.reload(chat_web)
    monkeypatch.setattr(chat_web, "REPO_ROOT", tmp_path)
    chat_web.args.runtime_autostart = 0
    chat_web.args.port = 0
    chat_web.args.host = "127.0.0.1"
    with TestClient(chat_web.app) as client:
        yield client


def test_corpus_text_file_round_trip_via_api(dashboard_client):
    write_response = dashboard_client.post(
        "/api/corpus/write",
        json={"path": "train/reference.txt", "content": "alpha\nbeta"},
    )
    assert write_response.status_code == 200
    written = write_response.json()
    assert written["path"] == "train/reference.txt"
    assert written["kind"] == "text"
    assert written["editable_as_text"] is True

    read_response = dashboard_client.get("/api/corpus/file", params={"path": "train/reference.txt"})
    assert read_response.status_code == 200
    loaded = read_response.json()
    assert loaded["content"] == "alpha\nbeta"
    assert loaded["kind"] == "text"

    files_response = dashboard_client.get("/api/corpus/files")
    assert files_response.status_code == 200
    assert files_response.json()["files"][0]["path"] == "train/reference.txt"

    delete_response = dashboard_client.post("/api/corpus/delete", json={"path": "train/reference.txt"})
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] == "train/reference.txt"


@pytest.mark.skipif(pa is None or pq is None, reason="pyarrow is required for parquet corpus API tests")
def test_corpus_parquet_file_round_trip_via_api(dashboard_client):
    records = [
        {"text": "first row", "source": "api"},
        {"text": "second row", "source": "api"},
    ]

    write_response = dashboard_client.post(
        "/api/corpus/write",
        json={"path": "train/reference.parquet", "records": records, "mode": "overwrite"},
    )
    assert write_response.status_code == 200
    written = write_response.json()
    assert written["path"] == "train/reference.parquet"
    assert written["kind"] == "parquet"
    assert written["row_count"] == 2
    assert written["sample_rows"] == records
    assert written["editable_as_text"] is False

    read_response = dashboard_client.get("/api/corpus/file", params={"path": "train/reference.parquet"})
    assert read_response.status_code == 200
    loaded = read_response.json()
    assert loaded["row_count"] == 2
    assert loaded["sample_rows"] == records
    assert loaded["columns"] == ["text", "source"]

    files_response = dashboard_client.get("/api/corpus/files")
    assert files_response.status_code == 200
    listed = files_response.json()["files"][0]
    assert listed["path"] == "train/reference.parquet"
    assert listed["kind"] == "parquet"
    assert listed["row_count"] == 2

    delete_response = dashboard_client.post("/api/corpus/delete", json={"path": "train/reference.parquet"})
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] == "train/reference.parquet"


def test_copy_sandbox_text_file_to_corpus_via_api(dashboard_client):
    sandbox_write = dashboard_client.post(
        "/api/sandbox/write",
        json={"path": "notes/reference.txt", "content": "copied from sandbox"},
    )
    assert sandbox_write.status_code == 200

    copy_response = dashboard_client.post(
        "/api/corpus/copy-from-sandbox",
        json={"source_path": "notes/reference.txt", "target_path": "train/copied.txt"},
    )
    assert copy_response.status_code == 200
    copied = copy_response.json()
    assert copied["source_path"] == "notes/reference.txt"
    assert copied["target_path"] == "train/copied.txt"

    read_response = dashboard_client.get("/api/corpus/file", params={"path": "train/copied.txt"})
    assert read_response.status_code == 200
    assert read_response.json()["content"] == "copied from sandbox"


@pytest.mark.skipif(pa is None or pq is None, reason="pyarrow is required for parquet corpus API tests")
def test_copy_sandbox_jsonl_objects_to_parquet_corpus_via_api(dashboard_client):
    sandbox_write = dashboard_client.post(
        "/api/sandbox/write",
        json={
            "path": "notes/reference.jsonl",
            "content": '{"text":"first row","source":"sandbox"}\n{"text":"second row","source":"sandbox"}',
        },
    )
    assert sandbox_write.status_code == 200

    copy_response = dashboard_client.post(
        "/api/corpus/copy-from-sandbox",
        json={"source_path": "notes/reference.jsonl", "target_path": "train/copied.parquet"},
    )
    assert copy_response.status_code == 200
    copied = copy_response.json()
    assert copied["target_path"] == "train/copied.parquet"

    read_response = dashboard_client.get("/api/corpus/file", params={"path": "train/copied.parquet"})
    assert read_response.status_code == 200
    loaded = read_response.json()
    assert loaded["kind"] == "parquet"
    assert loaded["row_count"] == 2
    assert loaded["sample_rows"] == [
        {"text": "first row", "source": "sandbox"},
        {"text": "second row", "source": "sandbox"},
    ]


@pytest.mark.skipif(pa is None or pq is None, reason="pyarrow is required for parquet corpus API tests")
def test_copy_sandbox_non_object_jsonl_to_parquet_returns_clear_error(dashboard_client):
    sandbox_write = dashboard_client.post(
        "/api/sandbox/write",
        json={
            "path": "notes/chat_train.jsonl",
            "content": '[{"role":"user","content":"hi"}]\n[{"role":"assistant","content":"hello"}]',
        },
    )
    assert sandbox_write.status_code == 200

    copy_response = dashboard_client.post(
        "/api/corpus/copy-from-sandbox",
        json={"source_path": "notes/chat_train.jsonl", "target_path": "train/copied.parquet"},
    )
    assert copy_response.status_code == 400
    assert "JSON object" in copy_response.json()["detail"]
