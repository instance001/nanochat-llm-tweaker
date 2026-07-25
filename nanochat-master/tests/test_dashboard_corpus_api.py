from __future__ import annotations

import importlib
import json
import sys

import pytest
from fastapi.testclient import TestClient

from nanochat import dashboard_tools
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


def test_sandbox_sft_validate_via_api(dashboard_client):
    content = '[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]\n'
    sandbox_write = dashboard_client.post(
        "/api/sandbox/write",
        json={"path": "chat_train.jsonl", "content": content},
    )
    assert sandbox_write.status_code == 200

    validate_response = dashboard_client.post(
        "/api/sandbox/sft/validate",
        json={"path": "chat_train.jsonl"},
    )

    assert validate_response.status_code == 200
    report = validate_response.json()
    assert report["ok"] is True
    assert report["record_count"] == 1


def test_sandbox_sft_validate_reports_bad_jsonl_via_api(dashboard_client):
    sandbox_write = dashboard_client.post(
        "/api/sandbox/write",
        json={"path": "chat_train.jsonl", "content": "not-json"},
    )
    assert sandbox_write.status_code == 200

    validate_response = dashboard_client.post(
        "/api/sandbox/sft/validate",
        json={"path": "chat_train.jsonl"},
    )

    assert validate_response.status_code == 200
    report = validate_response.json()
    assert report["ok"] is False
    assert "Line 1" in report["errors"][0]


def test_sandbox_sft_export_normalizes_jsonl_via_api(dashboard_client):
    content = '[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]\n'
    sandbox_write = dashboard_client.post(
        "/api/sandbox/write",
        json={"path": "chat_train.jsonl", "content": content},
    )
    assert sandbox_write.status_code == 200

    export_response = dashboard_client.post(
        "/api/sandbox/sft/export",
        json={"source_path": "chat_train.jsonl", "target_path": "exports/chat_train.normalized.jsonl", "format": "jsonl"},
    )

    assert export_response.status_code == 200
    exported = export_response.json()
    assert exported["format"] == "jsonl"
    assert exported["conversation_count"] == 1
    read_response = dashboard_client.get("/api/sandbox/file", params={"path": "exports/chat_train.normalized.jsonl"})
    assert read_response.status_code == 200
    assert json.loads(read_response.json()["content"].splitlines()[0])[0]["role"] == "user"


def test_sandbox_sft_export_previews_jsonl_without_writing_via_api(dashboard_client):
    content = '[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]\n'
    sandbox_write = dashboard_client.post(
        "/api/sandbox/write",
        json={"path": "chat_train.jsonl", "content": content},
    )
    assert sandbox_write.status_code == 200

    preview_response = dashboard_client.post(
        "/api/sandbox/sft/export",
        json={"source_path": "chat_train.jsonl", "target_path": "exports/unused.jsonl", "format": "preview_jsonl"},
    )

    assert preview_response.status_code == 200
    preview = preview_response.json()
    assert preview["format"] == "jsonl"
    assert preview["conversation_count"] == 1
    assert "sample_jsonl" in preview

    read_response = dashboard_client.get("/api/sandbox/file", params={"path": "exports/unused.jsonl"})
    assert read_response.status_code == 404


@pytest.mark.skipif(pa is None or pq is None, reason="pyarrow is required for SFT parquet export tests")
def test_sandbox_sft_export_parquet_via_api(dashboard_client):
    content = '[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]\n'
    sandbox_write = dashboard_client.post(
        "/api/sandbox/write",
        json={"path": "chat_train.jsonl", "content": content},
    )
    assert sandbox_write.status_code == 200

    export_response = dashboard_client.post(
        "/api/sandbox/sft/export",
        json={"source_path": "chat_train.jsonl", "target_path": "exports/chat_train.parquet", "format": "parquet"},
    )

    assert export_response.status_code == 200
    exported = export_response.json()
    assert exported["format"] == "parquet"
    assert exported["row_count"] == 1
    parquet_path = dashboard_client.app.state.sandbox.root / "exports" / "chat_train.parquet"
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    assert rows[0]["turn_count"] == 2
    assert json.loads(rows[0]["messages"])[1]["content"] == "hello"


@pytest.mark.skipif(pa is None or pq is None, reason="pyarrow is required for SFT parquet preview tests")
def test_sandbox_reads_exported_sft_parquet_preview_via_api(dashboard_client):
    content = '[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]\n'
    sandbox_write = dashboard_client.post(
        "/api/sandbox/write",
        json={"path": "chat_train.jsonl", "content": content},
    )
    assert sandbox_write.status_code == 200
    export_response = dashboard_client.post(
        "/api/sandbox/sft/export",
        json={"source_path": "chat_train.jsonl", "target_path": "exports/chat_train.parquet", "format": "parquet"},
    )
    assert export_response.status_code == 200

    read_response = dashboard_client.get("/api/sandbox/file", params={"path": "exports/chat_train.parquet"})
    files_response = dashboard_client.get("/api/sandbox/files")

    assert read_response.status_code == 200
    loaded = read_response.json()
    assert loaded["kind"] == "parquet"
    assert loaded["editable_as_text"] is False
    assert loaded["row_count"] == 1
    assert json.loads(loaded["sample_rows"][0]["messages"])[0]["content"] == "hi"
    listed = [item for item in files_response.json()["files"] if item["path"] == "exports/chat_train.parquet"][0]
    assert listed["kind"] == "parquet"
    assert listed["row_count"] == 1


def test_sandbox_sft_split_jsonl_via_api(dashboard_client):
    content = "\n".join(
        [
            '[{"role":"user","content":"u0"},{"role":"assistant","content":"a0"}]',
            '[{"role":"user","content":"u1"},{"role":"assistant","content":"a1"}]',
            '[{"role":"user","content":"u2"},{"role":"assistant","content":"a2"}]',
            '[{"role":"user","content":"u3"},{"role":"assistant","content":"a3"}]',
        ]
    )
    sandbox_write = dashboard_client.post(
        "/api/sandbox/write",
        json={"path": "chat_train.jsonl", "content": content},
    )
    assert sandbox_write.status_code == 200

    split_response = dashboard_client.post(
        "/api/sandbox/sft/export",
        json={
            "source_path": "chat_train.jsonl",
            "target_path": "exports/chat_train.split.jsonl",
            "val_target_path": "exports/chat_val.split.jsonl",
            "format": "split_jsonl",
            "val_ratio": 0.25,
            "seed": 7,
        },
    )

    assert split_response.status_code == 200
    split = split_response.json()
    assert split["format"] == "split_jsonl"
    assert split["train_count"] == 3
    assert split["val_count"] == 1
    assert sorted(split["train_indices"] + split["val_indices"]) == [0, 1, 2, 3]

    train_response = dashboard_client.get("/api/sandbox/file", params={"path": "exports/chat_train.split.jsonl"})
    val_response = dashboard_client.get("/api/sandbox/file", params={"path": "exports/chat_val.split.jsonl"})

    assert train_response.status_code == 200
    assert val_response.status_code == 200
    assert len(train_response.json()["content"].splitlines()) == 3
    assert len(val_response.json()["content"].splitlines()) == 1


def test_sandbox_sft_split_preview_via_api(dashboard_client):
    content = "\n".join(
        [
            '[{"role":"user","content":"u0"},{"role":"assistant","content":"a0"}]',
            '[{"role":"user","content":"u1"},{"role":"assistant","content":"a1"}]',
            '[{"role":"user","content":"u2"},{"role":"assistant","content":"a2"}]',
            '[{"role":"user","content":"u3"},{"role":"assistant","content":"a3"}]',
        ]
    )
    sandbox_write = dashboard_client.post(
        "/api/sandbox/write",
        json={"path": "chat_train.jsonl", "content": content},
    )
    assert sandbox_write.status_code == 200

    preview_response = dashboard_client.post(
        "/api/sandbox/sft/export",
        json={
            "source_path": "chat_train.jsonl",
            "format": "preview_split_jsonl",
            "val_ratio": 0.25,
            "seed": 7,
        },
    )

    assert preview_response.status_code == 200
    preview = preview_response.json()
    assert preview["format"] == "split_jsonl"
    assert preview["train_count"] == 3
    assert preview["val_count"] == 1
    assert sorted(preview["train_indices"] + preview["val_indices"]) == [0, 1, 2, 3]
    assert "sample_train_jsonl" in preview
    assert "sample_val_jsonl" in preview


def test_chat_transcript_export_json_via_api(dashboard_client):
    export_response = dashboard_client.post(
        "/api/sandbox/chat/export",
        json={
            "path": "transcripts/demo.json",
            "format": "json",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        },
    )

    assert export_response.status_code == 200
    exported = export_response.json()
    assert exported["path"] == "transcripts/demo.json"
    assert exported["message_count"] == 2

    read_response = dashboard_client.get("/api/sandbox/file", params={"path": "transcripts/demo.json"})
    assert read_response.status_code == 200
    payload = json.loads(read_response.json()["content"])
    assert payload["messages"][0]["content"] == "hi"


def test_chat_transcript_export_selected_turns_json_via_api(dashboard_client):
    export_response = dashboard_client.post(
        "/api/sandbox/chat/export",
        json={
            "path": "transcripts/selected.json",
            "format": "json",
            "messages": [
                {"role": "assistant", "content": "selected answer"},
            ],
        },
    )

    assert export_response.status_code == 200
    exported = export_response.json()
    assert exported["message_count"] == 1
    read_response = dashboard_client.get("/api/sandbox/file", params={"path": "transcripts/selected.json"})
    assert read_response.status_code == 200
    payload = json.loads(read_response.json()["content"])
    assert payload["messages"] == [{"role": "assistant", "content": "selected answer"}]


def test_chat_transcript_export_sft_jsonl_via_api(dashboard_client):
    export_response = dashboard_client.post(
        "/api/sandbox/chat/export",
        json={
            "path": "chat_train.jsonl",
            "format": "sft_jsonl",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        },
    )

    assert export_response.status_code == 200
    validate_response = dashboard_client.post("/api/sandbox/sft/validate", json={"path": "chat_train.jsonl"})
    assert validate_response.status_code == 200
    assert validate_response.json()["ok"] is True


def test_chat_transcript_export_sft_preview_does_not_write_via_api(dashboard_client):
    preview_response = dashboard_client.post(
        "/api/sandbox/chat/export",
        json={
            "path": "chat_train.jsonl",
            "format": "sft_jsonl",
            "preview": True,
            "messages": [
                {"role": "user", "content": "preview question"},
                {"role": "assistant", "content": "preview answer"},
            ],
        },
    )

    assert preview_response.status_code == 200
    preview = preview_response.json()
    assert preview["preview"] is True
    assert preview["message_count"] == 2
    assert json.loads(preview["content"].splitlines()[0]) == [
        {"role": "user", "content": "preview question"},
        {"role": "assistant", "content": "preview answer"},
    ]
    read_response = dashboard_client.get("/api/sandbox/file", params={"path": "chat_train.jsonl"})
    assert read_response.status_code == 404


def test_chat_transcript_export_selected_pair_sft_jsonl_via_api(dashboard_client):
    export_response = dashboard_client.post(
        "/api/sandbox/chat/export",
        json={
            "path": "chat_train.jsonl",
            "format": "sft_jsonl",
            "messages": [
                {"role": "user", "content": "selected question"},
                {"role": "assistant", "content": "selected answer"},
            ],
        },
    )

    assert export_response.status_code == 200
    exported = export_response.json()
    assert exported["message_count"] == 2
    read_response = dashboard_client.get("/api/sandbox/file", params={"path": "chat_train.jsonl"})
    assert read_response.status_code == 200
    saved = json.loads(read_response.json()["content"].splitlines()[0])
    assert saved == [
        {"role": "user", "content": "selected question"},
        {"role": "assistant", "content": "selected answer"},
    ]


def test_chat_transcript_export_sft_rejects_bad_role_order(dashboard_client):
    export_response = dashboard_client.post(
        "/api/sandbox/chat/export",
        json={
            "path": "chat_train.jsonl",
            "format": "sft_jsonl",
            "messages": [
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "hi"},
            ],
        },
    )

    assert export_response.status_code == 400
    assert "should be user" in export_response.json()["detail"]


def test_runtime_assist_review_mode_requires_approval_before_write(dashboard_client):
    class FakeRuntime:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, temperature=0.2, max_tokens=512, system_prompt=""):
            self.calls += 1
            if self.calls == 1:
                return {
                    "text": '<assistant_action>{"tool":"write_sandbox_file","args":{"path":"notes/approved.txt","content":"hello from approval"}}'
                    "</assistant_action>",
                    "raw": None,
                }
            return {"text": "Saved after approval.", "raw": None}

        def status(self):
            return {"ready": True, "model": {"path": "fake.gguf"}, "server": {"url": "http://127.0.0.1:0"}}

        def stop(self):
            return {"ready": False}

    dashboard_client.app.state.local_runtime = FakeRuntime()
    response = dashboard_client.post(
        "/api/runtime/assist",
        json={
            "messages": [{"role": "user", "content": "write a note"}],
            "action_mode": "review",
            "max_actions": 2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["approval_required"] is True
    assert payload["pending_action"]["tool"] == "write_sandbox_file"
    assert payload["pending_action"]["preview"]["requires_approval"] is True

    read_before_approval = dashboard_client.get("/api/sandbox/file", params={"path": "notes/approved.txt"})
    assert read_before_approval.status_code == 404

    approval_response = dashboard_client.post(
        "/api/runtime/assist",
        json={
            "messages": [{"role": "user", "content": "write a note"}],
            "action_mode": "review",
            "max_actions": 2,
            "approved_action": {
                "tool": payload["pending_action"]["tool"],
                "args": payload["pending_action"]["args"],
            },
            "approved_action_text": payload["pending_action"]["assistant_text"],
        },
    )

    assert approval_response.status_code == 200
    approved_payload = approval_response.json()
    assert approved_payload["text"] == "Saved after approval."
    assert approved_payload["actions"][0]["status"] == "ok"

    read_after_approval = dashboard_client.get("/api/sandbox/file", params={"path": "notes/approved.txt"})
    assert read_after_approval.status_code == 200
    assert read_after_approval.json()["content"] == "hello from approval"


def test_corpus_validate_via_api(dashboard_client):
    write_response = dashboard_client.post(
        "/api/corpus/write",
        json={"path": "train/reference.txt", "content": "alpha"},
    )
    assert write_response.status_code == 200

    validate_response = dashboard_client.post(
        "/api/corpus/validate",
        json={"path": ""},
    )

    assert validate_response.status_code == 200
    report = validate_response.json()
    assert report["ok"] is True
    assert report["record_count"] == 1
    assert report["file_count"] == 1
    assert any("local_corpus/train" in hint for hint in report["schema_hints"])
    assert "available" in report["active_tokenizer"]
    if report["active_tokenizer"]["available"]:
        assert report["active_token_count"] is not None
    else:
        assert report["active_token_count"] is None


def test_corpus_validate_reports_unsupported_files_via_api(dashboard_client):
    write_response = dashboard_client.post(
        "/api/corpus/write",
        json={"path": "train/reference.txt", "content": "alpha"},
    )
    assert write_response.status_code == 200
    ignored_path = dashboard_client.app.state.corpus.root / "train" / "ignored.bin"
    ignored_path.write_bytes(b"\x00\x01")

    validate_response = dashboard_client.post(
        "/api/corpus/validate",
        json={"path": ""},
    )

    assert validate_response.status_code == 200
    report = validate_response.json()
    assert report["ok"] is True
    assert report["unsupported_file_count"] == 1
    assert any("ignored.bin" in path for path in report["unsupported_files"])


def test_corpus_validate_reports_exact_duplicates_via_api(dashboard_client):
    first = dashboard_client.post(
        "/api/corpus/write",
        json={"path": "train/a.txt", "content": "duplicate text"},
    )
    second = dashboard_client.post(
        "/api/corpus/write",
        json={"path": "train/b.txt", "content": "duplicate   text\n"},
    )
    assert first.status_code == 200
    assert second.status_code == 200

    validate_response = dashboard_client.post(
        "/api/corpus/validate",
        json={"path": ""},
    )

    assert validate_response.status_code == 200
    report = validate_response.json()
    assert report["ok"] is True
    assert report["duplicate_document_count"] == 1
    assert report["duplicate_documents"][0]["preview"] == "duplicate text"


def test_corpus_inspect_via_api_matches_dataset_reader(dashboard_client):
    write_response = dashboard_client.post(
        "/api/corpus/write",
        json={
            "path": "train/records.jsonl",
            "content": '{"prompt":"alpha","response":"beta"}\n',
        },
    )
    assert write_response.status_code == 200

    inspect_response = dashboard_client.post(
        "/api/corpus/inspect",
        json={"split": "train", "show_docs": 5, "max_chars": 80},
    )

    assert inspect_response.status_code == 200
    report = inspect_response.json()
    assert report["split"] == "train"
    assert report["file_count"] == 1
    assert report["document_count"] == 2
    assert [doc["text"] for doc in report["documents"]] == ["alpha", "beta"]


def test_corpus_inspect_via_api_reports_statistics(dashboard_client):
    first = dashboard_client.post(
        "/api/corpus/write",
        json={"path": "train/a.txt", "content": "duplicate"},
    )
    second = dashboard_client.post(
        "/api/corpus/write",
        json={"path": "train/b.txt", "content": "duplicate"},
    )
    third = dashboard_client.post(
        "/api/corpus/write",
        json={"path": "train/c.jsonl", "content": '{"text":"' + ("x" * 30) + '"}\n'},
    )
    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 200

    inspect_response = dashboard_client.post(
        "/api/corpus/inspect",
        json={"split": "train", "show_docs": 2, "max_chars": 10, "long_doc_chars": 20},
    )

    assert inspect_response.status_code == 200
    report = inspect_response.json()
    assert report["document_count"] == 3
    assert report["duplicate_document_count"] == 1
    assert report["file_type_counts"] == {".jsonl": 1, ".txt": 2}
    assert report["very_long_document_count"] == 1
    assert report["estimated_tokens"] > 0


def test_corpus_split_preview_via_api(dashboard_client):
    response = dashboard_client.post(
        "/api/corpus/split/preview",
        json={
            "content": "one\n\ntwo\n\nthree\n\nfour",
            "split_mode": "paragraphs",
            "val_ratio": 0.25,
            "seed": 7,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["record_count"] == 4
    assert payload["train_count"] == 3
    assert payload["val_count"] == 1
    assert "train_content" not in payload
    assert sorted(payload["train_indices"] + payload["val_indices"]) == [0, 1, 2, 3]


def test_corpus_split_write_via_api(dashboard_client):
    response = dashboard_client.post(
        "/api/corpus/split/write",
        json={
            "train_path": "train/split.txt",
            "val_path": "val/split.txt",
            "content": "one\n\ntwo\n\nthree\n\nfour",
            "split_mode": "paragraphs",
            "val_ratio": 0.25,
            "seed": 7,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["train_path"] == "train/split.txt"
    assert payload["val_path"] == "val/split.txt"
    assert payload["train_count"] == 3
    assert payload["val_count"] == 1

    train_response = dashboard_client.get("/api/corpus/file", params={"path": "train/split.txt"})
    val_response = dashboard_client.get("/api/corpus/file", params={"path": "val/split.txt"})
    assert train_response.status_code == 200
    assert val_response.status_code == 200
    assert len(train_response.json()["content"].strip().split("\n\n")) == 3
    assert len(val_response.json()["content"].strip().split("\n\n")) == 1


def test_dashboard_job_preflight_reports_missing_corpus(dashboard_client):
    response = dashboard_client.post(
        "/api/dashboard/jobs/preflight",
        json={"job_type": "tokenizer_train", "params": {}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert any(check["code"] == "corpus_empty" for check in payload["checks"])


def test_dashboard_job_preflight_accepts_tokenizer_train_with_corpus(dashboard_client):
    write_response = dashboard_client.post(
        "/api/corpus/write",
        json={"path": "train/reference.txt", "content": "alpha"},
    )
    assert write_response.status_code == 200

    response = dashboard_client.post(
        "/api/dashboard/jobs/preflight",
        json={"job_type": "tokenizer_train", "params": {}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert any(check["code"] == "corpus_present" for check in payload["checks"])


def test_dashboard_readiness_reports_new_project_blockers(dashboard_client):
    response = dashboard_client.get("/api/dashboard/readiness")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert payload["status"] == "blocked"
    codes = {check["code"]: check for check in payload["checks"]}
    assert codes["corpus_train"]["status"] == "blocker"
    assert codes["sft_train_file"]["status"] == "warning"
    assert "local_corpus/train" in codes["corpus_train"]["next_step"]


def test_dashboard_readiness_included_in_bootstrap_after_dataset_creation(dashboard_client):
    write_response = dashboard_client.post(
        "/api/corpus/write",
        json={"path": "train/reference.txt", "content": "alpha"},
    )
    assert write_response.status_code == 200
    export_response = dashboard_client.post(
        "/api/sandbox/chat/export",
        json={
            "path": "chat_train.jsonl",
            "format": "sft_jsonl",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        },
    )
    assert export_response.status_code == 200

    response = dashboard_client.get("/api/dashboard/bootstrap")

    assert response.status_code == 200
    readiness = response.json()["readiness"]
    codes = {check["code"]: check for check in readiness["checks"]}
    assert codes["corpus_train"]["status"] == "ready"
    assert codes["sft_train_file"]["status"] == "ready"
    assert readiness["check_count"] >= 8


def test_dashboard_run_profiles_are_available_via_api_and_bootstrap(dashboard_client):
    response = dashboard_client.get("/api/dashboard/run-profiles")

    assert response.status_code == 200
    profiles = response.json()["profiles"]
    assert "tiny_smoke" in profiles
    assert profiles["tiny_smoke"]["forms"]["baseTrainForm"]["num_iterations"] == 40
    assert profiles["serious_run"]["forms"]["runtimeForm"]["parallel"] == 4

    bootstrap_response = dashboard_client.get("/api/dashboard/bootstrap")
    assert bootstrap_response.status_code == 200
    assert "gpu_prototype" in bootstrap_response.json()["builder"]["run_profiles"]


def test_dashboard_job_preview_returns_command_and_preflight(dashboard_client):
    write_response = dashboard_client.post(
        "/api/corpus/write",
        json={"path": "train/reference.txt", "content": "alpha"},
    )
    assert write_response.status_code == 200

    response = dashboard_client.post(
        "/api/dashboard/jobs/preview",
        json={"job_type": "tokenizer_train", "params": {"vocab_size": 16384}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_type"] == "tokenizer_train"
    assert payload["command"][1:3] == ["-m", "scripts.tok_train"]
    assert "--vocab-size" in payload["command"]
    assert "NANOCHAT_LOCAL_ONLY" in " ".join(payload["environment_notes"])
    assert payload["preflight"]["ok"] is True
    assert payload["form_validation"]["ok"] is True


def test_dashboard_job_validate_reports_risky_advanced_values(dashboard_client):
    response = dashboard_client.post(
        "/api/dashboard/jobs/validate",
        json={
            "job_type": "base_train",
            "params": {
                "device_type": "cpu",
                "device_batch_size": 8,
                "total_batch_size": 4,
                "max_seq_len": 1024,
                "fp8": 1,
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert any(check["field"] == "total_batch_size" and check["status"] == "error" for check in payload["checks"])
    assert any(check["field"] == "fp8" and check["status"] == "warning" for check in payload["checks"])


def test_dashboard_path_resolve_labels_sandbox_and_corpus_paths(dashboard_client):
    sandbox_response = dashboard_client.post(
        "/api/dashboard/paths/resolve",
        json={"path": "chat_train.jsonl", "kind": "sft"},
    )
    corpus_response = dashboard_client.post(
        "/api/dashboard/paths/resolve",
        json={"path": "train/reference.txt", "kind": "sft"},
    )

    assert sandbox_response.status_code == 200
    sandbox_path = sandbox_response.json()["paths"][0]
    assert sandbox_path["scope"] == "assistant_sandbox"
    assert sandbox_path["resolved_path"].endswith("assistant_sandbox\\chat_train.jsonl") or sandbox_path["resolved_path"].endswith("assistant_sandbox/chat_train.jsonl")

    assert corpus_response.status_code == 200
    corpus_path = corpus_response.json()["paths"][0]
    assert corpus_path["scope"] == "local_corpus"
    assert "usually expects assistant_sandbox" in corpus_path["note"]


def test_dashboard_job_preview_includes_path_hints(dashboard_client):
    write_response = dashboard_client.post(
        "/api/corpus/write",
        json={"path": "train/reference.txt", "content": "alpha"},
    )
    assert write_response.status_code == 200
    response = dashboard_client.post(
        "/api/dashboard/jobs/preview",
        json={"job_type": "chat_sft", "params": {"train_files": "chat_train.jsonl;train/reference.txt"}},
    )

    assert response.status_code == 200
    hints = response.json()["path_hints"]
    scopes = {hint["input"]: hint["scope"] for hint in hints}
    assert scopes["chat_train.jsonl"] == "assistant_sandbox"
    assert scopes["train/reference.txt"] == "local_corpus"


def test_dashboard_job_launch_rejects_invalid_form_values(dashboard_client):
    response = dashboard_client.post(
        "/api/dashboard/jobs",
        json={
            "job_type": "base_train",
            "params": {
                "device_batch_size": 8,
                "total_batch_size": 4,
                "eval_tokens": -1,
            },
        },
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["validation"]["ok"] is False


def test_dashboard_job_preview_rejects_unknown_job_type(dashboard_client):
    response = dashboard_client.post(
        "/api/dashboard/jobs/preview",
        json={"job_type": "unknown", "params": {}},
    )

    assert response.status_code == 400
    assert "Unsupported job_type" in response.json()["detail"]


def test_dashboard_resume_preview_via_api(dashboard_client, tmp_path, monkeypatch):
    base_dir = tmp_path / "cache"
    checkpoint_dir = base_dir / "base_checkpoints" / "d4"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model_000010.pt").write_bytes(b"demo")
    monkeypatch.setattr(dashboard_tools, "get_base_dir", lambda: str(base_dir))

    record = dashboard_tools.JobRecord(
        id="job126",
        label="base train",
        job_type="base_train",
        command=["python", "-m", "scripts.base_train"],
        created_at=0.0,
        cwd=str(tmp_path),
        params={"depth": 4},
        status="paused",
    )
    dashboard_client.app.state.job_manager._jobs[record.id] = record

    response = dashboard_client.get("/api/dashboard/jobs/job126/resume-preview")

    assert response.status_code == 200
    payload = response.json()
    assert payload["resume"]["checkpoint_step"] == 10
    assert "--resume-from-step" in payload["resume"]["command"]


def test_report_reset_and_generate_via_api(dashboard_client, tmp_path, monkeypatch):
    base_dir = tmp_path / "cache"
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(base_dir))

    reset_response = dashboard_client.post("/api/report/reset")
    assert reset_response.status_code == 200
    reset_payload = reset_response.json()
    assert reset_payload["header_exists"] is True
    assert (base_dir / "report" / "header.md").exists()

    generate_response = dashboard_client.post("/api/report/generate")
    assert generate_response.status_code == 200
    generate_payload = generate_response.json()
    assert generate_payload["report_exists"] is True
    assert "Total wall clock time" in generate_payload["preview"]


def test_corpus_parquet_preview_from_paragraphs_via_api(dashboard_client):
    preview_response = dashboard_client.post(
        "/api/corpus/parquet/preview",
        json={
            "path": "train/preview.parquet",
            "content": "alpha\n\nbeta\n\n",
            "conversion_mode": "paragraphs",
            "source": "api-test",
        },
    )

    assert preview_response.status_code == 200
    preview = preview_response.json()
    assert preview["row_count"] == 2
    assert preview["rejected_empty_rows"] == 1
    assert preview["sample_rows"][0] == {"text": "alpha", "source": "api-test", "row_index": 0}


def test_corpus_parquet_preview_reports_invalid_jsonl_via_api(dashboard_client):
    preview_response = dashboard_client.post(
        "/api/corpus/parquet/preview",
        json={
            "path": "train/preview.parquet",
            "content": '{"text":"ok"}\nnope',
            "conversion_mode": "jsonl",
        },
    )

    assert preview_response.status_code == 400
    assert "line 2" in preview_response.json()["detail"]


@pytest.mark.skipif(pa is None or pq is None, reason="pyarrow is required for parquet corpus API tests")
def test_corpus_parquet_write_from_markdown_sections_via_api(dashboard_client):
    write_response = dashboard_client.post(
        "/api/corpus/parquet/write",
        json={
            "path": "train/sections.parquet",
            "content": "# Intro\nAlpha\n\n## Next\nBeta",
            "conversion_mode": "markdown_sections",
            "mode": "overwrite",
            "source": "api-test",
        },
    )

    assert write_response.status_code == 200
    written = write_response.json()
    assert written["path"] == "train/sections.parquet"
    assert written["row_count"] == 2
    assert written["conversion"]["conversion_mode"] == "markdown_sections"
    assert written["sample_rows"][0]["heading"] == "Intro"

    read_response = dashboard_client.get("/api/corpus/file", params={"path": "train/sections.parquet"})
    assert read_response.status_code == 200
    assert read_response.json()["row_count"] == 2


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
