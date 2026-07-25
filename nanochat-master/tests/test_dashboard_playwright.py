from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

playwright_sync_api = pytest.importorskip(
    "playwright.sync_api",
    reason="Playwright smoke tests require `uv sync --extra browser-tests`.",
)

Error = playwright_sync_api.Error
sync_playwright = playwright_sync_api.sync_playwright


REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_HTML = REPO_ROOT / "nanochat" / "dashboard.html"
DASHBOARD_URL = "http://dashboard-smoke.local/"


def _checkpoint_sets() -> dict:
    return {
        "base": {"tags": []},
        "sft": {"tags": []},
        "rl": {"tags": []},
    }


def _run_profiles() -> dict:
    return {
        "tiny_smoke": {
            "label": "Tiny Smoke Test",
            "summary": "Small browser-test profile.",
            "forms": {
                "baseTrainForm": {
                    "corpus_dir": "C:/demo/local_corpus",
                    "depth": 4,
                    "num_iterations": 40,
                    "device_type": "cpu",
                }
            },
        }
    }


def _design() -> dict:
    return {
        "slug": "truth-first-teammate",
        "name": "Truth-First Teammate",
        "mission": "Help carefully.",
        "team_role": "Builder partner.",
        "tone": "Plain.",
        "uncertainty_policy": "Say when unsure.",
        "collaboration_policy": "Ask before risky work.",
        "guardrails": ["Do not invent facts."],
        "custom_notes": "",
        "identity_preview": "",
        "recipes": {},
        "draft_source": "manual",
    }


def _bootstrap_payload() -> dict:
    corpus_dir = "C:/demo/local_corpus"
    sandbox_dir = "C:/demo/assistant_sandbox"
    job = {
        "id": "job-failed",
        "label": "failed parquet run",
        "job_type": "base_train",
        "status": "failed",
        "created_at": 1_700_000_000,
        "latest_checkpoint_step": None,
        "checkpoint_tag": "",
    }
    return {
        "builder": {
            "repo_root": "C:/demo/nanochat-master",
            "base_dir": "C:/demo/cache",
            "dataset_dir": "C:/demo/local_corpus",
            "corpus_dir": corpus_dir,
            "sandbox_dir": sandbox_dir,
            "tokenizer_path": "C:/demo/cache/tokenizer/tokenizer.pkl",
            "identity_file": "C:/demo/cache/identity_conversations.jsonl",
            "default_chat_train_file": f"{sandbox_dir}/chat_train.jsonl",
            "default_chat_val_file": f"{sandbox_dir}/chat_val.jsonl",
            "dataset_shards": 1,
            "tokenizer_ready": False,
            "identity_exists": False,
            "local_only": True,
            "sandbox_file_count": 0,
            "devices": {"cuda_available": False, "mps_available": False, "gpu_names": []},
            "corpus_summary": {
                "exists": True,
                "splits": {
                    "train": {"file_count": 1, "sample_files": ["train/demo.txt"], "path": f"{corpus_dir}/train"},
                    "val": {"file_count": 0, "sample_files": [], "path": f"{corpus_dir}/val"},
                },
            },
            "checkpoint_sets": _checkpoint_sets(),
            "hardware_profile": {"preferred_device": "cpu", "gpus": [], "system_ram_gb": 32},
            "hardware_recommendations": {
                "summary": {"title": "CPU starter", "notes": ["safe local settings"]},
                "baseTrainForm": {
                    "corpus_dir": corpus_dir,
                    "depth": 4,
                    "max_seq_len": 256,
                    "device_batch_size": 2,
                    "total_batch_size": 512,
                    "save_every": 20,
                    "device_type": "cpu",
                },
                "chatSftForm": {"max_seq_len": 256, "device_batch_size": 2, "save_every": 20},
                "chatRlForm": {"save_every": 20},
                "runtimeForm": {"ctx_size": 2048, "threads": 4, "threads_http": 2, "parallel": 1},
            },
            "guided_presets": {
                "truth_first_teammate": {
                    "label": "Truth-First Teammate",
                    "summary": "Careful local helper.",
                    "design": _design(),
                    "recipes": {},
                }
            },
            "designs": [_design()],
            "run_profiles": _run_profiles(),
        },
        "jobs": [job],
        "chat": {"ready": False, "loading": False, "num_workers": 0, "config": {"source": "none"}},
        "runtime": {
            "ready": False,
            "running": False,
            "last_error": "",
            "config": {},
            "bundle": {
                "runtime_exists": True,
                "runtime_dir": "C:/demo/runtime/windows",
                "devices": [{"id": "cpu", "description": "CPU"}],
                "models": [
                    {
                        "name": "helper.gguf",
                        "path": "C:/demo/assistant_models/helper.gguf",
                        "size": 1024,
                        "updated_at": 1_700_000_000,
                    }
                ],
                "recommended_model": {
                    "name": "helper.gguf",
                    "path": "C:/demo/assistant_models/helper.gguf",
                    "size": 1024,
                    "updated_at": 1_700_000_000,
                },
            },
        },
        "readiness": {
            "status": "blocked",
            "ready_count": 1,
            "check_count": 2,
            "blocker_count": 1,
            "warning_count": 0,
            "next_step": "Add identity data.",
            "checks": [
                {"label": "Corpus", "status": "ready", "message": "Corpus has train files."},
                {"label": "Identity", "status": "blocker", "message": "Publish identity data."},
            ],
        },
        "ecg": _ecg_payload(),
        "sandbox": {"root": sandbox_dir, "files": [], "status": "ok"},
        "corpus": {"root": corpus_dir, "files": [{"path": "train/demo.txt", "kind": "text", "size": 12, "updated_at": 1_700_000_000}]},
        "activity": {"path": "builder_logs/activity.jsonl", "events": []},
        "benchmarks": {"path": "builder_logs/benchmark_history.jsonl", "record_count": 0},
        "report": {
            "report_exists": False,
            "report_file": "C:/demo/report.md",
            "root_copy_exists": False,
            "root_copy": "C:/demo/report.md",
            "ready_sections": 0,
            "expected_sections": 4,
            "preview": "",
        },
    }


def _ecg_payload() -> dict:
    return {
        "available": True,
        "supported": True,
        "status": "idle",
        "source": "cpu",
        "current_percent": 7,
        "history": [2, 5, 7],
        "label": "CPU",
        "note": "Browser smoke data.",
        "sources": {"cpu_percent": 7, "gpus": []},
    }


def _job_detail_payload() -> dict:
    return {
        "job": {
            "id": "job-failed",
            "label": "failed parquet run",
            "job_type": "base_train",
            "command": ["python", "-m", "scripts.base_train"],
            "display_command": "python -m scripts.base_train --device-type cpu",
            "cwd": "C:/demo/nanochat-master",
            "notes": "",
            "params": {},
            "status": "failed",
            "created_at": 1_700_000_000,
            "started_at": 1_700_000_001,
            "finished_at": 1_700_000_002,
            "exit_code": 1,
            "pid": None,
            "metrics": {},
            "latest_checkpoint_step": None,
            "checkpoint_tag": "",
            "checkpoint_dir": "",
            "requested_resume_step": None,
            "save_every": 20,
            "is_resumed_run": False,
            "can_pause": False,
            "can_resume": False,
            "failure_diagnosis": {
                "kind": "missing_pyarrow",
                "severity": "error",
                "title": "Parquet support is not installed yet",
                "detail": "This run tried to read a parquet corpus file.",
                "next_step": "Install parquet support with uv sync --extra parquet.",
                "evidence": ["ModuleNotFoundError: No module named 'pyarrow'"],
            },
            "logs": [
                "loading corpus",
                "ModuleNotFoundError: No module named 'pyarrow'",
                "job failed",
            ],
        }
    }


def _launched_job_payload(request_body: dict) -> dict:
    params = request_body.get("params") or {}
    job_type = request_body.get("job_type", "base_train")
    label = request_body.get("label") or job_type.replace("_", " ")
    return {
        "id": "job-launched",
        "label": label,
        "job_type": job_type,
        "command": ["python", "-m", "scripts.base_train"],
        "display_command": f"python -m scripts.base_train --run {params.get('run', 'builder-base')} --device-type {params.get('device_type', 'cpu')}",
        "cwd": "C:/demo/nanochat-master",
        "notes": "",
        "params": params,
        "status": "queued",
        "created_at": 1_700_000_200,
        "started_at": None,
        "finished_at": None,
        "exit_code": None,
        "pid": None,
        "metrics": {},
        "latest_checkpoint_step": None,
        "checkpoint_tag": "",
        "checkpoint_dir": "",
        "requested_resume_step": None,
        "save_every": params.get("save_every", 20),
        "is_resumed_run": False,
        "can_pause": False,
        "can_resume": False,
        "failure_diagnosis": None,
        "logs": ["[dashboard] queued browser launch"],
    }


def _preview_payload(request_body: dict) -> dict:
    params = request_body.get("params") or {}
    return {
        "job_type": request_body.get("job_type", "base_train"),
        "cwd": "C:/demo/nanochat-master",
        "command": ["python", "-m", "scripts.base_train", "--depth", str(params.get("depth", 4))],
        "display_command": f"python -m scripts.base_train --depth {params.get('depth', 4)} --device-type {params.get('device_type', 'cpu')}",
        "form_validation": {"ok": True, "checks": [], "warning_count": 0, "error_count": 0},
        "preflight": {"ok": True, "checks": [], "warning_count": 0, "error_count": 0},
        "path_hints": [],
        "environment_notes": ["local-only browser smoke"],
    }


def _text_file_payload(path: str, content: str, kind: str = "text") -> dict:
    return {
        "path": path,
        "kind": kind,
        "content": content,
        "size": len(content.encode("utf-8")),
        "updated_at": 1_700_000_100,
        "editable_as_text": True,
    }


def _wait_for_dashboard_ready(page) -> None:
    page.wait_for_function("window.__dashboardInitialized === true")


@pytest.fixture
def dashboard_page():
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except Error as exc:
            pytest.skip(f"Playwright Chromium is not installed. Run `uv run playwright install chromium`. ({exc})")
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        bootstrap = _bootstrap_payload()
        runtime_start_requests: list[dict] = []
        chat_requests: list[dict] = []
        chat_export_requests: list[dict] = []
        job_launch_requests: list[dict] = []
        sandbox_files: dict[str, str] = {}
        corpus_files: dict[str, str] = {"train/demo.txt": "hello corpus"}
        launched_jobs: dict[str, dict] = {}

        def update_workspace_listing(workspace: str) -> None:
            files = sandbox_files if workspace == "sandbox" else corpus_files
            target = bootstrap[workspace]
            target["files"] = [
                {
                    "path": path,
                    "kind": "text",
                    "size": len(content.encode("utf-8")),
                    "updated_at": 1_700_000_100,
                }
                for path, content in sorted(files.items())
            ]
            if workspace == "sandbox":
                bootstrap["builder"]["sandbox_file_count"] = len(files)
            if workspace == "corpus":
                train_count = sum(1 for path in files if path.startswith("train/"))
                val_count = sum(1 for path in files if path.startswith("val/"))
                bootstrap["builder"]["corpus_summary"]["splits"]["train"]["file_count"] = train_count
                bootstrap["builder"]["corpus_summary"]["splits"]["val"]["file_count"] = val_count

        update_workspace_listing("sandbox")
        update_workspace_listing("corpus")

        def handle_route(route):
            parsed = urlparse(route.request.url)
            path = parsed.path
            query = parse_qs(parsed.query)
            if path in {"", "/", "/dashboard.html"}:
                route.fulfill(status=200, content_type="text/html", body=DASHBOARD_HTML.read_text(encoding="utf-8"))
                return
            if path.endswith("/api/dashboard/bootstrap"):
                route.fulfill(status=200, content_type="application/json", body=json.dumps(bootstrap))
                return
            if path.endswith("/api/dashboard/ecg"):
                route.fulfill(status=200, content_type="application/json", body=json.dumps(_ecg_payload()))
                return
            if path.endswith("/api/dashboard/jobs/validate"):
                request_body = json.loads(route.request.post_data or "{}")
                params = request_body.get("params") or {}
                total_batch_size = int(params.get("total_batch_size") or 0)
                device_batch_size = int(params.get("device_batch_size") or 0)
                if total_batch_size < device_batch_size:
                    route.fulfill(
                        status=200,
                        content_type="application/json",
                        body=json.dumps(
                            {
                                "ok": False,
                                "checks": [
                                    {
                                        "status": "error",
                                        "field": "total_batch_size",
                                        "message": "must be at least device_batch_size",
                                    }
                                ],
                                "warning_count": 0,
                                "error_count": 1,
                            }
                        ),
                    )
                    return
                route.fulfill(status=200, content_type="application/json", body=json.dumps({"ok": True, "checks": [], "warning_count": 0, "error_count": 0}))
                return
            if path.endswith("/api/dashboard/jobs/preflight"):
                request_body = json.loads(route.request.post_data or "{}")
                params = request_body.get("params") or {}
                if params.get("run") == "blocked-preflight":
                    route.fulfill(
                        status=200,
                        content_type="application/json",
                        body=json.dumps(
                            {
                                "ok": False,
                                "checks": [
                                    {
                                        "status": "error",
                                        "message": "Corpus train split is missing",
                                        "detail": "Add train files before launch.",
                                    }
                                ],
                                "warning_count": 0,
                                "error_count": 1,
                            }
                        ),
                    )
                    return
                route.fulfill(status=200, content_type="application/json", body=json.dumps({"ok": True, "checks": [], "warning_count": 0, "error_count": 0}))
                return
            if path.endswith("/api/runtime/start"):
                request_body = json.loads(route.request.post_data or "{}")
                runtime_start_requests.append(request_body)
                bootstrap["runtime"] = {
                    **bootstrap["runtime"],
                    "ready": True,
                    "running": True,
                    "last_error": "",
                    "config": request_body,
                }
                route.fulfill(status=200, content_type="application/json", body=json.dumps(bootstrap["runtime"]))
                return
            if path.endswith("/api/runtime/chat"):
                request_body = json.loads(route.request.post_data or "{}")
                chat_requests.append(request_body)
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        {
                            "text": "Mock assistant reply for transcript smoke.",
                            "text_source": "mock",
                            "protocol_warning": "",
                            "raw": {"usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}},
                        }
                    ),
                )
                return
            if path.endswith("/api/sandbox/chat/export"):
                request_body = json.loads(route.request.post_data or "{}")
                chat_export_requests.append(request_body)
                content = json.dumps({"messages": request_body.get("messages", [])}, indent=2)
                if request_body.get("format") == "sft_jsonl":
                    content = json.dumps(request_body.get("messages", [])) + "\n"
                if request_body.get("preview"):
                    route.fulfill(
                        status=200,
                        content_type="application/json",
                        body=json.dumps(
                            {
                                "path": request_body["path"],
                                "format": request_body.get("format"),
                                "message_count": len(request_body.get("messages", [])),
                                "content": content,
                                "preview": True,
                            }
                        ),
                    )
                    return
                sandbox_files[request_body["path"]] = content
                update_workspace_listing("sandbox")
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        {
                            **_text_file_payload(request_body["path"], content, kind="jsonl" if request_body.get("format") == "sft_jsonl" else "json"),
                            "format": request_body.get("format"),
                            "message_count": len(request_body.get("messages", [])),
                        }
                    ),
                )
                return
            if path.endswith("/api/runtime/stop"):
                bootstrap["runtime"] = {
                    **bootstrap["runtime"],
                    "ready": False,
                    "running": False,
                    "config": {},
                }
                route.fulfill(status=200, content_type="application/json", body=json.dumps(bootstrap["runtime"]))
                return
            if path.endswith("/api/sandbox/file"):
                requested_path = query.get("path", [""])[0]
                route.fulfill(status=200, content_type="application/json", body=json.dumps(_text_file_payload(requested_path, sandbox_files[requested_path])))
                return
            if path.endswith("/api/corpus/file"):
                requested_path = query.get("path", [""])[0]
                route.fulfill(status=200, content_type="application/json", body=json.dumps(_text_file_payload(requested_path, corpus_files[requested_path])))
                return
            if path.endswith("/api/sandbox/write"):
                request_body = json.loads(route.request.post_data or "{}")
                sandbox_files[request_body["path"]] = request_body.get("content", "")
                update_workspace_listing("sandbox")
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(_text_file_payload(request_body["path"], sandbox_files[request_body["path"]])),
                )
                return
            if path.endswith("/api/corpus/write"):
                request_body = json.loads(route.request.post_data or "{}")
                corpus_files[request_body["path"]] = request_body.get("content", "")
                update_workspace_listing("corpus")
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(_text_file_payload(request_body["path"], corpus_files[request_body["path"]])),
                )
                return
            if path.endswith("/api/dashboard/jobs/job-launched"):
                route.fulfill(status=200, content_type="application/json", body=json.dumps({"job": launched_jobs["job-launched"]}))
                return
            if path.endswith("/api/dashboard/jobs/job-failed"):
                route.fulfill(status=200, content_type="application/json", body=json.dumps(_job_detail_payload()))
                return
            if path.endswith("/api/dashboard/jobs/preview"):
                request_body = json.loads(route.request.post_data or "{}")
                route.fulfill(status=200, content_type="application/json", body=json.dumps(_preview_payload(request_body)))
                return
            if path.endswith("/api/dashboard/jobs"):
                request_body = json.loads(route.request.post_data or "{}")
                job_launch_requests.append(request_body)
                job = _launched_job_payload(request_body)
                launched_jobs[job["id"]] = job
                bootstrap["jobs"] = [
                    {
                        "id": job["id"],
                        "label": job["label"],
                        "job_type": job["job_type"],
                        "status": job["status"],
                        "created_at": job["created_at"],
                        "latest_checkpoint_step": job["latest_checkpoint_step"],
                        "checkpoint_tag": job["checkpoint_tag"],
                    },
                    *[existing for existing in bootstrap["jobs"] if existing.get("id") != job["id"]],
                ]
                route.fulfill(status=200, content_type="application/json", body=json.dumps({"job": job, "resolved_params": job["params"]}))
                return
            route.fulfill(status=404, content_type="text/plain", body="not found")

        page.route("**/*", handle_route)
        page.runtime_start_requests = runtime_start_requests
        page.chat_requests = chat_requests
        page.chat_export_requests = chat_export_requests
        page.job_launch_requests = job_launch_requests
        yield page
        browser.close()


def test_dashboard_bootstrap_renders_with_mocked_api(dashboard_page):
    dashboard_page.goto(DASHBOARD_URL)
    dashboard_page.keyboard.press("Escape")
    _wait_for_dashboard_ready(dashboard_page)

    dashboard_page.get_by_text("Truth-First Teammate").first.wait_for()

    assert dashboard_page.locator("#datasetShardCount").inner_text() == "1"
    assert dashboard_page.locator("#deviceChips").inner_text()
    assert "First-Run Readiness" in dashboard_page.locator("#readinessChecklist").inner_text()
    assert "train/demo.txt" in dashboard_page.locator("#corpusFiles").inner_text()
    assert "failed parquet run" in dashboard_page.locator("#jobsList").inner_text()


def test_dashboard_job_log_filter_and_diagnosis_render(dashboard_page):
    dashboard_page.goto(DASHBOARD_URL)
    dashboard_page.keyboard.press("Escape")
    _wait_for_dashboard_ready(dashboard_page)
    dashboard_page.get_by_text("failed parquet run").click()

    dashboard_page.locator("#jobSummary").get_by_text("Parquet support is not installed yet").wait_for()
    assert "ModuleNotFoundError" in dashboard_page.locator("#jobLogs").input_value()

    dashboard_page.locator("#jobLogFilter").fill("pyarrow")

    assert "Showing 1 of 3 log line" in dashboard_page.locator("#jobLogMeta").inner_text()
    assert dashboard_page.locator("#jobLogs").input_value() == "ModuleNotFoundError: No module named 'pyarrow'"


def test_dashboard_base_train_command_preview_serializes_form(dashboard_page):
    dashboard_page.goto(DASHBOARD_URL)
    dashboard_page.keyboard.press("Escape")
    _wait_for_dashboard_ready(dashboard_page)
    dashboard_page.locator("#baseDepth").fill("6")

    dashboard_page.locator('[data-preview-job="base_train"][data-form="baseTrainForm"]').click()

    dashboard_page.locator("#jobSummary").get_by_text("Command preview for base_train").wait_for()
    command_preview = dashboard_page.locator("#jobCommandPreview").inner_text()
    assert "--depth 6" in command_preview
    assert "--device-type cpu" in command_preview


def test_dashboard_launch_blocks_on_form_validation_error(dashboard_page):
    dashboard_page.goto(DASHBOARD_URL)
    dashboard_page.keyboard.press("Escape")
    _wait_for_dashboard_ready(dashboard_page)
    dashboard_page.locator("#baseTotalBatch").fill("1")
    dashboard_page.locator("#baseDeviceBatch").fill("8")

    dashboard_page.locator('[data-job="base_train"][data-form="baseTrainForm"]').click()

    dashboard_page.locator("#jobSummary").get_by_text("Form validation stopped base_train.").wait_for()
    assert "total_batch_size" in dashboard_page.locator("#jobSummary").inner_text()
    assert dashboard_page.job_launch_requests == []


def test_dashboard_launch_blocks_on_preflight_error(dashboard_page):
    dashboard_page.goto(DASHBOARD_URL)
    dashboard_page.keyboard.press("Escape")
    _wait_for_dashboard_ready(dashboard_page)
    dashboard_page.locator("#baseRunName").fill("blocked-preflight")

    dashboard_page.locator('[data-job="base_train"][data-form="baseTrainForm"]').click()

    dashboard_page.locator("#jobSummary").get_by_text("Preflight stopped base_train.").wait_for()
    assert "Corpus train split is missing" in dashboard_page.locator("#jobSummary").inner_text()
    assert dashboard_page.job_launch_requests == []


def test_dashboard_launch_success_selects_new_job(dashboard_page):
    dashboard_page.goto(DASHBOARD_URL)
    dashboard_page.keyboard.press("Escape")
    _wait_for_dashboard_ready(dashboard_page)
    dashboard_page.locator("#baseRunName").fill("browser-launch")

    dashboard_page.locator('[data-job="base_train"][data-form="baseTrainForm"]').click()

    dashboard_page.locator("#jobsList").get_by_text("base train").wait_for()
    dashboard_page.locator("#jobSummary").get_by_text("Status: queued").wait_for()
    assert dashboard_page.job_launch_requests[-1]["job_type"] == "base_train"
    assert dashboard_page.job_launch_requests[-1]["params"]["run"] == "browser-launch"


def test_dashboard_sandbox_editor_save_updates_file_list(dashboard_page):
    dashboard_page.goto(DASHBOARD_URL)
    dashboard_page.keyboard.press("Escape")
    _wait_for_dashboard_ready(dashboard_page)

    dashboard_page.locator("#newSandboxFileBtn").click()
    dashboard_page.locator("#sandboxPath").fill("notes/browser-smoke.txt")
    dashboard_page.locator("#sandboxEditor").fill("saved from browser smoke")
    dashboard_page.locator("#saveSandboxBtn").click()

    dashboard_page.locator("#sandboxMeta").get_by_text("Saved notes/browser-smoke.txt").wait_for()
    dashboard_page.locator("#sandboxFiles").get_by_text("notes/browser-smoke.txt").wait_for()


def test_dashboard_corpus_editor_save_updates_file_list(dashboard_page):
    dashboard_page.goto(DASHBOARD_URL)
    dashboard_page.keyboard.press("Escape")
    _wait_for_dashboard_ready(dashboard_page)

    dashboard_page.locator("#newCorpusFileBtn").click()
    dashboard_page.locator("#corpusPath").fill("train/browser-smoke.txt")
    dashboard_page.locator("#corpusEditor").fill("browser corpus row")
    dashboard_page.locator("#saveCorpusBtn").click()

    dashboard_page.locator("#corpusMeta").get_by_text("Saved train/browser-smoke.txt").wait_for()
    dashboard_page.locator("#corpusFiles").get_by_text("train/browser-smoke.txt").wait_for()


def test_dashboard_runtime_start_and_stop_controls_update_status(dashboard_page):
    dashboard_page.goto(DASHBOARD_URL)
    dashboard_page.keyboard.press("Escape")
    _wait_for_dashboard_ready(dashboard_page)

    dashboard_page.locator("#runtimeModelPath").fill("C:/demo/assistant_models/helper.gguf")
    dashboard_page.locator("#runtimeCtxSize").fill("1024")
    dashboard_page.locator("#startRuntimeBtn").click()

    dashboard_page.locator("#runtimeStatusBox").get_by_text("Status: ready").wait_for()
    assert dashboard_page.locator("#chatProvider").input_value() == "local_runtime"
    assert dashboard_page.runtime_start_requests[-1]["model_path"] == "C:/demo/assistant_models/helper.gguf"
    assert dashboard_page.runtime_start_requests[-1]["ctx_size"] == 1024

    dashboard_page.locator("#stopRuntimeBtn").click()

    dashboard_page.locator("#runtimeStatusBox").get_by_text("Status: stopped").wait_for()


def test_dashboard_conversation_lab_sends_local_runtime_chat(dashboard_page):
    dashboard_page.goto(DASHBOARD_URL)
    dashboard_page.keyboard.press("Escape")
    _wait_for_dashboard_ready(dashboard_page)
    dashboard_page.locator("#chatProvider").select_option("local_runtime")
    dashboard_page.locator("#assistantToolsEnabled").select_option("0")

    dashboard_page.locator("#chatInput").fill("Say hello for the smoke test.")
    dashboard_page.locator("#sendChatBtn").click()

    dashboard_page.get_by_text("Mock assistant reply for transcript smoke.").wait_for()
    assert dashboard_page.chat_requests[-1]["messages"][-1]["content"] == "Say hello for the smoke test."


def test_dashboard_conversation_lab_saves_full_transcript_json(dashboard_page):
    dashboard_page.goto(DASHBOARD_URL)
    dashboard_page.keyboard.press("Escape")
    _wait_for_dashboard_ready(dashboard_page)
    dashboard_page.locator("#chatProvider").select_option("local_runtime")
    dashboard_page.locator("#assistantToolsEnabled").select_option("0")
    dashboard_page.locator("#chatInput").fill("Save this exchange.")
    dashboard_page.locator("#sendChatBtn").click()
    dashboard_page.get_by_text("Mock assistant reply for transcript smoke.").wait_for()

    dashboard_page.on("dialog", lambda dialog: dialog.accept("transcripts/browser-smoke.json"))
    dashboard_page.locator("#saveTranscriptBtn").click()

    dashboard_page.locator("#sandboxMeta").get_by_text("Loaded transcripts/browser-smoke.json").wait_for()
    export_request = dashboard_page.chat_export_requests[-1]
    assert export_request["format"] == "json"
    assert export_request["mode"] == "overwrite"
    assert [message["role"] for message in export_request["messages"]] == ["user", "assistant"]


def test_dashboard_conversation_lab_exports_selected_sft_jsonl(dashboard_page):
    dashboard_page.goto(DASHBOARD_URL)
    dashboard_page.keyboard.press("Escape")
    _wait_for_dashboard_ready(dashboard_page)
    dashboard_page.locator("#chatProvider").select_option("local_runtime")
    dashboard_page.locator("#assistantToolsEnabled").select_option("0")
    dashboard_page.locator("#chatInput").fill("Export this pair.")
    dashboard_page.locator("#sendChatBtn").click()
    dashboard_page.get_by_text("Mock assistant reply for transcript smoke.").wait_for()
    dashboard_page.locator("#selectAllChatBtn").click()

    dialog_responses = iter(["chat_train.jsonl", "yes"])
    dashboard_page.on("dialog", lambda dialog: dialog.accept(next(dialog_responses)))
    dashboard_page.locator("#exportChatSftBtn").click()

    dashboard_page.locator("#sandboxMeta").get_by_text("Loaded chat_train.jsonl").wait_for()
    preview_request, write_request = dashboard_page.chat_export_requests[-2:]
    assert preview_request["preview"] is True
    assert write_request["format"] == "sft_jsonl"
    assert write_request["mode"] == "append"
    assert [message["role"] for message in write_request["messages"]] == ["user", "assistant"]
