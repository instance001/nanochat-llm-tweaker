import ast
from html.parser import HTMLParser
from pathlib import Path

from nanochat import dashboard_tools
from nanochat.dashboard_tools import build_job_command, run_profiles, validate_job_params


REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_HTML = REPO_ROOT / "nanochat" / "dashboard.html"


class DashboardFormParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.current_form = None
        self.forms = {}
        self.job_forms = {}

    def handle_starttag(self, tag, attrs):
        attr_map = dict(attrs)
        if tag == "form":
            self.current_form = attr_map.get("id")
            if self.current_form:
                self.forms.setdefault(self.current_form, set())
            return
        if tag in {"input", "select", "textarea"} and self.current_form and attr_map.get("name"):
            self.forms.setdefault(self.current_form, set()).add(attr_map["name"])
            return
        job_type = attr_map.get("data-job") or attr_map.get("data-preview-job")
        form_id = attr_map.get("data-form")
        if job_type and form_id:
            self.job_forms.setdefault(job_type, set()).add(form_id)

    def handle_endtag(self, tag):
        if tag == "form":
            self.current_form = None


def dashboard_form_parser():
    parser = DashboardFormParser()
    parser.feed(DASHBOARD_HTML.read_text(encoding="utf-8"))
    return parser


def parser_long_flags(script_module):
    script_path = REPO_ROOT / (script_module.replace(".", "/") + ".py")
    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    flags = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                flags.add(arg.value[2:].replace("-", "_"))
    return flags


def test_dashboard_job_param_manifest_matches_script_argparse_flags():
    intentionally_unsupported = {
        ("base_eval", "hf_path"),
        ("benchmark_eval", "hf_path"),
    }
    for job_type, script_module in dashboard_tools.DASHBOARD_JOB_SCRIPT_MODULES.items():
        parser_flags = parser_long_flags(script_module)
        supported = dashboard_tools.DASHBOARD_JOB_PARAM_NAMES[job_type]
        missing = parser_flags - supported - {flag for mapped_job, flag in intentionally_unsupported if mapped_job == job_type}
        extra = supported - parser_flags
        assert missing == set(), f"{job_type} is missing dashboard support declarations for: {sorted(missing)}"
        assert extra == set(), f"{job_type} declares params not present in {script_module}: {sorted(extra)}"


def test_dashboard_job_param_manifest_matches_dashboard_html_forms():
    parser = dashboard_form_parser()
    implicit_forms = {
        "base_train": {"baseTrainAdvancedForm"},
        "chat_sft": {"chatSftAdvancedForm"},
    }
    intentionally_shared_form_fields = {
        "tokenizer_train": {"max_corpus_docs"},
        "tokenizer_eval": {"max_chars", "doc_cap", "vocab_size"},
    }

    for job_type, supported in dashboard_tools.DASHBOARD_JOB_PARAM_NAMES.items():
        form_ids = set(parser.job_forms.get(job_type, set())) | implicit_forms.get(job_type, set())
        assert form_ids, f"{job_type} has no launch or preview form in dashboard.html"
        form_names = set()
        for form_id in form_ids:
            assert form_id in parser.forms, f"{job_type} references missing dashboard form {form_id}"
            form_names.update(parser.forms[form_id])

        unsupported_form_names = form_names - supported - intentionally_shared_form_fields.get(job_type, set())
        missing_form_fields = supported - form_names
        assert unsupported_form_names == set(), f"{job_type} dashboard form fields are not declared in the manifest: {sorted(unsupported_form_names)}"
        assert missing_form_fields == set(), f"{job_type} manifest params are not represented in dashboard.html forms: {sorted(missing_form_fields)}"


def test_build_job_command_known_supported_params_emit_flags():
    params = {
        "device_type": "cpu",
        "model_tag": "demo",
        "resume_from_step": 10,
        "save_every": 20,
        "eval_examples": 40,
    }
    command = build_job_command("chat_rl", params)
    command_text = " ".join(command)

    for name in params:
        assert f"--{name.replace('_', '-')}" in command_text


def test_run_profiles_include_expected_starter_profiles():
    profiles = run_profiles()

    assert {"tiny_smoke", "laptop_overnight", "gpu_prototype", "serious_run"}.issubset(profiles)
    assert profiles["tiny_smoke"]["forms"]["baseTrainForm"]["num_iterations"] < profiles["serious_run"]["forms"]["baseTrainForm"]["num_iterations"]
    assert profiles["gpu_prototype"]["forms"]["runtimeForm"]["device_strategy"] == "gpu"


def test_run_profiles_returns_copy():
    profiles = run_profiles()
    profiles["tiny_smoke"]["forms"]["baseTrainForm"]["num_iterations"] = 999

    assert run_profiles()["tiny_smoke"]["forms"]["baseTrainForm"]["num_iterations"] == 40


def test_validate_job_params_flags_blocking_batch_error():
    report = validate_job_params(
        "base_train",
        {
            "device_type": "cpu",
            "device_batch_size": 8,
            "total_batch_size": 4,
            "max_seq_len": 513,
            "eval_tokens": -1,
        },
    )

    assert report["ok"] is False
    assert any(check["field"] == "total_batch_size" and check["status"] == "error" for check in report["checks"])
    assert any(check["field"] == "eval_tokens" and check["status"] == "error" for check in report["checks"])
    assert any(check["field"] == "device_batch_size" and check["status"] == "warning" for check in report["checks"])


def test_validate_job_params_warns_for_fp8_without_cuda():
    report = validate_job_params("base_train", {"device_type": "cpu", "fp8": 1, "device_batch_size": 2, "total_batch_size": 1024})

    assert report["ok"] is True
    assert any(check["field"] == "fp8" and check["status"] == "warning" for check in report["checks"])


def test_build_job_command_supports_chat_rl():
    command = build_job_command(
        "chat_rl",
        {
            "model_tag": "demo-sft",
            "num_epochs": 2,
            "device_type": "cpu",
        },
    )

    assert command[1:3] == ["-m", "scripts.chat_rl"]
    assert "--model-tag" in command
    assert "--num-epochs" in command


def test_build_job_command_supports_chat_rl_resume_flags():
    command = build_job_command(
        "chat_rl",
        {
            "model_tag": "demo-rl",
            "resume_from_step": 75,
            "save_every": 20,
        },
    )

    assert "--resume-from-step" in command
    assert "75" in command
    assert "--save-every" in command
    assert "20" in command


def test_build_job_command_supports_chat_eval():
    command = build_job_command(
        "chat_eval",
        {
            "source": "rl",
            "task_name": "GSM8K|MMLU",
            "device_type": "cpu",
        },
    )

    assert command[1:3] == ["-m", "scripts.chat_eval"]
    assert "--source" in command
    assert "rl" in command
    assert "--task-name" in command


def test_build_job_command_supports_tokenizer_eval_sample_depth():
    command = build_job_command(
        "tokenizer_eval",
        {
            "corpus_dir": "C:\\demo\\local_corpus",
            "max_corpus_docs": 12,
        },
    )

    assert command[1:3] == ["-m", "scripts.tok_eval"]
    assert "--corpus-dir" in command
    assert "--max-corpus-docs" in command
    assert "12" in command


def test_build_job_command_emits_fp8_flag_only_when_enabled():
    disabled = build_job_command("base_train", {"fp8": 0})
    enabled = build_job_command("base_train", {"fp8": 1})

    assert "--fp8" not in disabled
    assert "--fp8" in enabled


def test_build_job_command_supports_advanced_base_train_flags():
    command = build_job_command(
        "base_train",
        {
            "aspect_ratio": 96,
            "embedding_lr": 0.3,
            "unembedding_lr": 0.004,
            "matrix_lr": 0.02,
            "scalar_lr": 0.5,
            "weight_decay": 0.2,
            "adam_beta1": 0.8,
            "adam_beta2": 0.95,
            "warmup_ratio": 0.1,
            "warmdown_ratio": 0.4,
            "final_lr_frac": 0.05,
            "core_metric_max_per_task": 128,
        },
    )

    assert "--aspect-ratio" in command
    assert "--embedding-lr" in command
    assert "--unembedding-lr" in command
    assert "--matrix-lr" in command
    assert "--scalar-lr" in command
    assert "--weight-decay" in command
    assert "--adam-beta1" in command
    assert "--adam-beta2" in command
    assert "--warmup-ratio" in command
    assert "--warmdown-ratio" in command
    assert "--final-lr-frac" in command
    assert "--core-metric-max-per-task" in command


def test_build_job_command_supports_chat_sft_resume_flags():
    command = build_job_command(
        "chat_sft",
        {
            "model_tag": "demo-sft",
            "resume_from_step": 120,
            "save_every": 40,
        },
    )

    assert "--resume-from-step" in command
    assert "120" in command
    assert "--save-every" in command
    assert "40" in command


def test_build_job_command_supports_chat_sft_chatcore_flag():
    command = build_job_command(
        "chat_sft",
        {
            "model_tag": "demo-sft",
            "chatcore_every": 25,
        },
    )

    assert "--chatcore-every" in command
    assert "25" in command


def test_delete_design_removes_saved_file(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_tools, "DESIGNS_DIR", tmp_path)
    monkeypatch.setattr(dashboard_tools, "ASSISTANT_SANDBOX_DIR", tmp_path / "assistant_sandbox")
    monkeypatch.setattr(dashboard_tools, "LOCAL_CORPUS_DIR", tmp_path / "local_corpus")

    saved = dashboard_tools.save_design(
        {
            "name": "Draft Helper",
            "mission": "Help create a careful drafting assistant.",
            "team_role": "You help the user produce drafts.",
            "tone": "Calm",
            "uncertainty_policy": "Admit uncertainty.",
            "collaboration_policy": "Ask for missing inputs.",
            "guardrails": ["Do not invent facts."],
            "custom_notes": "",
            "recipes": {"tokenizerForm": {"vocab_size": 16384}},
        }
    )

    path = tmp_path / f"{saved['slug']}.json"
    assert path.exists()

    deleted = dashboard_tools.delete_design(saved["slug"])

    assert deleted["deleted"] is True
    assert deleted["slug"] == saved["slug"]
    assert not path.exists()


def test_recommend_forms_for_cpu_only_hardware_is_conservative():
    recommendations = dashboard_tools.recommend_forms_for_hardware(
        {
            "preferred_device": "cpu",
            "tier": "cpu-only",
            "system_ram_gb": 16.0,
            "logical_cpus": 8,
            "physical_cpus": 4,
            "gpus": [],
        }
    )

    assert recommendations["baseTrainForm"]["device_type"] == "cpu"
    assert recommendations["baseTrainForm"]["device_batch_size"] == 1
    assert recommendations["baseTrainForm"]["save_every"] > 0
    assert recommendations["chatSftForm"]["device_batch_size"] == 1
    assert recommendations["chatRlForm"]["device_batch_size"] <= 2
    assert recommendations["runtimeForm"]["ctx_size"] == 4096
    assert recommendations["runtimeForm"]["parallel"] == 1


def test_recommend_forms_for_consumer_mid_hardware_keeps_runtime_generous():
    recommendations = dashboard_tools.recommend_forms_for_hardware(
        {
            "preferred_device": "cuda",
            "tier": "consumer-mid",
            "system_ram_gb": 64.0,
            "logical_cpus": 16,
            "physical_cpus": 8,
            "gpus": [{"memory_gb": 12.0}],
        }
    )

    assert recommendations["runtimeForm"]["ctx_size"] == 8192
    assert recommendations["runtimeForm"]["parallel"] == 2
    assert recommendations["runtimeForm"]["threads"] >= 8


def test_latest_checkpoint_step_for_base_job_reads_last_saved_step(tmp_path, monkeypatch):
    base_dir = tmp_path / "cache"
    checkpoint_dir = base_dir / "base_checkpoints" / "d4"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model_000010.pt").write_bytes(b"demo")
    (checkpoint_dir / "model_000120.pt").write_bytes(b"demo")

    monkeypatch.setattr(dashboard_tools, "get_base_dir", lambda: str(base_dir))

    step = dashboard_tools.latest_checkpoint_step_for_job("base_train", {"depth": 4})

    assert step == 120


def test_latest_checkpoint_step_for_sft_job_reads_last_saved_step(tmp_path, monkeypatch):
    base_dir = tmp_path / "cache"
    checkpoint_dir = base_dir / "chatsft_checkpoints" / "demo-sft"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model_000005.pt").write_bytes(b"demo")
    (checkpoint_dir / "model_000090.pt").write_bytes(b"demo")

    monkeypatch.setattr(dashboard_tools, "get_base_dir", lambda: str(base_dir))

    step = dashboard_tools.latest_checkpoint_step_for_job("chat_sft", {"model_tag": "demo-sft"})

    assert step == 90


def test_latest_checkpoint_step_for_rl_job_reads_last_saved_step(tmp_path, monkeypatch):
    base_dir = tmp_path / "cache"
    checkpoint_dir = base_dir / "chatrl_checkpoints" / "demo-rl"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model_000003.pt").write_bytes(b"demo")
    (checkpoint_dir / "model_000055.pt").write_bytes(b"demo")

    monkeypatch.setattr(dashboard_tools, "get_base_dir", lambda: str(base_dir))

    step = dashboard_tools.latest_checkpoint_step_for_job("chat_rl", {"model_tag": "demo-rl"})

    assert step == 55


def test_job_snapshot_includes_resume_metadata(tmp_path, monkeypatch):
    base_dir = tmp_path / "cache"
    checkpoint_dir = base_dir / "chatsft_checkpoints" / "demo-sft"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model_000040.pt").write_bytes(b"demo")

    monkeypatch.setattr(dashboard_tools, "get_base_dir", lambda: str(base_dir))

    record = dashboard_tools.JobRecord(
        id="job123",
        label="chat sft resume",
        job_type="chat_sft",
        command=["python", "-m", "scripts.chat_sft"],
        created_at=0.0,
        cwd=str(tmp_path),
        params={"model_tag": "demo-sft", "resume_from_step": 32, "save_every": 20},
        status="failed",
    )

    snapshot = record.snapshot(include_logs=False)

    assert snapshot["requested_resume_step"] == 32
    assert snapshot["save_every"] == 20
    assert snapshot["is_resumed_run"] is True
    assert snapshot["latest_checkpoint_step"] == 40
    assert snapshot["can_resume"] is True


def test_job_snapshot_marks_non_resumed_run_without_resume_metadata(tmp_path, monkeypatch):
    base_dir = tmp_path / "cache"
    checkpoint_dir = base_dir / "base_checkpoints" / "d4"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model_000010.pt").write_bytes(b"demo")

    monkeypatch.setattr(dashboard_tools, "get_base_dir", lambda: str(base_dir))

    record = dashboard_tools.JobRecord(
        id="job124",
        label="base train",
        job_type="base_train",
        command=["python", "-m", "scripts.base_train"],
        created_at=0.0,
        cwd=str(tmp_path),
        params={"depth": 4, "save_every": ""},
        status="running",
    )

    snapshot = record.snapshot(include_logs=False)

    assert snapshot["requested_resume_step"] is None
    assert snapshot["save_every"] is None
    assert snapshot["is_resumed_run"] is False
    assert snapshot["latest_checkpoint_step"] == 10


def test_diagnose_job_failure_reports_missing_pyarrow():
    diagnosis = dashboard_tools.diagnose_job_failure(
        "base_train",
        "failed",
        ["python", "-m", "scripts.base_train", "--corpus-dir", "local_corpus"],
        [
            "Traceback (most recent call last):",
            "ModuleNotFoundError: No module named 'pyarrow'",
        ],
    )

    assert diagnosis is not None
    assert diagnosis["kind"] == "missing_pyarrow"
    assert "Parquet" in diagnosis["title"]
    assert diagnosis["evidence"]


def test_diagnose_job_failure_reports_missing_file_path():
    diagnosis = dashboard_tools.diagnose_job_failure(
        "chat_sft",
        "failed",
        ["python", "-m", "scripts.chat_sft", "--train-files", "assistant_sandbox/chat_train.jsonl"],
        ["FileNotFoundError: [Errno 2] No such file or directory: 'assistant_sandbox/chat_train.jsonl'"],
    )

    assert diagnosis is not None
    assert diagnosis["kind"] == "missing_file_or_path"
    assert "file or path" in diagnosis["title"].lower()


def test_diagnose_job_failure_returns_none_for_running_job():
    diagnosis = dashboard_tools.diagnose_job_failure(
        "base_train",
        "running",
        ["python", "-m", "scripts.base_train"],
        ["step 1"],
    )

    assert diagnosis is None


def test_job_snapshot_includes_failure_diagnosis(tmp_path):
    record = dashboard_tools.JobRecord(
        id="job126",
        label="base train failed",
        job_type="base_train",
        command=["python", "-m", "scripts.base_train"],
        created_at=0.0,
        cwd=str(tmp_path),
        status="failed",
        exit_code=1,
    )
    record.log_lines.append("RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB")

    snapshot = record.snapshot(include_logs=False)

    assert snapshot["failure_diagnosis"]["kind"] == "out_of_memory"
    assert snapshot["failure_diagnosis"]["evidence"]


def test_background_job_manager_previews_resume_command(tmp_path, monkeypatch):
    base_dir = tmp_path / "cache"
    checkpoint_dir = base_dir / "base_checkpoints" / "d4"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model_000010.pt").write_bytes(b"demo")

    monkeypatch.setattr(dashboard_tools, "get_base_dir", lambda: str(base_dir))
    manager = dashboard_tools.BackgroundJobManager(workdir=str(tmp_path))
    record = dashboard_tools.JobRecord(
        id="job125",
        label="base train",
        job_type="base_train",
        command=["python", "-m", "scripts.base_train"],
        created_at=0.0,
        cwd=str(tmp_path),
        params={"depth": 4, "save_every": ""},
        status="paused",
    )
    manager._jobs[record.id] = record

    preview = manager.preview_resume_job(record.id)

    assert preview["source_job"]["id"] == "job125"
    assert preview["resume"]["checkpoint_step"] == 10
    assert preview["resume"]["params"]["resume_from_step"] == 10
    assert preview["resume"]["params"]["save_every"] == 100
    assert "--resume-from-step" in preview["resume"]["command"]
    assert "10" in preview["resume"]["command"]
    diff = preview["resume"]["command_diff"]
    assert any(item["flag"] == "--resume-from-step" and item["value"] == "10" for item in diff["added"])
    assert any(item["flag"] == "--save-every" and item["value"] == "100" for item in diff["added"])


def test_background_job_manager_persists_and_reloads_finished_jobs(tmp_path):
    jobs_path = tmp_path / "builder_logs" / "jobs.json"
    manager = dashboard_tools.BackgroundJobManager(workdir=str(tmp_path), jobs_path=jobs_path)
    record = dashboard_tools.JobRecord(
        id="job-persisted",
        label="failed parquet run",
        job_type="base_train",
        command=["python", "-m", "scripts.base_train"],
        created_at=10.0,
        cwd=str(tmp_path),
        status="failed",
        exit_code=1,
    )
    record.log_lines.append("ModuleNotFoundError: No module named 'pyarrow'")
    manager._jobs[record.id] = record
    manager._persist_jobs()

    reloaded = dashboard_tools.BackgroundJobManager(workdir=str(tmp_path), jobs_path=jobs_path)
    snapshot = reloaded.get_job("job-persisted", include_logs=True)

    assert snapshot["status"] == "failed"
    assert snapshot["logs"] == ["ModuleNotFoundError: No module named 'pyarrow'"]
    assert snapshot["failure_diagnosis"]["kind"] == "missing_pyarrow"


def test_background_job_manager_marks_reloaded_running_job_stopped(tmp_path):
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text(
        """
        {
          "version": 1,
          "jobs": [
            {
              "id": "job-running",
              "label": "interrupted run",
              "job_type": "base_train",
              "command": ["python", "-m", "scripts.base_train"],
              "created_at": 11.0,
              "cwd": ".",
              "status": "running",
              "logs": ["step 12"]
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    manager = dashboard_tools.BackgroundJobManager(workdir=str(tmp_path), jobs_path=jobs_path)
    snapshot = manager.get_job("job-running", include_logs=True)

    assert snapshot["status"] == "stopped"
    assert snapshot["pid"] is None
    assert any("could not be reattached" in line for line in snapshot["logs"])


def test_background_job_manager_prunes_persisted_job_history(tmp_path):
    jobs_path = tmp_path / "jobs.json"
    manager = dashboard_tools.BackgroundJobManager(workdir=str(tmp_path), jobs_path=jobs_path, max_persisted_jobs=2)
    for index in range(4):
        record = dashboard_tools.JobRecord(
            id=f"job-{index}",
            label=f"job {index}",
            job_type="base_eval",
            command=["python", "-m", "scripts.base_eval"],
            created_at=float(index),
            cwd=str(tmp_path),
            status="completed",
            exit_code=0,
        )
        manager._jobs[record.id] = record

    manager._persist_jobs()
    reloaded = dashboard_tools.BackgroundJobManager(workdir=str(tmp_path), jobs_path=jobs_path, max_persisted_jobs=2)
    jobs = reloaded.list_jobs()

    assert [job["id"] for job in jobs] == ["job-3", "job-2"]


def test_background_job_manager_retains_active_jobs_over_history_limit(tmp_path):
    manager = dashboard_tools.BackgroundJobManager(workdir=str(tmp_path), jobs_path=tmp_path / "jobs.json", max_persisted_jobs=1)
    old_completed = dashboard_tools.JobRecord(
        id="old-completed",
        label="old completed",
        job_type="base_eval",
        command=["python", "-m", "scripts.base_eval"],
        created_at=10.0,
        cwd=str(tmp_path),
        status="completed",
    )
    active = dashboard_tools.JobRecord(
        id="active-running",
        label="active running",
        job_type="base_train",
        command=["python", "-m", "scripts.base_train"],
        created_at=1.0,
        cwd=str(tmp_path),
        status="running",
    )
    manager._jobs[old_completed.id] = old_completed
    manager._jobs[active.id] = active

    manager._persist_jobs()

    assert [job["id"] for job in manager.list_jobs()] == ["active-running"]


def test_command_diff_summary_reports_added_changed_removed_flags():
    diff = dashboard_tools.command_diff_summary(
        ["python", "-m", "scripts.base_train", "--depth", "4", "--fp8", "--save-every", "50"],
        ["python", "-m", "scripts.base_train", "--depth", "6", "--resume-from-step", "10"],
    )

    assert {"flag": "--resume-from-step", "value": "10"} in diff["added"]
    assert {"flag": "--depth", "from": "4", "to": "6"} in diff["changed"]
    assert {"flag": "--fp8", "value": "true"} in diff["removed"]
    assert {"flag": "--save-every", "value": "50"} in diff["removed"]
