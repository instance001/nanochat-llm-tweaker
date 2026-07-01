from nanochat import dashboard_tools
from nanochat.dashboard_tools import build_job_command


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
