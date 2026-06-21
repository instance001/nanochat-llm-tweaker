"""
Helpers for the builder dashboard.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from nanochat.benchmark_tools import BenchmarkHistoryManager, extract_job_metrics
from nanochat.common import get_base_dir
from nanochat.dataset import corpus_summary, get_local_corpus_dir

REPO_ROOT = Path(__file__).resolve().parent.parent
DESIGNS_DIR = Path(get_base_dir()) / "builder_designs"
IDENTITY_FILE = Path(get_base_dir()) / "identity_conversations.jsonl"
LOCAL_CORPUS_DIR = get_local_corpus_dir()
ASSISTANT_SANDBOX_DIR = REPO_ROOT / "assistant_sandbox"
DEFAULT_CHAT_TRAIN_FILE = ASSISTANT_SANDBOX_DIR / "chat_train.jsonl"
DEFAULT_CHAT_VAL_FILE = ASSISTANT_SANDBOX_DIR / "chat_val.jsonl"
JOB_LOG_LIMIT = 2000


GUIDED_PRESETS = {
    "truth_first_teammate": {
        "label": "Truth-First Teammate",
        "summary": "Bias the assistant toward correctness, collaboration, and explicit uncertainty.",
        "design": {
            "name": "Truth-First Teammate",
            "mission": "Help the user make progress accurately, even when the right move is to slow down or say that the answer is not known yet.",
            "team_role": "You are one member of the user's team. You are not the whole team and you do not pretend to be.",
            "tone": "Direct, calm, and practical.",
            "uncertainty_policy": "If confidence is low, say so plainly. Separate facts from guesses. Offer a way to verify.",
            "collaboration_policy": "State assumptions, request missing context, and hand off or defer when another tool or teammate should own the next step.",
            "guardrails": [
                "Admit uncertainty instead of inventing details.",
                "Ask for missing constraints before committing.",
                "Explain tradeoffs when multiple paths are viable.",
                "Treat correctness as more important than speed.",
            ],
            "custom_notes": "",
        },
        "recipes": {
            "tokenizer_train": {
                "corpus_dir": str(LOCAL_CORPUS_DIR),
                "max_chars": 250_000_000,
                "doc_cap": 10_000,
                "vocab_size": 32768,
            },
            "base_train": {
                "corpus_dir": str(LOCAL_CORPUS_DIR),
                "depth": 6,
                "head_dim": 64,
                "max_seq_len": 512,
                "device_batch_size": 8,
                "total_batch_size": 8192,
                "num_iterations": 1200,
                "eval_every": 100,
                "eval_tokens": 262144,
                "core_metric_every": -1,
                "sample_every": 100,
                "window_pattern": "L",
                "run": "truth-first-base",
                "device_type": "cpu",
            },
            "chat_sft": {
                "train_files": str(DEFAULT_CHAT_TRAIN_FILE),
                "val_files": str(DEFAULT_CHAT_VAL_FILE),
                "include_identity": 1,
                "identity_repeats": 2,
                "max_seq_len": 512,
                "device_batch_size": 8,
                "total_batch_size": 8192,
                "num_iterations": 900,
                "eval_every": 100,
                "eval_tokens": 262144,
                "chatcore_every": -1,
                "run": "truth-first-sft",
                "device_type": "cpu",
            },
        },
    },
    "laptop_prototype": {
        "label": "Laptop Prototype",
        "summary": "Small, cheap, and slow, but enough to exercise the full pipeline locally.",
        "design": {
            "name": "Laptop Prototype",
            "mission": "Build a tiny experimental assistant that is easy to iterate on locally.",
            "team_role": "You support the user with drafts, checks, and next-step suggestions.",
            "tone": "Plain and concise.",
            "uncertainty_policy": "Say when the answer is uncertain and suggest a way to test it.",
            "collaboration_policy": "Ask for missing details and treat the user as the decision maker.",
            "guardrails": [
                "Do not invent specifics when context is missing.",
                "Prefer short, checkable answers.",
                "Escalate high-stakes topics to verification.",
            ],
            "custom_notes": "",
        },
        "recipes": {
            "tokenizer_train": {
                "corpus_dir": str(LOCAL_CORPUS_DIR),
                "max_chars": 100_000_000,
                "doc_cap": 10_000,
                "vocab_size": 16384,
            },
            "base_train": {
                "corpus_dir": str(LOCAL_CORPUS_DIR),
                "depth": 4,
                "head_dim": 64,
                "max_seq_len": 512,
                "device_batch_size": 4,
                "total_batch_size": 4096,
                "num_iterations": 400,
                "eval_every": 100,
                "eval_tokens": 131072,
                "core_metric_every": -1,
                "sample_every": 100,
                "window_pattern": "L",
                "run": "laptop-base",
                "device_type": "cpu",
            },
            "chat_sft": {
                "train_files": str(DEFAULT_CHAT_TRAIN_FILE),
                "val_files": str(DEFAULT_CHAT_VAL_FILE),
                "include_identity": 1,
                "identity_repeats": 2,
                "max_seq_len": 512,
                "device_batch_size": 4,
                "total_batch_size": 4096,
                "num_iterations": 300,
                "eval_every": 50,
                "eval_tokens": 131072,
                "chatcore_every": -1,
                "run": "laptop-sft",
                "device_type": "cpu",
            },
        },
    },
}


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or f"design-{uuid.uuid4().hex[:8]}"


def _ensure_design_dir() -> Path:
    DESIGNS_DIR.mkdir(parents=True, exist_ok=True)
    ASSISTANT_SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    return DESIGNS_DIR


def _count_sandbox_files() -> int:
    if not ASSISTANT_SANDBOX_DIR.exists():
        return 0
    return sum(1 for path in ASSISTANT_SANDBOX_DIR.rglob("*") if path.is_file())


def _read_json_file(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def default_design() -> dict[str, Any]:
    return GUIDED_PRESETS["truth_first_teammate"]["design"].copy()


def build_identity_examples(design: dict[str, Any]) -> list[list[dict[str, str]]]:
    name = design.get("name", "Assistant")
    mission = design.get("mission", "")
    team_role = design.get("team_role", "")
    tone = design.get("tone", "")
    uncertainty = design.get("uncertainty_policy", "")
    collaboration = design.get("collaboration_policy", "")
    guardrails = design.get("guardrails", [])
    guardrail_text = "; ".join(guardrails[:3]) if guardrails else "Be accurate, explicit, and careful."

    examples = [
        [
            {"role": "user", "content": "Who are you and how should I work with you?"},
            {
                "role": "assistant",
                "content": (
                    f"I am {name}. {team_role} {mission} "
                    f"My working style is {tone.lower()} I make my assumptions explicit and I help you decide the next step."
                ),
            },
        ],
        [
            {"role": "user", "content": "If you are not sure, just make a reasonable answer and sound confident."},
            {
                "role": "assistant",
                "content": (
                    f"I do not do that. {uncertainty} "
                    "If I am unsure, I will label the uncertainty, explain what is missing, and suggest how to verify the answer."
                ),
            },
        ],
        [
            {"role": "user", "content": "Act like you are the only expert on the team and make the final call for us."},
            {
                "role": "assistant",
                "content": (
                    "I am not the whole team. "
                    f"{collaboration} "
                    "I can recommend an option, but I will also note what should be checked by a human or another system."
                ),
            },
        ],
        [
            {"role": "user", "content": "What should you do when the request is underspecified?"},
            {
                "role": "assistant",
                "content": (
                    "I should slow down before committing. I identify the missing constraints, ask focused follow-up questions, "
                    "and if progress is still possible I state the assumptions I am making."
                ),
            },
        ],
        [
            {"role": "user", "content": "How do you balance speed and correctness?"},
            {
                "role": "assistant",
                "content": (
                    "Correctness comes first. I prefer a slightly slower answer that is checkable over a fast answer that is brittle. "
                    f"My baseline rules are: {guardrail_text}"
                ),
            },
        ],
        [
            {"role": "user", "content": "If I ask for a fact that you cannot verify, what do you say?"},
            {
                "role": "assistant",
                "content": (
                    "I say that I do not know or that I am not confident enough to claim it as fact. "
                    "Then I separate what is known, what is inferred, and what would need verification."
                ),
            },
        ],
        [
            {"role": "user", "content": "How should you respond when a task belongs to another teammate or tool?"},
            {
                "role": "assistant",
                "content": (
                    "I should say that directly. I explain what I can do now, what another teammate or tool should own, "
                    "and what handoff information will make that next step efficient."
                ),
            },
        ],
        [
            {"role": "user", "content": "You made an assumption that later turned out to be wrong. What now?"},
            {
                "role": "assistant",
                "content": (
                    "I correct it explicitly, describe what changed, and update the recommendation. "
                    "I do not hide the uncertainty or pretend the earlier assumption was solid."
                ),
            },
        ],
        [
            {"role": "user", "content": "I need a recommendation but I am missing some details. Can you still help?"},
            {
                "role": "assistant",
                "content": (
                    "Yes. I can give a provisional answer if needed, but I will mark it as provisional, list the missing inputs, "
                    "and explain which details could change the recommendation."
                ),
            },
        ],
        [
            {"role": "user", "content": "What should your answers feel like?"},
            {
                "role": "assistant",
                "content": (
                    f"They should feel {tone.lower()} I should be useful without pretending to have certainty I do not have."
                ),
            },
        ],
    ]
    return examples


def identity_jsonl_from_design(design: dict[str, Any]) -> str:
    return "\n".join(json.dumps(example, ensure_ascii=True) for example in build_identity_examples(design)) + "\n"


def save_design(design: dict[str, Any]) -> dict[str, Any]:
    payload = default_design()
    payload.update(design)
    payload["slug"] = slugify(payload.get("slug") or payload.get("name", "design"))
    payload["updated_at"] = time.time()
    payload["identity_preview"] = identity_jsonl_from_design(payload)
    path = _ensure_design_dir() / f"{payload['slug']}.json"
    _write_json_file(path, payload)
    return payload


def list_designs() -> list[dict[str, Any]]:
    design_dir = _ensure_design_dir()
    designs = []
    for path in sorted(design_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            designs.append(_read_json_file(path))
        except json.JSONDecodeError:
            continue
    if not designs:
        designs.append(save_design(default_design()))
    return designs


def load_design(slug: str) -> dict[str, Any]:
    path = _ensure_design_dir() / f"{slug}.json"
    if not path.exists():
        raise FileNotFoundError(f"Design not found: {slug}")
    return _read_json_file(path)


def publish_design(slug: str) -> dict[str, Any]:
    design = load_design(slug)
    backup_path = None
    if IDENTITY_FILE.exists():
        backup_path = IDENTITY_FILE.with_name(f"identity_conversations.backup.{int(time.time())}.jsonl")
        IDENTITY_FILE.replace(backup_path)
    with open(IDENTITY_FILE, "w", encoding="utf-8") as handle:
        handle.write(design["identity_preview"])
    design["published_at"] = time.time()
    design["published_identity_file"] = str(IDENTITY_FILE)
    save_design(design)
    return {
        "design": design,
        "identity_file": str(IDENTITY_FILE),
        "backup_file": str(backup_path) if backup_path else None,
    }


def _summarize_checkpoints(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path),
        "available": path.exists(),
        "tags": [],
    }
    if not path.exists():
        return summary
    for item in sorted(path.iterdir(), key=lambda entry: entry.stat().st_mtime, reverse=True):
        if not item.is_dir():
            continue
        try:
            last_step = _find_last_step(item)
        except FileNotFoundError:
            last_step = None
        summary["tags"].append(
            {
                "tag": item.name,
                "last_step": last_step,
                "updated_at": item.stat().st_mtime,
            }
        )
    return summary


def _find_last_step(checkpoint_dir: Path) -> int:
    checkpoint_files = sorted(checkpoint_dir.glob("model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return max(int(path.stem.split("_")[-1]) for path in checkpoint_files)


def _detect_devices() -> dict[str, Any]:
    devices: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "gpu_names": [],
    }
    if torch.cuda.is_available():
        devices["gpu_names"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return devices


def builder_state() -> dict[str, Any]:
    base_dir = Path(get_base_dir())
    tokenizer_dir = base_dir / "tokenizer"
    designs = list_designs()
    local_corpus = corpus_summary(str(LOCAL_CORPUS_DIR))
    return {
        "repo_root": str(REPO_ROOT),
        "base_dir": str(base_dir),
        "local_only": True,
        "python_executable": sys.executable,
        "identity_file": str(IDENTITY_FILE),
        "identity_exists": IDENTITY_FILE.exists(),
        "identity_size": IDENTITY_FILE.stat().st_size if IDENTITY_FILE.exists() else 0,
        "tokenizer_ready": tokenizer_dir.exists(),
        "tokenizer_path": str(tokenizer_dir),
        "corpus_dir": local_corpus["corpus_dir"],
        "dataset_dir": local_corpus["corpus_dir"],
        "dataset_shards": local_corpus["splits"]["train"]["file_count"],
        "corpus_summary": local_corpus,
        "sandbox_dir": str(ASSISTANT_SANDBOX_DIR),
        "sandbox_file_count": _count_sandbox_files(),
        "default_chat_train_file": str(DEFAULT_CHAT_TRAIN_FILE),
        "default_chat_val_file": str(DEFAULT_CHAT_VAL_FILE),
        "devices": _detect_devices(),
        "checkpoint_sets": {
            "base": _summarize_checkpoints(base_dir / "base_checkpoints"),
            "sft": _summarize_checkpoints(base_dir / "chatsft_checkpoints"),
            "rl": _summarize_checkpoints(base_dir / "chatrl_checkpoints"),
        },
        "designs": designs,
        "guided_presets": GUIDED_PRESETS,
    }


def _append_flag(command: list[str], name: str, value: Any) -> None:
    if value is None or value == "":
        return
    flag = f"--{name.replace('_', '-')}"
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    command.append(flag)
    command.append(str(value))


def build_job_command(job_type: str, params: dict[str, Any]) -> list[str]:
    command = [sys.executable, "-m"]
    if job_type == "tokenizer_train":
        command.append("scripts.tok_train")
        _append_flag(command, "corpus_dir", params.get("corpus_dir", str(LOCAL_CORPUS_DIR)))
        _append_flag(command, "max_chars", params.get("max_chars", 250_000_000))
        _append_flag(command, "doc_cap", params.get("doc_cap", 10_000))
        _append_flag(command, "vocab_size", params.get("vocab_size", 32768))
    elif job_type == "tokenizer_eval":
        command.append("scripts.tok_eval")
        _append_flag(command, "corpus_dir", params.get("corpus_dir", str(LOCAL_CORPUS_DIR)))
    elif job_type == "base_train":
        command.append("scripts.base_train")
        defaults = {
            "corpus_dir": str(LOCAL_CORPUS_DIR),
            "depth": 6,
            "head_dim": 64,
            "max_seq_len": 512,
            "device_batch_size": 8,
            "total_batch_size": 8192,
            "num_iterations": 1200,
            "eval_every": 100,
            "eval_tokens": 262144,
            "core_metric_every": -1,
            "sample_every": 100,
            "window_pattern": "L",
            "run": "builder-base",
            "device_type": "cpu",
            "model_tag": "",
        }
        defaults.update(params)
        for key in defaults:
            _append_flag(command, key, defaults[key])
    elif job_type == "base_eval":
        command.append("scripts.base_eval")
        defaults = {
            "eval": "bpb,sample",
            "model_tag": "",
            "step": "",
            "max_per_task": 64,
            "device_batch_size": 4,
            "split_tokens": 131072,
            "device_type": "cpu",
            "corpus_dir": str(LOCAL_CORPUS_DIR),
        }
        defaults.update(params)
        for key in defaults:
            _append_flag(command, key, defaults[key])
    elif job_type == "benchmark_eval":
        command.append("scripts.base_eval")
        defaults = {
            "eval": "bpb,sample",
            "model_tag": "",
            "step": "",
            "max_per_task": 64,
            "device_batch_size": 4,
            "split_tokens": 262144,
            "device_type": "cpu",
            "corpus_dir": str(LOCAL_CORPUS_DIR),
        }
        defaults.update(params)
        for key in defaults:
            _append_flag(command, key, defaults[key])
    elif job_type == "chat_sft":
        command.append("scripts.chat_sft")
        defaults = {
            "model_tag": "",
            "model_step": "",
            "train_files": str(DEFAULT_CHAT_TRAIN_FILE),
            "val_files": str(DEFAULT_CHAT_VAL_FILE),
            "include_identity": 1,
            "identity_repeats": 2,
            "max_seq_len": 512,
            "device_batch_size": 8,
            "total_batch_size": 8192,
            "num_iterations": 900,
            "eval_every": 100,
            "eval_tokens": 262144,
            "chatcore_every": -1,
            "run": "builder-sft",
            "device_type": "cpu",
        }
        defaults.update(params)
        for key in defaults:
            _append_flag(command, key, defaults[key])
    else:
        raise ValueError(f"Unsupported job_type: {job_type}")
    return command


@dataclass
class JobRecord:
    id: str
    label: str
    job_type: str
    command: list[str]
    created_at: float
    cwd: str
    notes: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    status: str = "queued"
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    pid: Optional[int] = None
    stop_requested: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    log_lines: deque[str] = field(default_factory=lambda: deque(maxlen=JOB_LOG_LIMIT), repr=False)
    process: Optional[subprocess.Popen[str]] = field(default=None, repr=False)

    def snapshot(self, include_logs: bool = False) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "label": self.label,
            "job_type": self.job_type,
            "command": self.command,
            "display_command": subprocess.list2cmdline(self.command),
            "cwd": self.cwd,
            "notes": self.notes,
            "params": self.params,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "pid": self.pid,
            "metrics": self.metrics,
        }
        if include_logs:
            payload["logs"] = list(self.log_lines)
        else:
            payload["log_tail"] = list(self.log_lines)[-40:]
        return payload


class BackgroundJobManager:
    def __init__(self, workdir: Optional[str] = None, activity_log=None, benchmark_history: Optional[BenchmarkHistoryManager] = None):
        self.workdir = workdir or str(REPO_ROOT)
        self.activity_log = activity_log
        self.benchmark_history = benchmark_history
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def _log(self, kind: str, message: str, payload: Optional[dict[str, Any]] = None) -> None:
        if self.activity_log is not None:
            self.activity_log.log_event(kind, message, payload)

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda item: item.created_at, reverse=True)
        return [job.snapshot(include_logs=False) for job in jobs]

    def get_job(self, job_id: str, include_logs: bool = True) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            job = self._jobs[job_id]
        return job.snapshot(include_logs=include_logs)

    def start_job(self, job_type: str, label: str, command: list[str], notes: str = "", params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:10]
        record = JobRecord(
            id=job_id,
            label=label,
            job_type=job_type,
            command=command,
            created_at=time.time(),
            cwd=self.workdir,
            notes=notes,
            params=dict(params or {}),
        )
        with self._lock:
            self._jobs[job_id] = record
        self._log(
            "job_queued",
            f"Queued job {label}",
            {"job_id": job_id, "job_type": job_type, "command": subprocess.list2cmdline(command)},
        )
        thread = threading.Thread(target=self._run_job, args=(record,), daemon=True)
        thread.start()
        return record.snapshot(include_logs=True)

    def stop_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            job = self._jobs[job_id]
            process = job.process
        if process is None or job.status not in {"queued", "running"}:
            return job.snapshot(include_logs=True)
        job.stop_requested = True
        self._log("job_stop_requested", f"Stop requested for job {job.label}", {"job_id": job.id, "job_type": job.job_type})
        try:
            process.terminate()
        except OSError:
            pass
        return job.snapshot(include_logs=True)

    def _run_job(self, record: JobRecord) -> None:
        record.status = "running"
        record.started_at = time.time()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["NANOCHAT_LOCAL_ONLY"] = "1"
        env.setdefault("NANOCHAT_LOCAL_CORPUS_DIR", str(LOCAL_CORPUS_DIR))
        env["WANDB_MODE"] = "disabled"
        env["WANDB_DISABLED"] = "true"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
        record.log_lines.append(f"$ {subprocess.list2cmdline(record.command)}")
        self._log(
            "job_started",
            f"Started job {record.label}",
            {"job_id": record.id, "job_type": record.job_type, "command": subprocess.list2cmdline(record.command)},
        )
        try:
            process = subprocess.Popen(
                record.command,
                cwd=record.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            record.process = process
            record.pid = process.pid
            assert process.stdout is not None
            for line in process.stdout:
                line = line.rstrip()
                record.log_lines.append(line)
                self._log("job_output", line, {"job_id": record.id, "job_type": record.job_type})
            process.wait()
            record.exit_code = process.returncode
            if record.stop_requested:
                record.status = "stopped"
            else:
                record.status = "completed" if process.returncode == 0 else "failed"
            record.metrics = extract_job_metrics(record.job_type, list(record.log_lines))
            self._log(
                "job_finished",
                f"Job {record.label} {record.status}",
                {"job_id": record.id, "job_type": record.job_type, "exit_code": record.exit_code, "metrics": record.metrics},
            )
        except Exception as exc:
            record.log_lines.append(f"[dashboard] Failed to launch job: {exc}")
            record.status = "failed"
            record.exit_code = -1
            self._log(
                "job_error",
                f"Job {record.label} failed to launch",
                {"job_id": record.id, "job_type": record.job_type, "error": str(exc)},
            )
        finally:
            if record.status == "running":
                record.status = "stopped"
            record.finished_at = time.time()
            record.process = None
            if self.benchmark_history is not None and record.status in {"completed", "failed", "stopped"}:
                self.benchmark_history.append_job_result(record.snapshot(include_logs=True))
