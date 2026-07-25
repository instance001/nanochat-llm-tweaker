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
from copy import deepcopy
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
try:
    import psutil
except ImportError:
    psutil = None
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
JOB_HISTORY_LIMIT = 200
JOBS_FILE = REPO_ROOT / "builder_logs" / "jobs.json"


DASHBOARD_JOB_SCRIPT_MODULES = {
    "tokenizer_train": "scripts.tok_train",
    "tokenizer_eval": "scripts.tok_eval",
    "base_train": "scripts.base_train",
    "base_eval": "scripts.base_eval",
    "benchmark_eval": "scripts.base_eval",
    "chat_sft": "scripts.chat_sft",
    "chat_rl": "scripts.chat_rl",
    "chat_eval": "scripts.chat_eval",
}

DASHBOARD_JOB_PARAM_NAMES = {
    "tokenizer_train": {"corpus_dir", "max_chars", "doc_cap", "vocab_size"},
    "tokenizer_eval": {"corpus_dir", "max_corpus_docs"},
    "base_train": {
        "run",
        "device_type",
        "fp8",
        "fp8_recipe",
        "depth",
        "aspect_ratio",
        "head_dim",
        "max_seq_len",
        "window_pattern",
        "num_iterations",
        "target_flops",
        "target_param_data_ratio",
        "device_batch_size",
        "total_batch_size",
        "embedding_lr",
        "unembedding_lr",
        "weight_decay",
        "matrix_lr",
        "scalar_lr",
        "adam_beta1",
        "adam_beta2",
        "warmup_ratio",
        "warmdown_ratio",
        "final_lr_frac",
        "resume_from_step",
        "eval_every",
        "eval_tokens",
        "core_metric_every",
        "core_metric_max_per_task",
        "sample_every",
        "save_every",
        "model_tag",
        "corpus_dir",
    },
    "base_eval": {"eval", "model_tag", "step", "max_per_task", "device_batch_size", "split_tokens", "device_type", "corpus_dir"},
    "benchmark_eval": {"eval", "model_tag", "step", "max_per_task", "device_batch_size", "split_tokens", "device_type", "corpus_dir"},
    "chat_sft": {
        "run",
        "device_type",
        "model_tag",
        "model_step",
        "load_optimizer",
        "resume_from_step",
        "num_iterations",
        "max_seq_len",
        "device_batch_size",
        "total_batch_size",
        "embedding_lr",
        "unembedding_lr",
        "matrix_lr",
        "init_lr_frac",
        "warmup_ratio",
        "warmdown_ratio",
        "final_lr_frac",
        "eval_every",
        "eval_tokens",
        "chatcore_every",
        "save_every",
        "train_files",
        "val_files",
        "identity_file",
        "include_identity",
        "identity_repeats",
    },
    "chat_rl": {
        "run",
        "device_type",
        "model_tag",
        "model_step",
        "resume_from_step",
        "num_epochs",
        "device_batch_size",
        "examples_per_step",
        "num_samples",
        "max_new_tokens",
        "temperature",
        "top_k",
        "embedding_lr",
        "unembedding_lr",
        "matrix_lr",
        "weight_decay",
        "init_lr_frac",
        "eval_every",
        "eval_examples",
        "save_every",
    },
    "chat_eval": {"source", "task_name", "temperature", "max_new_tokens", "num_samples", "top_k", "batch_size", "model_tag", "step", "max_problems", "device_type"},
}


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


RUN_PROFILES = {
    "tiny_smoke": {
        "label": "Tiny Smoke Test",
        "summary": "Fastest possible local pipeline check with tiny iteration counts.",
        "forms": {
            "tokenizerForm": {
                "corpus_dir": str(LOCAL_CORPUS_DIR),
                "max_chars": 10_000_000,
                "doc_cap": 1_000,
                "vocab_size": 8192,
            },
            "baseTrainForm": {
                "corpus_dir": str(LOCAL_CORPUS_DIR),
                "depth": 2,
                "head_dim": 64,
                "max_seq_len": 256,
                "device_batch_size": 2,
                "total_batch_size": 1024,
                "num_iterations": 40,
                "eval_every": 20,
                "eval_tokens": 32768,
                "sample_every": 20,
                "save_every": 20,
                "run": "tiny-smoke-base",
                "device_type": "cpu",
            },
            "chatSftForm": {
                "train_files": str(DEFAULT_CHAT_TRAIN_FILE),
                "val_files": str(DEFAULT_CHAT_VAL_FILE),
                "include_identity": 1,
                "identity_repeats": 1,
                "max_seq_len": 256,
                "device_batch_size": 2,
                "total_batch_size": 1024,
                "num_iterations": 40,
                "eval_every": 20,
                "eval_tokens": 32768,
                "save_every": 20,
                "run": "tiny-smoke-sft",
                "device_type": "cpu",
            },
            "baseEvalForm": {
                "corpus_dir": str(LOCAL_CORPUS_DIR),
                "eval": "bpb,sample",
                "device_batch_size": 2,
                "split_tokens": 32768,
                "max_per_task": 16,
                "device_type": "cpu",
            },
            "chatRlForm": {
                "num_epochs": 1,
                "device_batch_size": 2,
                "total_batch_size": 1024,
                "save_every": 20,
                "device_type": "cpu",
            },
            "chatEvalForm": {
                "source": "sft",
                "task_name": "GSM8K",
                "limit": 16,
                "device_type": "cpu",
            },
        },
    },
    "laptop_overnight": {
        "label": "Laptop Overnight",
        "summary": "Conservative CPU-oriented settings for a longer unattended local run.",
        "forms": {
            "tokenizerForm": {
                "corpus_dir": str(LOCAL_CORPUS_DIR),
                "max_chars": 100_000_000,
                "doc_cap": 10_000,
                "vocab_size": 16384,
            },
            "baseTrainForm": {
                "corpus_dir": str(LOCAL_CORPUS_DIR),
                "depth": 4,
                "head_dim": 64,
                "max_seq_len": 512,
                "device_batch_size": 4,
                "total_batch_size": 4096,
                "num_iterations": 800,
                "eval_every": 100,
                "eval_tokens": 131072,
                "sample_every": 100,
                "save_every": 100,
                "run": "laptop-overnight-base",
                "device_type": "cpu",
            },
            "chatSftForm": {
                "train_files": str(DEFAULT_CHAT_TRAIN_FILE),
                "val_files": str(DEFAULT_CHAT_VAL_FILE),
                "include_identity": 1,
                "identity_repeats": 2,
                "max_seq_len": 512,
                "device_batch_size": 4,
                "total_batch_size": 4096,
                "num_iterations": 600,
                "eval_every": 100,
                "eval_tokens": 131072,
                "save_every": 100,
                "run": "laptop-overnight-sft",
                "device_type": "cpu",
            },
            "baseEvalForm": {
                "corpus_dir": str(LOCAL_CORPUS_DIR),
                "eval": "bpb,sample",
                "device_batch_size": 4,
                "split_tokens": 131072,
                "max_per_task": 64,
                "device_type": "cpu",
            },
        },
    },
    "gpu_prototype": {
        "label": "GPU Prototype",
        "summary": "Moderate starter settings for a consumer GPU prototype run.",
        "forms": {
            "tokenizerForm": {
                "corpus_dir": str(LOCAL_CORPUS_DIR),
                "max_chars": 250_000_000,
                "doc_cap": 25_000,
                "vocab_size": 32768,
            },
            "baseTrainForm": {
                "corpus_dir": str(LOCAL_CORPUS_DIR),
                "depth": 6,
                "head_dim": 64,
                "max_seq_len": 768,
                "device_batch_size": 8,
                "total_batch_size": 8192,
                "num_iterations": 1200,
                "eval_every": 100,
                "eval_tokens": 262144,
                "sample_every": 100,
                "save_every": 100,
                "run": "gpu-prototype-base",
                "device_type": "cuda",
            },
            "chatSftForm": {
                "train_files": str(DEFAULT_CHAT_TRAIN_FILE),
                "val_files": str(DEFAULT_CHAT_VAL_FILE),
                "include_identity": 1,
                "identity_repeats": 2,
                "max_seq_len": 768,
                "device_batch_size": 8,
                "total_batch_size": 8192,
                "num_iterations": 900,
                "eval_every": 100,
                "eval_tokens": 262144,
                "save_every": 100,
                "run": "gpu-prototype-sft",
                "device_type": "cuda",
            },
            "runtimeForm": {
                "ctx_size": 8192,
                "threads": 8,
                "threads_http": 4,
                "parallel": 2,
                "device_strategy": "gpu",
                "gpu_layers": "auto",
            },
        },
    },
    "serious_run": {
        "label": "Serious Run",
        "summary": "Larger local settings for when corpus and hardware are ready.",
        "forms": {
            "tokenizerForm": {
                "corpus_dir": str(LOCAL_CORPUS_DIR),
                "max_chars": 1_000_000_000,
                "doc_cap": 100_000,
                "vocab_size": 32768,
            },
            "baseTrainForm": {
                "corpus_dir": str(LOCAL_CORPUS_DIR),
                "depth": 8,
                "head_dim": 96,
                "max_seq_len": 1024,
                "device_batch_size": 8,
                "total_batch_size": 16384,
                "num_iterations": 5000,
                "eval_every": 250,
                "eval_tokens": 524288,
                "sample_every": 250,
                "save_every": 250,
                "run": "serious-base",
                "device_type": "cuda",
            },
            "chatSftForm": {
                "train_files": str(DEFAULT_CHAT_TRAIN_FILE),
                "val_files": str(DEFAULT_CHAT_VAL_FILE),
                "include_identity": 1,
                "identity_repeats": 2,
                "max_seq_len": 1024,
                "device_batch_size": 8,
                "total_batch_size": 16384,
                "num_iterations": 2500,
                "eval_every": 250,
                "eval_tokens": 524288,
                "save_every": 250,
                "run": "serious-sft",
                "device_type": "cuda",
            },
            "baseEvalForm": {
                "corpus_dir": str(LOCAL_CORPUS_DIR),
                "eval": "bpb,sample",
                "device_batch_size": 8,
                "split_tokens": 524288,
                "max_per_task": 256,
                "device_type": "cuda",
            },
            "runtimeForm": {
                "ctx_size": 12288,
                "threads": 12,
                "threads_http": 4,
                "parallel": 4,
                "device_strategy": "gpu",
                "gpu_layers": "auto",
            },
        },
    },
}


def run_profiles() -> dict[str, Any]:
    return deepcopy(RUN_PROFILES)


def _number_param(params: dict[str, Any], name: str) -> float | None:
    value = params.get(name)
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def validate_job_params(job_type: str, params: dict[str, Any], hardware_profile: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(status: str, field: str, message: str, detail: str = "") -> None:
        checks.append({"status": status, "field": field, "message": message, "detail": detail})

    device_type = str(params.get("device_type") or "cpu").lower()
    max_seq_len = _number_param(params, "max_seq_len")
    device_batch = _number_param(params, "device_batch_size")
    total_batch = _number_param(params, "total_batch_size")
    num_iterations = _number_param(params, "num_iterations")
    save_every = _number_param(params, "save_every")
    eval_tokens = _number_param(params, "eval_tokens")

    if max_seq_len is not None:
        if max_seq_len < 64:
            add("error", "max_seq_len", "Max sequence length must be at least 64.")
        if int(max_seq_len) % 64 != 0:
            add("warning", "max_seq_len", "Max sequence length is usually safest as a multiple of 64.")
        if device_type == "cpu" and max_seq_len > 512:
            add("warning", "max_seq_len", "Sequence length above 512 can make CPU runs very slow.", "Use a tiny/laptop profile unless this is intentional.")
        if max_seq_len > 2048:
            add("warning", "max_seq_len", "Very long sequence lengths can sharply increase memory use.")

    if device_batch is not None:
        if device_batch < 1:
            add("error", "device_batch_size", "Device batch size must be at least 1.")
        if device_type == "cpu" and device_batch > 4:
            add("warning", "device_batch_size", "Device batch size above 4 is usually heavy for CPU training.")
        if device_type == "cuda" and device_batch > 16:
            add("warning", "device_batch_size", "Device batch size above 16 may exceed consumer GPU memory.")

    if total_batch is not None:
        if total_batch < 1:
            add("error", "total_batch_size", "Total batch size must be at least 1.")
        if device_batch is not None and total_batch < device_batch:
            add("error", "total_batch_size", "Total batch size cannot be smaller than device batch size.")
        if device_batch is not None and total_batch >= 1 and device_batch >= 1 and int(total_batch) % int(device_batch) != 0:
            add("warning", "total_batch_size", "Total batch size is not divisible by device batch size.")

    if eval_tokens is not None:
        if eval_tokens < 0:
            add("error", "eval_tokens", "Eval tokens cannot be negative.")
        if eval_tokens > 1_000_000:
            add("warning", "eval_tokens", "Large eval token counts can dominate short runs.")

    if save_every is not None:
        if save_every < 1:
            add("error", "save_every", "Save Every must be positive when set.")
        if num_iterations is not None and save_every > num_iterations:
            add("warning", "save_every", "Save Every is larger than total iterations, so this run may not produce an intermediate resume checkpoint.")

    for name in ("embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr"):
        value = _number_param(params, name)
        if value is None:
            continue
        if value <= 0:
            add("error", name, f"{name.replace('_', ' ').title()} must be positive when set.")
        elif value > 1:
            add("warning", name, f"{name.replace('_', ' ').title()} is unusually high.")

    for name in ("warmup_ratio", "warmdown_ratio", "init_lr_frac", "final_lr_frac", "adam_beta1", "adam_beta2"):
        value = _number_param(params, name)
        if value is None:
            continue
        if value < 0 or value > 1:
            add("error", name, f"{name.replace('_', ' ').title()} should be between 0 and 1.")

    fp8_enabled = params.get("fp8") in {1, "1", True, "true", "True"}
    if fp8_enabled and device_type != "cuda":
        add("warning", "fp8", "FP8 training is intended for CUDA hardware.", "Disable FP8 for CPU/MPS runs.")

    error_count = sum(1 for check in checks if check["status"] == "error")
    warning_count = sum(1 for check in checks if check["status"] == "warning")
    return {
        "ok": error_count == 0,
        "job_type": job_type,
        "checks": checks,
        "error_count": error_count,
        "warning_count": warning_count,
        "hardware_tier": (hardware_profile or {}).get("tier"),
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


def _load_batch_environment(batch_path: Path, arguments: str = "") -> dict[str, str]:
    if not batch_path.exists():
        return {}
    command = ["cmd", "/c", "call", str(batch_path)]
    if arguments:
        command.extend(arguments.split())
    command.extend(["&&", "set"])
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        return {}
    env_overrides: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        env_overrides[key] = value
    return env_overrides


def _load_visual_studio_build_environment() -> tuple[dict[str, str], Optional[str]]:
    vs_dev_candidates = (
        Path(r"C:\BuildTools\Common7\Tools\VsDevCmd.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"),
    )
    for candidate in vs_dev_candidates:
        env_overrides = _load_batch_environment(candidate, "-arch=amd64 -host_arch=amd64")
        if env_overrides:
            return env_overrides, str(candidate)
    return {}, None


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


def delete_design(slug: str) -> dict[str, Any]:
    path = _ensure_design_dir() / f"{slug}.json"
    if not path.exists():
        raise FileNotFoundError(f"Design not found: {slug}")
    design = _read_json_file(path)
    path.unlink()
    return {
        "slug": slug,
        "name": design.get("name", slug),
        "deleted": True,
    }


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


def command_diff_summary(original: list[str], updated: list[str]) -> dict[str, Any]:
    original_tokens = [str(token) for token in original]
    updated_tokens = [str(token) for token in updated]
    added = []
    removed = []
    changed = []
    original_flags = _command_flag_map(original_tokens)
    updated_flags = _command_flag_map(updated_tokens)
    for flag, value in sorted(updated_flags.items()):
        if flag not in original_flags:
            added.append({"flag": flag, "value": value})
        elif original_flags[flag] != value:
            changed.append({"flag": flag, "from": original_flags[flag], "to": value})
    for flag, value in sorted(original_flags.items()):
        if flag not in updated_flags:
            removed.append({"flag": flag, "value": value})
    return {
        "original": subprocess.list2cmdline(original_tokens),
        "updated": subprocess.list2cmdline(updated_tokens),
        "added": added,
        "removed": removed,
        "changed": changed,
        "same_prefix": original_tokens[:2] == updated_tokens[:2],
    }


def _command_flag_map(tokens: list[str]) -> dict[str, str]:
    flags: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        if "=" in token:
            flag, value = token.split("=", 1)
            flags[flag] = value
            index += 1
        elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            flags[token] = tokens[index + 1]
            index += 2
        else:
            flags[token] = "true"
            index += 1
    return flags


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


def detect_hardware_profile() -> dict[str, Any]:
    preferred_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    system_ram_gb = None
    logical_cpus = os.cpu_count() or 1
    physical_cpus = None
    if psutil is not None:
        try:
            system_ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
        except Exception:
            system_ram_gb = None
        try:
            physical_cpus = psutil.cpu_count(logical=False) or logical_cpus
        except Exception:
            physical_cpus = None

    gpus: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": props.name,
                    "memory_gb": round(props.total_memory / (1024 ** 3), 1),
                    "major": props.major,
                    "minor": props.minor,
                }
            )

    gpu_memory_gb = max((gpu["memory_gb"] for gpu in gpus), default=0.0)
    if preferred_device == "cuda":
        if gpu_memory_gb >= 40:
            tier = "datacenter"
        elif gpu_memory_gb >= 20:
            tier = "workstation"
        elif gpu_memory_gb >= 10:
            tier = "consumer-mid"
        else:
            tier = "consumer-small"
    elif preferred_device == "mps":
        tier = "apple-silicon"
    else:
        tier = "cpu-only"

    return {
        "preferred_device": preferred_device,
        "tier": tier,
        "system_ram_gb": system_ram_gb,
        "logical_cpus": logical_cpus,
        "physical_cpus": physical_cpus,
        "gpus": gpus,
    }


def recommend_forms_for_hardware(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    preferred_device = profile.get("preferred_device", "cpu")
    tier = profile.get("tier", "cpu-only")
    ram_gb = float(profile.get("system_ram_gb") or 0.0)
    gpu_memory_gb = max((float(gpu.get("memory_gb") or 0.0) for gpu in profile.get("gpus", [])), default=0.0)

    tokenizer = {
        "max_chars": 250_000_000,
        "doc_cap": 10_000,
        "vocab_size": 32768,
    }
    base_train = {
        "device_type": preferred_device,
        "depth": 6,
        "head_dim": 64,
        "max_seq_len": 512,
        "device_batch_size": 8,
        "total_batch_size": 8192,
        "num_iterations": 1200,
        "eval_every": 100,
        "eval_tokens": 262144,
        "sample_every": 100,
        "window_pattern": "L",
        "save_every": 100,
    }
    chat_sft = {
        "device_type": preferred_device,
        "max_seq_len": 512,
        "device_batch_size": 8,
        "total_batch_size": 8192,
        "num_iterations": 900,
        "eval_every": 100,
        "save_every": 100,
    }
    chat_rl = {
        "device_type": preferred_device,
        "device_batch_size": 8,
        "examples_per_step": 16,
        "num_samples": 16,
        "eval_every": 60,
        "eval_examples": 400,
        "save_every": 60,
    }
    base_eval = {
        "device_type": preferred_device,
        "device_batch_size": 4,
        "split_tokens": 131072,
        "max_per_task": 64,
    }
    chat_eval = {
        "device_type": preferred_device,
        "batch_size": 8,
        "num_samples": 1,
        "max_new_tokens": 512,
    }
    runtime = {
        "ctx_size": 8192,
        "threads": min(max(int(profile.get("physical_cpus") or profile.get("logical_cpus") or 8), 4), 16),
        "threads_http": max(2, min(8, max(2, int(profile.get("logical_cpus") or 8) // 4 or 1))),
        "parallel": 2,
        "gpu_layers": "auto",
        "device_strategy": "auto",
    }

    if tier == "cpu-only" or (preferred_device == "cpu" and ram_gb and ram_gb <= 24):
        tokenizer.update({"max_chars": 50_000_000, "doc_cap": 4_000, "vocab_size": 16384})
        base_train.update({"depth": 4, "max_seq_len": 256, "device_batch_size": 1, "total_batch_size": 1024, "num_iterations": 300, "eval_tokens": 32768, "sample_every": -1, "save_every": 50})
        chat_sft.update({"max_seq_len": 256, "device_batch_size": 1, "total_batch_size": 1024, "num_iterations": 200, "eval_every": 50, "save_every": 50})
        chat_rl.update({"device_batch_size": 2, "examples_per_step": 4, "num_samples": 4, "eval_every": 20, "eval_examples": 40, "save_every": 20})
        base_eval.update({"device_batch_size": 1, "split_tokens": 16384, "max_per_task": 16})
        chat_eval.update({"batch_size": 1, "max_new_tokens": 256})
        runtime.update({"ctx_size": 4096 if ram_gb and ram_gb < 24 else 6144, "threads": min(runtime["threads"], 12), "threads_http": 2, "parallel": 1})
    elif tier == "apple-silicon":
        tokenizer.update({"max_chars": 100_000_000, "doc_cap": 6_000, "vocab_size": 16384})
        base_train.update({"depth": 4, "max_seq_len": 256, "device_batch_size": 2, "total_batch_size": 2048, "num_iterations": 400, "eval_tokens": 65536, "save_every": 50})
        chat_sft.update({"max_seq_len": 256, "device_batch_size": 2, "total_batch_size": 2048, "num_iterations": 300, "eval_every": 60, "save_every": 60})
        chat_rl.update({"device_batch_size": 2, "examples_per_step": 4, "num_samples": 4, "eval_examples": 80, "save_every": 20})
        base_eval.update({"device_batch_size": 2, "split_tokens": 32768, "max_per_task": 24})
        chat_eval.update({"batch_size": 2, "max_new_tokens": 320})
        runtime.update({"ctx_size": 6144, "parallel": 1, "threads_http": 2})
    elif tier == "consumer-small":
        tokenizer.update({"max_chars": 120_000_000, "doc_cap": 8_000, "vocab_size": 16384})
        base_train.update({"depth": 4, "max_seq_len": 384, "device_batch_size": 2, "total_batch_size": 2048, "num_iterations": 500, "eval_tokens": 65536, "save_every": 50})
        chat_sft.update({"max_seq_len": 384, "device_batch_size": 2, "total_batch_size": 2048, "num_iterations": 350, "eval_every": 60, "save_every": 60})
        chat_rl.update({"device_batch_size": 2, "examples_per_step": 8, "num_samples": 8, "eval_examples": 120, "save_every": 30})
        base_eval.update({"device_batch_size": 2, "split_tokens": 65536, "max_per_task": 32})
        chat_eval.update({"batch_size": 2, "max_new_tokens": 384})
        runtime.update({"ctx_size": 6144, "threads_http": 2, "parallel": 1})
    elif tier == "consumer-mid":
        base_train.update({"depth": 6, "max_seq_len": 512, "device_batch_size": 4, "total_batch_size": 4096, "num_iterations": 800, "save_every": 75})
        chat_sft.update({"max_seq_len": 512, "device_batch_size": 4, "total_batch_size": 4096, "num_iterations": 600, "eval_every": 80, "save_every": 80})
        chat_rl.update({"device_batch_size": 4, "examples_per_step": 8, "num_samples": 8, "eval_examples": 160, "save_every": 40})
        base_eval.update({"device_batch_size": 4, "split_tokens": 131072, "max_per_task": 48})
        chat_eval.update({"batch_size": 4})
        runtime.update({"ctx_size": 8192, "parallel": 2})
    elif tier == "workstation":
        base_train.update({"device_batch_size": 8, "total_batch_size": 8192, "num_iterations": 1200, "save_every": 100})
        chat_sft.update({"device_batch_size": 8, "total_batch_size": 8192, "num_iterations": 900, "save_every": 100})
        chat_rl.update({"device_batch_size": 8, "examples_per_step": 16, "num_samples": 16})
        runtime.update({"ctx_size": 8192, "parallel": 2})
    elif tier == "datacenter":
        base_train.update({"depth": 8, "head_dim": 64, "max_seq_len": 1024, "device_batch_size": 16, "total_batch_size": 16384, "num_iterations": 1600, "eval_tokens": 524288, "save_every": 100})
        chat_sft.update({"max_seq_len": 1024, "device_batch_size": 16, "total_batch_size": 16384, "num_iterations": 1200, "save_every": 120})
        chat_rl.update({"device_batch_size": 16, "examples_per_step": 32, "num_samples": 16, "eval_examples": 400, "save_every": 60})
        base_eval.update({"device_batch_size": 8, "split_tokens": 262144, "max_per_task": 96})
        chat_eval.update({"batch_size": 8, "max_new_tokens": 768})
        runtime.update({"ctx_size": 12288, "parallel": 4})

    summary = {
        "title": f"Hardware fit: {tier}",
        "notes": [
            f"Preferred device: {preferred_device}",
            f"System RAM: {ram_gb:.1f} GB" if ram_gb else "System RAM unavailable",
            f"Largest GPU memory: {gpu_memory_gb:.1f} GB" if gpu_memory_gb else "No CUDA GPU detected",
            "Checkpoint save cadence is enabled so long runs can be resumed later.",
        ],
    }
    return {
        "tokenizerForm": tokenizer,
        "baseTrainForm": base_train,
        "chatSftForm": chat_sft,
        "chatRlForm": chat_rl,
        "baseEvalForm": base_eval,
        "chatEvalForm": chat_eval,
        "runtimeForm": runtime,
        "summary": summary,
    }


def _resolve_base_checkpoint_tag(params: dict[str, Any]) -> str:
    model_tag = str(params.get("model_tag") or "").strip()
    if model_tag:
        return model_tag
    depth = int(params.get("depth") or 6)
    return f"d{depth}"


def _resolve_sft_checkpoint_tag(params: dict[str, Any]) -> str | None:
    model_tag = str(params.get("model_tag") or "").strip()
    if model_tag:
        return model_tag
    return None


def _resolve_rl_checkpoint_tag(params: dict[str, Any]) -> str | None:
    model_tag = str(params.get("model_tag") or "").strip()
    if model_tag:
        return model_tag
    return None


def _checkpoint_dir_for_job(job_type: str, params: dict[str, Any]) -> Path | None:
    base_dir = Path(get_base_dir())
    if job_type == "base_train":
        return base_dir / "base_checkpoints" / _resolve_base_checkpoint_tag(params)
    if job_type == "chat_sft":
        checkpoints_dir = base_dir / "chatsft_checkpoints"
        resolved_tag = _resolve_sft_checkpoint_tag(params)
        if resolved_tag:
            return checkpoints_dir / resolved_tag
        if not checkpoints_dir.exists():
            return None
        try:
            latest_dir = max((item for item in checkpoints_dir.iterdir() if item.is_dir()), key=lambda item: item.stat().st_mtime)
            return latest_dir
        except FileNotFoundError:
            return None
        except ValueError:
            return None
    if job_type == "chat_rl":
        checkpoints_dir = base_dir / "chatrl_checkpoints"
        resolved_tag = _resolve_rl_checkpoint_tag(params)
        if resolved_tag:
            return checkpoints_dir / resolved_tag
        if not checkpoints_dir.exists():
            return None
        try:
            latest_dir = max((item for item in checkpoints_dir.iterdir() if item.is_dir()), key=lambda item: item.stat().st_mtime)
            return latest_dir
        except FileNotFoundError:
            return None
        except ValueError:
            return None
    return None


def latest_checkpoint_step_for_job(job_type: str, params: dict[str, Any]) -> int | None:
    checkpoint_dir = _checkpoint_dir_for_job(job_type, params)
    if checkpoint_dir is None or not checkpoint_dir.exists():
        return None
    try:
        return _find_last_step(checkpoint_dir)
    except FileNotFoundError:
        return None


def checkpoint_info_for_job(job_type: str, params: dict[str, Any]) -> dict[str, Any]:
    checkpoint_dir = _checkpoint_dir_for_job(job_type, params)
    if checkpoint_dir is None:
        return {"tag": None, "path": None, "latest_step": None}
    latest_step = None
    if checkpoint_dir.exists():
        try:
            latest_step = _find_last_step(checkpoint_dir)
        except FileNotFoundError:
            latest_step = None
    return {
        "tag": checkpoint_dir.name,
        "path": str(checkpoint_dir),
        "latest_step": latest_step,
    }


def builder_state() -> dict[str, Any]:
    base_dir = Path(get_base_dir())
    tokenizer_dir = base_dir / "tokenizer"
    designs = list_designs()
    local_corpus = corpus_summary(str(LOCAL_CORPUS_DIR))
    hardware_profile = detect_hardware_profile()
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
        "hardware_profile": hardware_profile,
        "hardware_recommendations": recommend_forms_for_hardware(hardware_profile),
        "checkpoint_sets": {
            "base": _summarize_checkpoints(base_dir / "base_checkpoints"),
            "sft": _summarize_checkpoints(base_dir / "chatsft_checkpoints"),
            "rl": _summarize_checkpoints(base_dir / "chatrl_checkpoints"),
        },
        "designs": designs,
        "guided_presets": GUIDED_PRESETS,
        "run_profiles": run_profiles(),
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
        _append_flag(command, "max_corpus_docs", params.get("max_corpus_docs", 8))
    elif job_type == "base_train":
        command.append("scripts.base_train")
        defaults = {
            "corpus_dir": str(LOCAL_CORPUS_DIR),
            "depth": 6,
            "aspect_ratio": "",
            "head_dim": 64,
            "max_seq_len": 512,
            "device_batch_size": 8,
            "total_batch_size": 8192,
            "num_iterations": 1200,
            "embedding_lr": "",
            "unembedding_lr": "",
            "matrix_lr": "",
            "scalar_lr": "",
            "weight_decay": "",
            "adam_beta1": "",
            "adam_beta2": "",
            "warmup_ratio": "",
            "warmdown_ratio": "",
            "final_lr_frac": "",
            "eval_every": 100,
            "eval_tokens": 262144,
            "core_metric_every": -1,
            "core_metric_max_per_task": "",
            "sample_every": 100,
            "window_pattern": "L",
            "run": "builder-base",
            "device_type": "cpu",
            "model_tag": "",
            "target_flops": "",
            "target_param_data_ratio": "",
            "resume_from_step": "",
            "save_every": "",
            "fp8": 0,
            "fp8_recipe": "",
        }
        defaults.update(params)
        fp8_value = defaults.pop("fp8", 0)
        if fp8_value in {1, "1", True, "true", "True"}:
            command.append("--fp8")
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
            "load_optimizer": 1,
            "resume_from_step": "",
            "train_files": str(DEFAULT_CHAT_TRAIN_FILE),
            "val_files": str(DEFAULT_CHAT_VAL_FILE),
            "identity_file": "",
            "include_identity": 1,
            "identity_repeats": 2,
            "max_seq_len": 512,
            "device_batch_size": 8,
            "total_batch_size": 8192,
            "num_iterations": 900,
            "eval_every": 100,
            "eval_tokens": 262144,
            "chatcore_every": -1,
            "save_every": "",
            "run": "builder-sft",
            "device_type": "cpu",
            "embedding_lr": "",
            "unembedding_lr": "",
            "matrix_lr": "",
            "init_lr_frac": "",
            "warmup_ratio": "",
            "warmdown_ratio": "",
            "final_lr_frac": "",
        }
        defaults.update(params)
        for key in defaults:
            _append_flag(command, key, defaults[key])
    elif job_type == "chat_rl":
        command.append("scripts.chat_rl")
        defaults = {
            "model_tag": "",
            "model_step": "",
            "resume_from_step": "",
            "run": "builder-rl",
            "device_type": "cpu",
            "num_epochs": 1,
            "device_batch_size": 8,
            "examples_per_step": 16,
            "num_samples": 16,
            "max_new_tokens": 256,
            "temperature": 1.0,
            "top_k": 50,
            "embedding_lr": 0.2,
            "unembedding_lr": 0.004,
            "matrix_lr": 0.02,
            "weight_decay": 0.0,
            "init_lr_frac": 0.05,
            "eval_every": 60,
            "eval_examples": 400,
            "save_every": 60,
        }
        defaults.update(params)
        for key in defaults:
            _append_flag(command, key, defaults[key])
    elif job_type == "chat_eval":
        command.append("scripts.chat_eval")
        defaults = {
            "source": "sft",
            "task_name": "",
            "temperature": 0.0,
            "max_new_tokens": 512,
            "num_samples": 1,
            "top_k": 50,
            "batch_size": 8,
            "model_tag": "",
            "step": "",
            "max_problems": "",
            "device_type": "cpu",
        }
        defaults.update(params)
        for key in defaults:
            _append_flag(command, key, defaults[key])
    else:
        raise ValueError(f"Unsupported job_type: {job_type}")
    return command


def _failure_evidence(lines: list[str], patterns: list[str], limit: int = 3) -> list[str]:
    evidence: list[str] = []
    compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for line in lines:
        text = str(line).strip()
        if not text:
            continue
        if any(pattern.search(text) for pattern in compiled):
            evidence.append(text[:260])
            if len(evidence) >= limit:
                break
    return evidence


def diagnose_job_failure(
    job_type: str,
    status: str,
    command: list[str] | str,
    log_lines: list[str],
    metrics: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Return a friendly diagnosis for common failed dashboard jobs."""
    if status != "failed":
        return None
    display_command = subprocess.list2cmdline(command) if isinstance(command, list) else str(command)
    lines = [display_command, *[str(line) for line in log_lines]]
    combined = "\n".join(lines).lower()

    def has(*patterns: str) -> bool:
        return any(re.search(pattern, combined, re.IGNORECASE) for pattern in patterns)

    def diagnosis(kind: str, title: str, detail: str, next_step: str, patterns: list[str], severity: str = "error") -> dict[str, Any]:
        return {
            "kind": kind,
            "severity": severity,
            "title": title,
            "detail": detail,
            "next_step": next_step,
            "evidence": _failure_evidence(lines, patterns),
        }

    if has(r"parquet support requires pyarrow", r"no module named ['\"]pyarrow['\"]", r"modulenotfounderror: no module named ['\"]pyarrow['\"]"):
        patterns = [r"pyarrow", r"parquet support"]
        return diagnosis(
            "missing_pyarrow",
            "Parquet support is not installed yet",
            "This run tried to read a parquet corpus file, but the local Python environment does not have pyarrow.",
            "Install pyarrow locally, then retry. For a quick workaround, use .txt or .jsonl corpus files instead.",
            patterns,
        )
    if has(r"tokenizer[\\/]+tokenizer\.pkl", r"tokenizer[\\/]+tokenizer\.json", r"return rustbpetokenizer\.from_directory", r"get_tokenizer\(\)"):
        patterns = [r"tokenizer[\\/]+tokenizer\.(pkl|json)", r"RustBPETokenizer\.from_directory", r"get_tokenizer\(\)"]
        return diagnosis(
            "missing_tokenizer",
            "Tokenizer files are missing",
            "Base training or evaluation started before a tokenizer had been created in the nanochat cache.",
            "Run Stage 1 Train Tokenizer first, confirm the tokenizer shows as ready, then launch this job again.",
            patterns,
        )
    if has(r"no local documents found for split", r"populate the local corpus first"):
        patterns = [r"no local documents found for split", r"populate the local corpus first"]
        return diagnosis(
            "empty_corpus_split",
            "The local corpus is empty for this split",
            "The training script did not find usable documents under the local corpus path it expected.",
            "Add files under local_corpus, especially the train split, then retry. Text, JSONL, code files, and parquet are supported.",
            patterns,
        )
    if has(r"identity_conversations\.jsonl", r"published identity file", r"identity file"):
        patterns = [r"identity_conversations\.jsonl", r"published identity file", r"identity file"]
        return diagnosis(
            "missing_identity_file",
            "The active identity file is missing",
            "Chat SFT expected a published identity dataset, but it was not available when the run started.",
            "Open Assistant Studio, publish the identity file, and then rerun Chat SFT.",
            patterns,
        )
    if has(r"chatsft_checkpoints", r"no checkpoints found in .*chatsft_checkpoints", r"cannot find the path specified: .*chatsft_checkpoints"):
        patterns = [r"chatsft_checkpoints", r"no checkpoints found"]
        return diagnosis(
            "missing_sft_checkpoint",
            "A chat SFT checkpoint is missing",
            "This stage needs an existing supervised fine-tuning checkpoint before it can continue.",
            "Run Stage 3 Chat SFT first, then use Load Latest SFT or pick a valid SFT model tag before retrying Chat RL or Chat Eval.",
            patterns,
        )
    if job_type == "chat_rl" and has(r"base_checkpoints", r"no checkpoints found in .*base"):
        patterns = [r"base_checkpoints", r"no checkpoints found"]
        return diagnosis(
            "missing_base_checkpoint",
            "The required base checkpoint was not found",
            "Chat RL could not find the base model family it expected to build from.",
            "Run Stage 2 Base Model Training first, then complete Chat SFT before coming back to Chat RL.",
            patterns,
        )
    if has(r"out of memory", r"cuda out of memory", r"\boom\b") or bool((metrics or {}).get("oom_detected")):
        patterns = [r"out of memory", r"cuda out of memory", r"\boom\b"]
        return diagnosis(
            "out_of_memory",
            "The run ran out of memory",
            "The current batch size or model shape is too large for the selected device.",
            "Reduce Device Batch Size first. If needed, reduce sequence length, total batch size, or model depth and try again.",
            patterns,
        )
    if has(r"modulenotfounderror: no module named ['\"]([^'\"]+)['\"]", r"importerror:"):
        patterns = [r"modulenotfounderror", r"importerror"]
        return diagnosis(
            "missing_python_dependency",
            "A Python dependency is missing",
            "The job stopped while importing a required Python module.",
            "Install the missing package in this environment, then rerun the job.",
            patterns,
        )
    if has(r"filenotfounderror", r"no such file or directory", r"cannot find the path specified"):
        patterns = [r"filenotfounderror", r"no such file or directory", r"cannot find the path specified"]
        return diagnosis(
            "missing_file_or_path",
            "A required file or path was not found",
            "The command references a local file, checkpoint, corpus directory, or dataset path that does not exist.",
            "Check the path shown in the evidence, fix the form value or create the missing file, then retry.",
            patterns,
        )
    return {
        "kind": "unclassified",
        "severity": "error",
        "title": "The run stopped with an unclassified error",
        "detail": "The raw logs still contain the exact traceback or command output.",
        "next_step": "Read the final log lines, then use the assistant or process log to decide the next fix before rerunning.",
        "evidence": [line[:260] for line in log_lines[-3:] if str(line).strip()],
    }


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
    pause_requested: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    log_lines: deque[str] = field(default_factory=lambda: deque(maxlen=JOB_LOG_LIMIT), repr=False)
    process: Optional[subprocess.Popen[str]] = field(default=None, repr=False)

    def snapshot(self, include_logs: bool = False) -> dict[str, Any]:
        checkpoint_info = checkpoint_info_for_job(self.job_type, self.params)
        latest_checkpoint_step = checkpoint_info["latest_step"]
        resumable_job_types = {"base_train", "chat_sft", "chat_rl"}
        requested_resume_step = self.params.get("resume_from_step")
        try:
            requested_resume_step = int(requested_resume_step) if requested_resume_step not in {"", None} else None
        except (TypeError, ValueError):
            requested_resume_step = None
        save_every = self.params.get("save_every")
        try:
            save_every = int(save_every) if save_every not in {"", None} else None
        except (TypeError, ValueError):
            save_every = None
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
            "latest_checkpoint_step": latest_checkpoint_step,
            "checkpoint_tag": checkpoint_info["tag"],
            "checkpoint_dir": checkpoint_info["path"],
            "requested_resume_step": requested_resume_step,
            "save_every": save_every,
            "is_resumed_run": requested_resume_step is not None,
            "can_pause": self.job_type in resumable_job_types and self.status in {"queued", "running"},
            "can_resume": self.job_type in resumable_job_types and latest_checkpoint_step is not None and self.status in {"paused", "stopped", "failed"},
            "failure_diagnosis": diagnose_job_failure(self.job_type, self.status, self.command, list(self.log_lines), self.metrics),
        }
        if include_logs:
            payload["logs"] = list(self.log_lines)
        else:
            payload["log_tail"] = list(self.log_lines)[-40:]
        return payload


class BackgroundJobManager:
    def __init__(
        self,
        workdir: Optional[str] = None,
        activity_log=None,
        benchmark_history: Optional[BenchmarkHistoryManager] = None,
        jobs_path: Optional[str | Path] = None,
        max_persisted_jobs: int = JOB_HISTORY_LIMIT,
    ):
        self.workdir = workdir or str(REPO_ROOT)
        self.activity_log = activity_log
        self.benchmark_history = benchmark_history
        self.jobs_path = Path(jobs_path) if jobs_path is not None else JOBS_FILE
        self.max_persisted_jobs = max(1, int(max_persisted_jobs))
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._load_persisted_jobs()

    def _log(self, kind: str, message: str, payload: Optional[dict[str, Any]] = None) -> None:
        if self.activity_log is not None:
            self.activity_log.log_event(kind, message, payload)

    def _job_from_payload(self, payload: dict[str, Any]) -> Optional[JobRecord]:
        try:
            status = str(payload.get("status") or "stopped")
            logs = list(payload.get("logs") or payload.get("log_tail") or [])
            if status in {"queued", "running"}:
                status = "stopped"
                logs.append("[dashboard] Dashboard restarted before this job finished; the process could not be reattached.")
            record = JobRecord(
                id=str(payload["id"]),
                label=str(payload.get("label") or payload.get("job_type") or "job"),
                job_type=str(payload["job_type"]),
                command=list(payload.get("command") or []),
                created_at=float(payload.get("created_at") or 0.0),
                cwd=str(payload.get("cwd") or self.workdir),
                notes=str(payload.get("notes") or ""),
                params=dict(payload.get("params") or {}),
                status=status,
                started_at=payload.get("started_at"),
                finished_at=payload.get("finished_at"),
                exit_code=payload.get("exit_code"),
                pid=None,
                metrics=dict(payload.get("metrics") or {}),
            )
            record.log_lines.extend(str(line) for line in logs[-JOB_LOG_LIMIT:])
            return record
        except (KeyError, TypeError, ValueError):
            return None

    def _load_persisted_jobs(self) -> None:
        if not self.jobs_path.exists():
            return
        try:
            raw = json.loads(self.jobs_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        jobs = raw.get("jobs", raw if isinstance(raw, list) else [])
        if not isinstance(jobs, list):
            return
        for item in jobs:
            if not isinstance(item, dict):
                continue
            record = self._job_from_payload(item)
            if record is not None:
                self._jobs[record.id] = record
        self._prune_job_history()

    def _retained_jobs(self) -> list[JobRecord]:
        jobs = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
        active = [job for job in jobs if job.status in {"queued", "running"}]
        inactive = [job for job in jobs if job.status not in {"queued", "running"}]
        retained: list[JobRecord] = []
        seen: set[str] = set()
        for job in [*active, *inactive]:
            if job.id in seen:
                continue
            retained.append(job)
            seen.add(job.id)
            if len(retained) >= self.max_persisted_jobs and not active:
                break
        if len(retained) < self.max_persisted_jobs:
            return retained
        active_ids = {job.id for job in active}
        return [job for job in retained if job.id in active_ids][: len(active)] + [
            job for job in retained if job.id not in active_ids
        ][: max(0, self.max_persisted_jobs - len(active))]

    def _prune_job_history(self) -> None:
        retained = self._retained_jobs()
        retained_ids = {job.id for job in retained}
        self._jobs = {job_id: job for job_id, job in self._jobs.items() if job_id in retained_ids}

    def _persist_jobs(self) -> None:
        try:
            self.jobs_path.parent.mkdir(parents=True, exist_ok=True)
            self._prune_job_history()
            jobs = [job.snapshot(include_logs=True) for job in self._retained_jobs()]
            jobs.sort(key=lambda item: item.get("created_at") or 0.0, reverse=True)
            payload = {
                "version": 1,
                "saved_at": time.time(),
                "max_jobs": self.max_persisted_jobs,
                "jobs": jobs,
            }
            temp_path = self.jobs_path.with_suffix(f"{self.jobs_path.suffix}.tmp")
            temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temp_path.replace(self.jobs_path)
        except OSError:
            pass

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
            self._persist_jobs()
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
        self._persist_jobs()
        return job.snapshot(include_logs=True)

    def pause_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            job = self._jobs[job_id]
            process = job.process
        if job.job_type not in {"base_train", "chat_sft", "chat_rl"}:
            raise ValueError("Pause is currently supported only for base training, chat SFT, and chat RL runs.")
        if process is None or job.status not in {"queued", "running"}:
            return job.snapshot(include_logs=True)
        job.stop_requested = True
        job.pause_requested = True
        job.log_lines.append("[dashboard] Pause requested. The run will stop and can resume from the latest saved checkpoint.")
        self._log("job_pause_requested", f"Pause requested for job {job.label}", {"job_id": job.id, "job_type": job.job_type})
        try:
            process.terminate()
        except OSError:
            pass
        self._persist_jobs()
        return job.snapshot(include_logs=True)

    def _resume_plan(self, job: JobRecord) -> dict[str, Any]:
        if job.job_type not in {"base_train", "chat_sft", "chat_rl"}:
            raise ValueError("Resume is currently supported only for base training, chat SFT, and chat RL runs.")
        checkpoint_step = latest_checkpoint_step_for_job(job.job_type, job.params)
        if checkpoint_step is None:
            raise ValueError("No saved checkpoint was found for this run yet, so there is nothing to resume.")
        resume_params = dict(job.params)
        resume_params["resume_from_step"] = checkpoint_step
        if not resume_params.get("save_every"):
            resume_params["save_every"] = 100
        if job.job_type == "chat_sft":
            resume_params["model_step"] = ""
        if job.job_type == "chat_rl":
            resume_params["model_step"] = ""
        command = build_job_command(job.job_type, resume_params)
        return {
            "source_job_id": job.id,
            "source_label": job.label,
            "job_type": job.job_type,
            "checkpoint_step": checkpoint_step,
            "checkpoint": checkpoint_info_for_job(job.job_type, job.params),
            "params": resume_params,
            "command": command,
            "display_command": subprocess.list2cmdline(command),
            "command_diff": command_diff_summary(job.command, command),
            "label": f"{job.label} resume",
            "notes": f"resumed from {job.id} at step {checkpoint_step}",
        }

    def preview_resume_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            job = self._jobs[job_id]
        return {"source_job": job.snapshot(include_logs=False), "resume": self._resume_plan(job)}

    def resume_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            job = self._jobs[job_id]
        plan = self._resume_plan(job)
        resumed = self.start_job(job.job_type, plan["label"], plan["command"], notes=plan["notes"], params=plan["params"])
        self._log(
            "job_resumed",
            f"Resumed job {job.label}",
            {"job_id": resumed["id"], "source_job_id": job.id, "job_type": job.job_type, "resume_from_step": plan["checkpoint_step"]},
        )
        return resumed

    def _run_job(self, record: JobRecord) -> None:
        record.status = "running"
        record.started_at = time.time()
        env = os.environ.copy()
        vs_env_overrides, vs_env_source = _load_visual_studio_build_environment()
        if vs_env_overrides:
            env.update(vs_env_overrides)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["NANOCHAT_LOCAL_ONLY"] = "1"
        env.setdefault("NANOCHAT_LOCAL_CORPUS_DIR", str(LOCAL_CORPUS_DIR))
        env["WANDB_MODE"] = "disabled"
        env["WANDB_DISABLED"] = "true"
        env["WANDB_CONSOLE"] = "off"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
        if vs_env_source:
            record.log_lines.append(f"[dashboard] Loaded Visual Studio build environment from {vs_env_source}")
        else:
            record.log_lines.append("[dashboard] Visual Studio build environment was not found. CPU torch.compile may fail if cl.exe is unavailable.")
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
                checkpoint_step = latest_checkpoint_step_for_job(record.job_type, record.params)
                if record.pause_requested and checkpoint_step is not None:
                    record.status = "paused"
                    record.log_lines.append(f"[dashboard] Run paused. Latest saved checkpoint step: {checkpoint_step}.")
                else:
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
            self._persist_jobs()
            if self.benchmark_history is not None and record.status in {"completed", "failed", "stopped", "paused"}:
                self.benchmark_history.append_job_result(record.snapshot(include_logs=True))
