#!/usr/bin/env python3
"""
Unified web server for chat and the builder dashboard.

The dashboard is designed to work even before a chat model exists. If no chat
checkpoint is available yet, the app still starts and exposes setup, design,
training, and evaluation workflows.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import os
import random
import re
import subprocess
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, List, Optional

os.environ["NANOCHAT_LOCAL_ONLY"] = "1"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from nanochat.benchmark_tools import BenchmarkHistoryManager
from nanochat.common import autodetect_device_type, compute_init, get_base_dir
from nanochat.dataset import TEXT_SUFFIXES, inspect_local_corpus
from nanochat.dataset_sources import collect_huggingface_corpus_records, preview_huggingface_corpus_records, recommend_huggingface_corpus_sources
from nanochat.dashboard_tools import (
    BackgroundJobManager,
    GUIDED_PRESETS,
    builder_state,
    build_job_command,
    delete_design,
    list_designs,
    publish_design,
    run_profiles,
    save_design,
    validate_job_params,
)
from nanochat.activity_log import ActivityLogManager
from nanochat.ecg_monitor import EcgMonitor
from nanochat.local_runtime import LocalRuntimeManager
from nanochat.preflight import run_stage_preflight
from nanochat.report import EXPECTED_FILES, Report
from nanochat.sandbox_tools import CorpusManager, SandboxManager
from nanochat.sft_dataset_tools import (
    conversations_from_jsonl,
    conversations_to_jsonl,
    merge_jsonl,
    normalize_conversation,
    normalize_conversations,
    preview_sft_jsonl_export,
    preview_sft_parquet_export,
    preview_sft_split_export,
    sft_schema_payload,
    split_conversations,
    write_sft_parquet_file,
)
from nanochat.validation_tools import validate_corpus_path, validate_sft_jsonl_content

REPO_ROOT = Path(__file__).resolve().parent.parent
NANOCHAT_DIR = REPO_ROOT / "nanochat"

# Abuse prevention limits
MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 0
MAX_TOP_K = 200
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096
DEFAULT_LOCAL_RUNTIME_COCKPIT_PROTOCOL = (
    "Cockpit protocol for llm-tweaker:\n"
    "- You are operating inside llm-tweaker, a local LLM builder and tuning dashboard based on nanochat.\n"
    "- Your role is assistant and tutor for model creation, dataset shaping, evaluation, debugging, and safe next-step guidance.\n"
    "- Prefer direct, practical, user-visible answers over narration about your internal process.\n"
    "- Do not emit hidden chain-of-thought, scratchpad notes, or reasoning-only text in place of a final answer.\n"
    "- Never leave the assistant message blank when you can answer or ask one short clarifying question.\n"
    "- If tools or local actions are available, use the required assistant-action format only when an action is actually needed.\n"
    "- Treat dashboard controls, files, and logs as the operating environment you are helping the user navigate.\n"
    "- If something is missing or uncertain, say that plainly and name the next concrete thing to check."
)
DEFAULT_LOCAL_RUNTIME_SYSTEM_PROMPT = (
    "You are the local builder assistant for this dashboard. "
    "Help the user run the LLM builder, explain the next step, point out missing files or settings, "
    "and prefer practical, accurate guidance over speed. If you are unsure, say so plainly. "
    "When workspace files are provided, use only that workspace context. "
    "The writable local workspaces exposed through actions are assistant_sandbox and local_corpus."
)
ASSISTANT_ACTION_PATTERN = re.compile(r"<assistant_action>\s*(.*?)\s*</assistant_action>", re.DOTALL)
MAX_ASSISTANT_ACTIONS = 4
FIRST_RUN_REPO_DIRS = [
    "assistant_models",
    "models",
    "runtime",
    "runtime/windows",
    "runtime/models",
    "assistant_sandbox",
    "local_corpus",
    "local_corpus/train",
    "local_corpus/val",
    "builder_logs",
]
FIRST_RUN_BASE_DIRS = [
    "",
    "tokenizer",
    "base_checkpoints",
    "chatsft_checkpoints",
    "chatrl_checkpoints",
    "builder_designs",
    "report",
]


def _layout_path(root: Path, relative: str) -> Path:
    path = root
    for component in relative.split("/"):
        if component:
            path = path / component
    return path


def ensure_first_run_dirs() -> None:
    base_dir = Path(get_base_dir())
    for relative in FIRST_RUN_BASE_DIRS:
        _layout_path(base_dir, relative).mkdir(parents=True, exist_ok=True)
    for relative in FIRST_RUN_REPO_DIRS:
        _layout_path(REPO_ROOT, relative).mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser(description="NanoChat Web Server")
parser.add_argument("-n", "--num-gpus", type=int, default=1, help="Number of GPUs to use for chat inference")
parser.add_argument("-i", "--source", type=str, default="sft", help="Source of the chat model: sft|rl")
parser.add_argument("-t", "--temperature", type=float, default=0.8, help="Default temperature for generation")
parser.add_argument("-k", "--top-k", type=int, default=50, help="Default top-k sampling parameter")
parser.add_argument("-m", "--max-tokens", type=int, default=512, help="Default max tokens for generation")
parser.add_argument("-g", "--model-tag", type=str, default=None, help="Model tag to load")
parser.add_argument("-s", "--step", type=int, default=None, help="Step to load")
parser.add_argument("-p", "--port", type=int, default=8000, help="Port to run the server on")
parser.add_argument("--runtime-autostart", type=int, default=1, help="Auto-start local llama.cpp runtime when local GGUFs are available (1=yes, 0=no)")
parser.add_argument("--runtime-model", type=str, default="", help="Optional path to GGUF model for local runtime auto-start")
parser.add_argument("--runtime-port", type=int, default=8091, help="Port for the local llama.cpp runtime")
parser.add_argument("--runtime-device-strategy", type=str, default="auto", choices=["auto", "gpu", "cpu"], help="Device strategy for local runtime auto-start")
parser.add_argument(
    "--device-type",
    type=str,
    default="",
    choices=["cuda", "cpu", "mps"],
    help="Device type for evaluation: cuda|cpu|mps. empty => autodetect",
)
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)


@dataclass
class Worker:
    gpu_id: int
    device: torch.device
    engine: object
    tokenizer: object


class WorkerPool:
    """Pool of workers, each with a model replica on a specific device."""

    def __init__(self, num_gpus: Optional[int] = None):
        if num_gpus is None:
            if device_type == "cuda":
                num_gpus = torch.cuda.device_count()
            else:
                num_gpus = 1
        self.num_gpus = max(1, num_gpus)
        self.workers: List[Worker] = []
        self.available_workers: asyncio.Queue[Worker] = asyncio.Queue()

    async def initialize(self, source: str, model_tag: Optional[str] = None, step: Optional[int] = None):
        logger.info("Initializing chat worker pool with %s worker(s)", self.num_gpus)
        if self.num_gpus > 1:
            assert device_type == "cuda", "Only CUDA supports multiple chat workers."

        for gpu_id in range(self.num_gpus):
            if device_type == "cuda":
                worker_device = torch.device(f"cuda:{gpu_id}")
                logger.info("Loading chat model on GPU %s", gpu_id)
            else:
                worker_device = torch.device(device_type)
                logger.info("Loading chat model on %s", device_type)

            from nanochat.checkpoint_manager import load_model
            from nanochat.engine import Engine
            model, tokenizer, _ = load_model(source, worker_device, phase="eval", model_tag=model_tag, step=step)
            engine = Engine(model, tokenizer)
            worker = Worker(gpu_id=gpu_id, device=worker_device, engine=engine, tokenizer=tokenizer)
            self.workers.append(worker)
            await self.available_workers.put(worker)

        logger.info("Chat worker pool ready")

    async def acquire_worker(self) -> Worker:
        return await self.available_workers.get()

    async def release_worker(self, worker: Worker):
        await self.available_workers.put(worker)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None


class DesignRequest(BaseModel):
    slug: Optional[str] = None
    name: str
    mission: str
    team_role: str
    tone: str
    uncertainty_policy: str
    collaboration_policy: str
    guardrails: List[str] = Field(default_factory=list)
    custom_notes: str = ""
    recipes: dict[str, dict[str, Any]] = Field(default_factory=dict)


class DesignDraftRequest(BaseModel):
    goal: str
    temperature: float = 0.3
    max_tokens: int = 900


class JobRequest(BaseModel):
    job_type: str
    label: Optional[str] = None
    preset: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


class JobPreflightRequest(BaseModel):
    job_type: str
    params: dict[str, Any] = Field(default_factory=dict)


class ChatLoadRequest(BaseModel):
    source: Optional[str] = None
    model_tag: Optional[str] = None
    step: Optional[int] = None
    num_gpus: Optional[int] = None


class RuntimeStartRequest(BaseModel):
    model_path: str
    host: str = "127.0.0.1"
    port: int = 8091
    ctx_size: int = 8192
    threads: int = 8
    threads_http: int = 4
    parallel: int = 2
    alias: str = "local-runtime"
    device_strategy: str = "auto"
    gpu_layers: str = "auto"
    preferred_device: str = ""


class RuntimeChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.2
    max_tokens: int = 512
    system_prompt: Optional[str] = None
    sandbox_paths: List[str] = Field(default_factory=list)
    corpus_paths: List[str] = Field(default_factory=list)


class RuntimeAssistRequest(RuntimeChatRequest):
    max_actions: int = 3
    action_mode: str = "auto"
    approved_action: Optional[dict[str, Any]] = None
    approved_action_text: str = ""


class SandboxWriteRequest(BaseModel):
    path: str
    content: str = ""
    records: List[dict[str, Any]] = Field(default_factory=list)
    mode: str = "overwrite"


class CorpusParquetConvertRequest(BaseModel):
    path: str = "train/converted.parquet"
    content: str = ""
    conversion_mode: str = "paragraphs"
    mode: str = "overwrite"
    source: str = "dashboard"


class CorpusSplitRequest(BaseModel):
    train_path: str = "train/split.txt"
    val_path: str = "val/split.txt"
    content: str = ""
    split_mode: str = "paragraphs"
    val_ratio: float = 0.1
    seed: int = 1337


class CorpusHfImportRequest(BaseModel):
    dataset_id: str
    config: str = ""
    revision: str = ""
    split: str = "train"
    text_column: str = "text"
    limit_docs: int = 1000
    max_chars: int = 0
    sample_size: int = 12
    source: str = ""
    output_format: str = "parquet"
    train_val_ratio: float = 0.0
    shard_size_docs: int = 10000
    license_note: str = ""
    manifest_path: str = ""
    trust_remote_code: bool = False


class CorpusInspectRequest(BaseModel):
    split: str = "train"
    show_docs: int = 3
    max_chars: int = 240
    long_doc_chars: int = 8000


class SandboxDeleteRequest(BaseModel):
    path: str


class WorkspaceCopyRequest(BaseModel):
    source_path: str
    target_path: str


class PathResolveRequest(BaseModel):
    path: str = ""
    kind: str = "auto"
    multiple: bool = False


class ValidatePathRequest(BaseModel):
    path: str = ""


class ChatTranscriptExportRequest(BaseModel):
    path: str
    messages: List[ChatMessage]
    format: str = "json"
    mode: str = "overwrite"
    preview: bool = False


class SftDatasetExportRequest(BaseModel):
    source_path: str = "chat_train.jsonl"
    target_path: str = "exports/chat_train.parquet"
    val_target_path: str = "exports/chat_val.jsonl"
    format: str = "parquet"
    source: str = "assistant_sandbox"
    val_ratio: float = 0.1
    seed: int = 1337


class AutoTuneRequest(BaseModel):
    forms: dict[str, dict[str, Any]] = Field(default_factory=dict)


def validate_chat_request(request: ChatRequest):
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="At least one message is required")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many messages. Maximum {MAX_MESSAGES_PER_REQUEST} messages allowed per request",
        )

    total_length = 0
    for i, message in enumerate(request.messages):
        if not message.content:
            raise HTTPException(status_code=400, detail=f"Message {i} has empty content")
        msg_length = len(message.content)
        if msg_length > MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} is too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed per message",
            )
        total_length += msg_length

    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Total conversation is too long. Maximum {MAX_TOTAL_CONVERSATION_LENGTH} characters allowed",
        )

    for i, message in enumerate(request.messages):
        if message.role not in ["user", "assistant"]:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} has invalid role. Must be 'user' or 'assistant'",
            )

    if request.temperature is not None and not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
        raise HTTPException(
            status_code=400,
            detail=f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}",
        )

    if request.top_k is not None and not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
        raise HTTPException(status_code=400, detail=f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}")

    if request.max_tokens is not None and not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
        raise HTTPException(
            status_code=400,
            detail=f"max_tokens must be between {MIN_MAX_TOKENS} and {MAX_MAX_TOKENS}",
        )


def load_html(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def dashboard_ecg_activity_snapshot(app: FastAPI) -> dict[str, Any]:
    jobs = app.state.job_manager.list_jobs()
    runtime = app.state.local_runtime.status()
    latest_event = app.state.activity_log.latest_event()
    running_jobs = [job for job in jobs if job.get("status") == "running"]
    queued_jobs = [job for job in jobs if job.get("status") == "queued"]
    paused_jobs = [job for job in jobs if job.get("status") == "paused"]
    return {
        "running_jobs": len(running_jobs),
        "queued_jobs": len(queued_jobs),
        "paused_jobs": len(paused_jobs),
        "latest_job_label": running_jobs[0]["label"] if running_jobs else queued_jobs[0]["label"] if queued_jobs else None,
        "runtime_running": bool(runtime.get("running")),
        "runtime_ready": bool(runtime.get("ready")),
        "runtime_mode": runtime.get("mode"),
        "chat_loading": bool(app.state.chat_loading),
        "recent_event_age_s": (time.time() - float(latest_event["ts"])) if latest_event else None,
        "recent_event_kind": latest_event.get("kind") if latest_event else None,
    }


def chat_status_snapshot(app: FastAPI) -> dict[str, Any]:
    worker_pool = getattr(app.state, "worker_pool", None)
    config = getattr(app.state, "chat_config", {})
    workers = []
    if worker_pool is not None:
        workers = [{"gpu_id": worker.gpu_id, "device": str(worker.device)} for worker in worker_pool.workers]
    return {
        "ready": worker_pool is not None and len(worker_pool.workers) > 0,
        "loading": getattr(app.state, "chat_loading", False),
        "error": getattr(app.state, "chat_error", None),
        "config": config,
        "num_workers": len(workers),
        "available_workers": worker_pool.available_workers.qsize() if worker_pool is not None else 0,
        "workers": workers,
    }


def report_status_payload() -> dict[str, Any]:
    report_dir = Path(get_base_dir()) / "report"
    report_file = report_dir / "report.md"
    header_file = report_dir / "header.md"
    root_copy = REPO_ROOT / "report.md"
    section_files = []
    for file_name in EXPECTED_FILES:
        path = report_dir / file_name
        section_files.append(
            {
                "name": file_name,
                "path": str(path),
                "exists": path.exists(),
                "size": path.stat().st_size if path.exists() else 0,
                "updated_at": path.stat().st_mtime if path.exists() else None,
            }
        )
    preview = ""
    if report_file.exists():
        preview = report_file.read_text(encoding="utf-8", errors="replace")[:4000]
    return {
        "report_dir": str(report_dir),
        "report_file": str(report_file),
        "report_exists": report_file.exists(),
        "report_size": report_file.stat().st_size if report_file.exists() else 0,
        "header_file": str(header_file),
        "header_exists": header_file.exists(),
        "root_copy": str(root_copy),
        "root_copy_exists": root_copy.exists(),
        "section_files": section_files,
        "ready_sections": sum(1 for item in section_files if item["exists"]),
        "expected_sections": len(section_files),
        "preview": preview,
    }


def readiness_payload(app: FastAPI, builder: Optional[dict[str, Any]] = None, runtime: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    builder = builder or builder_state()
    runtime = runtime or app.state.local_runtime.status()
    checks: list[dict[str, Any]] = []

    def add_check(code: str, label: str, status: str, message: str, detail: str = "", next_step: str = "") -> None:
        checks.append(
            {
                "code": code,
                "label": label,
                "status": status,
                "message": message,
                "detail": detail,
                "next_step": next_step,
            }
        )

    corpus_files = app.state.corpus.list_files()
    train_files = sum(1 for file in corpus_files if str(file.get("path", "")).startswith("train/"))
    val_files = sum(1 for file in corpus_files if str(file.get("path", "")).startswith("val/"))
    add_check(
        "corpus_train",
        "Training Corpus",
        "ready" if train_files else "blocker",
        f"{train_files} train file(s) found.",
        f"{val_files} validation file(s) found.",
        "Add text, JSONL object rows, or parquet files under local_corpus/train.",
    )

    identity_exists = bool(builder.get("identity_exists"))
    add_check(
        "identity",
        "Identity Data",
        "ready" if identity_exists else "warning",
        "Identity conversation file is published." if identity_exists else "No published identity conversation file yet.",
        str(builder.get("identity_file", "")),
        "Save and publish an assistant design before Chat SFT.",
    )

    tokenizer_ready = bool(builder.get("tokenizer_ready"))
    add_check(
        "tokenizer",
        "Tokenizer",
        "ready" if tokenizer_ready else "blocker",
        "Tokenizer cache exists." if tokenizer_ready else "Tokenizer is not trained yet.",
        str(builder.get("tokenizer_path", "")),
        "Run tokenizer training after the corpus has at least one train file.",
    )

    checkpoint_sets = builder.get("checkpoint_sets", {})
    base_tags = checkpoint_sets.get("base", {}).get("tags", []) or []
    sft_tags = checkpoint_sets.get("sft", {}).get("tags", []) or []
    rl_tags = checkpoint_sets.get("rl", {}).get("tags", []) or []
    add_check(
        "base_checkpoint",
        "Base Checkpoint",
        "ready" if base_tags else "blocker",
        f"{len(base_tags)} base checkpoint tag(s) found.",
        "",
        "Run base training after the tokenizer is ready.",
    )
    add_check(
        "sft_checkpoint",
        "SFT Checkpoint",
        "ready" if sft_tags else "warning",
        f"{len(sft_tags)} SFT checkpoint tag(s) found.",
        "",
        "Draft/validate SFT JSONL, then run Chat SFT.",
    )
    add_check(
        "rl_checkpoint",
        "RL Checkpoint",
        "ready" if rl_tags else "optional",
        f"{len(rl_tags)} RL checkpoint tag(s) found.",
        "",
        "RL is optional; evaluate SFT first before deciding whether to run RL.",
    )

    sft_train_path = app.state.sandbox.root / "chat_train.jsonl"
    if sft_train_path.exists():
        try:
            sft_report = validate_sft_jsonl_content(sft_train_path.read_text(encoding="utf-8", errors="replace"), path=str(sft_train_path))
            sft_ready = bool(sft_report.get("ok")) and int(sft_report.get("record_count", 0) or 0) > 0
            add_check(
                "sft_train_file",
                "SFT Train File",
                "ready" if sft_ready else "warning",
                f"{sft_report.get('record_count', 0)} conversation row(s) found in chat_train.jsonl.",
                "; ".join((sft_report.get("errors") or sft_report.get("warnings") or [])[:2]),
                "Use Conversation Lab export or assistant SFT drafting to create valid JSONL.",
            )
        except ValueError as exc:
            add_check(
                "sft_train_file",
                "SFT Train File",
                "warning",
                "chat_train.jsonl exists but could not be validated.",
                str(exc),
                "Open the sandbox file and run Validate SFT Dataset.",
            )
    else:
        add_check(
            "sft_train_file",
            "SFT Train File",
            "warning",
            "No chat_train.jsonl file found in assistant_sandbox.",
            str(sft_train_path),
            "Export a Conversation Lab transcript or draft SFT pairs into chat_train.jsonl.",
        )

    bundle = runtime.get("bundle", {}) if isinstance(runtime, dict) else {}
    runtime_models = bundle.get("models", []) or []
    recommended_model = bundle.get("recommended_model")
    add_check(
        "helper_model",
        "Helper Model",
        "ready" if recommended_model or runtime_models else "warning",
        "A local GGUF helper model is available." if recommended_model or runtime_models else "No local GGUF helper model detected.",
        str((recommended_model or {}).get("path", "")) if isinstance(recommended_model, dict) else "",
        "Place a GGUF under assistant_models, models, or runtime/models.",
    )
    add_check(
        "local_runtime",
        "Local Runtime",
        "ready" if runtime.get("ready") else "warning",
        "Local runtime is ready." if runtime.get("ready") else "Local runtime is not ready.",
        runtime.get("last_error", "") or runtime.get("mode", "") or "",
        "Start the local runtime when you want Assistant Actions or GGUF chat.",
    )

    status_rank = {"blocker": 3, "warning": 2, "optional": 1, "ready": 0}
    blocker_count = sum(1 for check in checks if check["status"] == "blocker")
    warning_count = sum(1 for check in checks if check["status"] == "warning")
    ready_count = sum(1 for check in checks if check["status"] == "ready")
    next_check = next((check for check in checks if check["status"] == "blocker"), None)
    if next_check is None:
        next_check = next((check for check in checks if check["status"] == "warning"), None)
    if next_check is None:
        next_check = next((check for check in checks if check["status"] == "optional"), None)
    ordered_checks = sorted(checks, key=lambda check: status_rank.get(check["status"], 0), reverse=True)
    return {
        "ok": blocker_count == 0,
        "status": "blocked" if blocker_count else "attention" if warning_count else "ready",
        "ready_count": ready_count,
        "check_count": len(checks),
        "blocker_count": blocker_count,
        "warning_count": warning_count,
        "next_step": next_check["next_step"] if next_check else "Run an eval or compare checkpoints in Conversation Lab.",
        "checks": ordered_checks,
    }


def split_path_entries(value: str) -> list[str]:
    return [piece.strip() for piece in re.split(r"[;\n|]+", str(value or "")) if piece.strip()]


def resolve_path_reference(app: FastAPI, path_value: str, kind: str = "auto") -> dict[str, Any]:
    raw = str(path_value or "").strip()
    normalized_kind = (kind or "auto").strip().lower()
    result = {
        "input": raw,
        "kind": normalized_kind,
        "scope": "empty",
        "resolved_path": "",
        "exists": False,
        "is_absolute": False,
        "note": "No path provided.",
    }
    if not raw:
        return result

    candidate = Path(raw)
    is_absolute = candidate.is_absolute()
    result["is_absolute"] = is_absolute
    lower = raw.replace("\\", "/").lower()

    if normalized_kind in {"sandbox", "sandbox_file"}:
        if is_absolute:
            result.update({"scope": "absolute", "resolved_path": str(candidate.resolve()), "note": "Absolute path supplied where a sandbox-relative path is usually expected."})
        else:
            resolved = (app.state.sandbox.root / raw).resolve()
            result.update({"scope": "assistant_sandbox", "resolved_path": str(resolved), "note": "Resolved relative to assistant_sandbox."})
    elif normalized_kind in {"corpus", "corpus_file"}:
        if is_absolute:
            result.update({"scope": "absolute", "resolved_path": str(candidate.resolve()), "note": "Absolute path supplied where a corpus-relative path is usually expected."})
        else:
            resolved = (app.state.corpus.root / raw).resolve()
            result.update({"scope": "local_corpus", "resolved_path": str(resolved), "note": "Resolved relative to local_corpus."})
    elif normalized_kind in {"corpus_dir", "directory"}:
        resolved = candidate.resolve() if is_absolute else (REPO_ROOT / raw).resolve()
        result.update({"scope": "absolute" if is_absolute else "repo_relative", "resolved_path": str(resolved), "note": "Directory path used as provided." if is_absolute else "Relative directory path resolved from the repo root."})
    elif normalized_kind in {"model", "runtime_model"}:
        resolved = candidate.resolve() if is_absolute else (REPO_ROOT / raw).resolve()
        result.update({"scope": "absolute" if is_absolute else "repo_relative", "resolved_path": str(resolved), "note": "Runtime model path used as provided." if is_absolute else "Relative model path resolved from the repo root."})
    elif normalized_kind in {"sft", "sft_file"}:
        if is_absolute:
            result.update({"scope": "absolute", "resolved_path": str(candidate.resolve()), "note": "Absolute SFT dataset path."})
        elif lower.startswith("train/") or lower.startswith("val/"):
            resolved = (app.state.corpus.root / raw).resolve()
            result.update({"scope": "local_corpus", "resolved_path": str(resolved), "note": "Looks like a corpus split path; Chat SFT usually expects assistant_sandbox JSONL instead."})
        else:
            resolved = (app.state.sandbox.root / raw).resolve()
            result.update({"scope": "assistant_sandbox", "resolved_path": str(resolved), "note": "Resolved relative to assistant_sandbox for Chat SFT."})
    elif normalized_kind == "identity":
        resolved = candidate.resolve() if is_absolute else (Path(get_base_dir()) / raw).resolve()
        result.update({"scope": "absolute" if is_absolute else "cache_relative", "resolved_path": str(resolved), "note": "Identity path used as provided." if is_absolute else "Relative identity path resolved from the cache directory."})
    else:
        if is_absolute:
            result.update({"scope": "absolute", "resolved_path": str(candidate.resolve()), "note": "Absolute path."})
        elif lower.startswith("train/") or lower.startswith("val/"):
            resolved = (app.state.corpus.root / raw).resolve()
            result.update({"scope": "local_corpus", "resolved_path": str(resolved), "note": "Resolved relative to local_corpus because the path starts with train/ or val/."})
        else:
            resolved = (app.state.sandbox.root / raw).resolve()
            result.update({"scope": "assistant_sandbox", "resolved_path": str(resolved), "note": "Resolved relative to assistant_sandbox by default."})

    resolved_path = Path(result["resolved_path"])
    result["exists"] = resolved_path.exists()
    return result


def render_builder_brief(app: FastAPI) -> str:
    builder = builder_state()
    jobs = app.state.job_manager.list_jobs()[:6]
    runtime = app.state.local_runtime.status()
    sandbox = app.state.sandbox.status()
    corpus = app.state.corpus.status()
    benchmark = app.state.benchmark_history.snapshot(limit=6)
    lines = [
        f"Local-only mode: {builder.get('local_only', False)}",
        f"Corpus dir: {builder.get('corpus_dir', 'unknown')}",
        f"Corpus files: {corpus.get('file_count', 0)}",
        f"Corpus train files: {builder.get('corpus_summary', {}).get('splits', {}).get('train', {}).get('file_count', 0)}",
        f"Corpus val files: {builder.get('corpus_summary', {}).get('splits', {}).get('val', {}).get('file_count', 0)}",
        f"Tokenizer ready: {builder.get('tokenizer_ready', False)}",
        f"Identity ready: {builder.get('identity_exists', False)}",
        f"Sandbox dir: {builder.get('sandbox_dir', 'unknown')}",
        f"Sandbox files: {sandbox.get('file_count', 0)}",
        f"Runtime ready: {runtime.get('ready', False)}",
    ]
    latest_benchmark = benchmark.get("latest_benchmark")
    if latest_benchmark and latest_benchmark.get("metrics", {}).get("val_bpb") is not None:
        lines.append(f"Latest benchmark val bpb: {latest_benchmark['metrics']['val_bpb']:.6f}")
    if jobs:
        lines.append("Recent jobs:")
        for job in jobs:
            lines.append(f"- {job['label']} [{job['status']}] ({job['job_type']})")
    return "\n".join(lines)


def parse_assistant_action(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    match = ASSISTANT_ACTION_PATTERN.search(text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "tool" not in payload:
        return None
    payload.setdefault("args", {})
    if not isinstance(payload["args"], dict):
        return None
    return payload


def extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", candidate, re.IGNORECASE)
        if fence_match:
            candidate = fence_match.group(1)
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("The local GGUF did not return a JSON design draft.")
    parsed = json.loads(candidate[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("The local GGUF returned JSON, but it was not an object.")
    return parsed


def _default_design_recipes() -> dict[str, dict[str, Any]]:
    preset = GUIDED_PRESETS["truth_first_teammate"]["recipes"]
    return {
        "tokenizerForm": dict(preset["tokenizer_train"]),
        "baseTrainForm": dict(preset["base_train"]),
        "chatSftForm": dict(preset["chat_sft"]),
        "baseEvalForm": {
            "corpus_dir": preset["tokenizer_train"]["corpus_dir"],
            "eval": "bpb,sample",
            "device_batch_size": 4,
            "split_tokens": 131072,
            "max_per_task": 64,
            "device_type": "cpu",
        },
    }


def _coerce_design_draft_payload(payload: dict[str, Any], goal: str) -> dict[str, Any]:
    guardrails = payload.get("guardrails", [])
    if not isinstance(guardrails, list):
        guardrails = [str(guardrails)]
    recipes = payload.get("recipes", {})
    if not isinstance(recipes, dict):
        recipes = {}
    merged_recipes = _default_design_recipes()
    for key, value in recipes.items():
        if key in merged_recipes and isinstance(value, dict):
            merged_recipes[key].update(value)

    name = str(payload.get("name") or "").strip() or f"Goal Draft {random.randint(1000, 9999)}"
    return {
        "name": name,
        "mission": str(payload.get("mission") or goal).strip(),
        "team_role": str(payload.get("team_role") or "You support the user as a practical member of their team.").strip(),
        "tone": str(payload.get("tone") or "Direct, calm, and practical.").strip(),
        "uncertainty_policy": str(payload.get("uncertainty_policy") or "If confidence is low, say so plainly and separate facts from guesses.").strip(),
        "collaboration_policy": str(payload.get("collaboration_policy") or "State assumptions, ask for missing constraints, and suggest the next useful step.").strip(),
        "guardrails": [str(item).strip() for item in guardrails if str(item).strip()],
        "custom_notes": str(payload.get("custom_notes") or "").strip(),
        "recipes": merged_recipes,
        "draft_goal": goal,
        "draft_source": "local_gguf",
        "draft_generated_at": time.time(),
    }


def build_design_draft_prompt(goal: str) -> str:
    schema = {
        "name": "Short assistant name",
        "mission": "One or two sentences describing the intended outcome",
        "team_role": "How the assistant should position itself relative to the user",
        "tone": "Desired communication tone",
        "uncertainty_policy": "How it should behave when unsure",
        "collaboration_policy": "How it should work with humans, tools, and limits",
        "guardrails": ["Three to six concrete behavior rules"],
        "custom_notes": "Optional implementation or domain notes",
        "recipes": {
            "tokenizerForm": {"vocab_size": 32768, "max_chars": 250000000, "doc_cap": 10000},
            "baseTrainForm": {"depth": 6, "head_dim": 64, "max_seq_len": 512, "device_batch_size": 8, "total_batch_size": 8192, "num_iterations": 1200, "device_type": "cpu", "run": "builder-base"},
            "chatSftForm": {"include_identity": 1, "identity_repeats": 2, "max_seq_len": 512, "device_batch_size": 8, "total_batch_size": 8192, "num_iterations": 900, "device_type": "cpu", "run": "builder-sft"},
            "baseEvalForm": {"eval": "bpb,sample", "device_batch_size": 4, "split_tokens": 131072, "max_per_task": 64, "device_type": "cpu"},
        },
    }
    return (
        "You are drafting a local LLM builder blueprint from a plain-language user goal. "
        "Return JSON only, with no markdown and no explanation. "
        "Make the design practical, cautious, and realistic for a local-first workflow. "
        "Keep recipes conservative for a beginner machine unless the goal strongly implies otherwise. "
        "Do not invent unsupported stages or cloud dependencies.\n\n"
        f"User goal:\n{goal}\n\n"
        "Required JSON schema:\n"
        f"{json.dumps(schema, ensure_ascii=True, indent=2)}"
    )


def render_tool_help() -> str:
    tool_specs = [
        ("get_builder_state", "{}"),
        ("get_recent_activity", '{"limit": 40}'),
        ("get_benchmark_history", '{"limit": 20}'),
        ("autotune_settings", "{}"),
        ("get_corpus_schema", "{}"),
        ("recommend_corpus_sources", '{"goal": "build a model that explains Python errors to beginners", "max_results": 3}'),
        ("preview_hf_corpus_import", '{"dataset_id": "HuggingFaceFW/fineweb-edu", "split": "train", "text_column": "text", "limit_docs": 1000}'),
        ("import_hf_corpus", '{"dataset_id": "HuggingFaceFW/fineweb-edu", "split": "train", "text_column": "text", "limit_docs": 1000, "train_val_ratio": 0.1, "license_note": "Review dataset card/license before redistribution."}'),
        ("list_corpus_files", "{}"),
        ("read_corpus_file", '{"path": "train/reference.txt"}'),
        ("write_corpus_file", '{"path": "train/reference.parquet", "records": [{"text": "..."}, {"text": "..."}]}'),
        ("draft_corpus_file", '{"path": "train/reference.txt", "mode": "append", "content": "..."}'),
        ("delete_corpus_file", '{"path": "train/tmp.txt"}'),
        ("copy_sandbox_to_corpus", '{"source_path": "notes/reference.txt", "target_path": "train/reference.parquet"}'),
        ("get_sft_schema", "{}"),
        ("draft_sft_data", '{"path": "chat_train.jsonl", "mode": "append", "pairs": [{"user": "...", "assistant": "..."}]}'),
        ("list_sandbox_files", "{}"),
        ("read_sandbox_file", '{"path": "chat_train.jsonl"}'),
        ("write_sandbox_file", '{"path": "chat_train.jsonl", "content": "..."}'),
        ("delete_sandbox_file", '{"path": "drafts/tmp.txt"}'),
        ("list_jobs", "{}"),
        ("get_job_status", '{"job_id": "abc123"}'),
        ("launch_job", '{"job_type": "chat_sft", "params": {"train_files": "chat_train.jsonl", "val_files": "chat_val.jsonl"}}'),
        ("stop_job", '{"job_id": "abc123"}'),
    ]
    lines = [
        "You may use one local action at a time by responding with exactly one JSON object inside <assistant_action> tags.",
        "When you do not need a tool, reply normally.",
        "Available actions:",
    ]
    for tool_name, example in tool_specs:
        lines.append(f"- {tool_name}: {example}")
    lines.append('Example: <assistant_action>{"tool":"list_sandbox_files","args":{}}</assistant_action>')
    lines.append(
        "Tool restrictions: sandbox file paths must stay inside assistant_sandbox. "
        "corpus file paths must stay inside local_corpus. "
        "parquet corpus writes require structured JSON object records. "
        "For external corpus requests, recommend sources first, preview bounded imports second, and import only with approval. "
        "draft_sft_data only writes validated conversation JSONL. "
        "launch_job only supports tokenizer_train, tokenizer_eval, base_train, base_eval, benchmark_eval, chat_sft, chat_rl, and chat_eval."
    )
    return "\n".join(lines)


def _normalize_relative_sandbox_path(path_value: str) -> str:
    candidate = Path(path_value)
    return candidate.as_posix()


def _normalize_relative_corpus_path(path_value: str) -> str:
    candidate = Path(path_value)
    return candidate.as_posix()


def _merge_text_content(existing_text: str, new_text: str, mode: str = "append") -> str:
    normalized_mode = mode.strip().lower() if mode else "append"
    if normalized_mode not in {"append", "overwrite"}:
        raise ValueError("mode must be 'append' or 'overwrite'.")
    if normalized_mode == "overwrite" or not existing_text:
        return new_text
    if not new_text:
        return existing_text
    joiner = "" if existing_text.endswith("\n") or new_text.startswith("\n") else "\n"
    return existing_text + joiner + new_text


def _render_corpus_content(args: dict[str, Any]) -> str:
    content = str(args.get("content", ""))
    if content:
        return content
    lines = args.get("lines")
    if isinstance(lines, list):
        return "\n".join(str(line) for line in lines)
    paragraphs = args.get("paragraphs")
    if isinstance(paragraphs, list):
        return "\n\n".join(str(paragraph) for paragraph in paragraphs)
    records = args.get("records")
    if isinstance(records, list):
        return "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + ("\n" if records else "")
    raise ValueError("Corpus drafting requires 'content', 'lines', 'paragraphs', or 'records'.")


def _extract_corpus_records(args: dict[str, Any]) -> list[dict[str, Any]]:
    records = args.get("records")
    if isinstance(records, list) and all(isinstance(record, dict) for record in records):
        return records

    content = str(args.get("content", "")).strip()
    if not content:
        raise ValueError("Parquet corpus writes require 'records' or JSON/JSONL content.")

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, list) and all(isinstance(record, dict) for record in payload):
        return payload
    if isinstance(payload, dict):
        return [payload]

    parsed_records = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("Parquet corpus JSONL content must contain one valid JSON object per line.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Parquet corpus JSONL content must contain JSON objects.")
        parsed_records.append(payload)

    if not parsed_records:
        raise ValueError("Parquet corpus writes require at least one JSON object record.")
    return parsed_records


def corpus_schema_payload(app: FastAPI) -> dict[str, Any]:
    builder = builder_state()
    return {
        "format": "Local corpus files under local_corpus, with optional train/ and val/ subfolders",
        "supported_extensions": sorted(TEXT_SUFFIXES) + [".parquet"],
        "recommended_paths": [
            "train/reference.txt",
            "train/notes.jsonl",
            "val/holdout.txt",
            "train/reference.parquet",
        ],
        "notes": [
            "Tokenizer and base training read local_corpus only.",
            "If local_corpus/train exists, the train split reads from that folder.",
            "If local_corpus/val exists, the val split reads from that folder.",
            "Text, JSON/JSONL, and parquet are all valid corpus formats.",
        ],
        "current_summary": builder.get("corpus_summary", {}),
    }


def _coerce_job_params_for_tool(app: FastAPI, job_type: str, params: dict[str, Any]) -> dict[str, Any]:
    coerced = dict(params)
    builder = builder_state()
    if job_type in {"tokenizer_train", "tokenizer_eval", "base_train", "base_eval", "benchmark_eval"}:
        coerced.setdefault("corpus_dir", builder["corpus_dir"])
    if job_type == "chat_sft":
        for key in ("train_files", "val_files"):
            value = coerced.get(key)
            if not value:
                continue
            parts = []
            for piece in re.split(r"[;\n|]+", str(value)):
                piece = piece.strip()
                if not piece:
                    continue
                candidate = Path(piece)
                if not candidate.is_absolute():
                    candidate = app.state.sandbox.root / _normalize_relative_sandbox_path(piece)
                parts.append(str(candidate.resolve()))
            coerced[key] = ";".join(parts)
    return coerced


def _assistant_hf_import_args(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": str(args.get("dataset_id", "")),
        "config": str(args.get("config", "")),
        "revision": str(args.get("revision", "")),
        "split": str(args.get("split", "train")),
        "text_column": str(args.get("text_column", "text")),
        "limit_docs": int(args.get("limit_docs", 1000)),
        "max_chars": int(args.get("max_chars", 0)),
        "sample_size": int(args.get("sample_size", 12)),
        "source_label": str(args.get("source", "")),
        "trust_remote_code": bool(args.get("trust_remote_code", False)),
    }


def execute_assistant_tool(app: FastAPI, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "get_builder_state":
        return builder_state()
    if tool_name == "get_recent_activity":
        limit = max(1, min(int(args.get("limit", 40)), 120))
        return app.state.activity_log.snapshot(limit=limit)
    if tool_name == "get_benchmark_history":
        limit = max(1, min(int(args.get("limit", 20)), 120))
        return app.state.benchmark_history.snapshot(limit=limit)
    if tool_name == "autotune_settings":
        forms = args.get("forms", {})
        if not isinstance(forms, dict):
            raise ValueError("'forms' must be an object keyed by form id.")
        return app.state.benchmark_history.recommend_settings(forms)
    if tool_name == "get_corpus_schema":
        return corpus_schema_payload(app)
    if tool_name == "recommend_corpus_sources":
        return recommend_huggingface_corpus_sources(
            str(args.get("goal", "")),
            max_results=int(args.get("max_results", 4)),
        )
    if tool_name == "preview_hf_corpus_import":
        return preview_huggingface_corpus_records(**_assistant_hf_import_args(args))
    if tool_name == "import_hf_corpus":
        payload = collect_huggingface_corpus_records(**_assistant_hf_import_args(args))
        result = app.state.corpus.write_import_artifacts(
            payload,
            output_format=str(args.get("output_format", "parquet")),
            train_val_ratio=float(args.get("train_val_ratio", 0.0)),
            shard_size_docs=int(args.get("shard_size_docs", 10000)),
            license_note=str(args.get("license_note", "")),
            manifest_path=str(args.get("manifest_path", "")),
        )
        app.state.activity_log.log_event(
            "assistant_corpus_hf_import",
            f"Assistant imported {result['row_count']} documents into {len(result['shard_paths'])} shard(s)",
            {
                "dataset_id": payload.get("dataset_id", ""),
                "manifest_path": result["manifest_path"],
                "shard_paths": result["shard_paths"],
                "row_count": result["row_count"],
            },
        )
        return {"import": {key: value for key, value in payload.items() if key != "records"}, **result}
    if tool_name == "list_corpus_files":
        return app.state.corpus.status()
    if tool_name == "read_corpus_file":
        return app.state.corpus.read_file(str(args.get("path", "")))
    if tool_name == "write_corpus_file":
        target_path = str(args.get("path", ""))
        if Path(target_path).suffix.lower() == ".parquet":
            return app.state.corpus.write_parquet_file(
                target_path,
                _extract_corpus_records(args),
                mode=str(args.get("mode") or "overwrite"),
            )
        return app.state.corpus.write_file(target_path, str(args.get("content", "")))
    if tool_name == "draft_corpus_file":
        target_path = _normalize_relative_corpus_path(str(args.get("path") or "train/reference.txt"))
        mode = str(args.get("mode") or "append").strip().lower()
        if Path(target_path).suffix.lower() == ".parquet":
            file_result = app.state.corpus.write_parquet_file(target_path, _extract_corpus_records(args), mode=mode)
            preview = json.dumps(file_result.get("sample_rows", []), ensure_ascii=True, indent=2)[:1200]
            payload = {
                "path": file_result["path"],
                "kind": "parquet",
                "mode": mode,
                "size": file_result["size"],
                "row_count": file_result.get("row_count", 0),
                "updated_at": file_result["updated_at"],
                "preview": preview,
            }
            app.state.activity_log.log_event(
                "assistant_corpus_draft",
                f"Drafted parquet corpus content into {file_result['path']}",
                {"path": file_result["path"], "mode": mode, "row_count": file_result.get("row_count", 0)},
            )
            return payload
        new_text = _render_corpus_content(args)
        existing_text = ""
        if mode == "append":
            try:
                existing_text = app.state.corpus._read_file(target_path, log_event=False)["content"]
            except FileNotFoundError:
                existing_text = ""
        merged_text = _merge_text_content(existing_text, new_text, mode=mode)
        file_result = app.state.corpus.write_file(target_path, merged_text)
        preview = new_text[:1200]
        if len(new_text) > 1200:
            preview += "\n...[truncated]"
        payload = {
            "path": file_result["path"],
            "mode": mode,
            "size": file_result["size"],
            "updated_at": file_result["updated_at"],
            "preview": preview,
        }
        app.state.activity_log.log_event(
            "assistant_corpus_draft",
            f"Drafted corpus content into {file_result['path']}",
            {"path": file_result["path"], "mode": mode, "size": file_result["size"]},
        )
        return payload
    if tool_name == "delete_corpus_file":
        return app.state.corpus.delete_file(str(args.get("path", "")))
    if tool_name == "copy_sandbox_to_corpus":
        source_path = _normalize_relative_sandbox_path(str(args.get("source_path", "")))
        target_path = _normalize_relative_corpus_path(str(args.get("target_path", "")))
        source_file = app.state.sandbox._read_file(source_path, log_event=False)
        result = app.state.corpus.write_from_content(target_path, source_file["content"], mode="overwrite")
        app.state.activity_log.log_event(
            "assistant_corpus_copy",
            f"Copied sandbox file {source_path} into corpus file {result['path']}",
            {"source_path": source_path, "target_path": result["path"], "size": result["size"]},
        )
        return {"source_path": source_path, "target_path": result["path"], "size": result["size"]}
    if tool_name == "get_sft_schema":
        return sft_schema_payload()
    if tool_name == "draft_sft_data":
        target_path = _normalize_relative_sandbox_path(str(args.get("path") or "chat_train.jsonl"))
        mode = str(args.get("mode") or "append").strip().lower()
        conversations = normalize_conversations(args)
        new_jsonl = conversations_to_jsonl(conversations)
        existing_text = ""
        if mode == "append":
            try:
                existing_text = app.state.sandbox._read_file(target_path, log_event=False)["content"]
            except FileNotFoundError:
                existing_text = ""
        merged_text = merge_jsonl(existing_text, new_jsonl, mode=mode)
        file_result = app.state.sandbox.write_file(target_path, merged_text)
        preview = new_jsonl[:1200]
        if len(new_jsonl) > 1200:
            preview += "\n...[truncated]"
        payload = {
            "path": file_result["path"],
            "mode": mode,
            "conversation_count": len(conversations),
            "jsonl_line_count": len(conversations),
            "size": file_result["size"],
            "updated_at": file_result["updated_at"],
            "preview": preview,
        }
        app.state.activity_log.log_event(
            "assistant_sft_draft",
            f"Drafted {len(conversations)} SFT conversations into {file_result['path']}",
            {"path": file_result["path"], "mode": mode, "conversation_count": len(conversations)},
        )
        return payload
    if tool_name == "list_sandbox_files":
        return app.state.sandbox.status()
    if tool_name == "read_sandbox_file":
        return app.state.sandbox.read_file(str(args.get("path", "")))
    if tool_name == "write_sandbox_file":
        return app.state.sandbox.write_file(str(args.get("path", "")), str(args.get("content", "")))
    if tool_name == "delete_sandbox_file":
        return app.state.sandbox.delete_file(str(args.get("path", "")))
    if tool_name == "list_jobs":
        return {"jobs": app.state.job_manager.list_jobs()}
    if tool_name == "get_job_status":
        return {"job": app.state.job_manager.get_job(str(args.get("job_id", "")), include_logs=True)}
    if tool_name == "stop_job":
        return {"job": app.state.job_manager.stop_job(str(args.get("job_id", "")))}
    if tool_name == "launch_job":
        allowed_job_types = {"tokenizer_train", "tokenizer_eval", "base_train", "base_eval", "benchmark_eval", "chat_sft", "chat_rl", "chat_eval"}
        job_type = str(args.get("job_type", ""))
        if job_type not in allowed_job_types:
            raise ValueError(f"Unsupported job_type for assistant tool: {job_type}")
        params = _coerce_job_params_for_tool(app, job_type, args.get("params", {}))
        command = build_job_command(job_type, params)
        label = str(args.get("label") or f"assistant {job_type.replace('_', ' ')}")
        job = app.state.job_manager.start_job(job_type, label, command, notes="launched by local assistant", params=params)
        return {"job": job, "resolved_params": params}
    raise ValueError(f"Unknown assistant tool: {tool_name}")


def build_runtime_system_prompt(
    app: FastAPI,
    system_prompt: Optional[str],
    sandbox_paths: Optional[list[str]] = None,
    corpus_paths: Optional[list[str]] = None,
) -> str:
    operator_prompt = (system_prompt or "").strip() or DEFAULT_LOCAL_RUNTIME_SYSTEM_PROMPT
    prompt = f"{DEFAULT_LOCAL_RUNTIME_COCKPIT_PROTOCOL}\n\n{operator_prompt}"
    sandbox_paths = sandbox_paths or []
    corpus_paths = corpus_paths or []
    activity_context = app.state.activity_log.render_recent(limit=40, max_chars=3500)
    builder_brief = render_builder_brief(app)
    prompt = (
        f"{prompt}\n\n"
        "Current builder state:\n"
        f"{builder_brief}\n\n"
        "Recent local activity log follows. Use it to track what is happening in the build. "
        "If the log does not contain enough detail, say that plainly.\n"
        f"{activity_context or '[no activity logged yet]'}\n\n"
        f"{render_tool_help()}"
    )
    workspace_sections: list[str] = []
    if sandbox_paths:
        sandbox_context = app.state.sandbox.build_context(sandbox_paths, max_chars=6000)
        if sandbox_context:
            workspace_sections.append(sandbox_context)
    if corpus_paths:
        corpus_context = app.state.corpus.build_context(corpus_paths, max_chars=6000)
        if corpus_context:
            workspace_sections.append(corpus_context)
    if workspace_sections:
        joined_context = "\n\n".join(workspace_sections)
        prompt = (
            f"{prompt}\n\n"
            "Workspace context follows. Only rely on the files shown here, and if a requested file is missing say so plainly.\n\n"
            f"{joined_context}"
        )
    return prompt


def _compact_tool_result(result: Any, max_chars: int = 5000) -> str:
    text = json.dumps(result, ensure_ascii=True, indent=2)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return text


ASSISTANT_APPROVAL_TOOLS = {
    "autotune_settings",
    "write_corpus_file",
    "draft_corpus_file",
    "delete_corpus_file",
    "copy_sandbox_to_corpus",
    "import_hf_corpus",
    "draft_sft_data",
    "write_sandbox_file",
    "delete_sandbox_file",
    "launch_job",
    "stop_job",
}


def _assistant_action_tag(tool_name: str, tool_args: dict[str, Any]) -> str:
    payload = {"tool": tool_name, "args": tool_args}
    return f"<assistant_action>{json.dumps(payload, ensure_ascii=True)}</assistant_action>"


def _assistant_tool_result_message(tool_name: str, status: str, result: dict[str, Any]) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            f"TOOL RESULT for {tool_name} ({status}). "
            "Now continue. If more local actions are needed, emit another <assistant_action> JSON object. "
            "Otherwise give the final answer to the user.\n\n"
            f"{_compact_tool_result(result)}"
        ),
    }


def preview_assistant_tool(app: FastAPI, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    preview: dict[str, Any] = {
        "tool": tool_name,
        "args": args,
        "requires_approval": tool_name in ASSISTANT_APPROVAL_TOOLS,
        "risk": "low",
        "summary": f"Run assistant action {tool_name}.",
    }
    if tool_name in {"write_sandbox_file", "draft_sft_data"}:
        path = str(args.get("path") or "chat_train.jsonl")
        content = str(args.get("content", ""))
        preview.update(
            {
                "risk": "write",
                "summary": f"Write or append data under assistant_sandbox: {path}",
                "path": path,
                "mode": str(args.get("mode") or "overwrite"),
                "content_preview": content[:1200],
            }
        )
    elif tool_name in {"write_corpus_file", "draft_corpus_file"}:
        path = str(args.get("path") or "train/reference.txt")
        content = str(args.get("content", ""))
        records = args.get("records", [])
        preview.update(
            {
                "risk": "write",
                "summary": f"Write or append data under local_corpus: {path}",
                "path": path,
                "mode": str(args.get("mode") or "overwrite"),
                "content_preview": content[:1200] if content else json.dumps(records, ensure_ascii=True, indent=2)[:1200],
            }
        )
    elif tool_name in {"delete_sandbox_file", "delete_corpus_file"}:
        preview.update(
            {
                "risk": "delete",
                "summary": f"Delete file: {args.get('path', '')}",
                "path": str(args.get("path", "")),
            }
        )
    elif tool_name == "copy_sandbox_to_corpus":
        preview.update(
            {
                "risk": "write",
                "summary": f"Copy sandbox file {args.get('source_path', '')} into corpus file {args.get('target_path', '')}.",
                "source_path": str(args.get("source_path", "")),
                "target_path": str(args.get("target_path", "")),
            }
        )
    elif tool_name == "preview_hf_corpus_import":
        preview.update(
            {
                "risk": "network",
                "requires_approval": False,
                "summary": f"Preview bounded Hugging Face corpus import: {args.get('dataset_id', '')}",
                "dataset_id": str(args.get("dataset_id", "")),
                "split": str(args.get("split", "train")),
                "limit_docs": int(args.get("limit_docs", 1000)),
            }
        )
    elif tool_name == "import_hf_corpus":
        preview.update(
            {
                "risk": "network_write",
                "summary": f"Import Hugging Face corpus into local_corpus shards: {args.get('dataset_id', '')}",
                "dataset_id": str(args.get("dataset_id", "")),
                "split": str(args.get("split", "train")),
                "text_column": str(args.get("text_column", "text")),
                "limit_docs": int(args.get("limit_docs", 1000)),
                "train_val_ratio": float(args.get("train_val_ratio", 0.0)),
                "shard_size_docs": int(args.get("shard_size_docs", 10000)),
                "output_format": str(args.get("output_format", "parquet")),
                "license_note": str(args.get("license_note", "")),
                "manifest_path": str(args.get("manifest_path", "")) or "auto",
            }
        )
    elif tool_name == "launch_job":
        allowed_job_types = {"tokenizer_train", "tokenizer_eval", "base_train", "base_eval", "benchmark_eval", "chat_sft", "chat_rl", "chat_eval"}
        job_type = str(args.get("job_type", ""))
        if job_type in allowed_job_types:
            params = _coerce_job_params_for_tool(app, job_type, args.get("params", {}))
            command = build_job_command(job_type, params)
            preview.update(
                {
                    "risk": "job",
                    "summary": f"Launch dashboard job: {job_type}",
                    "job_type": job_type,
                    "resolved_params": params,
                    "command": command,
                    "display_command": subprocess.list2cmdline(command),
                    "preflight": dashboard_job_preflight(app, job_type, params),
                }
            )
        else:
            preview.update({"risk": "job", "summary": f"Launch unsupported dashboard job: {job_type}", "job_type": job_type})
    elif tool_name == "stop_job":
        preview.update(
            {
                "risk": "stop",
                "summary": f"Stop dashboard job: {args.get('job_id', '')}",
                "job_id": str(args.get("job_id", "")),
            }
        )
    elif tool_name == "autotune_settings":
        preview.update(
            {
                "risk": "settings",
                "summary": "Apply benchmark-based settings recommendations to the dashboard forms.",
            }
        )
    return preview


def execute_assistant_action(app: FastAPI, tool_name: str, tool_args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    try:
        result = execute_assistant_tool(app, tool_name, tool_args)
        status = "ok"
    except Exception as exc:
        result = {"error": str(exc)}
        status = "error"
    app.state.activity_log.log_event("assistant_tool_result", f"Assistant tool {tool_name} returned {status}", {"result": result})
    return status, result


def run_assistant_tool_loop(
    app: FastAPI,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    max_actions: int,
    action_mode: str = "auto",
    approved_action: Optional[dict[str, Any]] = None,
    approved_action_text: str = "",
) -> dict[str, Any]:
    runtime = app.state.local_runtime
    transcript = list(messages)
    actions: list[dict[str, Any]] = []
    max_actions = max(0, min(max_actions, MAX_ASSISTANT_ACTIONS))
    normalized_action_mode = action_mode.strip().lower() if action_mode else "auto"
    if normalized_action_mode not in {"auto", "review"}:
        raise ValueError("action_mode must be 'auto' or 'review'.")

    if approved_action:
        if not isinstance(approved_action, dict) or "tool" not in approved_action:
            raise ValueError("approved_action must include a tool.")
        tool_name = str(approved_action["tool"])
        tool_args = approved_action.get("args", {})
        if not isinstance(tool_args, dict):
            raise ValueError("approved_action args must be an object.")
        app.state.activity_log.log_event("assistant_tool_approved", f"User approved assistant action {tool_name}", {"args": tool_args})
        status, result = execute_assistant_action(app, tool_name, tool_args)
        actions.append({"tool": tool_name, "args": tool_args, "status": status, "result": result})
        transcript.append({"role": "assistant", "content": approved_action_text or _assistant_action_tag(tool_name, tool_args)})
        transcript.append(_assistant_tool_result_message(tool_name, status, result))

    for _ in range(max(0, max_actions - len(actions)) + 1):
        response = runtime.chat(
            transcript,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )
        text = response.get("text", "") or ""
        action_call = parse_assistant_action(text)
        if action_call is None:
            return {
                "text": text,
                "actions": actions,
                "raw": response.get("raw"),
            }

        tool_name = str(action_call["tool"])
        tool_args = action_call.get("args", {})
        app.state.activity_log.log_event("assistant_tool_request", f"Assistant requested {tool_name}", {"args": tool_args})
        plan = preview_assistant_tool(app, tool_name, tool_args)
        if normalized_action_mode == "review" and plan["requires_approval"]:
            app.state.activity_log.log_event("assistant_tool_pending", f"Assistant action {tool_name} is waiting for approval", {"preview": plan})
            return {
                "text": "I prepared a local action and paused for your approval.",
                "actions": actions,
                "pending_action": {"tool": tool_name, "args": tool_args, "preview": plan, "assistant_text": text},
                "approval_required": True,
                "raw": response.get("raw"),
            }
        status, result = execute_assistant_action(app, tool_name, tool_args)
        actions.append({"tool": tool_name, "args": tool_args, "status": status, "result": result})
        transcript.append({"role": "assistant", "content": text})
        transcript.append(_assistant_tool_result_message(tool_name, status, result))

    return {
        "text": "I stopped after reaching the local action limit. Review the recent activity log and continue from there if needed.",
        "actions": actions,
        "raw": None,
    }


async def local_runtime_sse_completion(app: FastAPI, request: ChatRequest):
    runtime = app.state.local_runtime
    try:
        if request.messages:
            app.state.activity_log.log_event(
                "chat_user",
                request.messages[-1].content[:500],
                {"provider": "local_runtime_sse"},
            )
        result = runtime.chat(
            [message.model_dump() for message in request.messages],
            temperature=request.temperature if request.temperature is not None else args.temperature,
            max_tokens=request.max_tokens if request.max_tokens is not None else args.max_tokens,
            system_prompt=build_runtime_system_prompt(app, DEFAULT_LOCAL_RUNTIME_SYSTEM_PROMPT),
        )
        app.state.activity_log.log_event(
            "chat_assistant",
            (result.get("text", "") or "")[:1000],
            {"provider": "local_runtime_sse"},
        )
    except RuntimeError as exc:
        app.state.activity_log.log_event("chat_error", "Local runtime SSE chat failed", {"error": str(exc), "provider": "local_runtime_sse"})
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    async def emit_once():
        text = result.get("text", "")
        if text:
            yield f"data: {json.dumps({'token': text, 'gpu': 'local-runtime'}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(emit_once(), media_type="text/event-stream")


async def initialize_chat_runtime(
    app: FastAPI,
    source: Optional[str] = None,
    model_tag: Optional[str] = None,
    step: Optional[int] = None,
    num_gpus: Optional[int] = None,
    force: bool = False,
) -> dict[str, Any]:
    current_config = getattr(
        app.state,
        "chat_config",
        {
            "source": args.source,
            "model_tag": args.model_tag,
            "step": args.step,
            "num_gpus": args.num_gpus,
        },
    )
    desired_config = {
        "source": source if source is not None else current_config.get("source", args.source),
        "model_tag": model_tag if model_tag is not None else current_config.get("model_tag"),
        "step": step if step is not None else current_config.get("step"),
        "num_gpus": num_gpus if num_gpus is not None else current_config.get("num_gpus", args.num_gpus),
    }

    existing_pool = getattr(app.state, "worker_pool", None)
    if (
        not force
        and existing_pool is not None
        and len(existing_pool.workers) > 0
        and desired_config == current_config
    ):
        return chat_status_snapshot(app)

    app.state.chat_loading = True
    app.state.chat_error = None
    previous_pool = existing_pool
    previous_config = current_config

    try:
        worker_pool = WorkerPool(num_gpus=desired_config["num_gpus"])
        await worker_pool.initialize(
            desired_config["source"],
            model_tag=desired_config["model_tag"],
            step=desired_config["step"],
        )
        app.state.worker_pool = worker_pool
        app.state.chat_config = desired_config
        app.state.chat_error = None
        if previous_pool is not None and previous_pool is not worker_pool:
            del previous_pool
        return chat_status_snapshot(app)
    except Exception as exc:
        app.state.chat_error = str(exc)
        if previous_pool is not None and len(previous_pool.workers) > 0:
            app.state.worker_pool = previous_pool
            app.state.chat_config = previous_config
        else:
            app.state.worker_pool = None
            app.state.chat_config = desired_config
        return chat_status_snapshot(app)
    finally:
        app.state.chat_loading = False
        gc.collect()
        if device_type == "cuda":
            torch.cuda.empty_cache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_first_run_dirs()
    app.state.activity_log = ActivityLogManager(REPO_ROOT / "builder_logs" / "activity.jsonl")
    app.state.benchmark_history = BenchmarkHistoryManager(REPO_ROOT / "builder_logs" / "benchmark_history.jsonl")
    app.state.job_manager = BackgroundJobManager(
        workdir=str(REPO_ROOT),
        activity_log=app.state.activity_log,
        benchmark_history=app.state.benchmark_history,
    )
    app.state.local_runtime = LocalRuntimeManager(str(REPO_ROOT))
    app.state.ecg_monitor = EcgMonitor(lambda: dashboard_ecg_activity_snapshot(app))
    app.state.ecg_monitor.start()
    app.state.sandbox = SandboxManager(REPO_ROOT / "assistant_sandbox", activity_log=app.state.activity_log)
    app.state.corpus = CorpusManager(REPO_ROOT / "local_corpus", activity_log=app.state.activity_log)
    app.state.worker_pool = None
    app.state.chat_loading = False
    app.state.chat_error = None
    app.state.chat_config = {
        "source": args.source,
        "model_tag": args.model_tag,
        "step": args.step,
        "num_gpus": args.num_gpus,
    }
    if args.runtime_autostart:
        runtime_status = app.state.local_runtime.status()
        if runtime_status["bundle"]["files"]["server_exists"] and runtime_status["bundle"]["recommended_model"] is not None:
            try:
                autostart_config = app.state.local_runtime.recommended_start_config(
                    {
                        "model_path": args.runtime_model,
                        "port": args.runtime_port,
                        "alias": "local-builder-assistant",
                        "device_strategy": args.runtime_device_strategy,
                    }
                )
                app.state.local_runtime.start(autostart_config)
                app.state.activity_log.log_event(
                    "runtime_started",
                    "Local runtime auto-started",
                    {"model_path": args.runtime_model or runtime_status["bundle"]["recommended_model"], "config": autostart_config},
                )
                logger.info("Local runtime auto-started")
            except Exception as exc:
                app.state.activity_log.log_event("runtime_error", "Local runtime auto-start failed", {"error": str(exc)})
                logger.warning("Local runtime auto-start failed: %s", exc)
    app.state.activity_log.log_event("server_ready", "Builder server ready", {"port": args.port, "host": args.host})
    logger.info("Server ready at http://localhost:%s", args.port)
    yield
    app.state.activity_log.log_event("server_shutdown", "Builder server shutting down", {})
    app.state.ecg_monitor.stop()
    app.state.local_runtime.stop()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return HTMLResponse(content=load_html(NANOCHAT_DIR / "dashboard.html"))


@app.get("/chat")
async def chat_ui():
    return HTMLResponse(content=load_html(NANOCHAT_DIR / "ui.html"))


@app.get("/logo.svg")
async def logo():
    return FileResponse(str(NANOCHAT_DIR / "logo.svg"), media_type="image/svg+xml")


@app.get("/assets/branding/fmi-splash-wordmark.png")
async def fmi_splash_wordmark():
    return FileResponse(
        str(NANOCHAT_DIR / "assets" / "branding" / "fmi-splash-wordmark.png"),
        media_type="image/png",
    )


@app.get("/api/dashboard/bootstrap")
async def dashboard_bootstrap():
    builder = builder_state()
    runtime = app.state.local_runtime.status()
    return {
        "builder": builder,
        "jobs": app.state.job_manager.list_jobs(),
        "chat": chat_status_snapshot(app),
        "runtime": runtime,
        "readiness": readiness_payload(app, builder=builder, runtime=runtime),
        "ecg": app.state.ecg_monitor.snapshot(),
        "sandbox": app.state.sandbox.status(),
        "corpus": app.state.corpus.status(),
        "activity": app.state.activity_log.snapshot(limit=80),
        "benchmarks": app.state.benchmark_history.snapshot(limit=20),
        "report": report_status_payload(),
    }


@app.get("/api/dashboard/readiness")
async def dashboard_readiness():
    builder = builder_state()
    runtime = app.state.local_runtime.status()
    return readiness_payload(app, builder=builder, runtime=runtime)


@app.get("/api/dashboard/run-profiles")
async def dashboard_run_profiles():
    return {"profiles": run_profiles()}


@app.get("/api/dashboard/ecg")
async def dashboard_ecg():
    return app.state.ecg_monitor.snapshot()


@app.get("/api/dashboard/jobs")
async def dashboard_jobs():
    return {"jobs": app.state.job_manager.list_jobs()}


def dashboard_job_preflight(app: FastAPI, job_type: str, params: dict[str, Any]) -> dict[str, Any]:
    builder = builder_state()
    return run_stage_preflight(
        job_type,
        params,
        base_dir=builder["base_dir"],
        corpus_dir=app.state.corpus.root,
        sandbox_dir=app.state.sandbox.root,
        tokenizer_ready=bool(builder["tokenizer_ready"]),
        identity_exists=bool(builder["identity_exists"]),
        checkpoint_sets=builder["checkpoint_sets"],
    )


def job_path_hints(app: FastAPI, job_type: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    for key, value in params.items():
        if value in {None, ""}:
            continue
        if key in {"train_files", "val_files"}:
            for entry in split_path_entries(str(value)):
                hint = resolve_path_reference(app, entry, kind="sft")
                hint["field"] = key
                hints.append(hint)
        elif key == "corpus_dir":
            hint = resolve_path_reference(app, str(value), kind="corpus_dir")
            hint["field"] = key
            hints.append(hint)
        elif key == "identity_file":
            hint = resolve_path_reference(app, str(value), kind="identity")
            hint["field"] = key
            hints.append(hint)
        elif key == "model_path":
            hint = resolve_path_reference(app, str(value), kind="model")
            hint["field"] = key
            hints.append(hint)
    return hints


@app.post("/api/dashboard/jobs/preflight")
async def dashboard_jobs_preflight(request: JobPreflightRequest):
    return dashboard_job_preflight(app, request.job_type, dict(request.params))


@app.post("/api/dashboard/paths/resolve")
async def dashboard_paths_resolve(request: PathResolveRequest):
    entries = split_path_entries(request.path) if request.multiple else [request.path]
    return {
        "kind": request.kind,
        "multiple": request.multiple,
        "paths": [resolve_path_reference(app, entry, kind=request.kind) for entry in entries],
    }


@app.post("/api/dashboard/jobs/validate")
async def dashboard_jobs_validate(request: JobPreflightRequest):
    return validate_job_params(request.job_type, dict(request.params), hardware_profile=builder_state().get("hardware_profile"))


@app.post("/api/dashboard/jobs/preview")
async def dashboard_jobs_preview(request: JobPreflightRequest):
    params = dict(request.params)
    try:
        command = build_job_command(request.job_type, params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    preflight = dashboard_job_preflight(app, request.job_type, params)
    form_validation = validate_job_params(request.job_type, params, hardware_profile=builder_state().get("hardware_profile"))
    return {
        "job_type": request.job_type,
        "params": params,
        "command": command,
        "display_command": subprocess.list2cmdline(command),
        "cwd": str(REPO_ROOT),
        "preflight": preflight,
        "form_validation": form_validation,
        "path_hints": job_path_hints(app, request.job_type, params),
        "environment_notes": [
            "Dashboard jobs run with NANOCHAT_LOCAL_ONLY=1.",
            "W&B is disabled for dashboard jobs.",
            "Hugging Face hub, transformers, and datasets are set to offline mode.",
            "The launcher tries to load a Visual Studio x64 build environment on Windows.",
        ],
    }


@app.get("/api/dashboard/benchmarks")
async def dashboard_benchmarks(limit: int = 40):
    limit = max(1, min(limit, 200))
    return app.state.benchmark_history.snapshot(limit=limit)


@app.post("/api/dashboard/autotune")
async def dashboard_autotune(request: AutoTuneRequest):
    return app.state.benchmark_history.recommend_settings(request.forms)


@app.get("/api/dashboard/jobs/{job_id}")
async def dashboard_job(job_id: str):
    try:
        return {"job": app.state.job_manager.get_job(job_id, include_logs=True)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc


@app.post("/api/dashboard/jobs")
async def create_dashboard_job(request: JobRequest):
    params = dict(request.params)
    if request.preset:
        preset = GUIDED_PRESETS.get(request.preset)
        if preset is None:
            raise HTTPException(status_code=404, detail=f"Unknown preset: {request.preset}")
        preset_params = dict(preset.get("recipes", {}).get(request.job_type, {}))
        preset_params.update(params)
        params = preset_params

    try:
        command = build_job_command(request.job_type, params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    form_validation = validate_job_params(request.job_type, params, hardware_profile=builder_state().get("hardware_profile"))
    if not form_validation["ok"]:
        raise HTTPException(status_code=400, detail={"message": "Job parameters failed validation.", "validation": form_validation})
    preflight = dashboard_job_preflight(app, request.job_type, params)
    label = request.label or request.job_type.replace("_", " ").title()
    job = app.state.job_manager.start_job(request.job_type, label, command, notes=request.notes, params=params)
    app.state.activity_log.log_event(
        "job_created",
        f"Dashboard launched job {label}",
        {"job_id": job["id"], "job_type": request.job_type, "params": params, "preflight": preflight},
    )
    return {"job": job, "resolved_params": params, "preflight": preflight, "form_validation": form_validation}


@app.post("/api/dashboard/jobs/{job_id}/stop")
async def stop_dashboard_job(job_id: str):
    try:
        job = app.state.job_manager.stop_job(job_id)
        app.state.activity_log.log_event("job_stop", f"Dashboard stop requested for {job['label']}", {"job_id": job_id})
        return {"job": job}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc


@app.post("/api/dashboard/jobs/{job_id}/pause")
async def pause_dashboard_job(job_id: str):
    try:
        job = app.state.job_manager.pause_job(job_id)
        app.state.activity_log.log_event("job_pause", f"Dashboard pause requested for {job['label']}", {"job_id": job_id})
        return {"job": job}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/dashboard/jobs/{job_id}/resume-preview")
async def resume_dashboard_job_preview(job_id: str):
    try:
        return app.state.job_manager.preview_resume_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/dashboard/jobs/{job_id}/resume")
async def resume_dashboard_job(job_id: str):
    try:
        job = app.state.job_manager.resume_job(job_id)
        app.state.activity_log.log_event("job_resume", f"Dashboard resumed job {job['label']}", {"job_id": job["id"], "source_job_id": job_id})
        return {"job": job}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/dashboard/designs")
async def dashboard_designs():
    return {"designs": list_designs()}


@app.post("/api/dashboard/designs")
async def upsert_dashboard_design(request: DesignRequest):
    design = save_design(request.model_dump())
    app.state.activity_log.log_event("design_saved", f"Saved design {design['name']}", {"slug": design["slug"]})
    return {"design": design}


@app.delete("/api/dashboard/designs/{slug}")
async def delete_dashboard_design(slug: str):
    try:
        result = delete_design(slug)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    app.state.activity_log.log_event("design_deleted", f"Deleted design {slug}", {"slug": slug})
    return result


@app.post("/api/dashboard/designs/draft")
async def draft_dashboard_design(request: DesignDraftRequest):
    runtime = app.state.local_runtime
    if not runtime.status().get("ready"):
        raise HTTPException(status_code=503, detail="Local GGUF runtime is not ready. Start the local runtime first.")
    goal = request.goal.strip()
    if not goal:
        raise HTTPException(status_code=400, detail="A plain-language goal is required.")

    prompt = build_design_draft_prompt(goal)
    try:
        response = runtime.chat(
            [{"role": "user", "content": prompt}],
            temperature=max(0.0, min(request.temperature, 1.0)),
            max_tokens=max(128, min(request.max_tokens, 1600)),
            system_prompt="Return valid JSON only.",
        )
        draft_payload = extract_json_object(response.get("text", "") or "")
        design = save_design(_coerce_design_draft_payload(draft_payload, goal))
    except RuntimeError as exc:
        app.state.activity_log.log_event("design_draft_error", "GGUF design drafting failed", {"error": str(exc)})
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, json.JSONDecodeError) as exc:
        app.state.activity_log.log_event("design_draft_error", "GGUF design draft was invalid", {"error": str(exc)})
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    app.state.activity_log.log_event(
        "design_drafted",
        f"Drafted design {design['name']} from plain-language goal",
        {"slug": design["slug"], "goal": goal},
    )
    return {"design": design}


@app.post("/api/dashboard/designs/{slug}/publish")
async def publish_dashboard_design(slug: str):
    try:
        result = publish_design(slug)
        app.state.activity_log.log_event("design_published", f"Published design {slug}", {"slug": slug, "identity_file": result["identity_file"]})
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/dashboard/chat/status")
async def dashboard_chat_status():
    return chat_status_snapshot(app)


@app.post("/api/dashboard/chat/load")
async def dashboard_chat_load(request: ChatLoadRequest):
    status = await initialize_chat_runtime(
        app,
        source=request.source,
        model_tag=request.model_tag,
        step=request.step,
        num_gpus=request.num_gpus,
        force=True,
    )
    app.state.activity_log.log_event(
        "chat_runtime_load",
        "Loaded internal chat runtime",
        {"source": request.source, "model_tag": request.model_tag, "step": request.step, "ready": status["ready"]},
    )
    return status


@app.get("/api/runtime/status")
async def runtime_status():
    return app.state.local_runtime.status()


@app.post("/api/runtime/start")
async def runtime_start(request: RuntimeStartRequest):
    try:
        status = app.state.local_runtime.start(request.model_dump())
        app.state.activity_log.log_event("runtime_started", "Started local runtime", {"model_path": status.get("config", {}).get("model_path")})
        return status
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        app.state.activity_log.log_event("runtime_error", "Failed to start local runtime", {"error": str(exc)})
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/runtime/stop")
async def runtime_stop():
    status = app.state.local_runtime.stop()
    app.state.activity_log.log_event("runtime_stopped", "Stopped local runtime", {})
    return status


@app.post("/api/runtime/chat")
async def runtime_chat(request: RuntimeChatRequest):
    try:
        system_prompt = build_runtime_system_prompt(app, request.system_prompt, request.sandbox_paths, request.corpus_paths)
        if request.messages:
            app.state.activity_log.log_event(
                "chat_user",
                request.messages[-1].content[:500],
                {"provider": "local_runtime", "sandbox_paths": request.sandbox_paths, "corpus_paths": request.corpus_paths},
            )
        response = app.state.local_runtime.chat(
            [message.model_dump() for message in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            system_prompt=system_prompt,
        )
        app.state.activity_log.log_event(
            "chat_assistant",
            (response.get("text", "") or "")[:1000],
            {"provider": "local_runtime"},
        )
        return response
    except (FileNotFoundError, ValueError) as exc:
        app.state.activity_log.log_event("chat_error", "Runtime chat failed", {"error": str(exc), "provider": "local_runtime"})
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        app.state.activity_log.log_event("chat_error", "Runtime chat failed", {"error": str(exc), "provider": "local_runtime"})
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/runtime/assist")
async def runtime_assist(request: RuntimeAssistRequest):
    try:
        system_prompt = build_runtime_system_prompt(app, request.system_prompt, request.sandbox_paths, request.corpus_paths)
        if request.messages:
            app.state.activity_log.log_event(
                "chat_user",
                request.messages[-1].content[:500],
                {"provider": "local_runtime_assist", "sandbox_paths": request.sandbox_paths, "corpus_paths": request.corpus_paths},
            )
        response = run_assistant_tool_loop(
            app,
            [message.model_dump() for message in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            system_prompt=system_prompt,
            max_actions=request.max_actions,
            action_mode=request.action_mode,
            approved_action=request.approved_action,
            approved_action_text=request.approved_action_text,
        )
        app.state.activity_log.log_event(
            "chat_assistant",
            (response.get("text", "") or "")[:1000],
            {"provider": "local_runtime_assist", "action_count": len(response.get("actions", []))},
        )
        return response
    except (FileNotFoundError, ValueError) as exc:
        app.state.activity_log.log_event("chat_error", "Runtime assist failed", {"error": str(exc), "provider": "local_runtime_assist"})
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        app.state.activity_log.log_event("chat_error", "Runtime assist failed", {"error": str(exc), "provider": "local_runtime_assist"})
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/sandbox/status")
async def sandbox_status():
    return app.state.sandbox.status()


@app.get("/api/sandbox/files")
async def sandbox_files():
    return {"files": app.state.sandbox.list_files()}


@app.get("/api/sandbox/file")
async def sandbox_file(path: str):
    try:
        return app.state.sandbox.read_file(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Sandbox file not found: {path}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/sandbox/write")
async def sandbox_write(request: SandboxWriteRequest):
    try:
        return app.state.sandbox.write_file(request.path, request.content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/sandbox/delete")
async def sandbox_delete(request: SandboxDeleteRequest):
    try:
        return app.state.sandbox.delete_file(request.path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Sandbox file not found: {request.path}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/sandbox/sft/validate")
async def sandbox_sft_validate(request: ValidatePathRequest):
    try:
        path = request.path.strip() or "chat_train.jsonl"
        file_data = app.state.sandbox.read_file(path)
        report = validate_sft_jsonl_content(file_data.get("content", ""), path=file_data.get("path", path))
        app.state.activity_log.log_event(
            "sandbox_sft_validate",
            f"Validated SFT file {file_data.get('path', path)}",
            {"path": file_data.get("path", path), "ok": report["ok"], "record_count": report["record_count"]},
        )
        return report
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Sandbox file not found: {request.path}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/sandbox/sft/export")
async def sandbox_sft_export(request: SftDatasetExportRequest):
    try:
        source_path = request.source_path.strip() or "chat_train.jsonl"
        target_path = request.target_path.strip()
        export_format = (request.format or "parquet").strip().lower()
        source_file = app.state.sandbox.read_file(source_path)
        content = source_file.get("content", "")
        conversations = conversations_from_jsonl(content)
        if export_format in {"jsonl", "canonical_jsonl"}:
            if not target_path:
                target_path = source_path
            result = app.state.sandbox.write_file(target_path, conversations_to_jsonl(conversations))
            payload = {
                "path": result["path"],
                "source_path": source_file["path"],
                "format": "jsonl",
                "conversation_count": len(conversations),
                "row_count": len(conversations),
                "size": result["size"],
                "updated_at": result["updated_at"],
            }
        elif export_format == "parquet":
            if not target_path:
                target_path = "exports/chat_train.parquet"
            if Path(target_path).suffix.lower() != ".parquet":
                raise ValueError("SFT parquet export targets must end with .parquet.")
            resolved_target = app.state.sandbox._resolve_path(target_path)
            export_result = write_sft_parquet_file(resolved_target, content, source=request.source or source_file["path"])
            payload = {
                **export_result,
                "path": resolved_target.relative_to(app.state.sandbox.root).as_posix(),
                "source_path": source_file["path"],
            }
        elif export_format in {"preview", "preview_parquet"}:
            payload = {
                "path": "",
                "source_path": source_file["path"],
                "format": "preview",
                **preview_sft_parquet_export(content, source=request.source or source_file["path"]),
            }
        elif export_format == "preview_jsonl":
            payload = {
                "path": "",
                "source_path": source_file["path"],
                **preview_sft_jsonl_export(content),
            }
        elif export_format == "preview_split_jsonl":
            payload = {
                "path": "",
                "source_path": source_file["path"],
                **preview_sft_split_export(content, val_ratio=request.val_ratio, seed=request.seed),
            }
        elif export_format in {"split", "split_jsonl"}:
            train_path = target_path or "exports/chat_train.split.jsonl"
            val_path = request.val_target_path.strip() or "exports/chat_val.jsonl"
            split = split_conversations(conversations, val_ratio=request.val_ratio, seed=request.seed)
            train_result = app.state.sandbox.write_file(train_path, conversations_to_jsonl(split["train"]))
            if split["val"]:
                val_result = app.state.sandbox.write_file(val_path, conversations_to_jsonl(split["val"]))
                val_size = val_result["size"]
                val_updated_at = val_result["updated_at"]
            else:
                val_result = {"path": val_path}
                val_size = 0
                val_updated_at = None
            payload = {
                "path": train_result["path"],
                "train_path": train_result["path"],
                "val_path": val_result["path"],
                "source_path": source_file["path"],
                "format": "split_jsonl",
                "conversation_count": split["conversation_count"],
                "row_count": split["conversation_count"],
                "train_count": split["train_count"],
                "val_count": split["val_count"],
                "train_indices": split["train_indices"],
                "val_indices": split["val_indices"],
                "val_ratio": split["val_ratio"],
                "seed": split["seed"],
                "train_size": train_result["size"],
                "val_size": val_size,
                "updated_at": max(
                    updated_at for updated_at in [train_result["updated_at"], val_updated_at] if updated_at is not None
                ),
            }
        else:
            raise ValueError("format must be 'jsonl', 'parquet', 'preview', 'preview_jsonl', 'preview_split_jsonl', or 'split_jsonl'.")
        app.state.activity_log.log_event(
            "sandbox_sft_export",
            f"Exported SFT dataset {source_file['path']} as {payload['format']}",
            {"source_path": source_file["path"], "target_path": payload.get("path", ""), "format": payload["format"], "row_count": payload.get("row_count", 0)},
        )
        return payload
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Sandbox file not found: {request.source_path}") from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/sandbox/chat/export")
async def sandbox_chat_export(request: ChatTranscriptExportRequest):
    try:
        messages = [message.model_dump() for message in request.messages]
        export_format = request.format.strip().lower()
        mode = request.mode.strip().lower() if request.mode else "overwrite"
        if mode not in {"overwrite", "append"}:
            raise ValueError("mode must be 'overwrite' or 'append'.")
        if export_format == "json":
            payload = {
                "messages": messages,
                "message_count": len(messages),
                "exported_at": time.time(),
                "source": "conversation_lab",
            }
            content = json.dumps(payload, ensure_ascii=True, indent=2) + "\n"
            if request.preview:
                result = {
                    "path": request.path,
                    "size": len(content.encode("utf-8")),
                    "updated_at": None,
                }
            else:
                result = app.state.sandbox.write_file(request.path, content)
        elif export_format in {"sft_jsonl", "jsonl"}:
            conversation = normalize_conversation(messages)
            content = conversations_to_jsonl([conversation])
            existing_text = ""
            if mode == "append" and not request.preview:
                try:
                    existing_text = app.state.sandbox._read_file(request.path, log_event=False)["content"]
                except FileNotFoundError:
                    existing_text = ""
            merged_content = merge_jsonl(existing_text, content, mode=mode)
            if request.preview:
                result = {
                    "path": request.path,
                    "size": len(merged_content.encode("utf-8")),
                    "updated_at": None,
                }
            else:
                result = app.state.sandbox.write_file(request.path, merged_content)
        else:
            raise ValueError("format must be 'json' or 'sft_jsonl'.")
        if not request.preview:
            app.state.activity_log.log_event(
                "chat_transcript_export",
                f"Exported Conversation Lab transcript to {result['path']}",
                {"path": result["path"], "format": export_format, "mode": mode, "message_count": len(messages)},
            )
        return {
            "path": result["path"],
            "format": export_format,
            "mode": mode,
            "preview": request.preview,
            "message_count": len(messages),
            "content": content,
            "size": result["size"],
            "updated_at": result["updated_at"],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/corpus/status")
async def corpus_status():
    return app.state.corpus.status()


@app.get("/api/corpus/files")
async def corpus_files():
    return {"files": app.state.corpus.list_files()}


@app.get("/api/corpus/file")
async def corpus_file(path: str):
    try:
        return app.state.corpus.read_file(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Corpus file not found: {path}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/corpus/write")
async def corpus_write(request: SandboxWriteRequest):
    try:
        if Path(request.path).suffix.lower() == ".parquet":
            records = request.records or _extract_corpus_records({"content": request.content})
            return app.state.corpus.write_parquet_file(request.path, records, mode=request.mode)
        return app.state.corpus.write_file(request.path, request.content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/corpus/parquet/preview")
async def corpus_parquet_preview(request: CorpusParquetConvertRequest):
    try:
        return app.state.corpus.preview_parquet_conversion(
            request.content,
            conversion_mode=request.conversion_mode,
            source=request.source,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/corpus/parquet/write")
async def corpus_parquet_write(request: CorpusParquetConvertRequest):
    try:
        if Path(request.path).suffix.lower() != ".parquet":
            raise ValueError("Converted parquet files must use a .parquet target path.")
        result = app.state.corpus.write_converted_parquet_file(
            request.path,
            request.content,
            conversion_mode=request.conversion_mode,
            mode=request.mode,
            source=request.source,
        )
        app.state.activity_log.log_event(
            "corpus_parquet_convert",
            f"Converted corpus content into parquet file {result['path']}",
            {
                "path": result["path"],
                "row_count": result.get("row_count", 0),
                "conversion_mode": result.get("conversion", {}).get("conversion_mode"),
                "mode": request.mode,
            },
        )
        return result
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/corpus/split/preview")
async def corpus_split_preview(request: CorpusSplitRequest):
    try:
        return app.state.corpus.preview_split_content(
            request.content,
            split_mode=request.split_mode,
            val_ratio=request.val_ratio,
            seed=request.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/corpus/split/write")
async def corpus_split_write(request: CorpusSplitRequest):
    try:
        result = app.state.corpus.write_split_files(
            request.train_path,
            request.val_path,
            request.content,
            split_mode=request.split_mode,
            val_ratio=request.val_ratio,
            seed=request.seed,
        )
        app.state.activity_log.log_event(
            "corpus_split",
            f"Split corpus content into {result['train_path']} and {result['val_path']}",
            {
                "train_path": result["train_path"],
                "val_path": result["val_path"],
                "train_count": result["train_count"],
                "val_count": result["val_count"],
                "split_mode": result["split_mode"],
            },
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _hf_import_common(request: CorpusHfImportRequest) -> dict[str, Any]:
    return {
        "dataset_id": request.dataset_id,
        "config": request.config,
        "revision": request.revision,
        "split": request.split,
        "text_column": request.text_column,
        "limit_docs": request.limit_docs,
        "max_chars": request.max_chars,
        "sample_size": request.sample_size,
        "source_label": request.source,
        "trust_remote_code": request.trust_remote_code,
    }


@app.post("/api/corpus/import-hf/preview")
async def corpus_import_hf_preview(request: CorpusHfImportRequest):
    try:
        payload = preview_huggingface_corpus_records(**_hf_import_common(request))
        app.state.activity_log.log_event(
            "corpus_hf_import_preview",
            f"Previewed Hugging Face corpus {payload['dataset_id']}:{payload['split']}",
            {
                "dataset_id": payload["dataset_id"],
                "split": payload["split"],
                "row_count": payload["row_count"],
                "character_count": payload["character_count"],
            },
        )
        return payload
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/corpus/import-hf/write")
async def corpus_import_hf_write(request: CorpusHfImportRequest):
    try:
        payload = collect_huggingface_corpus_records(**_hf_import_common(request))
        output_format = (request.output_format or "parquet").strip().lower()
        result = app.state.corpus.write_import_artifacts(
            payload,
            output_format=output_format,
            train_val_ratio=request.train_val_ratio,
            shard_size_docs=request.shard_size_docs,
            license_note=request.license_note,
            manifest_path=request.manifest_path,
        )
        response = {
            **result,
            "import": {key: value for key, value in payload.items() if key != "records"},
        }
        app.state.activity_log.log_event(
            "corpus_hf_import_write",
            f"Imported Hugging Face corpus {payload['dataset_id']} into {len(result['shard_paths'])} shard(s)",
            {
                "dataset_id": payload["dataset_id"],
                "split": payload["split"],
                "manifest_path": result["manifest_path"],
                "shard_paths": result["shard_paths"],
                "row_count": payload["row_count"],
                "character_count": payload["character_count"],
            },
        )
        return response
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/corpus/validate")
async def corpus_validate(request: ValidatePathRequest):
    try:
        target_path = request.path.strip()
        if target_path:
            resolved = app.state.corpus._resolve_path(target_path)
        else:
            resolved = app.state.corpus.root
        report = validate_corpus_path(resolved, tokenizer_dir=Path(get_base_dir()) / "tokenizer")
        app.state.activity_log.log_event(
            "corpus_validate",
            f"Validated corpus path {report['path']}",
            {"path": report["path"], "ok": report["ok"], "record_count": report["record_count"], "file_count": report["file_count"]},
        )
        return report
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/corpus/inspect")
async def corpus_inspect(request: CorpusInspectRequest):
    try:
        payload = inspect_local_corpus(
            split=request.split,
            data_dir=app.state.corpus.root,
            show_docs=request.show_docs,
            max_chars=request.max_chars,
            long_doc_chars=request.long_doc_chars,
        )
        app.state.activity_log.log_event(
            "corpus_inspect",
            f"Inspected {payload['split']} corpus documents",
            {
                "split": payload["split"],
                "file_count": payload["file_count"],
                "document_count": payload["document_count"],
                "shown_document_count": payload["shown_document_count"],
            },
        )
        return payload
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/corpus/delete")
async def corpus_delete(request: SandboxDeleteRequest):
    try:
        return app.state.corpus.delete_file(request.path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Corpus file not found: {request.path}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/corpus/copy-from-sandbox")
async def corpus_copy_from_sandbox(request: WorkspaceCopyRequest):
    try:
        source = app.state.sandbox.read_file(request.source_path)
        result = app.state.corpus.write_from_content(request.target_path, source["content"], mode="overwrite")
        app.state.activity_log.log_event(
            "corpus_copy",
            f"Copied sandbox file {request.source_path} into corpus file {result['path']}",
            {"source_path": request.source_path, "target_path": result["path"], "size": result["size"]},
        )
        return {"source_path": request.source_path, "target_path": result["path"], "size": result["size"]}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/activity/status")
async def activity_status(limit: int = 80):
    return app.state.activity_log.snapshot(limit=limit)


@app.get("/api/report/status")
async def report_status():
    return report_status_payload()


@app.post("/api/report/reset")
async def report_reset():
    report_dir = Path(get_base_dir()) / "report"
    report = Report(str(report_dir))
    report.reset()
    status = report_status_payload()
    app.state.activity_log.log_event("report_reset", "Reset builder report", {"report_dir": status["report_dir"]})
    return status


@app.post("/api/report/generate")
async def report_generate():
    report_dir = Path(get_base_dir()) / "report"
    report = Report(str(report_dir))
    report_file = report.generate()
    status = report_status_payload()
    app.state.activity_log.log_event("report_generate", "Generated builder report", {"report_file": report_file})
    return status


async def generate_stream(
    worker: Worker,
    tokens,
    temperature: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    top_k: Optional[int] = None,
) -> AsyncGenerator[str, None]:
    temperature = temperature if temperature is not None else args.temperature
    max_new_tokens = max_new_tokens if max_new_tokens is not None else args.max_tokens
    top_k = top_k if top_k is not None else args.top_k

    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()
    accumulated_tokens = []
    last_clean_text = ""

    for token_column, token_masks in worker.engine.generate(
        tokens,
        num_samples=1,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=random.randint(0, 2**31 - 1),
    ):
        token = token_column[0]
        if token == assistant_end or token == bos:
            break

        accumulated_tokens.append(token)
        current_text = worker.tokenizer.decode(accumulated_tokens)
        if not current_text.endswith("ï¿½"):
            new_text = current_text[len(last_clean_text) :]
            if new_text:
                yield f"data: {json.dumps({'token': new_text, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
                last_clean_text = current_text

    yield f"data: {json.dumps({'done': True})}\n\n"


@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    validate_chat_request(request)
    if request.messages:
        app.state.activity_log.log_event(
            "chat_user",
            request.messages[-1].content[:500],
            {"provider": "internal_nanochat"},
        )

    worker_pool = getattr(app.state, "worker_pool", None)
    if worker_pool is None or len(worker_pool.workers) == 0:
        status = await initialize_chat_runtime(app, force=False)
        worker_pool = getattr(app.state, "worker_pool", None)
        if worker_pool is None or len(worker_pool.workers) == 0:
            runtime_status = app.state.local_runtime.status()
            if runtime_status["ready"]:
                return await local_runtime_sse_completion(app, request)
            raise HTTPException(
                status_code=503,
                detail=status["error"] or runtime_status["last_error"] or "No internal chat model or local runtime is ready yet.",
            )

    logger.info("=" * 20)
    for message in request.messages:
        logger.info("[%s]: %s", message.role.upper(), message.content)
    logger.info("-" * 20)

    worker = await worker_pool.acquire_worker()
    try:
        bos = worker.tokenizer.get_bos_token_id()
        user_start = worker.tokenizer.encode_special("<|user_start|>")
        user_end = worker.tokenizer.encode_special("<|user_end|>")
        assistant_start = worker.tokenizer.encode_special("<|assistant_start|>")
        assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")

        conversation_tokens = [bos]
        for message in request.messages:
            if message.role == "user":
                conversation_tokens.append(user_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(user_end)
            elif message.role == "assistant":
                conversation_tokens.append(assistant_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(assistant_end)
        conversation_tokens.append(assistant_start)

        response_tokens = []

        async def stream_and_release():
            try:
                async for chunk in generate_stream(
                    worker,
                    conversation_tokens,
                    temperature=request.temperature,
                    max_new_tokens=request.max_tokens,
                    top_k=request.top_k,
                    ):
                    chunk_data = json.loads(chunk.replace("data: ", "").strip())
                    if "token" in chunk_data:
                        response_tokens.append(chunk_data["token"])
                    yield chunk
            finally:
                final_response = "".join(response_tokens)
                logger.info("[ASSISTANT] (GPU %s): %s", worker.gpu_id, final_response)
                app.state.activity_log.log_event(
                    "chat_assistant",
                    final_response[:1000],
                    {"provider": "internal_nanochat", "gpu": worker.gpu_id},
                )
                logger.info("=" * 20)
                await worker_pool.release_worker(worker)

        return StreamingResponse(stream_and_release(), media_type="text/event-stream")
    except Exception as exc:
        await worker_pool.release_worker(worker)
        raise exc


@app.get("/health")
async def health():
    chat = chat_status_snapshot(app)
    runtime = app.state.local_runtime.status()
    return {
        "status": "ok",
        "chat_ready": chat["ready"],
        "chat_error": chat["error"],
        "num_gpus": chat["config"].get("num_gpus", 0),
        "available_workers": chat["available_workers"],
        "runtime_ready": runtime["ready"],
        "runtime_error": runtime["last_error"],
    }


@app.get("/stats")
async def stats():
    chat = chat_status_snapshot(app)
    return {
        "chat": chat,
        "runtime": app.state.local_runtime.status(),
        "jobs": app.state.job_manager.list_jobs(),
        "sandbox": app.state.sandbox.status(),
        "activity": app.state.activity_log.snapshot(limit=80),
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting NanoChat Builder Server")
    logger.info("Generation defaults: temperature=%s top_k=%s max_tokens=%s", args.temperature, args.top_k, args.max_tokens)
    uvicorn.run(app, host=args.host, port=args.port)
