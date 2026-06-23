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
from nanochat.common import autodetect_device_type, compute_init
from nanochat.dataset import TEXT_SUFFIXES
from nanochat.dashboard_tools import (
    BackgroundJobManager,
    GUIDED_PRESETS,
    builder_state,
    build_job_command,
    list_designs,
    publish_design,
    save_design,
)
from nanochat.activity_log import ActivityLogManager
from nanochat.local_runtime import LocalRuntimeManager
from nanochat.sandbox_tools import CorpusManager, SandboxManager
from nanochat.sft_dataset_tools import (
    conversations_to_jsonl,
    merge_jsonl,
    normalize_conversations,
    sft_schema_payload,
)

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
DEFAULT_LOCAL_RUNTIME_SYSTEM_PROMPT = (
    "You are the local builder assistant for this dashboard. "
    "Help the user run the LLM builder, explain the next step, point out missing files or settings, "
    "and prefer practical, accurate guidance over speed. If you are unsure, say so plainly. "
    "When workspace files are provided, use only that workspace context. "
    "The writable local workspaces exposed through actions are assistant_sandbox and local_corpus."
)
ASSISTANT_ACTION_PATTERN = re.compile(r"<assistant_action>\s*(.*?)\s*</assistant_action>", re.DOTALL)
MAX_ASSISTANT_ACTIONS = 4

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


class JobRequest(BaseModel):
    job_type: str
    label: Optional[str] = None
    preset: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


class ChatLoadRequest(BaseModel):
    source: Optional[str] = None
    model_tag: Optional[str] = None
    step: Optional[int] = None
    num_gpus: Optional[int] = None


class RuntimeStartRequest(BaseModel):
    model_path: str
    host: str = "127.0.0.1"
    port: int = 8091
    ctx_size: int = 4096
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


class SandboxWriteRequest(BaseModel):
    path: str
    content: str


class SandboxDeleteRequest(BaseModel):
    path: str


class WorkspaceCopyRequest(BaseModel):
    source_path: str
    target_path: str


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


def render_tool_help() -> str:
    tool_specs = [
        ("get_builder_state", "{}"),
        ("get_recent_activity", '{"limit": 40}'),
        ("get_benchmark_history", '{"limit": 20}'),
        ("autotune_settings", "{}"),
        ("get_corpus_schema", "{}"),
        ("list_corpus_files", "{}"),
        ("read_corpus_file", '{"path": "train/reference.txt"}'),
        ("write_corpus_file", '{"path": "train/reference.txt", "content": "..."}'),
        ("draft_corpus_file", '{"path": "train/reference.txt", "mode": "append", "content": "..."}'),
        ("delete_corpus_file", '{"path": "train/tmp.txt"}'),
        ("copy_sandbox_to_corpus", '{"source_path": "notes/reference.txt", "target_path": "train/reference.txt"}'),
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
        "draft_sft_data only writes validated conversation JSONL. "
        "launch_job only supports tokenizer_train, tokenizer_eval, base_train, base_eval, benchmark_eval, and chat_sft."
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


def corpus_schema_payload(app: FastAPI) -> dict[str, Any]:
    builder = builder_state()
    return {
        "format": "Local corpus files under local_corpus, with optional train/ and val/ subfolders",
        "supported_extensions": sorted(TEXT_SUFFIXES) + [".parquet"],
        "recommended_paths": [
            "train/reference.txt",
            "train/notes.jsonl",
            "val/holdout.txt",
        ],
        "notes": [
            "Tokenizer and base training read local_corpus only.",
            "If local_corpus/train exists, the train split reads from that folder.",
            "If local_corpus/val exists, the val split reads from that folder.",
            "Plain text is the simplest starting format.",
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
    if tool_name == "list_corpus_files":
        return app.state.corpus.status()
    if tool_name == "read_corpus_file":
        return app.state.corpus.read_file(str(args.get("path", "")))
    if tool_name == "write_corpus_file":
        return app.state.corpus.write_file(str(args.get("path", "")), str(args.get("content", "")))
    if tool_name == "draft_corpus_file":
        target_path = _normalize_relative_corpus_path(str(args.get("path") or "train/reference.txt"))
        mode = str(args.get("mode") or "append").strip().lower()
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
        result = app.state.corpus.write_file(target_path, source_file["content"])
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
        allowed_job_types = {"tokenizer_train", "tokenizer_eval", "base_train", "base_eval", "benchmark_eval", "chat_sft"}
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
    prompt = system_prompt or DEFAULT_LOCAL_RUNTIME_SYSTEM_PROMPT
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


def run_assistant_tool_loop(
    app: FastAPI,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    max_actions: int,
) -> dict[str, Any]:
    runtime = app.state.local_runtime
    transcript = list(messages)
    actions: list[dict[str, Any]] = []
    max_actions = max(0, min(max_actions, MAX_ASSISTANT_ACTIONS))

    for _ in range(max_actions + 1):
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
        try:
            result = execute_assistant_tool(app, tool_name, tool_args)
            status = "ok"
        except Exception as exc:
            result = {"error": str(exc)}
            status = "error"
        app.state.activity_log.log_event("assistant_tool_result", f"Assistant tool {tool_name} returned {status}", {"result": result})
        actions.append({"tool": tool_name, "args": tool_args, "status": status, "result": result})
        transcript.append({"role": "assistant", "content": text})
        transcript.append(
            {
                "role": "user",
                "content": (
                    f"TOOL RESULT for {tool_name} ({status}). "
                    "Now continue. If more local actions are needed, emit another <assistant_action> JSON object. "
                    "Otherwise give the final answer to the user.\n\n"
                    f"{_compact_tool_result(result)}"
                ),
            }
        )

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
    app.state.activity_log = ActivityLogManager(REPO_ROOT / "builder_logs" / "activity.jsonl")
    app.state.benchmark_history = BenchmarkHistoryManager(REPO_ROOT / "builder_logs" / "benchmark_history.jsonl")
    app.state.job_manager = BackgroundJobManager(
        workdir=str(REPO_ROOT),
        activity_log=app.state.activity_log,
        benchmark_history=app.state.benchmark_history,
    )
    app.state.local_runtime = LocalRuntimeManager(str(REPO_ROOT))
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
                app.state.local_runtime.start(
                    {
                        "model_path": args.runtime_model,
                        "port": args.runtime_port,
                        "ctx_size": 4096,
                        "threads": max(4, (os.cpu_count() or 4)),
                        "threads_http": 4,
                        "parallel": 2,
                        "alias": "local-builder-assistant",
                        "device_strategy": args.runtime_device_strategy,
                    }
                )
                app.state.activity_log.log_event(
                    "runtime_started",
                    "Local runtime auto-started",
                    {"model_path": args.runtime_model or runtime_status["bundle"]["recommended_model"]},
                )
                logger.info("Local runtime auto-started")
            except Exception as exc:
                app.state.activity_log.log_event("runtime_error", "Local runtime auto-start failed", {"error": str(exc)})
                logger.warning("Local runtime auto-start failed: %s", exc)
    app.state.activity_log.log_event("server_ready", "Builder server ready", {"port": args.port, "host": args.host})
    logger.info("Server ready at http://localhost:%s", args.port)
    yield
    app.state.activity_log.log_event("server_shutdown", "Builder server shutting down", {})
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
    return {
        "builder": builder_state(),
        "jobs": app.state.job_manager.list_jobs(),
        "chat": chat_status_snapshot(app),
        "runtime": app.state.local_runtime.status(),
        "sandbox": app.state.sandbox.status(),
        "corpus": app.state.corpus.status(),
        "activity": app.state.activity_log.snapshot(limit=80),
        "benchmarks": app.state.benchmark_history.snapshot(limit=20),
    }


@app.get("/api/dashboard/jobs")
async def dashboard_jobs():
    return {"jobs": app.state.job_manager.list_jobs()}


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

    label = request.label or request.job_type.replace("_", " ").title()
    job = app.state.job_manager.start_job(request.job_type, label, command, notes=request.notes, params=params)
    app.state.activity_log.log_event(
        "job_created",
        f"Dashboard launched job {label}",
        {"job_id": job["id"], "job_type": request.job_type, "params": params},
    )
    return {"job": job, "resolved_params": params}


@app.post("/api/dashboard/jobs/{job_id}/stop")
async def stop_dashboard_job(job_id: str):
    try:
        job = app.state.job_manager.stop_job(job_id)
        app.state.activity_log.log_event("job_stop", f"Dashboard stop requested for {job['label']}", {"job_id": job_id})
        return {"job": job}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc


@app.get("/api/dashboard/designs")
async def dashboard_designs():
    return {"designs": list_designs()}


@app.post("/api/dashboard/designs")
async def upsert_dashboard_design(request: DesignRequest):
    design = save_design(request.model_dump())
    app.state.activity_log.log_event("design_saved", f"Saved design {design['name']}", {"slug": design["slug"]})
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
        return app.state.corpus.write_file(request.path, request.content)
    except ValueError as exc:
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
        result = app.state.corpus.write_file(request.target_path, source["content"])
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
