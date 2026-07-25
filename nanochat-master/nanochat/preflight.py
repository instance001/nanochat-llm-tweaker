"""
Preflight checks for dashboard-launched builder jobs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nanochat.dataset import corpus_summary
from nanochat.sft_dataset_tools import sft_schema_payload
from nanochat.validation_tools import validate_sft_jsonl_file


def _check(status: str, code: str, message: str, detail: str = "") -> dict[str, str]:
    return {
        "status": status,
        "code": code,
        "message": message,
        "detail": detail,
    }


def _split_file_list(raw_value: Any) -> list[str]:
    if not raw_value:
        return []
    pieces = []
    for separator in ("\n", "|"):
        raw_value = str(raw_value).replace(separator, ";")
    for piece in str(raw_value).split(";"):
        cleaned = piece.strip()
        if cleaned:
            pieces.append(cleaned)
    return pieces


def _resolve_workspace_file(path_value: str, sandbox_dir: Path) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate
    return sandbox_dir / candidate


def run_stage_preflight(
    job_type: str,
    params: dict[str, Any],
    *,
    base_dir: str | Path,
    corpus_dir: str | Path,
    sandbox_dir: str | Path,
    tokenizer_ready: bool,
    identity_exists: bool,
    checkpoint_sets: dict[str, Any],
) -> dict[str, Any]:
    base_dir = Path(base_dir)
    corpus_dir = Path(params.get("corpus_dir") or corpus_dir)
    sandbox_dir = Path(sandbox_dir)
    checks: list[dict[str, str]] = []

    def add(status: str, code: str, message: str, detail: str = "") -> None:
        checks.append(_check(status, code, message, detail))

    corpus = corpus_summary(corpus_dir)
    train_count = int(corpus.get("splits", {}).get("train", {}).get("file_count", 0))
    val_count = int(corpus.get("splits", {}).get("val", {}).get("file_count", 0))

    if job_type in {"tokenizer_train", "tokenizer_eval", "base_train", "base_eval", "benchmark_eval"}:
        if train_count <= 0:
            add("error", "corpus_empty", "No local corpus train files were found.", f"Add supported files under {corpus_dir / 'train'} or {corpus_dir}.")
        else:
            add("ok", "corpus_present", f"Found {train_count} local corpus train file(s).", str(corpus_dir))
        if val_count <= 0 and job_type in {"base_train", "base_eval", "benchmark_eval"}:
            add("warning", "corpus_val_missing", "No validation split files were found.", "The reader may fall back to the corpus root, but a separate val/ folder is safer.")

    if job_type == "base_train":
        if not tokenizer_ready:
            add("error", "tokenizer_missing", "Tokenizer is not ready.", f"Train a tokenizer first. Expected tokenizer directory under {base_dir / 'tokenizer'}.")
        else:
            add("ok", "tokenizer_ready", "Tokenizer is ready.")
        if not params.get("save_every"):
            add("warning", "save_every_missing", "Save Every is empty.", "Pause/resume is much more useful when checkpoints are saved during the run.")

    if job_type in {"base_eval", "benchmark_eval"}:
        base_tags = checkpoint_sets.get("base", {}).get("tags", [])
        if not base_tags and not params.get("model_tag"):
            add("error", "base_checkpoint_missing", "No base checkpoint is available for evaluation.", "Run base training first or provide a valid model tag.")
        else:
            add("ok", "base_checkpoint_available", "A base checkpoint target is available.")

    if job_type == "chat_sft":
        base_tags = checkpoint_sets.get("base", {}).get("tags", [])
        if not base_tags and not params.get("model_tag"):
            add("error", "base_checkpoint_missing", "No base checkpoint is available for Chat SFT.", "Run base training first or provide a valid base model tag.")
        else:
            add("ok", "base_checkpoint_available", "A base checkpoint target is available.")
        if not identity_exists and int(params.get("include_identity") or 0):
            add("warning", "identity_missing", "Identity examples are enabled but no published identity file exists.", "Publish a design first or set include_identity to 0.")
        _add_sft_file_checks(add, params.get("train_files"), sandbox_dir, required=True, label="training")
        _add_sft_file_checks(add, params.get("val_files"), sandbox_dir, required=False, label="validation")
        if not params.get("save_every"):
            add("warning", "save_every_missing", "Save Every is empty.", "Pause/resume is much more useful when checkpoints are saved during the run.")

    if job_type == "chat_rl":
        sft_tags = checkpoint_sets.get("sft", {}).get("tags", [])
        if not sft_tags and not params.get("model_tag"):
            add("error", "sft_checkpoint_missing", "No SFT checkpoint is available for Chat RL.", "Run Chat SFT first or provide a valid SFT model tag.")
        else:
            add("ok", "sft_checkpoint_available", "An SFT checkpoint target is available.")

    if job_type == "chat_eval":
        source = str(params.get("source") or "sft")
        key = "rl" if source == "rl" else "sft"
        tags = checkpoint_sets.get(key, {}).get("tags", [])
        if not tags and not params.get("model_tag"):
            add("error", f"{key}_checkpoint_missing", f"No {key.upper()} checkpoint is available for Chat Eval.", "Run the matching training stage first or provide a valid model tag.")
        else:
            add("ok", f"{key}_checkpoint_available", f"A {key.upper()} checkpoint target is available.")

    if not checks:
        add("ok", "no_preflight_rules", "No specific preflight rules are defined for this job type.")

    error_count = sum(1 for check in checks if check["status"] == "error")
    warning_count = sum(1 for check in checks if check["status"] == "warning")
    return {
        "job_type": job_type,
        "ok": error_count == 0,
        "error_count": error_count,
        "warning_count": warning_count,
        "checks": checks,
    }


def _add_sft_file_checks(add, raw_files: Any, sandbox_dir: Path, *, required: bool, label: str) -> None:
    files = _split_file_list(raw_files)
    if not files:
        status = "error" if required else "warning"
        add(status, f"sft_{label}_files_missing", f"No SFT {label} file was provided.", sft_schema_payload()["format"])
        return
    for path_value in files:
        lower = path_value.lower()
        if lower.endswith(".parquet"):
            add("error", "sft_file_is_parquet", f"{path_value} is a parquet file, but Chat SFT expects conversation JSONL.", "Use parquet for base corpus data, not SFT conversations.")
            continue
        if not (lower.endswith(".jsonl") or lower.endswith(".json")):
            add("warning", "sft_file_extension_unusual", f"{path_value} does not look like a JSONL/JSON conversation file.")
        resolved = _resolve_workspace_file(path_value, sandbox_dir)
        if not resolved.exists():
            add("error" if required else "warning", "sft_file_missing", f"SFT {label} file not found: {path_value}", str(resolved))
            continue
        report = validate_sft_jsonl_file(resolved)
        if report["ok"]:
            add("ok", f"sft_{label}_file_valid", f"SFT {label} file looks valid: {path_value}", f"{report['record_count']} conversation(s).")
        else:
            add("error" if required else "warning", "sft_file_invalid", f"SFT {label} file has validation issues: {path_value}", "; ".join(report["errors"][:3]))
