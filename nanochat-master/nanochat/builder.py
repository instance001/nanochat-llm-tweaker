"""
Small CLI wrapper for local builder utilities.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from nanochat.common import get_base_dir
from nanochat.dataset import corpus_summary, get_local_corpus_dir
from nanochat.dataset_sources import build_import_manifest, collect_huggingface_corpus_records, preview_huggingface_corpus_records, records_to_jsonl
from nanochat.dashboard_tools import build_job_command
from nanochat.local_runtime import LocalRuntimeManager
from nanochat.preflight import run_stage_preflight
from nanochat.sandbox_tools import CorpusManager
from nanochat.validation_tools import validate_corpus_path, validate_sft_jsonl_file

REPO_ROOT = Path(__file__).resolve().parent.parent


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=True, indent=2))


def _default_sandbox_dir() -> Path:
    return REPO_ROOT / "assistant_sandbox"


def _default_corpus_dir() -> Path:
    return get_local_corpus_dir()


def _load_params(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if getattr(args, "params_file", ""):
        params_path = Path(args.params_file).expanduser().resolve()
        payload.update(json.loads(params_path.read_text(encoding="utf-8")))
    if getattr(args, "params_json", ""):
        parsed = json.loads(args.params_json)
        if not isinstance(parsed, dict):
            raise ValueError("--params-json must decode to a JSON object.")
        payload.update(parsed)
    return payload


def _checkpoint_sets(base_dir: Path) -> dict[str, Any]:
    def summarize(path: Path) -> dict[str, Any]:
        tags = []
        if path.exists():
            for item in sorted((entry for entry in path.iterdir() if entry.is_dir()), key=lambda entry: entry.stat().st_mtime, reverse=True):
                tags.append({"tag": item.name, "path": str(item)})
        return {"path": str(path), "tags": tags}

    return {
        "base": summarize(base_dir / "base_checkpoints"),
        "sft": summarize(base_dir / "chatsft_checkpoints"),
        "rl": summarize(base_dir / "chatrl_checkpoints"),
    }


def _preflight_context(args: argparse.Namespace) -> dict[str, Any]:
    base_dir = Path(args.base_dir).expanduser().resolve() if getattr(args, "base_dir", "") else Path(get_base_dir())
    corpus_dir = Path(args.corpus_dir).expanduser().resolve() if getattr(args, "corpus_dir", "") else _default_corpus_dir()
    sandbox_dir = Path(args.sandbox_dir).expanduser().resolve() if getattr(args, "sandbox_dir", "") else _default_sandbox_dir()
    return {
        "base_dir": base_dir,
        "corpus_dir": corpus_dir,
        "sandbox_dir": sandbox_dir,
        "tokenizer_ready": (base_dir / "tokenizer").exists(),
        "identity_exists": (base_dir / "identity_conversations.jsonl").exists(),
        "checkpoint_sets": _checkpoint_sets(base_dir),
    }


def _status_payload(args: argparse.Namespace) -> dict[str, Any]:
    corpus_dir = Path(args.corpus_dir).expanduser().resolve() if args.corpus_dir else _default_corpus_dir()
    sandbox_dir = Path(args.sandbox_dir).expanduser().resolve() if args.sandbox_dir else _default_sandbox_dir()
    base_dir = Path(get_base_dir())
    return {
        "repo_root": str(REPO_ROOT),
        "base_dir": str(base_dir),
        "corpus": corpus_summary(corpus_dir),
        "sandbox_dir": str(sandbox_dir),
        "sandbox_exists": sandbox_dir.exists(),
        "tokenizer_ready": (base_dir / "tokenizer").exists(),
        "identity_exists": (base_dir / "identity_conversations.jsonl").exists(),
    }


def cmd_status(args: argparse.Namespace) -> int:
    _print_json(_status_payload(args))
    return 0


def cmd_corpus_validate(args: argparse.Namespace) -> int:
    target = Path(args.path).expanduser().resolve() if args.path else _default_corpus_dir()
    report = validate_corpus_path(target)
    _print_json(report)
    return 0 if report["ok"] else 1


def cmd_corpus_convert(args: argparse.Namespace) -> int:
    corpus_dir = Path(args.corpus_dir).expanduser().resolve() if args.corpus_dir else _default_corpus_dir()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    content = input_path.read_text(encoding="utf-8", errors="replace")
    manager = CorpusManager(corpus_dir)
    result = manager.write_converted_parquet_file(
        args.output,
        content,
        conversion_mode=args.mode,
        mode=args.write_mode,
        source=args.source or input_path.name,
    )
    _print_json(result)
    return 0


def cmd_corpus_import_hf(args: argparse.Namespace) -> int:
    corpus_dir = Path(args.corpus_dir).expanduser().resolve() if args.corpus_dir else _default_corpus_dir()
    common = {
        "dataset_id": args.dataset,
        "config": args.config,
        "revision": args.revision,
        "split": args.split,
        "text_column": args.text_column,
        "limit_docs": args.limit_docs,
        "max_chars": args.max_chars,
        "sample_size": args.sample_size,
        "source_label": args.source,
        "trust_remote_code": bool(args.trust_remote_code),
    }
    if args.preview:
        _print_json(preview_huggingface_corpus_records(**common))
        return 0

    payload = collect_huggingface_corpus_records(**common)
    manager = CorpusManager(corpus_dir)
    output_path = args.output.strip()
    output_format = args.output_format.strip().lower()
    if not output_path:
        result = manager.write_import_artifacts(
            payload,
            output_format=output_format,
            train_val_ratio=args.train_val_ratio,
            shard_size_docs=args.shard_size_docs,
            license_note=args.license_note,
            manifest_path=args.manifest_path,
        )
        _print_json({"import": {key: value for key, value in payload.items() if key != "records"}, **result})
        return 0

    if output_path.lower().endswith(".jsonl"):
        result = manager.write_file(output_path, records_to_jsonl(payload["records"]))
    else:
        if not output_path.lower().endswith(".parquet"):
            raise ValueError("Hugging Face imports write .parquet or .jsonl corpus files.")
        result = manager.write_parquet_file(output_path, payload["records"], mode=args.write_mode)
    shard_path = result["path"]
    shard_file = manager._resolve_path(shard_path)
    import hashlib

    digest = hashlib.sha256(shard_file.read_bytes()).hexdigest()
    written_shards = [
        {
            "split": "train",
            "path": shard_path,
            "row_count": payload["row_count"],
            "character_count": payload["character_count"],
            "size": result["size"],
            "sha256": digest,
        }
    ]
    manifest = build_import_manifest(
        payload,
        written_shards,
        train_val_ratio=0.0,
        shard_size_docs=payload["row_count"],
        output_format="jsonl" if shard_path.lower().endswith(".jsonl") else "parquet",
        license_note=args.license_note,
    )
    manifest_path = args.manifest_path or "hf_import_manifest.json"
    manifest_result = manager.write_file(manifest_path, json.dumps(manifest, ensure_ascii=True, indent=2) + "\n")
    _print_json(
        {
            **result,
            "manifest_path": manifest_result["path"],
            "manifest": manifest,
            "import": {key: value for key, value in payload.items() if key != "records"},
        }
    )
    return 0


def cmd_sft_validate(args: argparse.Namespace) -> int:
    report = validate_sft_jsonl_file(args.path)
    _print_json(report)
    return 0 if report["ok"] else 1


def cmd_job_preflight(args: argparse.Namespace) -> int:
    params = _load_params(args)
    report = run_stage_preflight(args.job_type, params, **_preflight_context(args))
    _print_json(report)
    return 0 if report["ok"] else 1


def cmd_job_preview(args: argparse.Namespace) -> int:
    params = _load_params(args)
    command = build_job_command(args.job_type, params)
    preflight = run_stage_preflight(args.job_type, params, **_preflight_context(args))
    payload = {
        "job_type": args.job_type,
        "params": params,
        "command": command,
        "display_command": subprocess.list2cmdline(command),
        "preflight": preflight,
        "environment_notes": [
            "Dashboard jobs also set NANOCHAT_LOCAL_ONLY=1 and disable W&B/Hugging Face online access.",
            "Running this CLI preview does not launch the job.",
        ],
    }
    _print_json(payload)
    return 0 if preflight["ok"] else 1


def _runtime_manager(args: argparse.Namespace) -> LocalRuntimeManager:
    repo_root = Path(args.repo_root).expanduser().resolve() if getattr(args, "repo_root", "") else REPO_ROOT
    return LocalRuntimeManager(str(repo_root))


def cmd_runtime_list_models(args: argparse.Namespace) -> int:
    runtime = _runtime_manager(args)
    payload = {
        "repo_root": str(runtime.repo_root),
        "models": runtime.list_models(),
    }
    _print_json(payload)
    return 0


def cmd_runtime_recommend(args: argparse.Namespace) -> int:
    runtime = _runtime_manager(args)
    status = runtime.bundle_status(refresh=True)
    payload = {
        "repo_root": str(runtime.repo_root),
        "bundle": status,
        "recommended_config": status.get("recommended_config", {}),
    }
    _print_json(payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local builder utility CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="Print local builder status as JSON")
    status.add_argument("--corpus-dir", default="", help="Override local corpus directory")
    status.add_argument("--sandbox-dir", default="", help="Override assistant sandbox directory")
    status.set_defaults(func=cmd_status)

    corpus = subparsers.add_parser("corpus", help="Corpus utilities")
    corpus_subparsers = corpus.add_subparsers(dest="corpus_command", required=True)

    corpus_validate = corpus_subparsers.add_parser("validate", help="Validate a corpus file or directory")
    corpus_validate.add_argument("path", nargs="?", default="", help="Corpus file/directory, default = local_corpus")
    corpus_validate.set_defaults(func=cmd_corpus_validate)

    corpus_convert = corpus_subparsers.add_parser("convert", help="Convert local text/JSON input into corpus parquet")
    corpus_convert.add_argument("--input", required=True, help="Source text, markdown, JSON, or JSONL file")
    corpus_convert.add_argument("--output", required=True, help="Relative parquet path under the corpus directory, e.g. train/notes.parquet")
    corpus_convert.add_argument("--corpus-dir", default="", help="Override output corpus directory")
    corpus_convert.add_argument(
        "--mode",
        default="paragraphs",
        choices=["paragraphs", "lines", "markdown_sections", "jsonl", "json_array"],
        help="Conversion mode",
    )
    corpus_convert.add_argument("--write-mode", default="overwrite", choices=["overwrite", "append"], help="Parquet write mode")
    corpus_convert.add_argument("--source", default="", help="Optional source label stored in generated text records")
    corpus_convert.set_defaults(func=cmd_corpus_convert)

    corpus_import_hf = corpus_subparsers.add_parser(
        "import-hf",
        help="Preview or import a bounded Hugging Face streaming dataset slice into local_corpus",
    )
    corpus_import_hf.add_argument("--dataset", required=True, help="Hugging Face dataset id, e.g. HuggingFaceFW/fineweb")
    corpus_import_hf.add_argument("--config", default="", help="Optional dataset config/name")
    corpus_import_hf.add_argument("--revision", default="", help="Optional dataset revision/commit for reproducibility")
    corpus_import_hf.add_argument("--split", default="train", help="Dataset split, default=train")
    corpus_import_hf.add_argument("--text-column", default="text", help="Text column or dotted path, default=text")
    corpus_import_hf.add_argument("--limit-docs", type=int, default=1000, help="Maximum usable documents to import")
    corpus_import_hf.add_argument("--max-chars", type=int, default=0, help="Optional total character cap, 0 = no cap")
    corpus_import_hf.add_argument("--sample-size", type=int, default=12, help="Preview sample rows to include")
    corpus_import_hf.add_argument("--source", default="", help="Optional source label stored in generated corpus rows")
    corpus_import_hf.add_argument("--output", default="", help="Optional legacy single-file .parquet or .jsonl path under local_corpus")
    corpus_import_hf.add_argument("--output-format", default="parquet", choices=["parquet", "jsonl"], help="Shard output format when --output is omitted")
    corpus_import_hf.add_argument("--corpus-dir", default="", help="Override output corpus directory")
    corpus_import_hf.add_argument("--write-mode", default="overwrite", choices=["overwrite", "append"], help="Parquet write mode")
    corpus_import_hf.add_argument("--train-val-ratio", type=float, default=0.0, help="Validation ratio for deterministic local shard split")
    corpus_import_hf.add_argument("--shard-size-docs", type=int, default=10000, help="Maximum documents per deterministic local shard")
    corpus_import_hf.add_argument("--license-note", default="", help="License/provenance note to store in the import manifest")
    corpus_import_hf.add_argument("--manifest-path", default="", help="Optional manifest path under local_corpus")
    corpus_import_hf.add_argument("--preview", action="store_true", help="Preview records without writing to local_corpus")
    corpus_import_hf.add_argument("--trust-remote-code", action="store_true", help="Allow dataset loading code when the dataset requires it")
    corpus_import_hf.set_defaults(func=cmd_corpus_import_hf)

    sft = subparsers.add_parser("sft", help="SFT dataset utilities")
    sft_subparsers = sft.add_subparsers(dest="sft_command", required=True)

    sft_validate = sft_subparsers.add_parser("validate", help="Validate an SFT JSONL conversation file")
    sft_validate.add_argument("path", help="SFT JSONL file")
    sft_validate.set_defaults(func=cmd_sft_validate)

    job = subparsers.add_parser("job", help="Job command utilities")
    job_subparsers = job.add_subparsers(dest="job_command", required=True)

    def add_job_common(job_parser: argparse.ArgumentParser) -> None:
        job_parser.add_argument("--job-type", required=True, help="Dashboard job type, e.g. tokenizer_train or chat_sft")
        job_parser.add_argument("--params-json", default="", help="JSON object of job parameters")
        job_parser.add_argument("--params-file", default="", help="Path to a JSON object of job parameters")
        job_parser.add_argument("--base-dir", default="", help="Override nanochat base/cache directory")
        job_parser.add_argument("--corpus-dir", default="", help="Override local corpus directory")
        job_parser.add_argument("--sandbox-dir", default="", help="Override assistant sandbox directory")

    job_preflight = job_subparsers.add_parser("preflight", help="Run preflight checks for a dashboard job")
    add_job_common(job_preflight)
    job_preflight.set_defaults(func=cmd_job_preflight)

    job_preview = job_subparsers.add_parser("preview", help="Preview the command and preflight for a dashboard job")
    add_job_common(job_preview)
    job_preview.set_defaults(func=cmd_job_preview)

    runtime = subparsers.add_parser("runtime", help="Managed GGUF runtime utilities")
    runtime_subparsers = runtime.add_subparsers(dest="runtime_command", required=True)

    runtime_list = runtime_subparsers.add_parser("list-models", help="List discovered local GGUF/GGML models")
    runtime_list.add_argument("--repo-root", default="", help="Override nanochat repo root")
    runtime_list.set_defaults(func=cmd_runtime_list_models)

    runtime_recommend = runtime_subparsers.add_parser("recommend", help="Print runtime bundle status and recommended start config")
    runtime_recommend.add_argument("--repo-root", default="", help="Override nanochat repo root")
    runtime_recommend.set_defaults(func=cmd_runtime_recommend)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        _print_json({"ok": False, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
