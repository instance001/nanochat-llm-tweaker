"""
Structured benchmark history and simple next-run recommendation helpers.
"""

from __future__ import annotations

import copy
import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Any

OOM_PATTERNS = (
    "out of memory",
    "cuda out of memory",
    "not enough memory",
    "oom",
)
TOKENIZER_ROW_RE = re.compile(r"^(?P<sample>\S+)\s+(?P<bytes>\d+)\s+(?P<tokens>\d+)\s+(?P<ratio>\d+(?:\.\d+)?)$")
VALUE_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


def _parse_first_float(line: str) -> float | None:
    match = VALUE_RE.search(line)
    if not match:
        return None
    return float(match.group(1))


def _find_line_value(log_lines: list[str], needle: str) -> float | None:
    for line in reversed(log_lines):
        if needle.lower() not in line.lower():
            continue
        return _parse_first_float(line)
    return None


def extract_job_metrics(job_type: str, log_lines: list[str]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    lowered_lines = [line.lower() for line in log_lines]
    metrics["oom_detected"] = any(pattern in line for pattern in OOM_PATTERNS for line in lowered_lines)

    if any("recommend using --window-pattern l" in line for line in lowered_lines):
        metrics["window_pattern_warning"] = True

    if job_type == "tokenizer_eval":
        rows = []
        for line in log_lines:
            match = TOKENIZER_ROW_RE.match(line.strip())
            if not match:
                continue
            row = {
                "sample": match.group("sample"),
                "bytes": int(match.group("bytes")),
                "tokens": int(match.group("tokens")),
                "bytes_per_token": float(match.group("ratio")),
            }
            rows.append(row)
        if rows:
            metrics["sample_count"] = len(rows)
            metrics["avg_bytes_per_token"] = sum(row["bytes_per_token"] for row in rows) / len(rows)
            metrics["rows"] = rows[:12]
        return metrics

    if job_type in {"base_eval", "benchmark_eval"}:
        train_bpb = _find_line_value(log_lines, "train bpb:")
        val_bpb = _find_line_value(log_lines, "val bpb:")
        core_metric = _find_line_value(log_lines, "CORE metric:")
        if train_bpb is not None:
            metrics["train_bpb"] = train_bpb
        if val_bpb is not None:
            metrics["val_bpb"] = val_bpb
        if core_metric is not None:
            metrics["core_metric"] = core_metric
        return metrics

    if job_type in {"base_train", "chat_sft"}:
        validation_values = []
        for line in log_lines:
            if "Validation bpb:" not in line:
                continue
            value = _parse_first_float(line.split("Validation bpb:", 1)[1])
            if value is not None:
                validation_values.append(value)
        if validation_values:
            metrics["last_val_bpb"] = validation_values[-1]
            metrics["min_val_bpb"] = min(validation_values)
        explicit_min = _find_line_value(log_lines, "Minimum validation bpb:")
        if explicit_min is not None:
            metrics["min_val_bpb"] = explicit_min
        total_time_m = _find_line_value(log_lines, "Total training time:")
        peak_memory_mib = _find_line_value(log_lines, "Peak memory usage:")
        if total_time_m is not None:
            metrics["total_training_time_m"] = total_time_m
        if peak_memory_mib is not None:
            metrics["peak_memory_mib"] = peak_memory_mib
        return metrics

    return metrics


class BenchmarkHistoryManager:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append_job_result(self, job_snapshot: dict[str, Any]) -> dict[str, Any]:
        logs = list(job_snapshot.get("logs") or [])
        entry = {
            "ts": time.time(),
            "job_id": job_snapshot.get("id"),
            "job_type": job_snapshot.get("job_type"),
            "label": job_snapshot.get("label"),
            "status": job_snapshot.get("status"),
            "exit_code": job_snapshot.get("exit_code"),
            "params": job_snapshot.get("params") or {},
            "metrics": extract_job_metrics(str(job_snapshot.get("job_type") or ""), logs),
            "notes": job_snapshot.get("notes", ""),
        }
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
        return entry

    def recent_records(self, limit: int = 80) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self._lock:
            with open(self.path, "r", encoding="utf-8", errors="replace") as handle:
                lines = handle.readlines()
        records = []
        for raw in lines[-limit:]:
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return records

    def snapshot(self, limit: int = 80) -> dict[str, Any]:
        records = self.recent_records(limit=limit)
        latest_benchmark = self._latest(records, "benchmark_eval", statuses={"completed"})
        latest_base_eval = self._latest(records, "base_eval", statuses={"completed"})
        latest_tokenizer_eval = self._latest(records, "tokenizer_eval", statuses={"completed"})
        return {
            "path": str(self.path),
            "record_count": len(records),
            "records": records,
            "latest_benchmark": latest_benchmark,
            "latest_base_eval": latest_base_eval,
            "latest_tokenizer_eval": latest_tokenizer_eval,
        }

    def recommend_settings(self, current_forms: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
        current_forms = copy.deepcopy(current_forms or {})
        records = self.recent_records(limit=240)
        recommendations = {
            "tokenizerForm": {"values": copy.deepcopy(current_forms.get("tokenizerForm", {})), "notes": []},
            "baseTrainForm": {"values": copy.deepcopy(current_forms.get("baseTrainForm", {})), "notes": []},
            "chatSftForm": {"values": copy.deepcopy(current_forms.get("chatSftForm", {})), "notes": []},
            "baseEvalForm": {"values": copy.deepcopy(current_forms.get("baseEvalForm", {})), "notes": []},
        }
        summary_notes: list[str] = []

        self._recommend_tokenizer(records, recommendations["tokenizerForm"])
        self._recommend_base_train(records, recommendations["baseTrainForm"])
        self._recommend_chat_sft(records, recommendations["chatSftForm"])
        self._recommend_eval(records, recommendations["baseEvalForm"])

        latest_benchmark = self._latest(records, "benchmark_eval", statuses={"completed"})
        if latest_benchmark and latest_benchmark.get("metrics", {}).get("val_bpb") is not None:
            summary_notes.append(
                f"Latest benchmark val bpb: {latest_benchmark['metrics']['val_bpb']:.6f} "
                f"from {latest_benchmark.get('label') or latest_benchmark.get('job_id')}"
            )
        else:
            summary_notes.append("No completed benchmark run yet. Start with a benchmark_eval run to make comparisons easier.")

        return {
            "recommended_forms": {key: value["values"] for key, value in recommendations.items()},
            "form_notes": {key: value["notes"] for key, value in recommendations.items()},
            "summary_notes": summary_notes,
            "history": self.snapshot(limit=20),
        }

    def _latest(self, records: list[dict[str, Any]], job_type: str, statuses: set[str] | None = None) -> dict[str, Any] | None:
        for record in reversed(records):
            if record.get("job_type") != job_type:
                continue
            if statuses is not None and record.get("status") not in statuses:
                continue
            return record
        return None

    def _best_by_metric(self, records: list[dict[str, Any]], job_type: str, metric_key: str) -> dict[str, Any] | None:
        candidates = [
            record
            for record in records
            if record.get("job_type") == job_type
            and record.get("status") == "completed"
            and record.get("metrics", {}).get(metric_key) is not None
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: float(item["metrics"][metric_key]))

    def _recommend_tokenizer(self, records: list[dict[str, Any]], target: dict[str, Any]) -> None:
        values = target["values"]
        notes = target["notes"]
        latest_train = self._latest(records, "tokenizer_train", statuses={"completed"})
        if latest_train:
            values.update({k: v for k, v in latest_train.get("params", {}).items() if v not in {"", None}})
        latest_eval = self._latest(records, "tokenizer_eval", statuses={"completed"})
        avg_bytes_per_token = latest_eval.get("metrics", {}).get("avg_bytes_per_token") if latest_eval else None
        if avg_bytes_per_token is not None:
            notes.append(f"Latest tokenizer eval averaged {avg_bytes_per_token:.2f} bytes/token.")
            current_vocab = int(values.get("vocab_size") or 32768)
            if avg_bytes_per_token > 5.5:
                values["vocab_size"] = min(65536, current_vocab + 4096)
                notes.append("Compression looks weak, so vocab size was increased slightly for the next tokenizer run.")
        if not notes:
            notes.append("No tokenizer benchmark history yet. Current tokenizer settings were left as-is.")

    def _recommend_base_train(self, records: list[dict[str, Any]], target: dict[str, Any]) -> None:
        values = target["values"]
        notes = target["notes"]
        latest = self._latest(records, "base_train")
        best = self._best_by_metric(records, "base_train", "min_val_bpb")
        preserve = {key: values.get(key) for key in ("corpus_dir", "run", "model_tag", "device_type") if values.get(key) not in {"", None}}

        if latest and latest.get("status") == "failed" and latest.get("metrics", {}).get("oom_detected"):
            values.update({k: v for k, v in latest.get("params", {}).items() if v not in {"", None}})
            values["device_batch_size"] = max(1, int(values.get("device_batch_size", 1)) // 2 or 1)
            max_seq_len = max(64, int(values.get("max_seq_len", 512)))
            total_batch_size = int(values.get("total_batch_size", max_seq_len * values["device_batch_size"]))
            values["total_batch_size"] = max(max_seq_len * values["device_batch_size"], total_batch_size // 2 or max_seq_len)
            if latest.get("metrics", {}).get("window_pattern_warning"):
                values["window_pattern"] = "L"
                notes.append("The previous run warned about sliding-window attention on the current hardware, so window_pattern was set to L.")
            notes.append("The latest base train run appears to have hit memory limits, so batch sizes were reduced.")
        elif best:
            values.update({k: v for k, v in best.get("params", {}).items() if v not in {"", None}})
            notes.append(f"Using the best completed base train settings so far (min val bpb {best['metrics']['min_val_bpb']:.6f}).")
            if latest and latest.get("job_id") == best.get("job_id"):
                num_iterations = int(values.get("num_iterations", 0) or 0)
                if num_iterations > 0:
                    values["num_iterations"] = max(num_iterations + 1, math.ceil(num_iterations * 1.15))
                    notes.append("That run is currently the best, so iterations were increased modestly for the next pass.")
        else:
            notes.append("No completed base train metrics yet. Current base settings were left in place.")

        for key, value in preserve.items():
            values[key] = value

    def _recommend_chat_sft(self, records: list[dict[str, Any]], target: dict[str, Any]) -> None:
        values = target["values"]
        notes = target["notes"]
        latest = self._latest(records, "chat_sft")
        best = self._best_by_metric(records, "chat_sft", "min_val_bpb")
        preserve = {
            key: values.get(key)
            for key in ("train_files", "val_files", "run", "model_tag", "model_step", "device_type")
            if values.get(key) not in {"", None}
        }

        if latest and latest.get("status") == "failed" and latest.get("metrics", {}).get("oom_detected"):
            values.update({k: v for k, v in latest.get("params", {}).items() if v not in {"", None}})
            values["device_batch_size"] = max(1, int(values.get("device_batch_size", 1)) // 2 or 1)
            max_seq_len = max(64, int(values.get("max_seq_len", 512)))
            total_batch_size = int(values.get("total_batch_size", max_seq_len * values["device_batch_size"]))
            values["total_batch_size"] = max(max_seq_len * values["device_batch_size"], total_batch_size // 2 or max_seq_len)
            notes.append("The latest chat SFT run appears to have hit memory limits, so batch sizes were reduced.")
        elif best:
            values.update({k: v for k, v in best.get("params", {}).items() if v not in {"", None}})
            notes.append(f"Using the best completed chat SFT settings so far (min val bpb {best['metrics']['min_val_bpb']:.6f}).")
            if latest and latest.get("job_id") == best.get("job_id"):
                num_iterations = int(values.get("num_iterations", 0) or 0)
                if num_iterations > 0:
                    values["num_iterations"] = max(num_iterations + 1, math.ceil(num_iterations * 1.10))
                    notes.append("That SFT run is currently the best, so iterations were nudged up slightly for the next pass.")
        else:
            notes.append("No completed chat SFT metrics yet. Current SFT settings were left in place.")

        if "include_identity" not in values:
            values["include_identity"] = 1
        if "identity_repeats" not in values:
            values["identity_repeats"] = 2

        for key, value in preserve.items():
            values[key] = value

    def _recommend_eval(self, records: list[dict[str, Any]], target: dict[str, Any]) -> None:
        values = target["values"]
        notes = target["notes"]
        latest_eval = self._latest(records, "benchmark_eval", statuses={"completed"}) or self._latest(records, "base_eval", statuses={"completed"})
        if latest_eval:
            values.update({k: v for k, v in latest_eval.get("params", {}).items() if v not in {"", None}})
        values["eval"] = "bpb,sample"
        if int(values.get("split_tokens", 0) or 0) < 262144:
            values["split_tokens"] = 262144
        if int(values.get("max_per_task", 0) or 0) <= 0:
            values["max_per_task"] = 64
        device_batch_size = int(values.get("device_batch_size", 0) or 0)
        if device_batch_size <= 0 or device_batch_size > 4:
            values["device_batch_size"] = 4
        notes.append("Benchmark eval was normalized to a stable local config so runs stay comparable.")
