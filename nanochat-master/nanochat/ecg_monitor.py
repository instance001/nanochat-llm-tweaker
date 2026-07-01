"""
Passive dashboard ECG monitor for hardware and builder activity.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from typing import Any, Callable

try:
    import psutil
except ImportError:
    psutil = None


def _clamp_percent(value: float | int | None) -> int:
    if value is None:
        return 0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, int(round(numeric))))


class EcgMonitor:
    def __init__(
        self,
        activity_provider: Callable[[], dict[str, Any]] | None = None,
        *,
        interval_s: float = 1.0,
        history_size: int = 60,
    ):
        self.activity_provider = activity_provider or (lambda: {})
        self.interval_s = max(0.25, float(interval_s))
        self.history_size = max(8, int(history_size))
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._history: deque[int] = deque([0] * min(24, self.history_size), maxlen=self.history_size)
        self._gpu_cache: list[dict[str, Any]] = []
        self._gpu_cache_ready = False
        self._gpu_cache_ts = 0.0
        self._gpu_cache_ttl_s = 2.0
        self._snapshot = self._build_snapshot(
            supported=False,
            available=False,
            source="none",
            label="ECG window offline",
            note="Waiting for the first local hardware sample.",
            current_percent=0,
            sources={},
            status="offline",
        )

    def start(self) -> None:
        if psutil is not None:
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, name="dashboard-ecg-monitor", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = None
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread is not None:
            thread.join(timeout=max(1.0, self.interval_s * 2))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._snapshot)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sample = self._sample()
            with self._lock:
                self._history.append(sample["current_percent"])
                self._snapshot = self._build_snapshot(
                    supported=sample["supported"],
                    available=sample["available"],
                    source=sample["source"],
                    label=sample["label"],
                    note=sample["note"],
                    current_percent=sample["current_percent"],
                    sources=sample["sources"],
                    status=sample["status"],
                )
            self._stop_event.wait(self.interval_s)

    def _build_snapshot(
        self,
        *,
        supported: bool,
        available: bool,
        source: str,
        label: str,
        note: str,
        current_percent: int,
        sources: dict[str, Any],
        status: str,
    ) -> dict[str, Any]:
        return {
            "supported": supported,
            "available": available,
            "source": source,
            "label": label,
            "note": note,
            "current_percent": current_percent,
            "history": list(self._history),
            "sources": sources,
            "status": status,
            "sampled_at": time.time(),
        }

    def _sample(self) -> dict[str, Any]:
        activity = self._safe_activity()
        cpu_percent = self._cpu_percent()
        ram_percent = self._ram_percent()
        gpus = self._gpu_samples()
        gpu_sample = max(gpus, key=lambda item: item.get("percent", -1), default=None)
        app_signal = self._app_signal(activity)

        source = "none"
        label = "ECG window unavailable"
        note = "No local CPU, GPU, or builder activity signal is available yet."
        current_percent = 0
        status = "offline"

        if gpu_sample and gpu_sample.get("percent") is not None:
            current_percent = _clamp_percent(gpu_sample.get("percent"))
            source = "gpu"
            label = "GPU activity"
            note = self._gpu_note(gpu_sample, activity)
        elif cpu_percent is not None:
            current_percent = _clamp_percent(cpu_percent)
            source = "cpu"
            label = "CPU activity"
            note = self._cpu_note(cpu_percent, ram_percent, activity)
        elif app_signal is not None:
            current_percent = _clamp_percent(app_signal)
            source = "app"
            label = "Builder activity"
            note = self._app_note(activity)

        if source != "none":
            if current_percent >= 70:
                status = "busy"
            elif current_percent >= 20:
                status = "active"
            elif self._has_active_work(activity):
                status = "quiet-active"
            else:
                status = "idle"

        return {
            "supported": source != "none" or self._has_any_signal(cpu_percent, gpus, app_signal),
            "available": source != "none",
            "source": source,
            "label": label,
            "note": note,
            "current_percent": current_percent,
            "sources": {
                "cpu_percent": _clamp_percent(cpu_percent) if cpu_percent is not None else None,
                "ram_percent": _clamp_percent(ram_percent) if ram_percent is not None else None,
                "gpus": gpus,
                "app": activity,
            },
            "status": status,
        }

    def _safe_activity(self) -> dict[str, Any]:
        try:
            payload = self.activity_provider() or {}
        except Exception as exc:
            return {"error": str(exc)}
        return payload if isinstance(payload, dict) else {}

    def _cpu_percent(self) -> float | None:
        if psutil is None:
            return None
        try:
            return float(psutil.cpu_percent(interval=None))
        except Exception:
            return None

    def _ram_percent(self) -> float | None:
        if psutil is None:
            return None
        try:
            return float(psutil.virtual_memory().percent)
        except Exception:
            return None

    def _gpu_samples(self) -> list[dict[str, Any]]:
        now = time.time()
        if self._gpu_cache_ready and (now - self._gpu_cache_ts) < self._gpu_cache_ttl_s:
            return list(self._gpu_cache)
        samples = self._gpu_samples_nvidia()
        if not samples:
            samples = self._gpu_samples_windows()
        self._gpu_cache = list(samples)
        self._gpu_cache_ready = True
        self._gpu_cache_ts = now
        return list(samples)

    def _gpu_samples_nvidia(self) -> list[dict[str, Any]]:
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            return []
        command = [
            nvidia_smi,
            "--query-gpu=name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=1.2,
                check=False,
            )
        except Exception:
            return []
        if result.returncode != 0:
            return []
        rows: list[dict[str, Any]] = []
        for raw in result.stdout.splitlines():
            parts = [part.strip() for part in raw.split(",")]
            if len(parts) < 4:
                continue
            name = parts[0]
            try:
                used_mib = float(parts[2])
                total_mib = float(parts[3])
                memory_percent = (used_mib / total_mib * 100.0) if total_mib > 0 else None
            except ValueError:
                used_mib = 0.0
                total_mib = 0.0
                memory_percent = None
            rows.append(
                {
                    "name": name,
                    "percent": _parse_percent(parts[1]),
                    "memory_used_mib": int(round(used_mib)),
                    "memory_total_mib": int(round(total_mib)),
                    "memory_percent": _clamp_percent(memory_percent) if memory_percent is not None else None,
                }
            )
        return rows

    def _gpu_samples_windows(self) -> list[dict[str, Any]]:
        if os.name != "nt":
            return []
        counters = self._read_windows_gpu_counters()
        if not counters:
            return []
        engines = counters.get("engines") or []
        memory = counters.get("memory") or []
        aggregates: dict[int, dict[str, Any]] = {}
        for entry in engines:
            phys = int(entry.get("phys", -1))
            if phys < 0:
                continue
            aggregate = aggregates.setdefault(
                phys,
                {"name": f"GPU {phys}", "percent": 0, "memory_used_mib": None, "memory_total_mib": None, "memory_percent": None},
            )
            aggregate["percent"] = max(aggregate["percent"], _clamp_percent(entry.get("percent")))
        for entry in memory:
            phys = int(entry.get("phys", -1))
            if phys < 0:
                continue
            aggregate = aggregates.setdefault(
                phys,
                {"name": f"GPU {phys}", "percent": 0, "memory_used_mib": None, "memory_total_mib": None, "memory_percent": None},
            )
            used_mib = int(round(float(entry.get("bytes", 0.0)) / (1024 ** 2)))
            aggregate["memory_used_mib"] = used_mib
        return [aggregate for _, aggregate in sorted(aggregates.items())]

    def _read_windows_gpu_counters(self) -> dict[str, Any] | None:
        script = r"""
$engines = @(Get-Counter '\GPU Engine(*)\Utilization Percentage' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty CounterSamples | ForEach-Object {
    $phys = -1
    if ($_.InstanceName -match 'phys_(\d+)') { $phys = [int]$matches[1] }
    [pscustomobject]@{
        instance = $_.InstanceName
        phys = $phys
        percent = [double]$_.CookedValue
    }
})
$memory = @(Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty CounterSamples | ForEach-Object {
    $phys = -1
    if ($_.InstanceName -match 'phys_(\d+)') { $phys = [int]$matches[1] }
    [pscustomobject]@{
        instance = $_.InstanceName
        phys = $phys
        bytes = [double]$_.CookedValue
    }
})
[pscustomobject]@{
    engines = $engines
    memory = $memory
} | ConvertTo-Json -Compress -Depth 5
"""
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=2.0,
                check=False,
            )
        except Exception:
            return None
        if result.returncode != 0 or not result.stdout.strip():
            return None
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _app_signal(self, activity: dict[str, Any]) -> int | None:
        if not activity:
            return None
        if activity.get("chat_loading"):
            return 70
        if activity.get("running_jobs"):
            return min(90, 30 + 15 * int(activity.get("running_jobs", 0)))
        if activity.get("queued_jobs"):
            return 25
        if activity.get("runtime_running"):
            return 15
        if activity.get("recent_event_age_s") is not None and float(activity["recent_event_age_s"]) < 10:
            return 10
        return 0

    def _has_any_signal(self, cpu_percent: float | None, gpus: list[dict[str, Any]], app_signal: int | None) -> bool:
        return cpu_percent is not None or bool(gpus) or app_signal is not None

    def _has_active_work(self, activity: dict[str, Any]) -> bool:
        return bool(
            activity.get("chat_loading")
            or activity.get("runtime_running")
            or activity.get("running_jobs")
            or activity.get("queued_jobs")
        )

    def _gpu_note(self, gpu_sample: dict[str, Any], activity: dict[str, Any]) -> str:
        parts = [
            f"{gpu_sample.get('name', 'GPU')} at {_clamp_percent(gpu_sample.get('percent'))}%",
        ]
        memory_percent = gpu_sample.get("memory_percent")
        if memory_percent is not None:
            parts.append(f"VRAM {memory_percent}%")
        parts.append(self._activity_suffix(activity))
        return " | ".join(part for part in parts if part)

    def _cpu_note(self, cpu_percent: float, ram_percent: float | None, activity: dict[str, Any]) -> str:
        parts = [f"CPU {_clamp_percent(cpu_percent)}%"]
        if ram_percent is not None:
            parts.append(f"RAM {_clamp_percent(ram_percent)}%")
        parts.append(self._activity_suffix(activity))
        return " | ".join(part for part in parts if part)

    def _app_note(self, activity: dict[str, Any]) -> str:
        suffix = self._activity_suffix(activity)
        if suffix:
            return suffix
        return "The dashboard has an app-level heartbeat but no local hardware telemetry."

    def _activity_suffix(self, activity: dict[str, Any]) -> str:
        if not activity:
            return ""
        parts: list[str] = []
        if activity.get("running_jobs"):
            parts.append(f"{activity['running_jobs']} job running")
        elif activity.get("queued_jobs"):
            parts.append(f"{activity['queued_jobs']} job queued")
        elif activity.get("runtime_running"):
            parts.append("runtime process up")
        else:
            parts.append("no active run")
        if activity.get("recent_event_age_s") is not None:
            age_s = int(max(0, round(float(activity["recent_event_age_s"]))))
            parts.append(f"last event {age_s}s ago")
        return " | ".join(parts)


def _parse_percent(raw: str) -> float | None:
    cleaned = str(raw).strip().replace("%", "")
    if not cleaned or cleaned.lower() in {"n/a", "na", "not supported"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None
