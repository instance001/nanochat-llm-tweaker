"""
Managed local llama.cpp runtime for the dashboard.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any

MODEL_PATTERNS = ("*.gguf", "*.GGUF", "*.ggml")
LOG_LIMIT = 2000


class LocalRuntimeManager:
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self.runtime_dir = self.repo_root / "runtime" / "windows"
        self.server_binary = self.runtime_dir / "llama-server.exe"
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._logs: deque[str] = deque(maxlen=LOG_LIMIT)
        self._config: dict[str, Any] = {}
        self._mode: str | None = None
        self._ready = False
        self._last_error: str | None = None
        self._thread: threading.Thread | None = None

    def bundle_status(self) -> dict[str, Any]:
        runtime_exists = self.runtime_dir.exists()
        models = self.list_models()
        files = {
            "server_binary": str(self.server_binary),
            "server_exists": self.server_binary.exists(),
            "vulkan_backend": str(self.runtime_dir / "ggml-vulkan.dll"),
            "vulkan_exists": (self.runtime_dir / "ggml-vulkan.dll").exists(),
            "llama_cli_exists": (self.runtime_dir / "llama-cli.exe").exists(),
        }
        return {
            "runtime_dir": str(self.runtime_dir),
            "runtime_exists": runtime_exists,
            "files": files,
            "devices": self._list_devices() if files["server_exists"] else [],
            "models": models,
            "recommended_model": models[0] if models else None,
        }

    def list_models(self) -> list[dict[str, Any]]:
        roots = [
            self.repo_root / "assistant_models",
            self.repo_root / "models",
            self.repo_root / "runtime" / "models",
            self.repo_root,
        ]
        seen: set[str] = set()
        results: list[dict[str, Any]] = []
        for root in roots:
            if not root.exists():
                continue
            for pattern in MODEL_PATTERNS:
                for path in root.rglob(pattern):
                    resolved = str(path.resolve())
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    try:
                        stat = path.stat()
                    except OSError:
                        continue
                    if stat.st_size <= 0:
                        continue
                    results.append(
                        {
                            "path": resolved,
                            "name": path.name,
                            "size": stat.st_size,
                            "updated_at": stat.st_mtime,
                            "score": self._score_model(path.name, stat.st_size),
                        }
                    )
        results.sort(key=lambda item: (item["score"], item["updated_at"]), reverse=True)
        for result in results:
            result.pop("score", None)
        return results[:200]

    def status(self) -> dict[str, Any]:
        bundle = self.bundle_status()
        with self._lock:
            process = self._process
            running = process is not None and process.poll() is None
            if process is not None and process.poll() is not None:
                self._ready = False
            return {
                "bundle": bundle,
                "running": running,
                "ready": self._ready and running,
                "mode": self._mode,
                "pid": process.pid if running else None,
                "config": self._config,
                "last_error": self._last_error,
                "log_tail": list(self._logs)[-80:],
            }

    def start(self, config: dict[str, Any]) -> dict[str, Any]:
        if not self.server_binary.exists():
            raise FileNotFoundError(f"Missing runtime binary: {self.server_binary}")

        model_path_value = config.get("model_path", "")
        if not model_path_value:
            recommended = self.bundle_status().get("recommended_model")
            if recommended is None:
                raise FileNotFoundError("No local GGUF models were found. Add one under assistant_models or models.")
            model_path_value = recommended["path"]

        model_path = Path(model_path_value).expanduser()
        if not model_path.is_absolute():
            model_path = (self.repo_root / model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        requested = {
            "model_path": str(model_path),
            "host": config.get("host", "127.0.0.1"),
            "port": int(config.get("port", 8091)),
            "ctx_size": int(config.get("ctx_size", 4096)),
            "threads": int(config.get("threads", max(4, os.cpu_count() or 4))),
            "threads_http": int(config.get("threads_http", 4)),
            "parallel": int(config.get("parallel", 2)),
            "alias": config.get("alias", model_path.stem),
            "device_strategy": config.get("device_strategy", "auto"),
            "gpu_layers": str(config.get("gpu_layers", "auto")),
            "preferred_device": config.get("preferred_device", ""),
        }

        attempts = self._build_attempts(requested)
        last_error = "Runtime failed to start"
        for attempt in attempts:
            self.stop()
            self._launch(requested, attempt)
            if self._wait_until_ready(requested["host"], requested["port"], timeout_s=24):
                with self._lock:
                    self._ready = True
                    self._mode = attempt["mode"]
                    self._config = requested | {"active_device": attempt.get("device_name")}
                    self._last_error = None
                return self.status()
            with self._lock:
                last_error = self._last_error or f"{attempt['mode']} launch did not become healthy"
            self.stop()

        raise RuntimeError(last_error)

    def stop(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            self._ready = False
            self._mode = None
            self._config = {}
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        with self._lock:
            self._process = None
        return self.status()

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        status = self.status()
        if not status["ready"]:
            raise RuntimeError("Local runtime is not ready")

        config = status["config"]
        payload_messages = []
        if system_prompt:
            payload_messages.append({"role": "system", "content": system_prompt})
        payload_messages.extend(messages)
        payload = {
            "model": config.get("alias") or "local-runtime",
            "messages": payload_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        url = f"http://{config['host']}:{config['port']}/v1/chat/completions"
        response = self._json_request(url, payload)
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError("Runtime returned no choices")
        message = choices[0].get("message", {})
        text = self._clean_response_text(message.get("content", ""))
        return {
            "text": text,
            "raw": response,
        }

    def _build_attempts(self, requested: dict[str, Any]) -> list[dict[str, Any]]:
        strategy = requested["device_strategy"]
        devices = self._list_devices()
        preferred = requested["preferred_device"]
        chosen_device = preferred or (devices[0]["id"] if devices else "")
        gpu_attempt = {"mode": "gpu", "device_name": chosen_device, "gpu_layers": requested["gpu_layers"]}
        cpu_attempt = {"mode": "cpu", "device_name": "none", "gpu_layers": "0"}

        if strategy == "cpu":
            return [cpu_attempt]
        if strategy == "gpu":
            if not chosen_device:
                raise RuntimeError("GPU mode requested, but no local runtime devices were found")
            return [gpu_attempt]
        if chosen_device:
            return [gpu_attempt, cpu_attempt]
        return [cpu_attempt]

    def _launch(self, requested: dict[str, Any], attempt: dict[str, Any]) -> None:
        command = [
            str(self.server_binary),
            "--model",
            requested["model_path"],
            "--host",
            requested["host"],
            "--port",
            str(requested["port"]),
            "--ctx-size",
            str(requested["ctx_size"]),
            "--threads",
            str(requested["threads"]),
            "--threads-http",
            str(requested["threads_http"]),
            "--parallel",
            str(requested["parallel"]),
            "--alias",
            requested["alias"],
            "--jinja",
            "--no-webui",
            "--log-colors",
            "off",
        ]
        if attempt["mode"] == "gpu":
            command.extend(["--device", attempt["device_name"], "--gpu-layers", str(attempt["gpu_layers"]), "--fit", "on"])
        else:
            command.extend(["--device", "none", "--gpu-layers", "0"])

        with self._lock:
            self._logs.clear()
            self._logs.append(f"$ {subprocess.list2cmdline(command)}")
            self._ready = False
            self._mode = attempt["mode"]
            self._last_error = None
            process = subprocess.Popen(
                command,
                cwd=str(self.runtime_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self._process = process
            self._thread = threading.Thread(target=self._pump_logs, args=(process,), daemon=True)
            self._thread.start()

    def _pump_logs(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            with self._lock:
                self._logs.append(line.rstrip())
        return_code = process.wait()
        with self._lock:
            if return_code != 0 and not self._last_error:
                self._last_error = f"Runtime exited with code {return_code}"
            self._ready = False

    def _wait_until_ready(self, host: str, port: int, timeout_s: int) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._lock:
                process = self._process
                if process is None:
                    self._last_error = "Runtime process was not created"
                    return False
                if process.poll() is not None:
                    self._last_error = f"Runtime exited with code {process.returncode}"
                    return False

            for url in (
                f"http://{host}:{port}/health",
                f"http://{host}:{port}/v1/models",
            ):
                try:
                    urllib.request.urlopen(url, timeout=2).read()
                    return True
                except Exception:
                    pass
            time.sleep(0.5)

        with self._lock:
            self._last_error = "Timed out waiting for runtime health"
        return False

    def _list_devices(self) -> list[dict[str, str]]:
        try:
            result = subprocess.run(
                [str(self.server_binary), "--list-devices"],
                cwd=str(self.runtime_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=8,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return []

        devices = []
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if ":" not in stripped:
                continue
            if not stripped.startswith(("Vulkan", "CUDA", "Metal", "OpenCL", "SYCL")):
                continue
            device_id, description = stripped.split(":", 1)
            devices.append({"id": device_id.strip(), "description": description.strip()})
        return devices

    def _json_request(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Runtime request failed: {exc.code} {body}") from exc

    def _clean_response_text(self, text: str) -> str:
        if not text:
            return ""

        markers = [
            "<|im_start|>user",
            "<|im_start|>assistant",
            "<|im_end|>",
            "<|assistant|>",
            "<|user|>",
            "<|end|>",
        ]
        cleaned = text
        for marker in markers:
            if marker in cleaned:
                cleaned = cleaned.split(marker, 1)[0]
        return cleaned.strip()

    def _score_model(self, name: str, size: int) -> int:
        lower = name.lower()
        score = 0

        if "coder" in lower:
            score += 55
        if "instruct" in lower or "chat" in lower or "-it" in lower:
            score += 35
        if "q4" in lower:
            score += 18
        if "q5" in lower:
            score += 10
        if "q8" in lower:
            score -= 12
        if any(token in lower for token in ("abliterated", "uncensored", "dolphin", "guanaco", "hermes")):
            score -= 40
        if "mixtral" in lower or "32b" in lower or "20b" in lower:
            score -= 45
        if "tinyllama" in lower:
            score -= 20
        if "mistral-7b-instruct-v0.2" in lower:
            score += 20
        if "deepseek-coder-6.7b-instruct" in lower:
            score += 30
        if "qwen2.5-1.5b-instruct" in lower:
            score += 8

        size_gb = size / (1024 ** 3)
        if 2.5 <= size_gb <= 5.0:
            score += 25
        elif 0.5 <= size_gb <= 2.0:
            score += 8
        elif size_gb > 8.5:
            score -= 30

        return score
