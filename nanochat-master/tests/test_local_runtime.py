from nanochat.local_runtime import LocalRuntimeManager
import pytest


def make_runtime():
    return LocalRuntimeManager(".")


def test_chat_prefers_standard_message_content(monkeypatch):
    runtime = make_runtime()
    monkeypatch.setattr(runtime, "status", lambda: {"ready": True, "config": {"alias": "demo", "host": "127.0.0.1", "port": 8080}})
    monkeypatch.setattr(
        runtime,
        "_json_request",
        lambda url, payload: {
            "choices": [
                {
                    "message": {
                        "content": "Final answer",
                        "reasoning_content": "scratchpad that should not be used",
                    }
                }
            ]
        },
    )

    result = runtime.chat([{"role": "user", "content": "hello"}])

    assert result["text"] == "Final answer"
    assert result["text_source"] == "content"
    assert result["protocol_warning"] is None


def test_chat_falls_back_to_reasoning_content_when_content_is_empty(monkeypatch):
    runtime = make_runtime()
    monkeypatch.setattr(runtime, "status", lambda: {"ready": True, "config": {"alias": "demo", "host": "127.0.0.1", "port": 8080}})
    monkeypatch.setattr(
        runtime,
        "_json_request",
        lambda url, payload: {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": "I help you build and tune local LLMs.",
                    }
                }
            ]
        },
    )

    result = runtime.chat([{"role": "user", "content": "hello"}])

    assert result["text"] == "I help you build and tune local LLMs."
    assert result["text_source"] == "reasoning_content"
    assert "reasoning-only" in result["protocol_warning"]


def test_chat_handles_structured_content_parts(monkeypatch):
    runtime = make_runtime()
    monkeypatch.setattr(runtime, "status", lambda: {"ready": True, "config": {"alias": "demo", "host": "127.0.0.1", "port": 8080}})
    monkeypatch.setattr(
        runtime,
        "_json_request",
        lambda url, payload: {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"text": "Line one"},
                            {"type": "output_text", "text": "Line two"},
                        ],
                    }
                }
            ]
        },
    )

    result = runtime.chat([{"role": "user", "content": "hello"}])

    assert result["text"] == "Line one\nLine two"
    assert result["text_source"] == "content"


def test_recommended_start_config_is_generous_but_single_slot_on_cpu(monkeypatch):
    runtime = make_runtime()

    class FakePsutil:
        @staticmethod
        def cpu_count(logical=False):
            return 8 if logical else 4

        class _VM:
            total = 64 * (1024 ** 3)

        @staticmethod
        def virtual_memory():
            return FakePsutil._VM()

    monkeypatch.setattr("nanochat.local_runtime.psutil", FakePsutil)
    monkeypatch.setattr("nanochat.local_runtime.os.cpu_count", lambda: 8)

    config = runtime.recommended_start_config(
        overrides={},
        models=[{"path": "C:\\demo\\model.gguf", "size": 16 * (1024 ** 3), "name": "demo.gguf"}],
        devices=[],
    )

    assert config["ctx_size"] == 8192
    assert config["parallel"] == 1
    assert config["threads"] == 4


def test_recommended_start_config_scales_up_on_large_gpu(monkeypatch):
    runtime = make_runtime()

    class FakePsutil:
        @staticmethod
        def cpu_count(logical=False):
            return 16 if logical else 8

        class _VM:
            total = 128 * (1024 ** 3)

        @staticmethod
        def virtual_memory():
            return FakePsutil._VM()

    monkeypatch.setattr("nanochat.local_runtime.psutil", FakePsutil)
    monkeypatch.setattr("nanochat.local_runtime.os.cpu_count", lambda: 16)

    config = runtime.recommended_start_config(
        overrides={},
        models=[{"path": "C:\\demo\\model.gguf", "size": 8 * (1024 ** 3), "name": "demo.gguf"}],
        devices=[{"description": "RTX 6000 (24576 MiB, 20000 MiB free)"}],
    )

    assert config["ctx_size"] == 8192
    assert config["parallel"] == 2
    assert config["threads"] == 8


def test_bundle_status_uses_cache(monkeypatch, tmp_path):
    runtime = make_runtime()
    calls = {"models": 0, "devices": 0}
    runtime.runtime_dir = tmp_path / "runtime"
    runtime.runtime_dir.mkdir()
    runtime.server_binary = runtime.runtime_dir / "llama-server.exe"
    runtime.server_binary.write_bytes(b"demo")

    def fake_models():
        calls["models"] += 1
        return [{"path": "C:\\demo\\model.gguf", "name": "demo.gguf", "size": 1, "updated_at": 1.0}]

    def fake_devices():
        calls["devices"] += 1
        return [{"id": "Vulkan0", "description": "Demo GPU (8192 MiB, 4096 MiB free)"}]

    monkeypatch.setattr(runtime, "list_models", fake_models)
    monkeypatch.setattr(runtime, "_list_devices", fake_devices)

    first = runtime.bundle_status()
    second = runtime.bundle_status()

    assert first["recommended_model"]["name"] == "demo.gguf"
    assert second["recommended_model"]["name"] == "demo.gguf"
    assert calls == {"models": 1, "devices": 1}


def test_start_rejects_port_conflict(monkeypatch, tmp_path):
    runtime = make_runtime()
    model_path = tmp_path / "demo.gguf"
    model_path.write_bytes(b"demo")
    runtime.runtime_dir = tmp_path / "runtime"
    runtime.runtime_dir.mkdir()
    runtime.server_binary = runtime.runtime_dir / "llama-server.exe"
    runtime.server_binary.write_bytes(b"demo")
    monkeypatch.setattr(runtime, "_port_in_use", lambda host, port: True)

    with pytest.raises(RuntimeError, match="already in use"):
        runtime.start({"model_path": str(model_path), "host": "127.0.0.1", "port": 8091})
