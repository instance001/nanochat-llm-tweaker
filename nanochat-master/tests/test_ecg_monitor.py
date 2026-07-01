from nanochat.ecg_monitor import EcgMonitor


def test_ecg_monitor_prefers_gpu_signal(monkeypatch):
    monitor = EcgMonitor(lambda: {"running_jobs": 1, "recent_event_age_s": 2}, interval_s=10)
    monkeypatch.setattr(monitor, "_cpu_percent", lambda: 18.0)
    monkeypatch.setattr(monitor, "_ram_percent", lambda: 45.0)
    monkeypatch.setattr(
        monitor,
        "_gpu_samples",
        lambda: [{"name": "RTX Demo", "percent": 72.0, "memory_percent": 61, "memory_used_mib": 4000, "memory_total_mib": 8000}],
    )

    sample = monitor._sample()

    assert sample["source"] == "gpu"
    assert sample["current_percent"] == 72
    assert sample["status"] == "busy"
    assert "VRAM 61%" in sample["note"]


def test_ecg_monitor_falls_back_to_cpu_signal(monkeypatch):
    monitor = EcgMonitor(lambda: {"runtime_running": True, "recent_event_age_s": 5}, interval_s=10)
    monkeypatch.setattr(monitor, "_cpu_percent", lambda: 27.0)
    monkeypatch.setattr(monitor, "_ram_percent", lambda: 52.0)
    monkeypatch.setattr(monitor, "_gpu_samples", lambda: [])

    sample = monitor._sample()

    assert sample["source"] == "cpu"
    assert sample["current_percent"] == 27
    assert sample["status"] == "active"
    assert "CPU 27%" in sample["note"]


def test_ecg_monitor_uses_app_signal_when_hardware_is_unavailable(monkeypatch):
    monitor = EcgMonitor(lambda: {"queued_jobs": 1, "latest_job_label": "Tokenizer", "recent_event_age_s": 1}, interval_s=10)
    monkeypatch.setattr(monitor, "_cpu_percent", lambda: None)
    monkeypatch.setattr(monitor, "_ram_percent", lambda: None)
    monkeypatch.setattr(monitor, "_gpu_samples", lambda: [])

    sample = monitor._sample()

    assert sample["source"] == "app"
    assert sample["available"] is True
    assert sample["current_percent"] == 25
    assert "queued" in sample["note"]


def test_ecg_monitor_falls_back_to_windows_gpu_counters(monkeypatch):
    monitor = EcgMonitor(lambda: {"runtime_running": True, "recent_event_age_s": 3}, interval_s=10)
    monkeypatch.setattr("nanochat.ecg_monitor.os.name", "nt")
    monkeypatch.setattr(monitor, "_cpu_percent", lambda: 11.0)
    monkeypatch.setattr(monitor, "_ram_percent", lambda: 50.0)
    monkeypatch.setattr(monitor, "_gpu_samples_nvidia", lambda: [])
    monkeypatch.setattr(
        monitor,
        "_read_windows_gpu_counters",
        lambda: {
            "engines": [{"phys": 0, "percent": 63.0}],
            "memory": [{"phys": 0, "bytes": 1024 * 1024 * 512}],
        },
    )

    sample = monitor._sample()

    assert sample["source"] == "gpu"
    assert sample["current_percent"] == 63
    assert sample["sources"]["gpus"][0]["memory_used_mib"] == 512
