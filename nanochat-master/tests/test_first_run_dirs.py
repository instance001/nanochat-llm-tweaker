from __future__ import annotations

import importlib
import sys


def test_chat_web_first_run_dirs_bootstrap_packaged_layout(tmp_path, monkeypatch):
    base_dir = tmp_path / "builder_data"
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(base_dir))
    monkeypatch.setattr(sys, "argv", ["chat_web_first_run_test"])

    chat_web = importlib.import_module("scripts.chat_web")
    chat_web = importlib.reload(chat_web)
    monkeypatch.setattr(chat_web, "REPO_ROOT", tmp_path)

    chat_web.ensure_first_run_dirs()

    for relative in chat_web.FIRST_RUN_REPO_DIRS:
        assert chat_web._layout_path(tmp_path, relative).is_dir()
    for relative in chat_web.FIRST_RUN_BASE_DIRS:
        assert chat_web._layout_path(base_dir, relative).is_dir()
