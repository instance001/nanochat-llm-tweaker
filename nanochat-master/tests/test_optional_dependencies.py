from __future__ import annotations

from pathlib import Path

import pytest

from nanochat import sft_dataset_tools

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_declares_parquet_extra():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    parquet_extra = pyproject["project"]["optional-dependencies"]["parquet"]

    assert any(requirement.startswith("pyarrow") for requirement in parquet_extra)


def test_sft_parquet_missing_pyarrow_error_mentions_project_extra(monkeypatch, tmp_path):
    monkeypatch.setattr(sft_dataset_tools, "pa", None)
    monkeypatch.setattr(sft_dataset_tools, "pq", None)

    with pytest.raises(RuntimeError, match="uv sync --extra parquet"):
        sft_dataset_tools.write_sft_parquet_file(tmp_path / "chat.parquet", "[]")
