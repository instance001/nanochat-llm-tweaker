"""
Strict file operations for scoped local workspaces.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class WorkspaceManager:
    def __init__(self, root: str | os.PathLike[str], workspace_name: str, event_prefix: str, activity_log=None):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.workspace_name = workspace_name
        self.event_prefix = event_prefix
        self.activity_log = activity_log

    def _log(self, kind: str, message: str, payload: dict[str, Any] | None = None) -> None:
        if self.activity_log is not None:
            self.activity_log.log_event(kind, message, payload)

    def _resolve_path(self, relative_path: str) -> Path:
        if not relative_path:
            raise ValueError(f"A {self.workspace_name} path is required.")
        candidate = Path(relative_path)
        if candidate.is_absolute():
            raise ValueError(f"{self.workspace_name} paths must be relative.")
        resolved = (self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Access is restricted to {self.workspace_name}.") from exc
        return resolved

    def status(self) -> dict[str, Any]:
        files = self.list_files()
        return {
            "workspace": self.workspace_name,
            "root": str(self.root),
            "exists": self.root.exists(),
            "file_count": len(files),
            "files": files,
        }

    def list_files(self) -> list[dict[str, Any]]:
        files = []
        if not self.root.exists():
            return files
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(self.root).as_posix()
            files.append(
                {
                    "path": relative,
                    "size": path.stat().st_size,
                    "updated_at": path.stat().st_mtime,
                }
            )
        return files

    def _read_file(self, relative_path: str, log_event: bool) -> dict[str, Any]:
        path = self._resolve_path(relative_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(relative_path)
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            content = handle.read()
        result = {
            "path": path.relative_to(self.root).as_posix(),
            "content": content,
            "size": path.stat().st_size,
            "updated_at": path.stat().st_mtime,
        }
        if log_event:
            self._log(
                f"{self.event_prefix}_read",
                f"Read {self.workspace_name} file {result['path']}",
                {"path": result["path"], "size": result["size"]},
            )
        return result

    def read_file(self, relative_path: str) -> dict[str, Any]:
        return self._read_file(relative_path, log_event=True)

    def write_file(self, relative_path: str, content: str) -> dict[str, Any]:
        path = self._resolve_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        result = self._read_file(path.relative_to(self.root).as_posix(), log_event=False)
        self._log(
            f"{self.event_prefix}_write",
            f"Wrote {self.workspace_name} file {result['path']}",
            {"path": result["path"], "size": result["size"]},
        )
        return result

    def append_file(self, relative_path: str, content: str) -> dict[str, Any]:
        path = self._resolve_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(content)
        result = self._read_file(path.relative_to(self.root).as_posix(), log_event=False)
        self._log(
            f"{self.event_prefix}_write",
            f"Appended {self.workspace_name} file {result['path']}",
            {"path": result["path"], "size": result["size"]},
        )
        return result

    def delete_file(self, relative_path: str) -> dict[str, Any]:
        path = self._resolve_path(relative_path)
        if not path.exists():
            raise FileNotFoundError(relative_path)
        if path.is_dir():
            raise ValueError(f"Only files can be deleted from {self.workspace_name}.")
        path.unlink()
        result = {
            "deleted": relative_path,
            "root": str(self.root),
        }
        self._log(
            f"{self.event_prefix}_delete",
            f"Deleted {self.workspace_name} file {relative_path}",
            {"path": relative_path},
        )
        return result

    def build_context(self, relative_paths: list[str], max_chars: int = 12000) -> str:
        sections = []
        remaining = max_chars
        for relative_path in relative_paths:
            file_data = self._read_file(relative_path, log_event=False)
            content = file_data["content"]
            if len(content) > remaining:
                content = content[:remaining]
            sections.append(f"[{self.workspace_name} file: {file_data['path']}]\n{content}")
            remaining -= len(content)
            if remaining <= 0:
                break
        return "\n\n".join(sections)


class SandboxManager(WorkspaceManager):
    def __init__(self, root: str | os.PathLike[str], activity_log=None):
        super().__init__(root=root, workspace_name="assistant_sandbox", event_prefix="sandbox", activity_log=activity_log)


class CorpusManager(WorkspaceManager):
    def __init__(self, root: str | os.PathLike[str], activity_log=None):
        super().__init__(root=root, workspace_name="local_corpus", event_prefix="corpus", activity_log=activity_log)
