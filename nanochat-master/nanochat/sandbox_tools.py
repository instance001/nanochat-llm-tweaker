"""
Strict file operations for scoped local workspaces.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None


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

    def list_files(self) -> list[dict[str, Any]]:
        files = []
        if not self.root.exists():
            return files
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(self.root).as_posix()
            item = {
                "path": relative,
                "size": path.stat().st_size,
                "updated_at": path.stat().st_mtime,
                "kind": self._detect_kind(path),
            }
            if item["kind"] == "parquet":
                try:
                    item.update(self._parquet_summary(path))
                except RuntimeError as exc:
                    item["preview_error"] = str(exc)
            files.append(item)
        return files

    def _detect_kind(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return "parquet"
        if suffix == ".json":
            return "json"
        if suffix == ".jsonl":
            return "jsonl"
        return "text"

    def _require_pyarrow(self) -> None:
        if pq is None or pa is None:
            raise RuntimeError(
                "Parquet support requires pyarrow, but it is not installed locally."
            )

    def _parquet_summary(self, path: Path) -> dict[str, Any]:
        self._require_pyarrow()
        parquet_file = pq.ParquetFile(path)
        row_count = parquet_file.metadata.num_rows if parquet_file.metadata is not None else None
        column_names = parquet_file.schema.names if parquet_file.schema is not None else []
        return {
            "row_count": row_count,
            "column_count": len(column_names),
            "columns": column_names,
        }

    def _read_parquet_file(self, relative_path: str, log_event: bool) -> dict[str, Any]:
        path = self._resolve_path(relative_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(relative_path)
        self._require_pyarrow()
        parquet_file = pq.ParquetFile(path)
        sample_rows = []
        if parquet_file.num_row_groups > 0:
            sample_table = parquet_file.read_row_group(0)
            sample_rows = sample_table.slice(0, min(12, sample_table.num_rows)).to_pylist()
        preview = json.dumps(sample_rows, ensure_ascii=True, indent=2)
        result = {
            "path": path.relative_to(self.root).as_posix(),
            "kind": "parquet",
            "size": path.stat().st_size,
            "updated_at": path.stat().st_mtime,
            "row_count": parquet_file.metadata.num_rows if parquet_file.metadata is not None else 0,
            "row_group_count": parquet_file.num_row_groups,
            "column_count": len(parquet_file.schema.names),
            "columns": parquet_file.schema.names,
            "schema": str(parquet_file.schema_arrow),
            "sample_rows": sample_rows,
            "preview": preview,
            "content": preview,
            "editable_as_text": False,
        }
        if log_event:
            self._log(
                f"{self.event_prefix}_read",
                f"Read {self.workspace_name} parquet file {result['path']}",
                {"path": result["path"], "size": result["size"], "row_count": result["row_count"]},
            )
        return result

    def read_file(self, relative_path: str) -> dict[str, Any]:
        path = self._resolve_path(relative_path)
        if path.suffix.lower() == ".parquet":
            return self._read_parquet_file(path.relative_to(self.root).as_posix(), log_event=True)
        result = super().read_file(relative_path)
        result["kind"] = self._detect_kind(path)
        result["editable_as_text"] = True
        return result

    def write_file(self, relative_path: str, content: str) -> dict[str, Any]:
        path = self._resolve_path(relative_path)
        if path.suffix.lower() == ".parquet":
            raise ValueError("Parquet corpus files must be written from structured records, not plain text.")
        result = super().write_file(relative_path, content)
        result["kind"] = self._detect_kind(path)
        result["editable_as_text"] = True
        return result

    def _parse_parquet_records_from_content(self, content: str) -> list[dict[str, Any]]:
        text = str(content).strip()
        if not text:
            raise ValueError("Parquet corpus writes require JSON/JSONL content with at least one object record.")

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list) and all(isinstance(record, dict) for record in payload):
            return payload
        if isinstance(payload, dict):
            return [payload]

        parsed_records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("Parquet corpus JSONL content must contain one valid JSON object per line.") from exc
            if not isinstance(payload, dict):
                raise ValueError("Parquet corpus JSONL content must contain JSON objects.")
            parsed_records.append(payload)

        if not parsed_records:
            raise ValueError("Parquet corpus writes require at least one JSON object record.")
        return parsed_records

    def write_from_content(self, relative_path: str, content: str, mode: str = "overwrite") -> dict[str, Any]:
        path = self._resolve_path(relative_path)
        if path.suffix.lower() == ".parquet":
            records = self._parse_parquet_records_from_content(content)
            return self.write_parquet_file(path.relative_to(self.root).as_posix(), records, mode=mode)
        return self.write_file(path.relative_to(self.root).as_posix(), content)

    def write_parquet_file(self, relative_path: str, records: list[dict[str, Any]], mode: str = "overwrite") -> dict[str, Any]:
        path = self._resolve_path(relative_path)
        if path.suffix.lower() != ".parquet":
            raise ValueError("Structured parquet writes require a .parquet target path.")
        if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
            raise ValueError("Parquet writes require a list of object records.")
        if not records:
            raise ValueError("Parquet writes require at least one record.")
        self._require_pyarrow()
        normalized_mode = mode.strip().lower() if mode else "overwrite"
        if normalized_mode not in {"overwrite", "append"}:
            raise ValueError("mode must be 'overwrite' or 'append'.")

        path.parent.mkdir(parents=True, exist_ok=True)
        final_records = list(records)
        if normalized_mode == "append" and path.exists():
            existing_table = pq.read_table(path)
            final_records = existing_table.to_pylist() + final_records
        table = pa.Table.from_pylist(final_records)
        pq.write_table(table, path)
        result = self._read_parquet_file(path.relative_to(self.root).as_posix(), log_event=False)
        self._log(
            f"{self.event_prefix}_write",
            f"Wrote {self.workspace_name} parquet file {result['path']}",
            {
                "path": result["path"],
                "size": result["size"],
                "row_count": result["row_count"],
                "mode": normalized_mode,
            },
        )
        return result

    def build_context(self, relative_paths: list[str], max_chars: int = 12000) -> str:
        sections = []
        remaining = max_chars
        for relative_path in relative_paths:
            path = self._resolve_path(relative_path)
            if path.suffix.lower() == ".parquet":
                file_data = self._read_parquet_file(path.relative_to(self.root).as_posix(), log_event=False)
                header = (
                    f"[{self.workspace_name} parquet file: {file_data['path']}]\n"
                    f"Schema:\n{file_data.get('schema', '')}\n\nSample rows:\n"
                )
                content = header + file_data.get("content", "")
            else:
                file_data = self._read_file(path.relative_to(self.root).as_posix(), log_event=False)
                content = file_data.get("content", "")
                content = f"[{self.workspace_name} file: {file_data['path']}]\n{content}"
            if len(content) > remaining:
                content = content[:remaining]
            sections.append(content)
            remaining -= len(content)
            if remaining <= 0:
                break
        return "\n\n".join(sections)
