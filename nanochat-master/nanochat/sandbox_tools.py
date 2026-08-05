"""
Strict file operations for scoped local workspaces.
"""

from __future__ import annotations

import json
import os
import random
import hashlib
from pathlib import Path
from typing import Any

from nanochat.dataset_sources import build_import_manifest, records_to_jsonl, shard_import_records, slugify_import_name, split_records_for_import

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None

PARQUET_INSTALL_HINT = "Install parquet support with `uv sync --extra parquet` from nanochat-master."


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
            raise RuntimeError(f"Parquet support requires pyarrow, but it is not installed locally. {PARQUET_INSTALL_HINT}")

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
            raise ValueError("Sandbox parquet files must be written through an export action, not plain text.")
        result = super().write_file(relative_path, content)
        result["kind"] = self._detect_kind(path)
        result["editable_as_text"] = True
        return result


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
                f"Parquet support requires pyarrow, but it is not installed locally. {PARQUET_INSTALL_HINT}"
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

    def convert_content_to_parquet_records(
        self,
        content: str,
        conversion_mode: str = "paragraphs",
        source: str = "dashboard",
    ) -> dict[str, Any]:
        text = str(content or "")
        normalized_mode = (conversion_mode or "paragraphs").strip().lower().replace("-", "_")
        if normalized_mode in {"paragraph", "paragraphs"}:
            records, rejected = self._records_from_paragraphs(text, source)
        elif normalized_mode in {"line", "lines"}:
            records, rejected = self._records_from_lines(text, source)
        elif normalized_mode in {"markdown", "markdown_sections", "heading_sections"}:
            records, rejected = self._records_from_markdown_sections(text, source)
        elif normalized_mode in {"jsonl", "json_lines"}:
            records = self._parse_jsonl_records(text)
            rejected = 0
        elif normalized_mode in {"json", "json_array", "json_object"}:
            records = self._parse_json_records(text)
            rejected = 0
        else:
            raise ValueError(
                "Unsupported parquet conversion mode. Use paragraphs, lines, markdown_sections, jsonl, or json_array."
            )
        if not records:
            raise ValueError("Parquet conversion produced no records. Add usable text or structured rows first.")
        columns = sorted({key for record in records for key in record.keys()})
        return {
            "conversion_mode": normalized_mode,
            "row_count": len(records),
            "column_count": len(columns),
            "columns": columns,
            "sample_rows": records[:12],
            "rejected_empty_rows": rejected,
            "records": records,
        }

    def preview_parquet_conversion(
        self,
        content: str,
        conversion_mode: str = "paragraphs",
        source: str = "dashboard",
    ) -> dict[str, Any]:
        payload = self.convert_content_to_parquet_records(content, conversion_mode=conversion_mode, source=source)
        preview = dict(payload)
        preview.pop("records", None)
        return preview

    def split_content(
        self,
        content: str,
        split_mode: str = "paragraphs",
        val_ratio: float = 0.1,
        seed: int = 1337,
    ) -> dict[str, Any]:
        if val_ratio <= 0 or val_ratio >= 1:
            raise ValueError("val_ratio must be greater than 0 and less than 1.")
        mode = (split_mode or "paragraphs").strip().lower().replace("-", "_")
        items = self._split_items_from_content(content, mode)
        if not items:
            raise ValueError("Corpus split produced no records. Add usable content first.")
        indexed = list(enumerate(items))
        rng = random.Random(seed)
        rng.shuffle(indexed)
        if len(indexed) == 1:
            val_count = 0
        else:
            val_count = max(1, round(len(indexed) * val_ratio))
            val_count = min(val_count, len(indexed) - 1)
        val_items = indexed[:val_count]
        train_items = indexed[val_count:]
        train_values = [item for _, item in train_items]
        val_values = [item for _, item in val_items]
        return {
            "split_mode": mode,
            "record_count": len(items),
            "train_count": len(train_values),
            "val_count": len(val_values),
            "train_indices": [index for index, _ in train_items],
            "val_indices": [index for index, _ in val_items],
            "val_ratio": val_ratio,
            "seed": seed,
            "train_content": self._serialize_split_items(train_values, mode),
            "val_content": self._serialize_split_items(val_values, mode),
            "sample_train": train_values[:5],
            "sample_val": val_values[:5],
        }

    def preview_split_content(
        self,
        content: str,
        split_mode: str = "paragraphs",
        val_ratio: float = 0.1,
        seed: int = 1337,
    ) -> dict[str, Any]:
        payload = self.split_content(content, split_mode=split_mode, val_ratio=val_ratio, seed=seed)
        preview = dict(payload)
        preview.pop("train_content", None)
        preview.pop("val_content", None)
        return preview

    def write_split_files(
        self,
        train_path: str,
        val_path: str,
        content: str,
        split_mode: str = "paragraphs",
        val_ratio: float = 0.1,
        seed: int = 1337,
    ) -> dict[str, Any]:
        payload = self.split_content(content, split_mode=split_mode, val_ratio=val_ratio, seed=seed)
        train_result = self.write_file(train_path, payload["train_content"])
        val_result = self.write_file(val_path, payload["val_content"]) if payload["val_count"] else {
            "path": val_path,
            "size": 0,
            "updated_at": None,
        }
        return {
            **payload,
            "train_path": train_result["path"],
            "val_path": val_result["path"],
            "train_size": train_result["size"],
            "val_size": val_result["size"],
            "updated_at": max(
                updated_at for updated_at in [train_result["updated_at"], val_result["updated_at"]] if updated_at is not None
            ),
        }

    def write_converted_parquet_file(
        self,
        relative_path: str,
        content: str,
        conversion_mode: str = "paragraphs",
        mode: str = "overwrite",
        source: str = "dashboard",
    ) -> dict[str, Any]:
        payload = self.convert_content_to_parquet_records(content, conversion_mode=conversion_mode, source=source)
        result = self.write_parquet_file(relative_path, payload["records"], mode=mode)
        result["conversion"] = {
            "conversion_mode": payload["conversion_mode"],
            "rejected_empty_rows": payload["rejected_empty_rows"],
        }
        return result

    def _records_from_paragraphs(self, text: str, source: str) -> tuple[list[dict[str, Any]], int]:
        chunks = [chunk.strip() for chunk in text.replace("\r\n", "\n").split("\n\n")]
        return self._text_records_from_chunks(chunks, source)

    def _records_from_lines(self, text: str, source: str) -> tuple[list[dict[str, Any]], int]:
        chunks = [line.strip() for line in text.splitlines()]
        return self._text_records_from_chunks(chunks, source)

    def _records_from_markdown_sections(self, text: str, source: str) -> tuple[list[dict[str, Any]], int]:
        sections: list[dict[str, Any]] = []
        current_heading = ""
        current_lines: list[str] = []
        rejected = 0

        def flush() -> None:
            nonlocal rejected
            body = "\n".join(current_lines).strip()
            heading = current_heading.strip()
            if not heading and not body:
                rejected += 1
                return
            section_text = f"{heading}\n\n{body}".strip() if heading else body
            if not section_text:
                rejected += 1
                return
            sections.append(
                {
                    "text": section_text,
                    "heading": heading,
                    "source": source,
                    "row_index": len(sections),
                }
            )

        for raw_line in text.replace("\r\n", "\n").splitlines():
            line = raw_line.rstrip()
            if line.startswith("#"):
                marker, _, title = line.partition(" ")
                if marker and all(char == "#" for char in marker):
                    flush()
                    current_heading = title.strip() or marker
                    current_lines = []
                    continue
            current_lines.append(line)
        flush()
        return sections, rejected

    def _text_records_from_chunks(self, chunks: list[str], source: str) -> tuple[list[dict[str, Any]], int]:
        records = []
        rejected = 0
        for chunk in chunks:
            normalized = chunk.strip()
            if not normalized:
                rejected += 1
                continue
            records.append(
                {
                    "text": normalized,
                    "source": source,
                    "row_index": len(records),
                }
            )
        return records, rejected

    def _split_items_from_content(self, content: str, mode: str) -> list[Any]:
        text = str(content or "")
        if mode in {"paragraph", "paragraphs"}:
            return [chunk.strip() for chunk in text.replace("\r\n", "\n").split("\n\n") if chunk.strip()]
        if mode in {"line", "lines"}:
            return [line.strip() for line in text.splitlines() if line.strip()]
        if mode in {"markdown", "markdown_sections", "heading_sections"}:
            records, _ = self._records_from_markdown_sections(text, source="split")
            return [record["text"] for record in records]
        if mode in {"jsonl", "json_lines"}:
            return self._parse_jsonl_records(text)
        if mode in {"json", "json_array", "json_object"}:
            return self._parse_json_records(text)
        raise ValueError("Unsupported corpus split mode. Use paragraphs, lines, markdown_sections, jsonl, or json_array.")

    def _serialize_split_items(self, items: list[Any], mode: str) -> str:
        if not items:
            return ""
        if mode in {"jsonl", "json_lines", "json", "json_array", "json_object"}:
            return "\n".join(json.dumps(item, ensure_ascii=True) for item in items) + "\n"
        separator = "\n" if mode in {"line", "lines"} else "\n\n"
        return separator.join(str(item).strip() for item in items if str(item).strip()) + "\n"

    def _parse_json_records(self, text: str) -> list[dict[str, Any]]:
        stripped = text.strip()
        if not stripped:
            raise ValueError("JSON parquet conversion requires a JSON object or array of objects.")
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError("JSON parquet conversion requires valid JSON.") from exc
        if isinstance(payload, dict):
            return [payload]
        if isinstance(payload, list) and all(isinstance(record, dict) for record in payload):
            return payload
        raise ValueError("JSON parquet conversion requires a JSON object or an array of objects.")

    def _parse_jsonl_records(self, text: str) -> list[dict[str, Any]]:
        records = []
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL parquet conversion failed on line {line_number}.") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL parquet conversion line {line_number} must be a JSON object.")
            records.append(payload)
        if not records:
            raise ValueError("JSONL parquet conversion requires at least one JSON object line.")
        return records

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

    def write_import_artifacts(
        self,
        import_payload: dict[str, Any],
        *,
        output_format: str = "parquet",
        train_val_ratio: float = 0.0,
        shard_size_docs: int = 10000,
        license_note: str = "",
        manifest_path: str = "",
    ) -> dict[str, Any]:
        records = import_payload.get("records", [])
        if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
            raise ValueError("Import artifacts require collected object records.")
        normalized_format = (output_format or "parquet").strip().lower()
        if normalized_format not in {"parquet", "jsonl"}:
            raise ValueError("output_format must be parquet or jsonl.")
        records_by_split = split_records_for_import(records, train_val_ratio=train_val_ratio)
        shards = shard_import_records(
            records_by_split,
            dataset_id=str(import_payload.get("dataset_id", "hf_import")),
            source_split=str(import_payload.get("split", "train")),
            shard_size_docs=shard_size_docs,
            output_format=normalized_format,
        )
        written_shards = []
        for shard in shards:
            path = shard["path"]
            if normalized_format == "jsonl":
                write_result = self.write_file(path, records_to_jsonl(shard["records"]))
            else:
                write_result = self.write_parquet_file(path, shard["records"], mode="overwrite")
            resolved = self._resolve_path(path)
            digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
            written_shards.append(
                {
                    "split": shard["split"],
                    "path": write_result["path"],
                    "row_count": shard["row_count"],
                    "character_count": shard["character_count"],
                    "size": write_result["size"],
                    "sha256": digest,
                }
            )
        default_manifest_name = f"{slugify_import_name(str(import_payload.get('dataset_id', 'hf_import')))}_import_manifest.json"
        manifest_relative_path = manifest_path.strip() or default_manifest_name
        manifest = build_import_manifest(
            import_payload,
            written_shards,
            train_val_ratio=train_val_ratio,
            shard_size_docs=shard_size_docs,
            output_format=normalized_format,
            license_note=license_note,
        )
        manifest_result = self.write_file(manifest_relative_path, json.dumps(manifest, ensure_ascii=True, indent=2) + "\n")
        manifest_hash = hashlib.sha256(self._resolve_path(manifest_result["path"]).read_bytes()).hexdigest()
        return {
            "format": normalized_format,
            "manifest_path": manifest_result["path"],
            "manifest_sha256": manifest_hash,
            "manifest": manifest,
            "shards": written_shards,
            "shard_paths": [shard["path"] for shard in written_shards],
            "row_count": manifest["actual_documents"],
            "character_count": manifest["actual_characters"],
            "train_count": sum(shard["row_count"] for shard in written_shards if shard["split"] == "train"),
            "val_count": sum(shard["row_count"] for shard in written_shards if shard["split"] == "val"),
        }

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
