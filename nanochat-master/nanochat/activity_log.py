"""
Append-only local activity log for builder state, jobs, sandbox edits, and chat.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class ActivityLogManager:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_event: dict[str, Any] | None = None

    def log_event(self, kind: str, message: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        event = {
            "ts": time.time(),
            "kind": kind,
            "message": message,
            "payload": payload or {},
        }
        line = json.dumps(event, ensure_ascii=True)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            self._last_event = event
        return event

    def recent_events(self, limit: int = 80) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self._lock:
            with open(self.path, "r", encoding="utf-8", errors="replace") as handle:
                lines = handle.readlines()
        events = []
        for raw in lines[-limit:]:
            raw = raw.strip()
            if not raw:
                continue
            try:
                events.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return events

    def render_recent(self, limit: int = 40, max_chars: int = 6000) -> str:
        events = self.recent_events(limit=limit)
        lines: list[str] = []
        for event in events:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event["ts"]))
            line = f"{timestamp} | {event['kind']} | {event['message']}"
            payload = event.get("payload") or {}
            if payload:
                compact = json.dumps(payload, ensure_ascii=True, sort_keys=True)
                line = f"{line} | {compact}"
            lines.append(line)
        rendered = "\n".join(lines)
        if len(rendered) > max_chars:
            rendered = rendered[-max_chars:]
        return rendered

    def snapshot(self, limit: int = 80) -> dict[str, Any]:
        events = self.recent_events(limit=limit)
        return {
            "path": str(self.path),
            "event_count": len(events),
            "events": events,
        }

    def latest_event(self) -> dict[str, Any] | None:
        with self._lock:
            if self._last_event is not None:
                return dict(self._last_event)
        events = self.recent_events(limit=1)
        latest = events[-1] if events else None
        if latest is not None:
            with self._lock:
                self._last_event = latest
            return dict(latest)
        return None
