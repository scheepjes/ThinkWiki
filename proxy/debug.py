"""In-memory, thread-safe ring buffer of captured request/response exchanges.

Used by the proxy's debug mode so operators can inspect every query sent to the
upstream (including the augmented prompt) and the response that came back.
"""

from __future__ import annotations

import itertools
import threading
import time
from typing import Any


class DebugStore:
    def __init__(self, max_entries: int = 200) -> None:
        self._max = max(1, int(max_entries))
        self._items: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._counter = itertools.count(1)

    def record(self, entry: dict[str, Any]) -> dict[str, Any]:
        entry = dict(entry)
        entry.setdefault("id", next(self._counter))
        entry.setdefault("timestamp", time.time())
        with self._lock:
            self._items.append(entry)
            if len(self._items) > self._max:
                self._items = self._items[-self._max :]
        return entry

    def entries(self) -> list[dict[str, Any]]:
        """All captured exchanges, most recent first."""
        with self._lock:
            return list(reversed(self._items))

    def get(self, entry_id: int) -> dict[str, Any] | None:
        with self._lock:
            for item in self._items:
                if item.get("id") == entry_id:
                    return item
        return None

    def clear(self) -> int:
        with self._lock:
            n = len(self._items)
            self._items.clear()
            return n

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
