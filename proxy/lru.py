"""A tiny thread-safe LRU cache (no external dependencies)."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Hashable
from typing import Any


class LRUCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self._data: OrderedDict[Hashable, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> Any | None:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key: Hashable, value: Any) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            while len(self._data) > self.capacity:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
