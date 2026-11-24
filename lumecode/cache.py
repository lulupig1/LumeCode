from __future__ import annotations

import threading


class FileCache:
    def __init__(self) -> None:
        # 值存为 (content, mtime_ns)。mtime 让读取方可以判断缓存是否已过期，
        # 避免文件被外部修改后仍然返回陈旧内容。
        self._store: dict[str, tuple[str, int]] = {}
        self._lock = threading.Lock()

    def get(self, path: str) -> str | None:
        with self._lock:
            entry = self._store.get(path)
        return entry[0] if entry is not None else None

    def get_fresh(self, path: str, mtime_ns: int) -> str | None:
        """仅当缓存记录的 mtime 与 *mtime_ns* 一致时才返回内容。

        文件已被外部修改（mtime 变化）时返回 None，强制调用方重新读盘——
        否则模型会拿到陈旧内容，并据此做出错误的编辑决策。
        """
        with self._lock:
            entry = self._store.get(path)
        if entry is None:
            return None
        content, cached_mtime_ns = entry
        return content if cached_mtime_ns == mtime_ns else None

    def put(self, path: str, content: str, mtime_ns: int = 0) -> None:
        with self._lock:
            self._store[path] = (content, mtime_ns)

    def invalidate(self, path: str) -> None:
        with self._lock:
            self._store.pop(path, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
