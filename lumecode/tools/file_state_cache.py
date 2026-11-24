from __future__ import annotations

from pathlib import Path

from lumecode.tools.file_io import read_text_file


class FileStateCache:
    """Tracks which files have been read, enforcing read-before-edit.

    Stores { absolute_path: (content, mtime_ns, newline) } after each ReadFile call.
    EditFile and WriteFile check the cache before proceeding:
      - Gate 1: file must have been read (present in cache).
      - Gate 2: file must not have been modified since the read (mtime_ns matches).

    `newline` 记录原文件的行尾风格，写回时由 WriteFile/EditFile 还原，
    避免在 Windows 上把 LF 文件悄悄改写成 CRLF。
    """

    def __init__(self) -> None:
        self._cache: dict[str, tuple[str, int, str]] = {}

    def record(self, path: str, content: str, mtime_ns: int, newline: str = "\n") -> None:
        """Record a file's content and mtime after a successful read."""
        self._cache[path] = (content, mtime_ns, newline)

    def newline_for(self, path: str) -> str:
        """Return the newline style observed when *path* was last read."""
        entry = self._cache.get(path)
        return entry[2] if entry is not None else "\n"

    def check(self, path: str) -> tuple[bool, str]:
        """Check whether the file is safe to edit/write.

        Returns (ok, error_message). If ok is True, error_message is empty.
        """
        entry = self._cache.get(path)
        if entry is None:
            return False, "Error: file has not been read yet. Read it first before editing."

        _, cached_mtime_ns, _ = entry
        try:
            current_mtime_ns = Path(path).stat().st_mtime_ns
        except OSError:
            # File may have been deleted; allow the write to proceed
            # (WriteFile will create it, EditFile will fail on its own).
            return True, ""

        if current_mtime_ns != cached_mtime_ns:
            return (
                False,
                "Error: file has been modified since last read. Read it again before editing.",
            )

        return True, ""

    def update(self, path: str) -> None:
        """Update the cache entry after a successful edit/write."""
        try:
            p = Path(path)
            text, newline, _ = read_text_file(p, max_bytes=None)
            mtime_ns = p.stat().st_mtime_ns
            self._cache[path] = (text, mtime_ns, newline)
        except OSError:
            # If we can't read it back, just remove the stale entry.
            self._cache.pop(path, None)
        except UnicodeDecodeError:
            self._cache.pop(path, None)
