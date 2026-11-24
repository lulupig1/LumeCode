from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from lumecode.tools.base import Tool, ToolResult
from lumecode.tools.file_io import BinaryFileError, read_text_file

if TYPE_CHECKING:
    from lumecode.cache import FileCache
    from lumecode.tools.file_state_cache import FileStateCache


class Params(BaseModel):
    file_path: str = Field(description="Absolute or relative path to the file to read")
    offset: int = Field(default=0, description="Line offset to start reading from (0-based)")
    limit: int = Field(default=2000, description="Maximum number of lines to read")


class ReadFile(Tool):
    name = "ReadFile"
    description = "Read a file and return its contents with line numbers."
    params_model = Params
    category = "read"
    is_concurrency_safe = True

    def __init__(
        self, file_cache: FileCache | None = None, file_state_cache: FileStateCache | None = None
    ) -> None:
        self._cache = file_cache
        self._state_cache = file_state_cache

    async def execute(self, params: Params) -> ToolResult:
        path = Path(params.file_path)
        if not path.exists():
            return ToolResult(output=f"Error: file not found: {params.file_path}", is_error=True)
        if not path.is_file():
            return ToolResult(output=f"Error: not a file: {params.file_path}", is_error=True)

        resolved = str(path.resolve())

        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError as e:
            return ToolResult(output=f"Error reading file: {e}", is_error=True)

        # 缓存按 mtime 校验：文件被外部改动后必须重新读盘，否则会把陈旧内容
        # 交给模型，同时让 read-before-edit 门槛记录到错配的 (内容, mtime)。
        text = self._cache.get_fresh(resolved, mtime_ns) if self._cache else None
        newline = "\n"
        truncated = False
        if text is None:
            try:
                text, newline, truncated = read_text_file(path)
            except BinaryFileError as e:
                return ToolResult(output=f"Error: {e}", is_error=True)
            except UnicodeDecodeError:
                return ToolResult(
                    output=(
                        f"Error: {params.file_path} is not valid UTF-8 text. "
                        "Use Bash (e.g. `file`, `xxd`, `iconv`) to inspect it."
                    ),
                    is_error=True,
                )
            except OSError as e:
                return ToolResult(output=f"Error reading file: {e}", is_error=True)
            if self._cache:
                self._cache.put(resolved, text, mtime_ns)

        if self._state_cache:
            self._state_cache.record(resolved, text, mtime_ns, newline)

        lines = text.splitlines()
        selected = lines[params.offset : params.offset + params.limit]
        numbered = [f"{i + params.offset + 1}\t{line}" for i, line in enumerate(selected)]
        output = "\n".join(numbered)
        if truncated:
            output += "\n… (file truncated: exceeds the read size limit)"
        return ToolResult(output=output)
