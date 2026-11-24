"""共享的文件读写助手：原子写入 + 行尾保持 + 二进制检测。

WriteFile / EditFile 原先各自调用 ``Path.write_text``，有两个问题：
  1. 截断后重写——进程在中途崩溃会留下被破坏的文件；
  2. 文本模式会做行尾转换——一个 LF 文件在 Windows 上被编辑后会被
     整体改写成 CRLF（与仓库的 ``* text=auto`` 约定相冲突）。

这里统一为：读时把行尾归一化成 ``\\n``（便于与模型给出的字符串做精确匹配），
并记住原文件的行尾风格；写时用临时文件 + ``os.replace`` 原子替换，并还原行尾。
"""

from __future__ import annotations

from pathlib import Path

from lumecode.fileutil import atomic_write_bytes

# 单次读取的字节上限：超出部分截断。防止误读巨型文件（日志、二进制
# 产物）把整个内容物化进内存。
MAX_READ_BYTES = 8 * 1024 * 1024


class BinaryFileError(Exception):
    """文件看起来是二进制（含 NUL 字节），不应按文本读取。"""


def detect_newline(raw: bytes) -> str:
    """返回 *raw* 使用的主要行尾风格（``"\\r\\n"`` 或 ``"\\n"``）。"""
    return "\r\n" if b"\r\n" in raw else "\n"


def read_text_file(path: Path, max_bytes: int | None = MAX_READ_BYTES) -> tuple[str, str, bool]:
    """按 UTF-8 读取 *path*，返回 ``(text, newline, truncated)``。

    ``text`` 的行尾已归一化为 ``"\\n"``；``newline`` 是原文件的行尾风格，
    供写回时还原；``truncated`` 表示内容因超过 *max_bytes* 被截断。

    含 NUL 字节时抛 :class:`BinaryFileError`；非 UTF-8 时抛 ``UnicodeDecodeError``。
    """
    raw = path.read_bytes()
    truncated = False
    if max_bytes is not None and len(raw) > max_bytes:
        raw = raw[:max_bytes]
        truncated = True

    if b"\x00" in raw:
        raise BinaryFileError("file appears to be binary (contains NUL bytes)")

    # 截断可能切在多字节序列中间，此时用 replace 容错；完整读取则严格解码，
    # 避免把编码问题静默成替换字符。
    text = raw.decode("utf-8", errors="replace" if truncated else "strict")
    newline = detect_newline(raw)
    if newline != "\n":
        text = text.replace("\r\n", "\n")
    return text, newline, truncated


def write_text_file(path: Path, text: str, newline: str = "\n") -> None:
    """把 *text* 原子地写入 *path*，并按 *newline* 还原行尾风格。

    原子性由 :func:`lumecode.fileutil.atomic_write_bytes` 保证（临时文件 +
    fsync + ``os.replace``）。
    """
    data = text if newline == "\n" else text.replace("\n", newline)
    atomic_write_bytes(path, data.encode("utf-8"))
