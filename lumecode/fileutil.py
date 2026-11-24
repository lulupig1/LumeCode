"""跨进程文件工具：文件锁 + 原子写。

Mailbox、SharedTaskStore 以及文件写入工具都需要这两件事，之前各自实现了
一份（Mailbox 的锁还有个"抢锁失败后无锁继续写"的 bug）。这里统一实现。

- :func:`file_lock` —— 基于 ``O_CREAT|O_EXCL`` 的跨进程互斥锁。
- :func:`atomic_write_bytes` / :func:`atomic_write_text` —— 临时文件 +
  ``os.replace``，崩溃或并发读都不会看到半截内容。
"""

from __future__ import annotations

import contextlib
import os
import random
import time
import uuid
from collections.abc import Iterator
from pathlib import Path


@contextlib.contextmanager
def file_lock(
    lock_path: str | Path,
    *,
    attempts: int = 10,
    stale_after: float = 10.0,
) -> Iterator[None]:
    """获取 *lock_path* 表示的跨进程锁，退出 with 块时释放。

    锁文件用 ``O_CREAT|O_EXCL`` 原子创建。若文件已存在但其 mtime 超过
    *stale_after* 秒，视为持有者已崩溃，可抢占。抢不到就抛异常——
    绝不能"抢不到就无锁继续"，那会破坏互斥并在释放时误删别人的锁。
    """
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    acquired = False
    last_err: Exception | None = None
    for _ in range(attempts):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.close(fd)
            acquired = True
            break
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > stale_after:
                    lock_path.unlink(missing_ok=True)
            except OSError:
                pass
            # 抖动退避，避免多个进程同步重试。
            time.sleep((5 + random.randint(0, 95)) / 1000)
        except OSError as e:
            last_err = e
            break

    if not acquired:
        raise last_err or TimeoutError(
            f"failed to acquire lock {lock_path} after {attempts} attempts"
        )

    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """把 *data* 原子地写入 *path*。

    先写同目录下的临时文件并 fsync，再 ``os.replace`` 覆盖目标。同目录保证
    同一文件系统，因此 replace 是原子的：进程崩溃不会留下半截文件，并发读者
    也只会看到完整的旧内容或完整的新内容。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp_path, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    """:func:`atomic_write_bytes` 的文本版本。"""
    atomic_write_bytes(path, text.encode(encoding))
