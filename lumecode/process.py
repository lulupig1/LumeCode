"""子进程管理的共享助手：独立进程组 + 整组终止。

Bash 工具和 hook 的 command 动作都需要这两件事：
  1. 让子进程脱离父进程的进程组，这样超时时可以连同它的后代一起杀掉；
  2. 超时/取消时真的把整组杀干净，而不是只杀掉最外层的 shell。

之前这两处各自实现了一遍，且都只调用了 ``proc.kill()``——那只杀掉
``sh -c`` 这一层，``sleep 1000 &`` 之类的后台子进程会变成孤儿继续存活。
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess


def spawn_kwargs() -> dict:
    """返回让子进程成为独立进程组/会话的 Popen 参数。

    配合 :func:`terminate_process_tree` 使用：只有子进程自成一个进程组，
    ``killpg`` 才能精确地只杀掉它和它的后代，不波及父进程。
    """
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


async def terminate_process_tree(proc: asyncio.subprocess.Process) -> None:
    """终止 *proc* 及其所有子进程。

    POSIX 下杀掉整个进程组；Windows 下用 ``taskkill /T`` 连带子树。
    任何"进程已退出/无权限"的情况都被吞掉——终止是尽力而为的清理动作，
    不该掩盖调用方真正的错误。
    """
    if proc.returncode is not None:
        return
    try:
        if os.name == "nt":
            killer = await asyncio.create_subprocess_shell(
                f"taskkill /F /T /PID {proc.pid}",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
        else:
            # start_new_session=True 让子进程成为新进程组的组长，
            # 因此 getpgid(pid) == pid，整组一起杀。
            # killpg/getpgid/SIGKILL 是 POSIX 专有，Windows 的 typeshed 里没有
            # 这些属性；本分支只在 POSIX 下执行，故忽略 attr-defined。
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)  # type: ignore[attr-defined]
    except (ProcessLookupError, PermissionError, OSError):
        pass

    with contextlib.suppress(ProcessLookupError, OSError):
        proc.kill()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(proc.wait(), timeout=5)
