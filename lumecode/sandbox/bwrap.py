"""Linux bubblewrap（bwrap）沙箱实现。

通过 bwrap 创建隔离的用户命名空间，将根文件系统以只读方式挂载，
仅对白名单路径开放写权限。支持网络隔离（unshare-net）。
"""

from __future__ import annotations

import shlex
import shutil
from pathlib import Path

from lumecode.sandbox import Sandbox, SandboxConfig


class BwrapSandbox(Sandbox):
    """Linux bubblewrap 沙箱。"""

    def wrap(self, command: str, config: SandboxConfig) -> str:
        """构建 bwrap 命令行。

        基本结构：
          bwrap --unshare-user --unshare-pid
                --ro-bind / /                   # 根文件系统只读
                --bind <writable> <writable>     # 可写路径（读写挂载）
                --ro-bind <protected> <protected> # 禁写路径（强制只读覆盖）
                [--unshare-net]                  # 网络隔离
                --proc /proc                     # 挂载 /proc
                --dev /dev                       # 挂载 /dev
                -- bash -c <command>
        """
        args: list[str] = [
            "bwrap",
            "--unshare-user",
            "--unshare-pid",
            # 父进程退出时一并终止沙箱进程，避免留下孤儿。
            "--die-with-parent",
            # 根目录只读挂载
            "--ro-bind",
            "/",
            "/",
        ]

        # 可写路径：用 --bind（读写）覆盖只读挂载。
        # --bind 的源路径必须已存在，否则 bwrap 会让整条命令失败；不存在的
        # 路径（还没创建、或本就无需开放）直接跳过，而不是拖垮整次调用。
        for path in config.allow_write:
            resolved = Path(path).resolve()
            if not resolved.exists():
                continue
            args.extend(["--bind", str(resolved), str(resolved)])

        # 禁写路径：用 --ro-bind 强制只读（覆盖上面的 --bind）。
        # 同理，--ro-bind 不能指向不存在的源——例如 .lumecode/config.yaml
        # 尚未创建时，一个"保护"路径不该导致命令跑不起来。
        for path in config.deny_write:
            resolved = Path(path).resolve()
            if not resolved.exists():
                continue
            args.extend(["--ro-bind", str(resolved), str(resolved)])

        # 网络隔离
        if not config.network_enabled:
            args.append("--unshare-net")

        # 挂载 /proc 和 /dev（很多工具需要）
        args.extend(["--proc", "/proc"])
        args.extend(["--dev", "/dev"])

        # 分隔符 + 实际命令
        args.append("--")
        args.extend(["bash", "-c", command])

        return " ".join(shlex.quote(a) for a in args)

    def available(self) -> bool:
        """检查 bwrap 是否在 PATH 中。"""
        return shutil.which("bwrap") is not None
