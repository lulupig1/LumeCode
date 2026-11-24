"""路径级沙箱（沙箱策略轴）。

与 permissions/checker.py 的**审批策略轴**正交：

* 本模块（沙箱策略）回答"技术上允许触碰哪些路径"——可写根 + 只读例外。
* PermissionChecker 回答"边界之内的动作要不要问用户"。

把这两轴分开的原因（对齐 Codex 的 SandboxPolicy vs approval policy）：
越界是事实层面的边界，越界的处理方式（询问 / 拒绝）属于审批策略的选择，
不应该被硬编码在路径检查里。当前审批侧对越界仍走"询问"，见 checker.check。
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class SandboxPolicy:
    """沙箱策略：定义"技术上允许触碰哪些路径"。

    * ``writable_roots``：允许写入的根目录。
    * ``readonly_paths``：位于可写根**之内**也被强制只读的例外，
      优先级高于 ``writable_roots``（如项目里的 .git / .lumecode）。
    """

    writable_roots: tuple[Path, ...] = ()
    readonly_paths: tuple[Path, ...] = ()

    def decide(self, real_path: Path) -> tuple[bool, str]:
        """对**已解析的真实路径**给出是否放行及原因。"""
        # 只读例外优先于可写根：先判例外，再判根。
        for ro in self.readonly_paths:
            if real_path == ro:
                return False, f"路径 {real_path} 在只读列表中"
            try:
                real_path.relative_to(ro)
                return False, f"路径 {real_path} 在只读列表中"
            except ValueError:
                continue

        for root in self.writable_roots:
            try:
                real_path.relative_to(root)
                return True, ""
            except ValueError:
                continue

        return False, f"路径 {real_path} 超出沙箱范围"


class PathSandbox:
    # 默认为只读的路径（相对项目根解析）：即便身处可写根内也禁止写入。
    # .git 保护仓库完整性（否则 Agent 能改写历史/引用）；.lumecode 保护
    # Agent 自身的配置、技能、会话与计划元数据。计划文件（.lumecode/plans/）
    # 是这条规则之外的例外——Plan 模式在审批侧（checker 的 Layer 0）先行放行。
    _DEFAULT_READONLY: ClassVar[list[str]] = [
        ".git",
        ".lumecode",
    ]

    def __init__(
        self,
        project_root: str,
        extra_allowed: list[str] | None = None,
        deny_write: list[str] | None = None,
    ) -> None:
        root = Path(project_root).resolve()
        writable_roots: list[Path] = [root, Path(tempfile.gettempdir()).resolve()]
        if extra_allowed:
            writable_roots.extend(Path(p).resolve() for p in extra_allowed)

        # 只读路径列表：相对路径基于 project_root 解析。
        readonly_paths: list[Path] = []
        for dp in deny_write or self._DEFAULT_READONLY:
            dp_path = Path(dp)
            if not dp_path.is_absolute():
                dp_path = root / dp_path
            readonly_paths.append(dp_path.resolve())

        self.policy = SandboxPolicy(
            writable_roots=tuple(writable_roots),
            readonly_paths=tuple(readonly_paths),
        )

    @property
    def project_root(self) -> Path:
        return self.policy.writable_roots[0]

    def check(self, path: str) -> tuple[bool, str]:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self.project_root / p
        abs_path = p.absolute()

        try:
            real_path = abs_path.resolve(strict=True)
        except OSError:
            # 目标可能尚不存在（如新建文件）：解析出存在的最近祖先，
            # 再拼回剩余部分，避免对不存在路径的 resolve 直接失败。
            ancestor = abs_path
            while not ancestor.exists():
                parent = ancestor.parent
                if parent == ancestor:
                    return False, f"无法解析路径: {path}"
                ancestor = parent
            try:
                resolved_ancestor = ancestor.resolve(strict=True)
            except OSError:
                return False, f"无法解析路径: {path}"
            real_path = resolved_ancestor / abs_path.relative_to(ancestor)

        return self.policy.decide(real_path)
