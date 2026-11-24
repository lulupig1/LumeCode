from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from lumecode.permissions.dangerous import DangerousCommandDetector, is_safe_command
from lumecode.permissions.modes import DecisionEffect, PermissionMode, mode_decide
from lumecode.permissions.rules import RuleEngine, extract_content
from lumecode.permissions.sandbox import PathSandbox
from lumecode.tools.base import Tool

log = logging.getLogger(__name__)

_PLAN_MODE_ALLOWED_TOOLS = frozenset({"Agent", "ToolSearch", "AskUserQuestion", "ExitPlanMode"})


@dataclass
class Decision:
    effect: DecisionEffect
    reason: str


class PermissionChecker:
    """权限判定，分两条正交的轴（对齐 Codex）：

    * **沙箱策略**（``self.sandbox`` / SandboxPolicy）：技术上允许触碰哪些路径。
    * **审批策略**（规则引擎 + 权限模式 + 会话级放行）：边界之内的动作是否询问。

    ``check`` 自上而下依次应用这两轴，越界与否在 Layer 2 由沙箱策略决定。
    """

    def __init__(
        self,
        detector: DangerousCommandDetector,
        sandbox: PathSandbox,
        rule_engine: RuleEngine,
        mode: PermissionMode = PermissionMode.DEFAULT,
        sandbox_enabled: bool = False,
    ) -> None:
        self.detector = detector
        self.sandbox = sandbox
        self.rule_engine = rule_engine
        self.mode = mode
        self.plan_file_path: str = ""
        # OS 级沙箱是否启用（开启后命令类工具可自动放行，因为内核会兜底）
        self.sandbox_enabled = sandbox_enabled
        # Layer 4b: 会话级 allow-always 集合（内存中，不持久化）
        # 存放格式为 "ToolName:pattern"，用户选择 "don't ask again" 时记录
        self._session_allowed: set[str] = set()

    def add_session_allow(self, tool_name: str, content: str) -> None:
        """将工具+内容模式加入会话级放行集合（Layer 4b）。

        比持久化规则引擎优先级更高，但不写入磁盘——会话结束即消失。
        """
        key = f"{tool_name}:{content}"
        self._session_allowed.add(key)

    def _check_session_allowed(self, tool_name: str, content: str) -> bool:
        """检查是否匹配会话级放行记录。"""
        if not self._session_allowed:
            return False
        key = f"{tool_name}:{content}"
        if key in self._session_allowed:
            return True
        # 前缀匹配：已记录的 pattern 可能带通配尾缀
        for allowed in self._session_allowed:
            if allowed.endswith("*") and key.startswith(allowed[:-1]):
                return True
        return False

    @staticmethod
    def describe_tool_action(tool_name: str, arguments: dict[str, Any]) -> str:
        """为 HITL 确认生成人类可读的操作描述（对齐 Go 版 ExtractContent + formatToolArgs）。"""
        content = extract_content(tool_name, arguments)
        if content:
            return content
        # 无法从标准字段提取时，拼接参数摘要
        parts = []
        for k, v in arguments.items():
            sv = str(v)
            if len(sv) > 80:
                sv = sv[:77] + "..."
            parts.append(f"{k}={sv}")
        return ", ".join(parts) if parts else tool_name

    def check(self, tool: Tool, arguments: dict[str, Any]) -> Decision:
        content = extract_content(tool.name, arguments)

        # Layer 0: Plan 模式例外放行
        if self.mode == PermissionMode.PLAN:
            if tool.name in _PLAN_MODE_ALLOWED_TOOLS:
                return Decision(effect="allow", reason="Plan mode: allowed tool")
            if tool.name in ("WriteFile", "EditFile") and content and self._is_plan_file(content):
                return Decision(effect="allow", reason="Plan mode: plan file write")

        # Layer 1: 安全的只读命令（自动放行）
        if tool.category == "command" and is_safe_command(content or ""):
            return Decision(effect="allow", reason="Safe read-only command")

        # Layer 1b: 危险命令黑名单（仅 Bash）
        if tool.category == "command":
            hit, reason = self.detector.detect(content)
            if hit:
                return Decision(effect="deny", reason=f"危险命令拦截: {reason}")

        # Layer 1c: OS 沙箱自动放行
        # 沙箱开启时，命令类工具通过了危险命令检查后直接放行——
        # 内核级隔离会阻止越权写入，无需再弹确认。
        # 拆分复合命令逐条检查，防止通过命令拼接绕过权限检查，
        # deny 规则和 ask 规则不受沙箱影响。
        if self.sandbox_enabled and tool.category == "command":
            import re

            subcommands = [
                s.strip() for s in re.split(r"\s*(?:&&|\|\||[;|])\s*", content) if s.strip()
            ]
            if not subcommands:
                subcommands = [content]
            has_ask = False
            for sub in subcommands:
                rule_result = self.rule_engine.evaluate(tool.name, sub)
                if rule_result == "deny":
                    return Decision(effect="deny", reason="权限规则拒绝")
                if rule_result == "ask":
                    has_ask = True
            if has_ask:
                return Decision(effect="ask", reason="权限规则要求确认")
            return Decision(effect="allow", reason="OS 沙箱自动放行")

        # Layer 2: 沙箱策略轴（仅文件类工具）。
        # 这一层问的是"该路径在技术上是否允许"——属于**沙箱策略**，
        # 与"边界之内的动作要不要问用户"（审批策略）正交。当前策略下越界仍
        # 交给审批轴处理（effect="ask"，由用户定夺）；BYPASS 模式整体关闭沙箱。
        sandbox_decision = self._check_sandbox(tool, content)
        if sandbox_decision is not None:
            return sandbox_decision

        # Layer 3: 规则引擎匹配
        rule_result = self.rule_engine.evaluate(tool.name, content)
        if rule_result == "allow":
            return Decision(effect="allow", reason="权限规则放行")
        if rule_result == "deny":
            return Decision(effect="deny", reason="权限规则拒绝")

        # Layer 4b: 会话级放行（内存中，优先于模式兜底）
        if self._check_session_allowed(tool.name, content or ""):
            return Decision(effect="allow", reason="会话级放行（session allow-always）")

        # Layer 4: 权限模式兜底判定
        effect = mode_decide(self.mode, tool.category)
        if effect == "allow":
            return Decision(effect="allow", reason=f"权限模式 {self.mode.value} 放行")
        if effect == "deny":
            return Decision(effect="deny", reason=f"权限模式 {self.mode.value} 拒绝")

        # Layer 5: 触发人工确认（HITL）
        return Decision(effect="ask", reason="需要用户确认")

    def _check_sandbox(self, tool: Tool, content: str) -> Decision | None:
        """沙箱策略轴判定。

        返回 ``Decision`` 表示越界（当前策略下交审批轴，即 effect="ask"）；
        返回 ``None`` 表示路径在沙箱允许范围内、或本工具/模式不走路径检查。
        越界的处理方式（询问 vs 拒绝）属于审批策略，故集中在此以便将来调整。
        """
        if self.mode == PermissionMode.BYPASS:
            return None
        if tool.category not in ("read", "write") or not content:
            return None
        ok, reason = self.sandbox.check(content)
        if ok:
            return None
        return Decision(effect="ask", reason=f"路径沙箱拦截: {reason}")

    def _is_plan_file(self, target_path: str) -> bool:
        if not self.plan_file_path or not target_path:
            return ".lumecode/plans/" in target_path
        try:
            abs_target = os.path.abspath(target_path)
            abs_plan = os.path.abspath(self.plan_file_path)
            if abs_target == abs_plan:
                return True
        except Exception as e:
            log.debug("plan path containment check failed: %s", e)
        if os.path.basename(target_path) == os.path.basename(self.plan_file_path):
            return True
        return ".lumecode/plans/" in target_path
