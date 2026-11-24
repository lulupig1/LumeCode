from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from typing import Any

from lumecode.hooks.conditions import ConditionGroup


def _quote_for_shell(value: str) -> str:
    """把 *value* 转义成 shell 中的单个字面量参数。

    POSIX 下直接用 ``shlex.quote``，完全可靠。

    Windows 的 cmd.exe 没有可靠的通用转义机制，这里做保守处理：双引号包裹
    并转义 cmd 的元字符。这覆盖了 ``&``/``|``/``<``/``>``/``^``/``%`` 这些
    最常见的注入面，但不等价于 POSIX 的严格性——在 Windows 上编写 hook 时，
    仍应优先让被插值的值只作为普通参数出现。
    """
    if os.name != "nt":
        return shlex.quote(value)
    escaped = value.replace("^", "^^").replace("%", "%%")
    for ch in ('"', "&", "|", "<", ">", "(", ")"):
        escaped = escaped.replace(ch, "^" + ch)
    return f'"{escaped}"'


@dataclass
class Action:
    type: str
    command: str = ""
    message: str = ""
    url: str = ""
    method: str = "POST"
    body: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    prompt: str = ""
    timeout: int = 30


@dataclass
class ActionResult:
    output: str = ""
    success: bool = True


@dataclass
class Hook:
    id: str
    event: str
    action: Action
    condition: ConditionGroup | None = None
    reject: bool = False
    once: bool = False
    async_exec: bool = False
    executed: bool = False

    def should_run(self) -> bool:
        return not (self.once and self.executed)

    def mark_executed(self) -> None:
        self.executed = True


@dataclass
class HookContext:
    event_name: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    file_path: str = ""
    message: str = ""
    error: str = ""

    def get_field(self, name: str) -> str:
        if name == "tool":
            return self.tool_name
        if name == "event":
            return self.event_name
        if name.startswith("args."):
            key = name[5:]
            value = self.tool_args.get(key, "")
            return str(value) if value else ""
        return ""

    def expand(self, template: str, *, shell_quote: bool = False) -> str:
        """把 ``$EVENT`` / ``$TOOL_ARGS.x`` 等占位符替换为上下文值。

        ``shell_quote=True`` 时对**每个被替换进去的值**做 shell 转义，使其在
        结果中只作为单个字面量参数出现；模板里作者自己写的字面量（管道、
        重定向、变量引用等）不受影响。

        这是防注入的关键：工具参数由模型控制，若不转义，
        ``command: "echo $TOOL_ARGS.file_path"`` 遇到
        ``file_path = "; rm -rf ~"`` 就会把 rm 执行掉。
        """

        def sub(value: str) -> str:
            return _quote_for_shell(value) if shell_quote else value

        result = template
        result = result.replace("$EVENT", sub(self.event_name))
        result = result.replace("$TOOL_NAME", sub(self.tool_name))
        result = result.replace("$FILE_PATH", sub(self.file_path))
        result = result.replace("$MESSAGE", sub(self.message))
        result = result.replace("$ERROR", sub(self.error))
        for key, value in self.tool_args.items():
            result = result.replace(f"$TOOL_ARGS.{key}", sub(str(value)))
        return result


class ToolRejectedError(Exception):
    def __init__(self, tool: str, reason: str, hook_id: str) -> None:
        self.tool = tool
        self.reason = reason
        self.hook_id = hook_id
        super().__init__(f"Tool '{tool}' rejected by hook '{hook_id}': {reason}")
