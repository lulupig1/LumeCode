from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# 写入回调：ConversationManager 每次追加一条消息时同步调用。
# 参数为 (message, kind)，kind ∈ {"user", "assistant", "tool_results", "system_reminder"}。
# 用于让会话持久化与内存变更处在同一时刻，避免"回合结束后批量补抄"的时间窗。
AppendSink = Callable[["Message", str], None]
ReplaceSink = Callable[["HistoryReplacement"], None]


@dataclass
class HistoryReplacement:
    """描述一次历史替换的语义，供持久化层记录结构性事件。

    kind="rollback"：截断到 message_index（/rewind）。
    kind="compact"：新历史 = 摘要 + 保留尾部，keep 原样写入 boundary。
    kind="other"：无法用结构化事件表达，持久化层应退化为"截断到当前长度"。
    """

    kind: str
    message_index: int = 0
    summary: str = ""
    keep: list[Message] = field(default_factory=list)
    # compact 时重建历史的第一条消息（摘要 user 消息，可能携带恢复附件）。
    # 持久化层把它写进 boundary 的内联尾部，保证恢复出的历史与压缩后的内存历史一致。
    summary_message: Message | None = None


@dataclass
class ToolUseBlock:
    tool_use_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: bool = False


@dataclass
class ThinkingBlock:
    thinking: str
    signature: str


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str
    tool_uses: list[ToolUseBlock] = field(default_factory=list)
    tool_results: list[ToolResultBlock] = field(default_factory=list)
    thinking_blocks: list[ThinkingBlock] = field(default_factory=list)


# 估算最后一次 API 用量锚点之后追加的消息 token 开销时使用的字符/token 比率。
# 与 context.manager 中的恢复状态启发值保持一致，全代码库统一使用同一比率。
_CHARS_PER_TOKEN = 3.5


def _message_chars(m: Message) -> int:
    n = len(m.content)
    for tb in m.thinking_blocks:
        n += len(tb.thinking)
    for tu in m.tool_uses:
        n += len(tu.tool_name) + len(json.dumps(tu.arguments, ensure_ascii=False))
    for tr in m.tool_results:
        n += len(tr.content)
    return n


def estimate_tokens(messages: list[Message]) -> int:
    """基于字符数对一组消息做 token 估算。

    刻意做得粗略——它只覆盖那些尚未锚定到真实 API 用量数值的消息，这部分的
    精确度本就无关紧要。统计内容包括消息正文、thinking、工具调用参数以及
    工具结果内容。
    """
    total = sum(_message_chars(m) for m in messages)
    return int(total / _CHARS_PER_TOKEN)


@dataclass
class ConversationManager:
    history: list[Message] = field(default_factory=list)
    env_injected: bool = field(default=False, init=False)
    ltm_injected: bool = field(default=False, init=False)
    # API 报告的每轮真实 prompt 大小，保留用于向后兼容。
    # 现在与 baseline_tokens 一致（input + cache_read + cache_creation + output）。
    last_input_tokens: int = field(default=0, init=False)
    # 真实用量锚点。baseline_tokens 是上一轮 API 计费的完整 prompt+output 大小；
    # anchor_count 是记录该数值时的消息数量。两者配合让 current_tokens() 在
    # anchor_count 以内信任 API 数据，只对之后追加的消息做字符估算。
    # baseline_tokens == 0 表示"尚无锚点"（冷启动），此时退化为纯字符估算。
    baseline_tokens: int = field(default=0, init=False)
    anchor_count: int = field(default=0, init=False)
    # 写穿钩子。add_* 在更新内存的同一时刻调用 append_sink，replace_history
    # 调用 replace_sink。两者都为空时退化为纯内存对话（测试、临时摘要对话）。
    # inject_environment / inject_long_term_memory 刻意不通知：它们是每次
    # 启动/压缩后重新注入的派生上下文，落盘会导致恢复时重复注入。
    _append_sink: AppendSink | None = field(default=None, init=False, repr=False)
    _replace_sink: ReplaceSink | None = field(default=None, init=False, repr=False)
    # 当前轮次标识，由 Agent 在每轮开始时设置；写盘时随记录一起持久化，
    # 使恢复后能按轮次定位（rewind、审计）。
    turn_id: str = field(default="", init=False)

    def attach_sink(
        self,
        append: AppendSink | None,
        replace: ReplaceSink | None = None,
    ) -> None:
        """安装写穿钩子。已安装时重复调用会替换旧钩子。"""
        self._append_sink = append
        self._replace_sink = replace

    @property
    def has_sink(self) -> bool:
        return self._append_sink is not None or self._replace_sink is not None

    def _notify_append(self, message: Message, kind: str) -> None:
        sink = self._append_sink
        if sink is None:
            return
        try:
            sink(message, kind)
        except Exception:
            # 写盘失败时保留内存中的这一轮（像 Gemini CLI 处理 ENOSPC 那样），
            # 但必须留下明显日志：静默丢记录正是这套设计要消灭的问题。
            log.exception("conversation append sink failed (kind=%s); message kept in memory", kind)

    def record_usage_anchor(
        self,
        input_tokens: int,
        output_tokens: int = 0,
        cache_read: int = 0,
        cache_creation: int = 0,
    ) -> None:
        """根据一次 API 响应钉下一个真实用量锚点。

        baseline = input + cache_read + cache_creation + output。各家服务商
        返回的 input_tokens 已经排除了命中缓存的 token，所以这三个 input 分量
        是相加关系，合起来才是真正的 prompt 大小；之所以再加上 output，是因为
        assistant 的回复此刻已成为历史的一部分。anchor_count 对齐到当前的消息
        数量，这样后续新追加的消息就成了唯一需要估算的部分。
        """
        self.baseline_tokens = input_tokens + cache_read + cache_creation + output_tokens
        self.anchor_count = len(self.history)
        # 保持旧字段同步，兼容仍在使用它的读取方。
        self.last_input_tokens = self.baseline_tokens

    def current_tokens(self) -> int:
        """对当前对话中的 token 数量做出最佳估算。

        有锚点时：baseline（真实用量）+ 仅对锚点之后追加的那些消息做字符估算。
        没有锚点时（冷启动，或刚经历一次压缩重置）：对整个历史做字符估算，
        这样在第一次 API 响应到来之前阈值检查依然能正常工作。
        """
        if self.baseline_tokens <= 0:
            return estimate_tokens(self.history)
        tail = self.history[self.anchor_count :]
        return self.baseline_tokens + estimate_tokens(tail)

    def add_user_message(self, content: str) -> None:
        message = Message(role="user", content=content)
        self.history.append(message)
        self._notify_append(message, "user")

    def add_assistant_message(
        self,
        content: str,
        tool_uses: list[ToolUseBlock] | None = None,
        thinking_blocks: list[ThinkingBlock] | None = None,
    ) -> None:
        message = Message(
            role="assistant",
            content=content,
            tool_uses=tool_uses or [],
            thinking_blocks=thinking_blocks or [],
        )
        self.history.append(message)
        self._notify_append(message, "assistant")

    def add_system_reminder(self, content: str) -> None:
        message = Message(
            role="user",
            content=f"<system-reminder>\n{content}\n</system-reminder>",
        )
        self.history.append(message)
        self._notify_append(message, "system_reminder")

    def add_tool_results_message(self, tool_results: list[ToolResultBlock]) -> None:
        message = Message(role="user", content="", tool_results=tool_results)
        self.history.append(message)
        self._notify_append(message, "tool_results")

    def inject_environment(self, context: str) -> None:
        if not self.env_injected:
            self.history.insert(0, Message(role="user", content=context))
            self.env_injected = True

    def inject_long_term_memory(self, instructions: str, memories: str) -> None:
        if self.ltm_injected:
            return
        sections: list[str] = []
        if instructions:
            sections.append(
                "# lumecodeMd\n"
                "Codebase and user instructions are shown below. "
                "Be sure to adhere to these instructions. "
                "IMPORTANT: These instructions OVERRIDE any default behavior "
                "and you MUST follow them exactly as written.\n\n" + instructions
            )
        if memories:
            sections.append("# autoMemory\n" + memories)
        if not sections:
            return
        from datetime import date

        sections.append(f"# currentDate\nToday's date is {date.today().isoformat()}.")
        body = "\n\n".join(sections)
        wrapped = (
            "<system-reminder>\n"
            "As you answer the user's questions, you can use the following context:\n"
            + body
            + "\n\n      IMPORTANT: this context may or may not be relevant to your tasks."
            " You should not respond to this context unless it is highly relevant to your task.\n"
            "</system-reminder>"
        )
        pos = 1 if self.env_injected else 0
        self.history.insert(pos, Message(role="user", content=wrapped))
        self.ltm_injected = True

    def replace_history(
        self,
        new_messages: list[Message],
        replacement: HistoryReplacement | None = None,
    ) -> None:
        self.history = new_messages
        self.env_injected = False
        self.ltm_injected = False
        # 旧的用量锚点描述的是压缩前的对话记录，这里清除它，
        # 使 current_tokens() 退化为字符估算，直到下次 API 响应
        # 基于摘要后的历史重新建立锚点。
        self.baseline_tokens = 0
        self.anchor_count = 0
        self.last_input_tokens = 0
        if self._replace_sink is not None:
            info = replacement or HistoryReplacement(kind="other", message_index=len(new_messages))
            try:
                self._replace_sink(info)
            except Exception:
                log.exception("conversation replace sink failed; history replaced in memory only")

    def get_messages(self) -> list[Message]:
        return list(self.history)
