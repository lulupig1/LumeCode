from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from lumecode.commands.registry import Command, CommandContext, CommandType
from lumecode.telemetry import (
    TelemetryRecorder,
    TraceStore,
    format_cost_report,
    format_session_list,
    format_trace_tree,
)

if TYPE_CHECKING:
    from lumecode.agents.trace import TraceManager


def _format_elapsed(start: float, end: float | None) -> str:
    elapsed = (end or time.monotonic()) - start
    if elapsed >= 60:
        return f"{elapsed / 60:.1f}m"
    return f"{elapsed:.0f}s"


def _status_icon(status: str) -> str:
    return {"running": "⏳", "completed": "✓", "failed": "✗"}.get(status, "?")


def _render_legacy_trace(trace_manager: TraceManager, lead_agent_id: str = "") -> str:
    """没有本地 trace 时，退回旧的实时 Agent 树视图。"""
    nodes = list(trace_manager._nodes.values())
    if not nodes:
        return "没有 Agent 追踪记录"

    parent_map: dict[str | None, list] = {}
    for n in nodes:
        parent_map.setdefault(n.parent_id, []).append(n)

    lines = ["Agent 追踪树（实时）:"]

    def _render(parent_id: str | None, indent: int) -> None:
        children = parent_map.get(parent_id, [])
        for n in children:
            icon = _status_icon(n.status)
            elapsed = _format_elapsed(n.start_time, n.end_time)
            tokens = (
                f"↑{n.input_tokens} ↓{n.output_tokens}" if n.input_tokens or n.output_tokens else ""
            )
            prefix = "  " * indent
            lines.append(
                f"{prefix}{icon} [{n.agent_id[:8]}] {n.agent_type} — {n.status} ({elapsed}) {tokens}"
            )
            _render(n.agent_id, indent + 1)

    roots = [n for n in nodes if n.parent_id is None or n.parent_id not in trace_manager._nodes]
    if not roots:
        roots = nodes[:1]

    if lead_agent_id:
        lines.append(f"  Lead: {lead_agent_id[:8]}")

    for root in roots:
        icon = _status_icon(root.status)
        elapsed = _format_elapsed(root.start_time, root.end_time)
        tokens = (
            f"↑{root.input_tokens} ↓{root.output_tokens}"
            if root.input_tokens or root.output_tokens
            else ""
        )
        lines.append(
            f"  {icon} [{root.agent_id[:8]}] {root.agent_type} — {root.status} ({elapsed}) {tokens}"
        )
        _render(root.agent_id, 2)

    total_in = sum(n.input_tokens for n in nodes)
    total_out = sum(n.output_tokens for n in nodes)
    lines.append(f"\n  合计: {len(nodes)} agents, ↑{total_in} ↓{total_out}")
    return "\n".join(lines)


def create_trace_command(
    trace_manager: TraceManager,
    lead_agent_id: str = "",
    telemetry_getter: Callable[[], TelemetryRecorder | None] | None = None,
) -> Command:
    """创建 /trace 命令。

    用法：
      /trace                 当前会话的完整调用树
      /trace cost            当前会话的成本报告
      /trace list            最近的 trace 会话
      /trace <session_id>    指定会话的调用树
      /trace cleanup         清理超过保留期的 trace 文件
    """

    async def handler(ctx: CommandContext) -> None:
        recorder = telemetry_getter() if telemetry_getter is not None else None
        store = recorder.store if recorder is not None else TraceStore()
        current_session = (
            recorder.session_id
            if recorder is not None
            else getattr(ctx.agent, "session_id", "") or ""
        )

        parts = ctx.args.strip().split()
        if parts and parts[0] in {"list", "ls"}:
            ctx.ui.add_system_message(format_session_list(store.list_sessions()))
            return

        if parts and parts[0] == "cleanup":
            removed = store.cleanup()
            ctx.ui.add_system_message(f"已清理 {removed} 个过期 trace 文件")
            return

        wants_cost = bool(parts) and parts[0] == "cost"
        if wants_cost:
            session_id = parts[1] if len(parts) > 1 else current_session
        else:
            session_id = parts[0] if parts else current_session
            wants_cost = len(parts) > 1 and parts[1] == "cost"

        if not session_id:
            ctx.ui.add_system_message(_render_legacy_trace(trace_manager, lead_agent_id))
            return

        # 当前会话可能还有 span 在后台队列里，先 flush 再读，避免“刚跑完看不到”。
        if recorder is not None and session_id == recorder.session_id:
            recorder.flush()

        records = store.read(session_id)
        if not records:
            if session_id == current_session:
                ctx.ui.add_system_message(_render_legacy_trace(trace_manager, lead_agent_id))
            else:
                ctx.ui.add_system_message(f"没有找到会话 {session_id} 的 trace 记录")
            return

        if wants_cost:
            ctx.ui.add_system_message(format_cost_report(records))
        else:
            ctx.ui.add_system_message(format_trace_tree(records))

    return Command(
        name="trace",
        description="查看本地 trace 树/成本（/trace [cost|list|cleanup|<session_id>]）",
        type=CommandType.LOCAL,
        handler=handler,
        aliases=["tree"],
        usage="/trace [cost|list|cleanup|<session_id>]",
    )
