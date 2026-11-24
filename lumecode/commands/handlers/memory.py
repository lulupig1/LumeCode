from __future__ import annotations

from pathlib import Path
from typing import Any

from lumecode.commands.registry import Command, CommandContext, CommandType


async def handle_memory(ctx: CommandContext) -> None:
    mm = ctx.memory_manager
    if mm is None:
        ctx.ui.add_system_message("记忆管理器未初始化")
        return

    parts = ctx.args.split(None, 1)
    sub = parts[0] if parts else ""

    if sub == "" or sub == "list":
        display = mm.get_display_text()
        ctx.ui.add_system_message(display)

    elif sub == "clear":
        mm.clear()
        ctx.ui.add_system_message("所有自动记忆已清空。")

    elif sub == "edit":
        ctx.ui.add_system_message(
            f"编辑记忆文件：\n  用户级目录: {mm.user_mem_dir}\n  项目级目录: {mm.project_mem_dir}"
        )

    elif sub == "history":
        ctx.ui.add_system_message(_history_text(mm))

    else:
        ctx.ui.add_system_message("用法: /memory [list | clear | edit | history]")


def _history_text(mm: Any) -> str:
    """列出记忆历史快照（抽取追加、dream 整理前自动保留）。"""
    from lumecode.memory.history import HISTORY_DIR_NAME

    lines = ["记忆历史快照（只读，写入前自动保存）："]
    found = False
    for label, mem_dir in (("用户级", mm.user_mem_dir), ("项目级", mm.project_mem_dir)):
        root = Path(str(mem_dir)) / HISTORY_DIR_NAME
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            count = len(list(entry.glob("*.md"))) if entry.name != "dream" else None
            if entry.name == "dream":
                count = len(list(entry.iterdir()))
                detail = f"{count} 次整理快照"
            else:
                detail = f"{count} 份"
            lines.append(f"  [{label}] {entry.name}: {detail}  ({entry})")
            found = True
    if not found:
        return "还没有任何历史快照。"
    lines.append("")
    lines.append("恢复：直接把对应快照文件复制回记忆目录即可。")
    return "\n".join(lines)


MEMORY_COMMAND = Command(
    name="memory",
    description="记忆管理",
    usage="/memory [list | clear | edit | history]",
    type=CommandType.LOCAL,
    handler=handle_memory,
)
