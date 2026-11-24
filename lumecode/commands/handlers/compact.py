from __future__ import annotations

from lumecode.commands.registry import Command, CommandContext, CommandType


async def handle_compact(ctx: CommandContext) -> None:
    if ctx.agent is None:
        ctx.ui.add_system_message("Agent 未初始化")
        return

    input_tokens, _ = ctx.ui.get_token_count()
    if input_tokens < 5000:
        ctx.ui.add_system_message(f"当前 token 数 {input_tokens:,}，无需压缩")
        return

    from lumecode.agent import CompactNotification, ErrorEvent

    result = await ctx.agent.manual_compact(ctx.conversation)
    if isinstance(result, CompactNotification):
        # 持久化已由 ConversationManager 的 replace sink 在压缩发生时完成
        # （写一条 compact_boundary）。这里不再补写，避免 resume 重放出重复 boundary。
        ctx.ui.add_system_message(result.message)
    elif isinstance(result, ErrorEvent):
        ctx.ui.add_system_message(f"压缩失败: {result.message}")


COMPACT_COMMAND = Command(
    name="compact",
    aliases=["c"],
    description="压缩上下文",
    usage="/compact [保留重点]",
    type=CommandType.LOCAL,
    handler=handle_compact,
)
