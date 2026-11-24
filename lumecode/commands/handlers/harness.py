"""``/harness``、``/distill``、``/evolve`` 三个命令。

设计原则：

- ``/harness`` 只读查看，外加人工改写基础分节（``/harness edit``）。
- ``/distill`` 只产提议，绝不写 harness 文件。
- ``/evolve`` 负责评审、写入、回滚——写入前一定先快照。
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from lumecode.commands.registry import Command, CommandContext, CommandHandler, CommandType
from lumecode.harness.attribution import FailureCase, group_failures, load_failures
from lumecode.harness.distill import run_distill
from lumecode.harness.metrics import collect_metrics, compute_experience_usage
from lumecode.harness.models import Proposal
from lumecode.harness.proposals import AcceptError, accept_by_id, reject_by_id
from lumecode.harness.store import HarnessStore, HarnessWriteError


def _store(ctx: CommandContext) -> HarnessStore:
    harness = getattr(ctx.agent, "harness", None)
    if isinstance(harness, HarnessStore):
        return harness
    work_dir = getattr(ctx.agent, "work_dir", None) or os.getcwd()
    return HarnessStore(work_dir)


def _work_dir(ctx: CommandContext) -> str:
    return getattr(ctx.agent, "work_dir", None) or os.getcwd()


def _render_proposal(index: int, proposal: Proposal) -> str:
    kind = "经验" if proposal.kind == "experience" else "提示词补丁"
    lines = [
        f"[{index}] {proposal.id}  ({kind} → {proposal.target})",
        f"    {proposal.title}",
    ]
    if proposal.rationale:
        lines.append(f"    理由: {_one_line(proposal.rationale)}")
    if proposal.expected_effect:
        lines.append(f"    预期: {_one_line(proposal.expected_effect)}")
    if proposal.evidence:
        ev = ", ".join(
            f"{e.get('session', '?')}:{e.get('span', '?')}" for e in proposal.evidence[:3]
        )
        lines.append(f"    证据: {ev}")
    lines.append(f"    正文: {_one_line(proposal.body, 160)}")
    return "\n".join(lines)


def _one_line(text: str, limit: int = 100) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# /harness
# ---------------------------------------------------------------------------


async def handle_harness(ctx: CommandContext) -> None:
    store = _store(ctx)
    args = ctx.args.strip()
    parts = args.split(maxsplit=2)
    sub = parts[0] if parts else ""

    if sub in {"", "status"}:
        _harness_status(ctx, store)
        return

    if sub == "edit":
        if len(parts) < 3:
            keys = ", ".join(
                store.list_base_sections()
                or [
                    "identity",
                    "system",
                    "doing-tasks",
                    "executing-actions",
                    "using-tools",
                    "tone-style",
                    "text-output",
                ]
            )
            ctx.ui.add_system_message(f"用法: /harness edit <分节> <文件路径>\n可用分节: {keys}")
            return
        key, source = parts[1], parts[2]
        try:
            content = await asyncio.to_thread(Path(source).read_text, encoding="utf-8")
        except OSError as e:
            ctx.ui.add_system_message(f"读取失败: {e}")
            return
        try:
            store.write_base_section(key, content)
        except HarnessWriteError as e:
            ctx.ui.add_system_message(f"写入被拒绝: {e}")
            return
        ctx.ui.add_system_message(f"已更新基础提示词分节 prompt/{key}.md（新会话生效）")
        return

    if sub == "base":
        sections = store.list_base_sections()
        if not sections:
            ctx.ui.add_system_message(
                "尚未文件化：基础提示词仍使用内置常量。\n"
                "用 /harness edit <key> <file> 从文件导入一个分节即可开始。"
            )
            return
        ctx.ui.add_system_message(
            "已文件化分节:\n" + "\n".join(f"  - prompt/{s}.md" for s in sections)
        )
        return

    ctx.ui.add_system_message("用法: /harness [status|base|edit <key> <file>]")


def _harness_status(ctx: CommandContext, store: HarnessStore) -> None:
    stats = store.stats()
    entries = store.list_experiences()
    lines = [
        "Harness 状态",
        "─────────────",
        f"版本:      v{stats['version']}",
        f"经验:      {stats['experiences']} / {stats['max']}（retired {stats['retired']}）",
        f"注入预算:  {stats['inject_chars']} / {stats['inject_max']} 字符",
        f"待评审:    {stats['pending']} 条（/evolve review）",
        f"快照:      {stats['snapshots']} 个",
        f"目录:      {stats['root']}",
    ]
    sections = store.list_base_sections()
    lines.append(f"基础分节:  {', '.join(sections) if sections else '（内置常量）'}")
    if entries:
        lines.append("")
        lines.append("已生效经验:")
        for entry in entries:
            lines.append(
                f"  - {entry.name}"
                + (f"  [sig={entry.signature}]" if entry.signature else "")
                + f" — {_one_line(entry.description, 80)}"
            )
    ctx.ui.add_system_message("\n".join(lines))


# ---------------------------------------------------------------------------
# /distill
# ---------------------------------------------------------------------------


async def handle_distill(ctx: CommandContext) -> None:
    args = ctx.args.strip().split()
    since = "7d"
    dry_run = False
    for arg in args:
        if arg == "--dry-run":
            dry_run = True
        elif arg.startswith("--since"):
            since = arg.split("=", 1)[1] if "=" in arg else since
        elif arg.endswith(("d", "h", "m")) and arg[:-1].isdigit():
            since = arg

    work_dir = _work_dir(ctx)
    store = _store(ctx)
    cases = load_failures(work_dir, since)
    if not cases:
        ctx.ui.add_system_message(f"最近 {since} 没有失败案例，不需要复盘。")
        return

    groups = group_failures(cases)
    if dry_run:
        ctx.ui.add_system_message(
            _render_cases(cases, groups, since) + "\n\n（--dry-run：不调用模型）"
        )
        return

    client = getattr(ctx.agent, "client", None)
    if client is None:
        ctx.ui.add_system_message("当前 Agent 没有可用的模型客户端，无法生成提议。")
        return

    ctx.ui.add_system_message(f"正在复盘最近 {since} 的 {len(cases)} 例失败…")
    protocol = getattr(ctx.agent, "protocol", "anthropic")
    try:
        proposals, dropped, _ = await run_distill(client, protocol, work_dir, since, store=store)
    except Exception as e:
        ctx.ui.add_system_message(f"复盘失败: {e}")
        return

    lines = [_render_cases(cases, groups, since), ""]
    if not proposals:
        lines.append("没有生成可用的提议。")
    else:
        lines.append(f"生成 {len(proposals)} 条提议（已存入 proposals/）：")
        for index, proposal in enumerate(proposals, 1):
            lines.append("")
            lines.append(_render_proposal(index, proposal))
    if dropped:
        lines.append("")
        lines.append("丢弃：")
        lines.extend(f"  - {reason}" for reason in dropped)
    if proposals:
        lines.append("")
        lines.append(
            "用 /evolve review 查看详情，/evolve accept <id> 或 /evolve reject <id> 处理。"
        )
    ctx.ui.add_system_message("\n".join(lines))


def _render_cases(
    cases: list[FailureCase],
    groups: list[tuple[str, list[FailureCase]]],
    since: str,
) -> str:
    recovered = sum(1 for c in cases if c.recovered)
    lines = [f"最近 {since}：{len(cases)} 例失败（自愈型 {recovered} 例），分 {len(groups)} 类："]
    for signature, group in groups[:10]:
        lines.append(f"  {signature}  ×{len(group)}")
    if len(groups) > 10:
        lines.append(f"  … 另有 {len(groups) - 10} 类")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# /evolve
# ---------------------------------------------------------------------------


async def handle_evolve(ctx: CommandContext) -> None:
    store = _store(ctx)
    args = ctx.args.strip()
    parts = args.split(maxsplit=1)
    sub = parts[0] if parts else "status"
    rest = parts[1].strip() if len(parts) > 1 else ""

    if sub == "status":
        _evolve_status(ctx, store)
        return
    if sub == "log":
        _evolve_log(ctx, store)
        return
    if sub in {"review", "pending"}:
        _evolve_review(ctx, store)
        return
    if sub == "accept":
        _evolve_accept(ctx, store, rest)
        return
    if sub == "reject":
        _evolve_reject(ctx, store, rest)
        return
    if sub in {"rollback", "undo"}:
        _evolve_rollback(ctx, store)
        return
    if sub == "retire":
        _evolve_retire(ctx, store, rest)
        return

    ctx.ui.add_system_message(
        "用法:\n"
        "  /evolve status            指标与经验命中\n"
        "  /evolve log               改动历史\n"
        "  /evolve review            待评审提议\n"
        "  /evolve accept <id>       采纳\n"
        "  /evolve reject <id>       拒绝\n"
        "  /evolve retire <name>     停用一条经验\n"
        "  /evolve rollback          回滚到上一个快照"
    )


def _evolve_status(ctx: CommandContext, store: HarnessStore) -> None:
    work_dir = _work_dir(ctx)
    try:
        metrics = collect_metrics(work_dir, since="7d")
    except Exception as e:
        ctx.ui.add_system_message(f"读取指标失败: {e}")
        return

    lines = [
        f"进化状态  （harness v{store.current_version()}）",
        "─────────────",
        "近 7 天:",
        f"  任务数             {metrics.task_count}",
        f"  token / 任务       {metrics.tokens_per_task:,.0f}",
        f"  工具调用 / 任务    {metrics.tool_calls_per_task:.1f}",
        f"  首次尝试成功率     {metrics.first_try_rate:.0%}",
    ]

    # 经验命中
    try:
        usage = compute_experience_usage(work_dir, since="30d")
    except Exception:
        usage = None
    if usage is not None and usage.items:
        lines.append("")
        lines.append(f"经验（僵尸率 {usage.zombie_rate:.0%}）:")
        for item in usage.items:
            state = ""
            if item.name in usage.tainted_names:
                state = "  ⚠ 创建后仍复发"
            elif item.hits == 0:
                state = "  (未命中)"
            lines.append(
                f"  - {item.name}: 命中 {item.hits}，复发 {item.identical_mistakes}{state}"
            )
    lines.append("")
    lines.append(
        f"待评审 {len(store.list_proposals())} 条。判断是否真进化：看「创建后仍复发」是否降到 0。"
    )
    ctx.ui.add_system_message("\n".join(lines))


def _evolve_log(ctx: CommandContext, store: HarnessStore) -> None:
    versions = store.versions()
    if not versions:
        ctx.ui.add_system_message("还没有任何 harness 改动记录。")
        return
    lines = ["Harness 改动历史", "─────────────"]
    for version in reversed(versions[-20:]):
        lines.append(f"v{version.version}  {version.created_at}  {version.note}")
        for change in version.changes:
            lines.append(f"    + {change}")
        if version.proposal_ids:
            lines.append(f"    提议: {', '.join(version.proposal_ids)}")
    lines.append("")
    lines.append(f"快照 {len(store.snapshots())} 个；/evolve rollback 可退回上一版。")
    ctx.ui.add_system_message("\n".join(lines))


def _evolve_review(ctx: CommandContext, store: HarnessStore) -> None:
    pending = store.list_proposals()
    if not pending:
        ctx.ui.add_system_message("没有待评审的提议。先运行 /distill。")
        return
    blocks = [_render_proposal(index, p) for index, p in enumerate(pending, 1)]
    ctx.ui.add_system_message("待评审提议：\n\n" + "\n\n".join(blocks))


def _evolve_accept(ctx: CommandContext, store: HarnessStore, rest: str) -> None:
    if not rest:
        ctx.ui.add_system_message("用法: /evolve accept <id>  或  /evolve accept all")
        return
    if rest == "all":
        pending = store.list_proposals()
        if not pending:
            ctx.ui.add_system_message("没有待评审的提议。")
            return
        results: list[str] = []
        for proposal in pending:
            try:
                results.append(accept_by_id(proposal.id, store).summary)
            except AcceptError as e:
                results.append(f"{proposal.id}: {e}")
        ctx.ui.add_system_message("\n".join(results))
        return

    try:
        result = accept_by_id(rest, store)
    except AcceptError as e:
        ctx.ui.add_system_message(f"采纳失败: {e}")
        return
    ctx.ui.add_system_message(
        f"{result.summary}\nharness 版本 → v{store.current_version()}（新会话生效）"
    )


def _evolve_reject(ctx: CommandContext, store: HarnessStore, rest: str) -> None:
    if not rest:
        ctx.ui.add_system_message("用法: /evolve reject <id>")
        return
    try:
        reject_by_id(rest, store)
    except AcceptError as e:
        ctx.ui.add_system_message(f"拒绝失败: {e}")
        return
    ctx.ui.add_system_message(f"已拒绝 {rest}（不会删除，下次 distill 不会被重复提议）")


def _evolve_rollback(ctx: CommandContext, store: HarnessStore) -> None:
    snaps = store.snapshots()
    if not snaps:
        ctx.ui.add_system_message("没有可回滚的快照。")
        return
    target = snaps[-1]
    if not store.restore_snapshot(target):
        ctx.ui.add_system_message("回滚失败。")
        return
    ctx.ui.add_system_message(f"已回滚到 {target.name}（回滚前的状态也已快照，可再次回滚撤销）")


def _evolve_retire(ctx: CommandContext, store: HarnessStore, rest: str) -> None:
    if not rest:
        ctx.ui.add_system_message("用法: /evolve retire <name>")
        return
    if not store.retire_experience(rest):
        ctx.ui.add_system_message(f"找不到经验 {rest}")
        return
    ctx.ui.add_system_message(f"已停用经验 {rest}（保留文件，避免下次 distill 重复提议）")


# ---------------------------------------------------------------------------
# 命令注册
# ---------------------------------------------------------------------------


def _make_command(name: str, handler: CommandHandler, description: str, usage: str) -> Command:
    return Command(
        name=name,
        description=description,
        usage=usage,
        type=CommandType.LOCAL,
        handler=handler,
    )


HARNESS_COMMANDS = [
    _make_command(
        "harness",
        handle_harness,
        "查看/编辑 harness 基础提示词分节与经验库",
        "/harness [status|base|edit <key> <file>]",
    ),
    _make_command(
        "distill",
        handle_distill,
        "复盘最近的失败轨迹，生成 harness 改动提议",
        "/distill [--since 7d] [--dry-run]",
    ),
    _make_command(
        "evolve",
        handle_evolve,
        "评审/采纳/回滚 harness 改动，查看进化指标",
        "/evolve [status|log|review|accept|reject|retire|rollback]",
    ),
]
