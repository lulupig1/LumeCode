"""把 JSONL span 记录渲染成人类可读的树和成本报告。

这一层刻意不依赖 UI：``/trace`` 命令只负责把字符串交给 Textual，单元测试
可以直接断言输出。所有函数都容忍缺字段/坏记录，trace 是调试辅助，不能因为
一条旧记录就让命令报错。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lumecode.telemetry.store import TraceFileInfo

_KIND_ICONS = {
    "session": "◆",
    "agent": "◆",
    "generation": "✦",
    "tool": "⚙",
    "compact": "⇣",
    "memory_recall": "◈",
    "memory_extract": "✎",
    "memory_consolidation": "✎",
    "session_summary": "✎",
    "hook": "⚑",
}


@dataclass
class SpanNode:
    record: dict[str, Any]
    children: list[SpanNode] = field(default_factory=list)


def _str(record: dict[str, Any], key: str, default: str = "") -> str:
    value = record.get(key, default)
    return str(value) if value is not None else default


def _int(record: dict[str, Any], key: str) -> int:
    value = record.get(key, 0)
    return int(value) if isinstance(value, (int, float)) else 0


def _float(record: dict[str, Any], key: str) -> float:
    value = record.get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def build_span_tree(records: list[dict[str, Any]]) -> list[SpanNode]:
    """按 parent_id 构建 span 树。父节点缺失的 span 视为根。"""
    nodes: dict[str, SpanNode] = {}
    ordered: list[dict[str, Any]] = [r for r in records if r.get("type", "span") == "span"]
    for record in ordered:
        span_id = _str(record, "span_id")
        if not span_id:
            continue
        nodes[span_id] = SpanNode(record=record)

    roots: list[SpanNode] = []
    for record in ordered:
        span_id = _str(record, "span_id")
        node = nodes.get(span_id)
        if node is None:
            continue
        parent_id = _str(record, "parent_id")
        parent = nodes.get(parent_id) if parent_id else None
        if parent is None or parent is node:
            roots.append(node)
        else:
            parent.children.append(node)
    return roots


def _format_usage(record: dict[str, Any]) -> str:
    usage = record.get("usage")
    if not isinstance(usage, dict):
        return ""
    input_tokens = _int(usage, "input_tokens")
    output_tokens = _int(usage, "output_tokens")
    total = _int(usage, "total_tokens")
    if not total and not input_tokens and not output_tokens:
        return ""
    return f" ↑{input_tokens} ↓{output_tokens}"


def _format_cost(record: dict[str, Any]) -> str:
    cost = record.get("cost_usd")
    if not isinstance(cost, (int, float)):
        return ""
    return f" ${cost:.6f}"


def format_span_line(record: dict[str, Any]) -> str:
    kind = _str(record, "kind", "span")
    icon = _KIND_ICONS.get(kind, "·")
    name = _str(record, "name", kind)
    status = _str(record, "status", "ok")
    status_icon = {"ok": "✓", "error": "✗", "cancelled": "⊘"}.get(status, "?")
    duration = record.get("duration_ms")
    duration_text = f" {duration:.0f}ms" if isinstance(duration, (int, float)) else ""
    model = _str(record, "model")
    model_text = f" [{model}]" if model and kind == "generation" else ""
    return (
        f"{icon} {name}{model_text}{_format_usage(record)}{_format_cost(record)}"
        f" — {status_icon}{duration_text}"
    )


def format_trace_tree(records: list[dict[str, Any]], max_depth: int = 12) -> str:
    spans = [r for r in records if r.get("type", "span") == "span"]
    roots = build_span_tree(spans)
    if not roots:
        return "当前会话还没有 trace 记录。"

    lines: list[str] = ["Trace 树:"]

    def render(node: SpanNode, depth: int) -> None:
        if depth > max_depth:
            lines.append("  " * depth + "…")
            return
        prefix = "  " * depth
        lines.append(prefix + format_span_line(node.record))
        for child in node.children:
            render(child, depth + 1)

    for root in roots:
        render(root, 0)

    span_count = len(spans)
    total_in = sum(_int(r.get("usage") or {}, "input_tokens") for r in spans)
    total_out = sum(_int(r.get("usage") or {}, "output_tokens") for r in spans)
    total_cost = sum(_float(r, "cost_usd") for r in spans)
    lines.append("")
    lines.append(
        f"合计: {span_count} spans, ↑{total_in} ↓{total_out}"
        + (f", 估算成本 ${total_cost:.6f}" if total_cost else "")
    )
    return "\n".join(lines)


def format_cost_report(records: list[dict[str, Any]]) -> str:
    spans = [r for r in records if r.get("type", "span") == "span"]
    if not spans:
        return "当前会话还没有 trace 记录。"

    by_kind: dict[str, list[dict[str, Any]]] = {}
    by_model: dict[str, list[dict[str, Any]]] = {}
    for record in spans:
        by_kind.setdefault(_str(record, "kind", "span"), []).append(record)
        if record.get("model"):
            by_model.setdefault(_str(record, "model"), []).append(record)

    total_cost = sum(_float(r, "cost_usd") for r in spans)
    total_in = sum(_int(r.get("usage") or {}, "input_tokens") for r in spans)
    total_out = sum(_int(r.get("usage") or {}, "output_tokens") for r in spans)
    total_cache_read = sum(_int(r.get("usage") or {}, "cache_read_tokens") for r in spans)
    total_cache_creation = sum(_int(r.get("usage") or {}, "cache_creation_tokens") for r in spans)

    lines = [
        "Trace 成本报告:",
        (
            f"  总 token: ↑{total_in:,} ↓{total_out:,} "
            f"cache_read={total_cache_read:,} cache_creation={total_cache_creation:,}"
        ),
        f"  估算成本: ${total_cost:.6f}",
        "",
        "按类型:",
    ]
    for kind, items in sorted(
        by_kind.items(), key=lambda kv: -sum(_float(r, "cost_usd") for r in kv[1])
    ):
        cost = sum(_float(r, "cost_usd") for r in items)
        tokens = sum(_int(r.get("usage") or {}, "total_tokens") for r in items)
        lines.append(f"  {kind:<22} {len(items):>3} spans  {tokens:>9,} tokens  ${cost:.6f}")

    if by_model:
        lines.extend(["", "按模型:"])
        for model, items in sorted(
            by_model.items(), key=lambda kv: -sum(_float(r, "cost_usd") for r in kv[1])
        ):
            cost = sum(_float(r, "cost_usd") for r in items)
            tokens = sum(_int(r.get("usage") or {}, "total_tokens") for r in items)
            lines.append(f"  {model:<28} {len(items):>3} 次  {tokens:>9,} tokens  ${cost:.6f}")

    tools = by_kind.get("tool", [])
    if tools:
        lines.extend(["", "工具耗时 Top 5:"])
        ranked = sorted(tools, key=lambda r: -_float(r, "duration_ms"))[:5]
        for record in ranked:
            lines.append(
                f"  {_str(record, 'name', 'tool'):<24} {_float(record, 'duration_ms'):>9.0f}ms"
                f"  {'error' if _str(record, 'status') == 'error' else 'ok'}"
            )

    # 只警告“真的有 token 消耗但没匹配到价格”的 span。usage 全为 0 的
    # span（例如 provider 没返回 usage、或 selector 没被调用）不算无法计价。
    unknown_cost: list[dict[str, Any]] = []
    for record in spans:
        usage = record.get("usage")
        if not isinstance(usage, dict) or record.get("cost_usd") is not None:
            continue
        if _int(usage, "total_tokens") > 0:
            unknown_cost.append(record)
    if unknown_cost:
        lines.extend(
            [
                "",
                f"注意: {len(unknown_cost)} 个带用量的 span 无法匹配价格表，成本未计入。",
                "可以稍后在 PricingTable 中补充模型价格。",
            ]
        )
    return "\n".join(lines)


def format_session_list(infos: list[TraceFileInfo]) -> str:
    if not infos:
        return "没有找到 trace 文件。"
    lines = ["Trace 会话列表（按最近修改排序）:"]
    for info in infos[:20]:
        size_kb = info.size_bytes / 1024
        lines.append(f"  {info.session_id:<40} {info.age_days:>3} 天前  {size_kb:>8.1f} KB")
    if len(infos) > 20:
        lines.append(f"  … 另有 {len(infos) - 20} 个会话")
    return "\n".join(lines)
