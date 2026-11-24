"""harness 相关的指标统计。

判断"是否真的进化"只信客观计数器，不信感觉。这里只读 trace 与版本日志，
不做任何模型调用，所以随时可跑、成本为零。

四个核心指标：

- ``errors_by_signature``：某类失败出现了几次（经验生效的直接证据）
- ``tokens_per_task`` / ``tool_calls_per_task``：防退化（不能涨）
- ``first_try_success``：首次尝试就成功的会话占比
- ``experience_hits`` / ``zombie``：经验是否在占地方
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lumecode.harness.attribution import (
    _is_error_span,
    _result_text,
    _signature,
    _tool_name,
    load_failures,
)
from lumecode.harness.store import HarnessStore
from lumecode.telemetry.store import TRACES_DIR, TraceStore


@dataclass
class SessionStats:
    session_id: str
    tokens: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    model_errors: int = 0
    mtime: float = 0.0

    @property
    def first_try_ok(self) -> bool:
        # 一次失败都没有 = 首次尝试就成功（简化定义，随工具调用为 0 时不算）
        return self.tool_calls > 0 and self.tool_errors == 0 and self.model_errors == 0


@dataclass
class MetricsReport:
    sessions: list[SessionStats] = field(default_factory=list)

    @property
    def task_count(self) -> int:
        return len(self.sessions)

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens for s in self.sessions)

    @property
    def total_tool_calls(self) -> int:
        return sum(s.tool_calls for s in self.sessions)

    @property
    def tokens_per_task(self) -> float:
        return self.total_tokens / self.task_count if self.task_count else 0.0

    @property
    def tool_calls_per_task(self) -> float:
        return self.total_tool_calls / self.task_count if self.task_count else 0.0

    @property
    def first_try_rate(self) -> float:
        usable = [s for s in self.sessions if s.tool_calls > 0]
        if not usable:
            return 0.0
        return sum(1 for s in usable if s.first_try_ok) / len(usable)


def _usage_tokens(record: dict[str, Any]) -> int:
    """优先读 ``usage``，回退到 ``attributes`` 里的 ``gen_ai.usage.*``。

    某些客户端路径只把用量写进 attributes（``usage`` 为 null），两条都读
    才能避免 token 指标恒为 0。
    """
    usage = record.get("usage")
    total = 0
    if isinstance(usage, dict):
        for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_creation_tokens"):
            value = usage.get(key)
            if isinstance(value, (int, float)):
                total += int(value)
    if total:
        return total
    attrs = record.get("attributes")
    if isinstance(attrs, dict):
        for key in (
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.output_tokens",
            "gen_ai.usage.cache_read_tokens",
            "gen_ai.usage.cache_creation_tokens",
        ):
            value = attrs.get(key)
            if isinstance(value, (int, float)):
                total += int(value)
    return total


def compute_session_stats(records: list[dict[str, Any]], session_id: str) -> SessionStats:
    stats = SessionStats(session_id=session_id)
    for record in records:
        kind = record.get("kind")
        if kind in {"generation", "session_summary"}:
            stats.tokens += _usage_tokens(record)
            if _is_error_span(record):
                # 模型调用本身失败（鉴权/网络/限流）：说明 harness 没挡住，
                # 不是工具用法问题，distill 不应该把它当成行为样本。
                stats.model_errors += 1
        elif kind == "tool":
            stats.tool_calls += 1
            if _is_error_span(record):
                stats.tool_errors += 1
        start = record.get("start_time_ns")
        if isinstance(start, (int, float)):
            stats.mtime = max(stats.mtime, start / 1_000_000_000)
    return stats


def collect_metrics(
    work_dir: str | Path,
    *,
    since: str = "7d",
    store: TraceStore | None = None,
) -> MetricsReport:
    from lumecode.harness.attribution import _parse_since

    cutoff = _parse_since(since)
    trace_store = store or TraceStore(str(Path(work_dir) / TRACES_DIR))
    report = MetricsReport()
    for info in trace_store.list_sessions():
        if cutoff is not None and info.mtime < cutoff:
            continue
        records = trace_store.read(info.session_id)
        stats = compute_session_stats(records, info.session_id)
        stats.mtime = info.mtime
        report.sessions.append(stats)
    return report


# ---------------------------------------------------------------------------
# 经验命中率
# ---------------------------------------------------------------------------


@dataclass
class ExperienceUsage:
    name: str
    signature: str
    created_at: str
    hits: int = 0
    identical_mistakes: int = 0


@dataclass
class ExperienceReport:
    items: list[ExperienceUsage] = field(default_factory=list)

    @property
    def zombie_names(self) -> list[str]:
        return [item.name for item in self.items if item.hits == 0]

    @property
    def zombie_rate(self) -> float:
        if not self.items:
            return 0.0
        return len(self.zombie_names) / len(self.items)

    @property
    def tainted_names(self) -> list[str]:
        """创建之后仍然犯同样错误的经验——它们没有起作用。"""
        return [item.name for item in self.items if item.identical_mistakes > 0]


def _parse_iso(value: str) -> float:
    """ISO 时间戳 → epoch 秒。容忍 ``Z`` 后缀和带小数秒的变体。"""
    from datetime import datetime

    text = value.strip().replace("Z", "+00:00")
    if not text:
        return 0.0
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return 0.0


def compute_experience_usage(work_dir: str | Path, since: str = "30d") -> ExperienceReport:
    """统计每条经验：创建后同类失败是否还在发生。

    - ``hits``：经验创建后，同类失败次数相比创建前下降（>0 记为命中）
    - ``identical_mistakes``：经验创建后仍出现的同类失败次数
    """
    harness = HarnessStore(work_dir)
    cases = load_failures(work_dir, since)
    report = ExperienceReport()

    for entry in harness.list_experiences():
        usage = ExperienceUsage(
            name=entry.name,
            signature=entry.signature,
            created_at=entry.created_at,
        )
        created_ts = _parse_iso(entry.created_at)
        if entry.signature and entry.status == "active":
            before = sum(
                1 for c in cases if c.signature == entry.signature and c.start_ns / 1e9 < created_ts
            )
            after = sum(
                1
                for c in cases
                if c.signature == entry.signature and c.start_ns / 1e9 >= created_ts
            )
            usage.identical_mistakes = after
            usage.hits = max(0, before - after)
        report.items.append(usage)
    return report


def error_counts_by_signature(work_dir: str | Path, since: str = "30d") -> Counter[str]:
    cases = load_failures(work_dir, since)
    return Counter(c.signature for c in cases)


def daily_error_counts(work_dir: str | Path, since: str = "30d") -> dict[str, int]:
    """按天聚合失败数，用于看趋势。"""
    from datetime import UTC, datetime

    cases = load_failures(work_dir, since)
    counts: Counter[str] = Counter()
    for case in cases:
        if case.start_ns <= 0:
            continue
        day = datetime.fromtimestamp(case.start_ns / 1e9, UTC).strftime("%m-%d")
        counts[day] += 1
    return dict(sorted(counts.items()))


def tool_error_rate(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """按工具统计调用次数与失败次数。"""
    result: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "errors": 0})
    for record in records:
        if record.get("kind") != "tool":
            continue
        tool = _tool_name(record)
        result[tool]["calls"] += 1
        if _is_error_span(record):
            result[tool]["errors"] += 1
            # 触发一次签名计算，保证坏数据不会在统计视图里静默通过
            _signature(tool, record.get("input"), _result_text(record))
    return dict(result)
