"""从本地 trace 中抽取失败案例。

原始 span 是流水账，直接喂给模型只会得到泛泛而谈的总结。这里做结构化
抽取，重点是三件事：

1. **自愈型优先**：失败之后 agent 自己换了办法并成功——这一对里自带
   正解，是最有价值的样本，不需要模型去猜。
2. **稳定签名**：把错误归到稳定的 ``signature``（例如
   ``tool:bash:pip install:missing_module``），让后续统计能回答
   "这条经验生效了吗"。
3. **可溯源**：每条案例都带 ``session_id + span_id``，写不出证据的
   经验在评审阶段会被丢弃。
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lumecode.telemetry.store import TRACES_DIR, TraceStore

log = logging.getLogger(__name__)

# 失败的归因类别 -> 稳定错误分类
_ERROR_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"command not found|not recognized as an internal", re.I), "not_found"),
    (re.compile(r"no such file or directory|cannot find the path", re.I), "not_found"),
    (re.compile(r"no module named|modulenotfounderror", re.I), "missing_module"),
    (re.compile(r"permission denied|access is denied|not permitted", re.I), "permission"),
    (
        re.compile(
            r"old_string not found|file has not been read|string to replace not found", re.I
        ),
        "edit_mismatch",
    ),
    (re.compile(r"syntaxerror|syntax error|invalid syntax", re.I), "syntax"),
    (re.compile(r"timed out|timeout", re.I), "timeout"),
    (
        re.compile(r"connection refused|could not resolve|name or service not known", re.I),
        "network",
    ),
    (re.compile(r"already exists|file exists", re.I), "exists"),
    (re.compile(r"is a directory|not a directory", re.I), "path_type"),
]

# 错误类别 -> 默认归因。模型能力型不进入可改进候选。
_ATTRIBUTION_BY_CATEGORY: dict[str, str] = {
    "not_found": "prompt",
    "missing_module": "prompt",
    "edit_mismatch": "prompt",
    "permission": "tool_schema",
    "exists": "prompt",
    "path_type": "prompt",
    "syntax": "prompt",
    "timeout": "unknown",
    "network": "unknown",
}

_SHELL_WRAPPERS = {"sudo", "env", "time", "nohup", "exec"}

_IGNORE_SHELL_TOKENS = {
    "-m",
    "-c",
    "-e",
    "--",
    "&&",
    "||",
    "|",
    ";",
}


@dataclass
class FailureCase:
    session_id: str
    span_id: str
    tool: str
    signature: str
    category: str
    error: str
    action: str
    recovered: bool
    repeated: int = 1
    attribution: str = "prompt"
    task_summary: str = ""
    recovered_action: str = ""
    start_ns: int = 0

    @property
    def value_score(self) -> float:
        if self.recovered:
            return 1.0
        if self.repeated >= 2:
            return 0.8
        return 0.4

    def evidence(self) -> dict[str, str]:
        return {
            "session": self.session_id,
            "span": self.span_id,
            "tool": self.tool,
            "error": self.error[:300],
        }


def _classify_error(text: str) -> str:
    for pattern, category in _ERROR_PATTERNS:
        if pattern.search(text):
            return category
    return "other"


def _shell_head(command: str) -> str:
    """取命令的前两个有意义的词，作为签名的稳定部分。"""
    tokens = [t for t in re.split(r"\s+", command.strip()) if t and t not in _IGNORE_SHELL_TOKENS]
    while tokens and tokens[0].split("/")[-1] in _SHELL_WRAPPERS:
        tokens = tokens[1:]
    head = [Path(t).name if "/" in t or "\\" in t else t for t in tokens[:2]]
    return " ".join(head).lower()


def _signature(tool: str, args: Any, error: str) -> tuple[str, str]:
    category = _classify_error(error)
    if tool.lower() == "bash":
        command = ""
        if isinstance(args, dict):
            command = str(args.get("command") or "")
        elif isinstance(args, str):
            command = args
        head = _shell_head(command)
        if head:
            return f"tool:bash:{head}:{category}", category
    return f"tool:{tool.lower()}:{category}", category


def _tool_name(record: dict[str, Any]) -> str:
    attrs = record.get("attributes") or {}
    if isinstance(attrs, dict):
        name = attrs.get("gen_ai.tool.name")
        if name:
            return str(name)
    return str(record.get("name") or "unknown")


def _result_text(record: dict[str, Any]) -> str:
    for key in ("error", "output"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _task_summary(records: list[dict[str, Any]]) -> str:
    """取会话开头那条 GENERATION 的输入作为任务概述。"""
    for record in records:
        if record.get("kind") != "generation":
            continue
        value = record.get("input")
        if isinstance(value, str) and value.strip():
            text = " ".join(value.split())
            return text[:200]
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    return " ".join(item["text"].split())[:200]
    return ""


def _parse_since(value: str) -> float | None:
    """解析 ``7d`` / ``24h`` / ``30m`` 为时间戳；非法返回 None。"""
    if not value:
        return None
    match = re.fullmatch(r"(\d+)\s*([dhm])", value.strip().lower())
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2)
    seconds = {"d": 86_400, "h": 3_600, "m": 60}[unit] * amount
    return time.time() - seconds


def _is_error_span(record: dict[str, Any]) -> bool:
    if record.get("status") == "error":
        return True
    attrs = record.get("attributes")
    return isinstance(attrs, dict) and bool(attrs.get("tool.is_error"))


def extract_failures_from_records(
    records: list[dict[str, Any]], session_id: str
) -> list[FailureCase]:
    """从一个会话的 span 列表抽取失败案例（纯函数，便于测试）。"""
    if not records:
        return []
    ordered = sorted(records, key=lambda r: int(r.get("start_time_ns") or 0))
    summary = _task_summary(ordered)

    cases: list[FailureCase] = []
    for index, record in enumerate(ordered):
        if record.get("kind") != "tool" or not _is_error_span(record):
            continue
        tool = _tool_name(record)
        error = _result_text(record)
        signature, category = _signature(tool, record.get("input"), error)

        # 自愈：同一工具在后续 10 个 span 内成功，且本会话不再复发同签名失败。
        tail = ordered[index + 1 : index + 11]
        recovered_action = ""
        for later in tail:
            if later.get("kind") != "tool" or _is_error_span(later):
                continue
            if _tool_name(later) != tool:
                continue
            later_sig, _ = _signature(tool, later.get("input"), _result_text(later))
            if later_sig == signature:
                continue
            recovered_action = _action_text(later)
            break

        still_repeats = sum(
            1
            for later in ordered[index + 1 :]
            if later.get("kind") == "tool"
            and _is_error_span(later)
            and _signature(_tool_name(later), later.get("input"), _result_text(later))[0]
            == signature
        )

        cases.append(
            FailureCase(
                session_id=session_id,
                span_id=str(record.get("span_id") or ""),
                tool=tool,
                signature=signature,
                category=category,
                error=error[:600],
                action=_action_text(record),
                recovered=bool(recovered_action),
                repeated=1 + still_repeats,
                attribution=_ATTRIBUTION_BY_CATEGORY.get(category, "prompt"),
                task_summary=summary,
                recovered_action=recovered_action,
                start_ns=int(record.get("start_time_ns") or 0),
            )
        )
    return cases


def _action_text(record: dict[str, Any]) -> str:
    args = record.get("input")
    if isinstance(args, str):
        return args[:200]
    if isinstance(args, dict):
        for key in ("command", "path", "file_path", "pattern", "query"):
            value = args.get(key)
            if isinstance(value, str) and value:
                return value[:200]
        return json.dumps(args, ensure_ascii=False)[:200]
    return ""


def load_failures(
    work_dir: str | Path,
    since: str = "7d",
    *,
    store: TraceStore | None = None,
) -> list[FailureCase]:
    """读取 ``.lumecode/traces`` 并抽取失败案例。"""
    cutoff = _parse_since(since)
    trace_store = store or TraceStore(str(Path(work_dir) / TRACES_DIR))

    cases: list[FailureCase] = []
    for info in trace_store.list_sessions():
        if cutoff is not None and info.mtime < cutoff:
            continue
        records = trace_store.read(info.session_id)
        cases.extend(extract_failures_from_records(records, info.session_id))

    cases.sort(key=lambda c: (-c.value_score, -c.start_ns))
    return cases


def group_failures(cases: list[FailureCase]) -> list[tuple[str, list[FailureCase]]]:
    """按签名分组，组内按时间倒序。组间按价值分总和降序。"""
    groups: dict[str, list[FailureCase]] = defaultdict(list)
    for case in cases:
        groups[case.signature].append(case)
    return sorted(
        groups.items(),
        key=lambda kv: sum(c.value_score for c in kv[1]),
        reverse=True,
    )


def signature_counts(cases: list[FailureCase]) -> Counter[str]:
    return Counter(c.signature for c in cases)


@dataclass
class FailureSummary:
    total: int = 0
    recovered: int = 0
    groups: list[str] = field(default_factory=list)

    def render(self) -> str:
        if not self.total:
            return "没有发现失败案例。"
        head = f"共 {self.total} 例失败，其中自愈型 {self.recovered} 例"
        if not self.groups:
            return head
        return head + "\n" + "\n".join(f"  {g}" for g in self.groups)
