"""``/distill``：从失败轨迹里提炼候选改动。

流程对齐 Harness² 的 Propose 阶段，但刻意做了两点保守化：

1. **自带正解的样本优先**：自愈型失败（错→报错→换办法→成功）不需要模型
   推理，直接把「错误做法 / 正确做法」成对喂给 editor。
2. **没有证据的提议直接丢弃**：editor 必须引用 ``session_id:span_id``，
   引用不到的提议在解析阶段就被过滤，不进 pending 队列。

这里只产提议，不写任何 harness 文件。写入发生在 ``/evolve accept``。
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any

from lumecode.harness.attribution import FailureCase, group_failures
from lumecode.harness.models import Proposal
from lumecode.harness.store import HarnessStore, split_frontmatter

log = logging.getLogger(__name__)

BASE_PROMPT_KEYS = [
    "identity",
    "system",
    "doing-tasks",
    "executing-actions",
    "using-tools",
    "tone-style",
    "text-output",
]

DISTILL_INSTRUCTIONS = """\
You are the harness editor for a terminal coding agent called LumeCode.

The agent's behaviour comes from a harness: the system prompt sections plus a
small library of "experiences" (short behavioural rules learned from past
failures). Your job is to turn observed failures into concrete harness changes.

You get:
1. Real failure cases from the agent's execution traces, grouped by signature.
   Each case carries evidence: session_id and span_id.
2. The current harness: prompt section keys and existing experiences.

Output rules (strict):
- Emit at most 3 proposals.
- Prefer `kind: experience` — a short rule that prevents the failure.
- Use `kind: prompt_patch` only when the base prompt itself is wrong, cannot be
  fixed by an experience, and `target` is one of the prompt section keys.
- Every proposal MUST cite at least one evidence pair from the input, written
  exactly as `<session_id>:<span_id>`. Proposals without evidence are dropped.
- Only propose changes to prompt / skill selection / tool usage conventions.
  Never propose anything about permissions, sandboxing, or disabling checks.
- `target` for an experience is a kebab-case slug (a-z0-9-), for a prompt patch
  it is a section key from the input.
- `body` is the actual text to add. Keep it under 400 characters.
- `rationale` explains why this prevents the failure. `expected_effect` states
  the measurable difference (e.g. "no repeated pip install failure").

Emit proposals as fenced yaml blocks, one per proposal:

```yaml
id: exp-20260101-001
kind: experience
target: use-uv-for-python-packages
title: Use uv for Python package installs
body: This project's venv has no pip; install packages with `uv pip install --python .venv/Scripts/python.exe <pkg>`.
signature: tool:bash:pip install:missing_module
attribution: prompt
evidence:
  - session: abc123
    span: 9f2c1d
rationale: ...
expected_effect: ...
risk: low
```
"""


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:48] or "experience"


def _next_id(store: HarnessStore) -> str:
    existing = {p.id for p in store.list_proposals(status=None)}
    day = datetime.now(UTC).strftime("%Y%m%d")
    index = 1
    while True:
        candidate = f"exp-{day}-{index:03d}"
        if candidate not in existing:
            return candidate
        index += 1


def format_failure_cases(
    groups: list[tuple[str, list[FailureCase]]], limit_per_group: int = 3
) -> str:
    blocks: list[str] = []
    for signature, cases in groups:
        lines = [f"### {signature}  ({len(cases)} case(s))"]
        for case in cases[:limit_per_group]:
            lines.append(f"- task: {case.task_summary or '(unknown)'}")
            lines.append(f"  wrong action: {case.action}")
            lines.append(f"  error: {case.error.splitlines()[0][:200] if case.error else ''}")
            if case.recovered:
                lines.append(f"  recovered action: {case.recovered_action}")
            lines.append(f"  evidence: {case.session_id}:{case.span_id}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_current_harness(store: HarnessStore) -> str:
    keys = store.list_base_sections() or BASE_PROMPT_KEYS
    lines = ["Prompt section keys:", ", ".join(keys), "", "Existing experiences:"]
    entries = store.list_experiences(include_retired=True)
    if not entries:
        lines.append("(none)")
    for entry in entries:
        state = "retired" if entry.status == "retired" else "active"
        lines.append(
            f"- [{state}] {entry.name} (signature={entry.signature or '-'}): {entry.description}"
        )
    return "\n".join(lines)


def build_editor_prompt(
    work_dir: str,
    since: str,
    groups: list[tuple[str, list[FailureCase]]],
    store: HarnessStore,
) -> str:
    return "\n\n".join(
        [
            DISTILL_INSTRUCTIONS,
            "## Failure cases",
            format_failure_cases(groups),
            "## Current harness",
            format_current_harness(store),
        ]
    )


_YAML_BLOCK_RE = re.compile(r"```ya?ml\s*\n(.*?)```", re.DOTALL)

_CLOSED_SECTION_WORDS = re.compile(
    r"\b(permission|sandbox|approval|bypass|disable\s+checks?|allow\s+all)\b", re.I
)


def _evidence_ok(evidence: list[Any], valid: set[str]) -> bool:
    for item in evidence:
        if not isinstance(item, dict):
            continue
        key = f"{item.get('session', '')}:{item.get('span', '')}"
        if key in valid:
            return True
    return False


def parse_proposals(
    text: str,
    store: HarnessStore,
    valid_evidence: set[str],
    *,
    max_proposals: int = 3,
) -> tuple[list[Proposal], list[str]]:
    """解析 editor 输出。返回 (通过的提议, 被丢弃的原因)。"""
    proposals: list[Proposal] = []
    dropped: list[str] = []
    prompt_keys = set(store.list_base_sections()) | set(BASE_PROMPT_KEYS)

    for index, block in enumerate(_YAML_BLOCK_RE.findall(text)):
        meta, body = split_frontmatter(block) if block.lstrip().startswith("---") else ({}, block)
        if not meta:
            import yaml

            try:
                loaded = yaml.safe_load(block)
            except yaml.YAMLError as e:
                dropped.append(f"#{index + 1}: YAML 解析失败 ({e})")
                continue
            meta = loaded if isinstance(loaded, dict) else {}
        if not meta:
            dropped.append(f"#{index + 1}: 空 YAML 块")
            continue
        if len(proposals) >= max_proposals:
            dropped.append(f"{meta.get('id', index + 1)}: 超出单轮 {max_proposals} 条上限")
            continue

        kind = "prompt_patch" if meta.get("kind") == "prompt_patch" else "experience"
        body_text = str(meta.get("body") or body or "").strip()
        if not body_text:
            dropped.append(f"{meta.get('id', index + 1)}: 缺少 body")
            continue
        if _CLOSED_SECTION_WORDS.search(body_text):
            dropped.append(f"{meta.get('id', index + 1)}: 触及权限/沙箱红线")
            continue

        evidence = [e for e in meta.get("evidence") or [] if isinstance(e, dict)]
        if not _evidence_ok(evidence, valid_evidence):
            dropped.append(f"{meta.get('id', index + 1)}: 证据无法溯源到具体 span")
            continue

        target = str(meta.get("target") or "").strip()
        if kind == "prompt_patch":
            if target not in prompt_keys:
                dropped.append(f"{meta.get('id', index + 1)}: prompt 分节 {target!r} 不存在")
                continue
        else:
            target = _slug(target or meta.get("title") or "experience")

        proposals.append(
            Proposal(
                id=str(meta.get("id") or ""),
                kind=kind,  # type: ignore[arg-type]
                target=target,
                title=str(meta.get("title") or target),
                body=body_text[:4000],
                rationale=str(meta.get("rationale") or ""),
                expected_effect=str(meta.get("expected_effect") or ""),
                risk=str(meta.get("risk") or "low"),
                signature=str(meta.get("signature") or ""),
                attribution=str(meta.get("attribution") or "unknown"),  # type: ignore[arg-type]
                evidence=[
                    {
                        str(k): str(v)
                        for k, v in e.items()
                        if k in {"session", "span", "tool", "error"}
                    }
                    for e in evidence
                ],
            )
        )

    # 分配 id（editor 可能没给或给重）
    used: set[str] = set()
    for proposal in proposals:
        if not proposal.id or proposal.id in used:
            proposal.id = _next_id(store) if not used else f"{_next_id(store)}-{len(used)}"
        used.add(proposal.id)
    return proposals, dropped


async def run_distill(
    client: Any,
    protocol: str,
    work_dir: str,
    since: str = "7d",
    *,
    store: HarnessStore | None = None,
    min_cases: int = 1,
) -> tuple[list[Proposal], list[str], list[FailureCase]]:
    """执行一次 distill。返回 (提议, 丢弃原因, 输入的失败案例)。"""
    from lumecode.agent import Agent
    from lumecode.conversation import ConversationManager
    from lumecode.tools import ToolRegistry

    harness = store or HarnessStore(work_dir)
    harness.ensure_scaffold()

    cases = load_failures_for_distill(work_dir, since)
    if len(cases) < min_cases:
        return [], ["没有足够的失败案例"], cases

    groups = group_failures(cases)
    prompt = build_editor_prompt(work_dir, since, groups, harness)

    # editor 必须是「全新上下文」：踩坑的 agent 自己总结会自我辩护。
    # 这里不给任何工具，只让它读文本、产出文本。
    editor = Agent(
        client=client,
        registry=ToolRegistry(),
        protocol=protocol,
        work_dir=work_dir,
        max_iterations=2,
    )
    conv = ConversationManager()
    conv.add_user_message(prompt)

    chunks: list[str] = []
    async for event in editor.run(conv):
        text = getattr(event, "text", None)
        if isinstance(text, str):
            chunks.append(text)

    valid = {f"{c.session_id}:{c.span_id}" for c in cases}
    proposals, dropped = parse_proposals("".join(chunks), harness, valid)
    for proposal in proposals:
        harness.save_proposal(proposal)
    return proposals, dropped, cases


def load_failures_for_distill(work_dir: str, since: str) -> list[FailureCase]:
    from lumecode.harness.attribution import load_failures

    return load_failures(work_dir, since)
