"""提议的评审与应用。

``/evolve accept`` 走这里。三条不可绕过：

1. 应用前先 ``snapshot()``——回滚能力是这套系统能存在的前提。
2. 经验写入同样受 ``HarnessStore`` 的上限约束。
3. 每次写入都追加一条版本记录，方便回答"从哪次改动开始变差"。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from lumecode.harness.models import ExperienceEntry, Proposal
from lumecode.harness.store import HarnessStore, HarnessWriteError

log = logging.getLogger(__name__)


class AcceptError(Exception):
    pass


@dataclass
class ApplyResult:
    proposal: Proposal
    summary: str


def apply_proposal(proposal: Proposal, store: HarnessStore) -> ApplyResult:
    if proposal.status != "pending":
        raise AcceptError(f"提议 {proposal.id} 已经是 {proposal.status} 状态")

    store.ensure_scaffold()

    # 必须先快照：即使 write 过程抛异常，也能回滚。
    store.snapshot(reason=f"accept:{proposal.id}")

    if proposal.kind == "prompt_patch":
        summary = _apply_prompt_patch(proposal, store)
    else:
        summary = _apply_experience(proposal, store)

    store.update_proposal_status(proposal, "accepted")
    store.append_version(
        changes=[f"{proposal.kind}:{proposal.target}"],
        proposal_ids=[proposal.id],
        note=proposal.title,
    )
    return ApplyResult(proposal=proposal, summary=summary)


def reject_proposal(proposal: Proposal, store: HarnessStore) -> None:
    if proposal.status != "pending":
        raise AcceptError(f"提议 {proposal.id} 已经是 {proposal.status} 状态")
    store.update_proposal_status(proposal, "rejected")


def _apply_experience(proposal: Proposal, store: HarnessStore) -> str:
    entry = ExperienceEntry(
        name=proposal.target,
        description=proposal.title,
        body=proposal.body,
        signature=proposal.signature,
        attribution=proposal.attribution,
        evidence=proposal.evidence,
        source=f"distill:{proposal.id}",
    )
    try:
        path = store.add_experience(entry)
    except HarnessWriteError as e:
        store.restore_snapshot()
        raise AcceptError(str(e)) from e
    return f"已写入经验 {entry.name} → {path}"


def _apply_prompt_patch(proposal: Proposal, store: HarnessStore) -> str:
    """基础提示词分节替换。人工确认后才会走到这里，但仍保留快照。"""
    try:
        store.write_base_section(proposal.target, proposal.body)
    except HarnessWriteError as e:
        store.restore_snapshot()
        raise AcceptError(str(e)) from e
    return f"已更新基础提示词分节 prompt/{proposal.target}.md"


def accept_by_id(proposal_id: str, store: HarnessStore) -> ApplyResult:
    proposal = store.get_proposal(proposal_id)
    if proposal is None:
        raise AcceptError(f"找不到提议 {proposal_id}")
    return apply_proposal(proposal, store)


def reject_by_id(proposal_id: str, store: HarnessStore) -> None:
    proposal = store.get_proposal(proposal_id)
    if proposal is None:
        raise AcceptError(f"找不到提议 {proposal_id}")
    reject_proposal(proposal, store)
