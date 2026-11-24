"""harness 子系统用到的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ExperienceStatus = Literal["active", "retired"]
ProposalKind = Literal["experience", "prompt_patch"]
ProposalStatus = Literal["pending", "accepted", "rejected"]

# 失败归因目标。只有 harness 自己能改的目标才允许成为经验的归属；
# tool_bug / model_limit 只记录，不作为可自动改进的对象。
Attribution = Literal["prompt", "skill", "tool_schema", "tool_bug", "model_limit", "unknown"]


@dataclass
class ExperienceEntry:
    """一条经验 = 一条以后要遵守的行为规则。

    ``signature`` 是经验所针对的失败类别的稳定标识（例如
    ``tool:bash:pip-install``）。它让后续统计能回答"这条经验生效了吗"——
    看同类失败在经验创建之后还有没有再发生。
    """

    name: str
    description: str
    body: str = ""
    status: ExperienceStatus = "active"
    signature: str = ""
    attribution: Attribution = "unknown"
    evidence: list[dict[str, str]] = field(default_factory=list)
    created_at: str = ""
    source: str = "manual"

    def to_frontmatter(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at,
            "source": self.source,
        }
        if self.signature:
            data["signature"] = self.signature
        if self.attribution and self.attribution != "unknown":
            data["attribution"] = self.attribution
        if self.evidence:
            data["evidence"] = self.evidence
        return data


@dataclass
class Proposal:
    """一条待评审的改动提议。``kind == "prompt_patch"`` 时 ``diff`` 是对
    基础提示词分节的替换内容（人工确认后才写入）。"""

    id: str
    kind: ProposalKind
    target: str  # experience 时为经验名；prompt_patch 时为 prompt 分节 key
    title: str
    body: str
    rationale: str = ""
    expected_effect: str = ""
    risk: str = "low"
    signature: str = ""
    attribution: Attribution = "unknown"
    evidence: list[dict[str, str]] = field(default_factory=list)
    created_at: str = ""
    status: ProposalStatus = "pending"

    def to_frontmatter(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "target": self.target,
            "title": self.title,
            "status": self.status,
            "created_at": self.created_at,
            "risk": self.risk,
        }
        if self.rationale:
            data["rationale"] = self.rationale
        if self.expected_effect:
            data["expected_effect"] = self.expected_effect
        if self.signature:
            data["signature"] = self.signature
        if self.attribution and self.attribution != "unknown":
            data["attribution"] = self.attribution
        if self.evidence:
            data["evidence"] = self.evidence
        return data


@dataclass
class HarnessVersion:
    """``versions.jsonl`` 的一行。出问题时靠它回答"从哪次改动开始变差"。"""

    version: int
    created_at: str
    changes: list[str] = field(default_factory=list)
    proposal_ids: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "changes": self.changes,
            "proposal_ids": self.proposal_ids,
            "note": self.note,
        }
