"""harness 目录的读写。

目录布局::

    .lumecode/harness/
      prompt/<key>.md          # 基础提示词分节，只读（人改）
      experience/INDEX.md      # 经验索引，自动重建
      experience/<name>.md     # 单条经验
      proposals/<id>.md        # 待评审的改动提议
      versions.jsonl           # 改动历史
      .history/<stamp>/        # 每次自动写入前的整目录快照

所有写操作都先快照，且受硬上限约束：条数、总字符、单文件大小。
这些上限写在代码里，不依赖提示词——即使 Agent 被诱导，也无法越过。
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from lumecode.harness.models import ExperienceEntry, HarnessVersion, Proposal

log = logging.getLogger(__name__)

HARNESS_DIR = ".lumecode/harness"

# 硬上限（代码层强制）
MAX_EXPERIENCES = 30
MAX_INJECT_CHARS = 6000
MAX_ENTRY_CHARS = 1200
MAX_BODY_CHARS = 4000
MAX_VERSION_SNAPSHOTS = 10

INDEX_NAME = "INDEX.md"
VERSIONS_NAME = "versions.jsonl"

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class HarnessWriteError(Exception):
    """写入被拒绝（超出上限、名称非法、目标只读等）。"""


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def split_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    """分离 YAML frontmatter 和正文。没有 frontmatter 时返回 ({}, raw)。"""
    text = raw.lstrip("﻿").lstrip()
    if not text.startswith("---"):
        return {}, raw
    end = text.find("\n---", 3)
    if end == -1:
        return {}, raw
    block = text[3:end]
    body = text[end + 4 :].lstrip("\n")
    try:
        meta = yaml.safe_load(block)
    except yaml.YAMLError:
        return {}, raw
    return (meta if isinstance(meta, dict) else {}), body


def _render_doc(frontmatter: dict[str, Any], body: str) -> str:
    header = yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False).strip()
    return f"---\n{header}\n---\n\n{body.rstrip()}\n"


class HarnessStore:
    def __init__(self, work_dir: str | Path) -> None:
        self.root = Path(work_dir) / HARNESS_DIR
        self.prompt_dir = self.root / "prompt"
        self.experience_dir = self.root / "experience"
        self.proposals_dir = self.root / "proposals"
        self.history_dir = self.root / ".history"

    # ---------------------------------------------------------------- 基础层

    def base_section(self, key: str) -> str | None:
        """读取基础提示词分节。文件不存在时返回 None（调用方回退常量）。"""
        path = self.prompt_dir / f"{key}.md"
        try:
            if not path.is_file():
                return None
            return path.read_text(encoding="utf-8").strip()
        except OSError as e:
            log.debug("[harness] read base section failed for %s: %s", path, e)
            return None

    def list_base_sections(self) -> list[str]:
        try:
            return sorted(p.stem for p in self.prompt_dir.glob("*.md") if p.is_file())
        except OSError:
            return []

    def write_base_section(self, key: str, content: str) -> None:
        """改写基础分节。仅供人工命令（``/harness edit``）使用，Agent 不可达。"""
        if not _NAME_RE.match(key):
            raise HarnessWriteError(f"非法的分节名: {key!r}")
        self.snapshot(reason=f"prompt/{key}")
        self.prompt_dir.mkdir(parents=True, exist_ok=True)
        (self.prompt_dir / f"{key}.md").write_text(content.rstrip() + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- 经验层

    def _entry_path(self, name: str) -> Path:
        return self.experience_dir / f"{name}.md"

    def list_experiences(self, include_retired: bool = False) -> list[ExperienceEntry]:
        entries: list[ExperienceEntry] = []
        try:
            paths = sorted(self.experience_dir.glob("*.md"))
        except OSError:
            return entries
        for path in paths:
            if path.name == INDEX_NAME or not path.is_file():
                continue
            entry = self._read_entry(path)
            if entry is None:
                continue
            if not include_retired and entry.status != "active":
                continue
            entries.append(entry)
        return entries

    def _read_entry(self, path: Path) -> ExperienceEntry | None:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as e:
            log.debug("[harness] read experience failed for %s: %s", path, e)
            return None
        meta, body = split_frontmatter(raw)
        name = str(meta.get("name") or path.stem)
        description = str(meta.get("description") or "")
        return ExperienceEntry(
            name=name,
            description=description,
            body=body.strip(),
            status="retired" if meta.get("status") == "retired" else "active",
            signature=str(meta.get("signature") or ""),
            attribution=str(meta.get("attribution") or "unknown"),  # type: ignore[arg-type]
            evidence=[e for e in meta.get("evidence") or [] if isinstance(e, dict)],
            created_at=str(meta.get("created_at") or ""),
            source=str(meta.get("source") or "manual"),
        )

    def get_experience(self, name: str) -> ExperienceEntry | None:
        path = self._entry_path(name)
        if not path.is_file():
            return None
        return self._read_entry(path)

    def add_experience(self, entry: ExperienceEntry) -> Path:
        """写入一条经验。超上限或名称非法时抛 HarnessWriteError。"""
        if not _NAME_RE.match(entry.name):
            raise HarnessWriteError(
                f"非法的经验名 {entry.name!r}：只允许小写字母、数字、点、短横线、下划线"
            )
        if len(entry.description) > MAX_ENTRY_CHARS // 2:
            raise HarnessWriteError("经验描述过长")
        if len(entry.body) > MAX_BODY_CHARS:
            raise HarnessWriteError("经验正文过长")

        existing = self.list_experiences(include_retired=True)
        if not any(e.name == entry.name for e in existing) and len(existing) >= MAX_EXPERIENCES:
            raise HarnessWriteError(
                f"经验条数已达上限 {MAX_EXPERIENCES}，请先精简（/evolve retire <name>）"
            )

        self.snapshot(reason=f"experience:{entry.name}")
        self.experience_dir.mkdir(parents=True, exist_ok=True)
        if not entry.created_at:
            entry.created_at = _now()
        path = self._entry_path(entry.name)
        path.write_text(_render_doc(entry.to_frontmatter(), entry.body), encoding="utf-8")
        self.rebuild_index()
        return path

    def retire_experience(self, name: str) -> bool:
        """把经验标记为 retired（不删除，避免下次 distill 又重复提议）。"""
        entry = self.get_experience(name)
        if entry is None:
            return False
        entry.status = "retired"
        self.snapshot(reason=f"retire:{name}")
        self._entry_path(name).write_text(
            _render_doc(entry.to_frontmatter(), entry.body), encoding="utf-8"
        )
        self.rebuild_index()
        return True

    def rebuild_index(self) -> None:
        entries = self.list_experiences(include_retired=True)
        lines = ["# Harness 经验索引", ""]
        if not entries:
            lines.append("（暂无经验。运行 `/distill` 从失败轨迹中提炼。）")
        for entry in entries:
            suffix = " `retired`" if entry.status == "retired" else ""
            lines.append(
                f"- [{entry.description or entry.name}]({entry.name}.md) — {entry.name}{suffix}"
            )
        self.experience_dir.mkdir(parents=True, exist_ok=True)
        (self.experience_dir / INDEX_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def render_injection(self) -> str:
        """生成注入 system prompt 的经验段。空库时返回空串。"""
        entries = self.list_experiences()
        if not entries:
            return ""

        parts: list[str] = []
        used = 0
        for entry in entries:
            text = f"- {entry.description}"
            if entry.body:
                text += f"\n  {entry.body.strip()}"
            if used + len(text) > MAX_INJECT_CHARS:
                break
            parts.append(text)
            used += len(text)

        if not parts:
            return ""
        return "# Learned conventions for this project\n\n" + "\n".join(parts)

    # ---------------------------------------------------------------- 提议

    def save_proposal(self, proposal: Proposal) -> Path:
        if not _NAME_RE.match(proposal.id):
            raise HarnessWriteError(f"非法的提议 id: {proposal.id!r}")
        if not proposal.created_at:
            proposal.created_at = _now()
        self.proposals_dir.mkdir(parents=True, exist_ok=True)
        path = self.proposals_dir / f"{proposal.id}.md"
        body = proposal.body
        path.write_text(_render_doc(proposal.to_frontmatter(), body), encoding="utf-8")
        return path

    def list_proposals(self, status: str | None = "pending") -> list[Proposal]:
        result: list[Proposal] = []
        try:
            paths = sorted(self.proposals_dir.glob("*.md"))
        except OSError:
            return result
        for path in paths:
            if not path.is_file():
                continue
            try:
                meta, body = split_frontmatter(path.read_text(encoding="utf-8"))
            except OSError:
                continue
            if not meta:
                continue
            proposal = Proposal(
                id=str(meta.get("id") or path.stem),
                kind="prompt_patch" if meta.get("kind") == "prompt_patch" else "experience",
                target=str(meta.get("target") or ""),
                title=str(meta.get("title") or ""),
                body=body.strip(),
                rationale=str(meta.get("rationale") or ""),
                expected_effect=str(meta.get("expected_effect") or ""),
                risk=str(meta.get("risk") or "low"),
                signature=str(meta.get("signature") or ""),
                attribution=str(meta.get("attribution") or "unknown"),  # type: ignore[arg-type]
                evidence=[e for e in meta.get("evidence") or [] if isinstance(e, dict)],
                created_at=str(meta.get("created_at") or ""),
                status=(
                    meta["status"]
                    if meta.get("status") in {"pending", "accepted", "rejected"}
                    else "pending"
                ),
            )
            if status is None or proposal.status == status:
                result.append(proposal)
        return result

    def get_proposal(self, proposal_id: str) -> Proposal | None:
        for proposal in self.list_proposals(status=None):
            if proposal.id == proposal_id:
                return proposal
        return None

    def update_proposal_status(self, proposal: Proposal, status: str) -> None:
        proposal.status = status  # type: ignore[assignment]
        self.save_proposal(proposal)

    # ---------------------------------------------------------------- 版本

    def versions(self) -> list[HarnessVersion]:
        path = self.root / VERSIONS_NAME
        result: list[HarnessVersion] = []
        try:
            with path.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(data, dict):
                        continue
                    result.append(
                        HarnessVersion(
                            version=int(data.get("version") or 0),
                            created_at=str(data.get("created_at") or ""),
                            changes=list(data.get("changes") or []),
                            proposal_ids=list(data.get("proposal_ids") or []),
                            note=str(data.get("note") or ""),
                        )
                    )
        except OSError:
            return []
        return result

    def current_version(self) -> int:
        return max((v.version for v in self.versions()), default=0)

    def append_version(
        self, changes: list[str], proposal_ids: list[str], note: str = ""
    ) -> HarnessVersion:
        version = HarnessVersion(
            version=self.current_version() + 1,
            created_at=_now(),
            changes=changes,
            proposal_ids=proposal_ids,
            note=note,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / VERSIONS_NAME).open("a", encoding="utf-8") as f:
            f.write(json.dumps(version.to_dict(), ensure_ascii=False) + "\n")
        return version

    # ---------------------------------------------------------------- 快照

    def snapshot(self, reason: str = "") -> Path | None:
        """把 harness 目录整份复制到 ``.history/<stamp>/``。

        任何自动写入前都必须调用。没有可快照内容时返回 None。
        """
        if not self.root.is_dir():
            return None
        files = [p for p in self.root.rglob("*") if p.is_file()]
        # 排除历史目录自身，避免快照套娃
        files = [p for p in files if self.history_dir not in p.parents]
        if not files:
            return None

        dest_root = self.history_dir / _stamp()
        copied = 0
        for src in files:
            try:
                dest = dest_root / src.relative_to(self.root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                copied += 1
            except OSError as e:
                log.debug("[harness] snapshot failed for %s: %s", src, e)
        if reason:
            with contextlib.suppress(OSError):
                (dest_root / ".reason").write_text(reason, encoding="utf-8")
        self._prune_snapshots()
        return dest_root if copied else None

    def snapshots(self) -> list[Path]:
        try:
            return sorted(p for p in self.history_dir.iterdir() if p.is_dir())
        except OSError:
            return []

    def _prune_snapshots(self) -> None:
        snaps = self.snapshots()
        for old in snaps[:-MAX_VERSION_SNAPSHOTS] if MAX_VERSION_SNAPSHOTS > 0 else snaps:
            with contextlib.suppress(OSError):
                shutil.rmtree(old)

    def restore_snapshot(self, snapshot: Path | None = None) -> bool:
        """把 harness 目录回滚到某个快照（默认最近一次）。

        回滚前先把当前状态也快照一份，保证回滚本身可以撤销。
        """
        if snapshot is None:
            snaps = self.snapshots()
            if not snaps:
                return False
            snapshot = snaps[-1]
        if not snapshot.is_dir():
            return False

        self.snapshot(reason="pre-rollback")
        # 清掉当前内容（保留 .history）
        for child in list(self.root.iterdir()) if self.root.is_dir() else []:
            if child == self.history_dir:
                continue
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            except OSError as e:
                log.debug("[harness] rollback cleanup failed for %s: %s", child, e)

        for src in snapshot.rglob("*"):
            if not src.is_file() or src.name == ".reason":
                continue
            try:
                dest = self.root / src.relative_to(snapshot)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
            except OSError as e:
                log.debug("[harness] rollback copy failed for %s: %s", src, e)
        return True

    # ---------------------------------------------------------------- 元信息

    def ensure_scaffold(self) -> None:
        """创建目录骨架（幂等）。"""
        for d in (self.prompt_dir, self.experience_dir, self.proposals_dir):
            d.mkdir(parents=True, exist_ok=True)
        index = self.experience_dir / INDEX_NAME
        if not index.is_file():
            self.rebuild_index()

    def stats(self) -> dict[str, Any]:
        experiences = self.list_experiences(include_retired=True)
        active = [e for e in experiences if e.status == "active"]
        injection = self.render_injection()
        return {
            "version": self.current_version(),
            "experiences": len(active),
            "retired": len(experiences) - len(active),
            "max": MAX_EXPERIENCES,
            "inject_chars": len(injection),
            "inject_max": MAX_INJECT_CHARS,
            "pending": len(self.list_proposals()),
            "snapshots": len(self.snapshots()),
            "root": str(self.root),
        }
