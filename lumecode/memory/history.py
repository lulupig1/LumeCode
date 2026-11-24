"""记忆文件历史快照。

记忆会被两条路径改写：``MemoryManager.extract``（追加新事实）和后台
dream 整理（合并/删除，BY PASS 权限的子 Agent 直接写文件）。两者都可能
改坏一条记忆，因此每次写入前保留旧版本，出问题时可回溯。

布局（都在记忆目录内，扫描记忆时必须跳过）::

    .history/
      <stem>/<UTC 时间戳>.md        # 单个文件被改写前的快照，保留最近 N 份
      dream/<UTC 时间戳>/<相对路径>  # 每次 dream 前的整目录快照，保留最近 N 次
"""

from __future__ import annotations

import contextlib
import logging
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

HISTORY_DIR_NAME = ".history"
MAX_SNAPSHOTS_PER_FILE = 5
MAX_DREAM_SNAPSHOTS = 5
ENTRYPOINT_NAME = "MEMORY.md"


def history_dir(mem_dir: str | Path) -> Path:
    return Path(mem_dir) / HISTORY_DIR_NAME


def is_hidden_rel(rel: Path) -> bool:
    """记忆目录内的相对路径是否落在隐藏子目录（如 .history）里。"""
    return any(part.startswith(".") for part in rel.parts)


def iter_memory_files(mem_dir: str | Path, *, include_index: bool = False) -> Iterator[Path]:
    """遍历记忆目录里的 .md 文件，跳过 `.history` 等隐藏目录。"""
    root = Path(mem_dir)
    if not root.is_dir():
        return
    try:
        candidates = sorted(root.rglob("*.md"))
    except OSError:
        return
    for path in candidates:
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if is_hidden_rel(rel):
            continue
        if not include_index and path.name == ENTRYPOINT_NAME:
            continue
        if path.is_file():
            yield path


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def snapshot_file(
    mem_dir: str | Path,
    file_path: str | Path,
    *,
    keep: int = MAX_SNAPSHOTS_PER_FILE,
) -> Path | None:
    """把文件的当前内容复制到 `.history/<stem>/<时间戳>.md`。

    文件不存在或复制失败时返回 None——快照是尽力而为，绝不能打断写入。
    """
    root = Path(mem_dir)
    src = Path(file_path)
    try:
        rel = src.relative_to(root)
    except ValueError:
        rel = Path(src.name)
    dest_dir = history_dir(root) / rel.parent / rel.stem
    dest = dest_dir / f"{_stamp()}{rel.suffix or '.md'}"
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    except OSError as e:
        logger.debug("memory snapshot failed for %s: %s", src, e)
        return None
    _prune_dir(dest_dir, keep)
    return dest


def snapshot_tree(
    mem_dir: str | Path,
    *,
    keep: int = MAX_DREAM_SNAPSHOTS,
) -> Path | None:
    """在 dream 整理前对整个记忆目录做一次快照（含 MEMORY.md）。

    返回快照目录；没有可快照的文件时返回 None。
    """
    root = Path(mem_dir)
    if not root.is_dir():
        return None
    files = list(iter_memory_files(root, include_index=True))
    if not files:
        return None

    stamp = _stamp()
    dest_root = history_dir(root) / "dream" / stamp
    copied = 0
    for src in files:
        dest = dest_root / src.relative_to(root)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied += 1
        except OSError as e:
            logger.debug("dream snapshot failed for %s: %s", src, e)
    _prune_children(history_dir(root) / "dream", keep)
    return dest_root if copied else None


def clear_history(mem_dir: str | Path) -> None:
    """删除记忆目录下的全部快照（`/memory clear` 时调用）。"""
    target = history_dir(mem_dir)
    if not target.is_dir():
        return
    try:
        shutil.rmtree(target)
    except OSError as e:
        logger.debug("failed to clear memory history at %s: %s", target, e)


def _prune_dir(dest_dir: Path, keep: int) -> None:
    try:
        snapshots = sorted(dest_dir.glob("*.md"))
    except OSError:
        return
    for old in snapshots[:-keep] if keep > 0 else snapshots:
        with contextlib.suppress(OSError):
            old.unlink()


def _prune_children(parent: Path, keep: int) -> None:
    try:
        children = sorted(p for p in parent.iterdir() if p.is_dir())
    except OSError:
        return
    for old in children[:-keep] if keep > 0 else children:
        with contextlib.suppress(OSError):
            shutil.rmtree(old)
