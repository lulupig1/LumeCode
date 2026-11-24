"""Harness 自我改进子系统。

设计目标（对齐 Google Harness² 的「不碰权重，只改外壳」思路）：

1. **基础层只读**：``.lumecode/harness/prompt/*.md`` 只能由人修改，
   Agent 永远不写这里。
2. **经验层可写**：``.lumecode/harness/experience/`` 由 ``/evolve accept``
   写入，每条都必须能溯源到具体 span。
3. **写入前必快照**：所有自动写入都先做整目录快照，``/evolve rollback``
   可退回。
4. **有上限**：条数、总字符、单轮改动数都在代码里卡死，不依赖提示词约束。
"""

from lumecode.harness.models import ExperienceEntry, HarnessVersion, Proposal
from lumecode.harness.store import (
    MAX_EXPERIENCES,
    MAX_INJECT_CHARS,
    HarnessStore,
    HarnessWriteError,
)

__all__ = [
    "MAX_EXPERIENCES",
    "MAX_INJECT_CHARS",
    "ExperienceEntry",
    "HarnessStore",
    "HarnessVersion",
    "HarnessWriteError",
    "Proposal",
]
