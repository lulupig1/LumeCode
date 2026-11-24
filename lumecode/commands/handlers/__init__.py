from __future__ import annotations

from lumecode.commands.handlers.clear import CLEAR_COMMAND
from lumecode.commands.handlers.compact import COMPACT_COMMAND
from lumecode.commands.handlers.harness import HARNESS_COMMANDS
from lumecode.commands.handlers.help import HELP_COMMAND
from lumecode.commands.handlers.mcp import MCP_COMMAND
from lumecode.commands.handlers.memory import MEMORY_COMMAND
from lumecode.commands.handlers.permission import PERMISSION_COMMAND
from lumecode.commands.handlers.plan import PLAN_COMMAND
from lumecode.commands.handlers.rewind import REWIND_COMMAND
from lumecode.commands.handlers.sandbox import SANDBOX_COMMAND
from lumecode.commands.handlers.session import SESSION_COMMAND
from lumecode.commands.handlers.skill import SKILL_COMMAND
from lumecode.commands.handlers.status import STATUS_COMMAND
from lumecode.commands.registry import CommandRegistry

ALL_COMMANDS = [
    HELP_COMMAND,
    COMPACT_COMMAND,
    CLEAR_COMMAND,
    PLAN_COMMAND,
    SESSION_COMMAND,
    MCP_COMMAND,
    MEMORY_COMMAND,
    PERMISSION_COMMAND,
    SANDBOX_COMMAND,
    REWIND_COMMAND,
    STATUS_COMMAND,
    SKILL_COMMAND,
    *HARNESS_COMMANDS,
]


def register_all_commands(registry: CommandRegistry) -> None:
    for cmd in ALL_COMMANDS:
        registry.register_sync(cmd)
