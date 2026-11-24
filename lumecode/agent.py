from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from lumecode.client import LLMClient
from lumecode.context import (
    CompactBoundary,
    CompactCircuitBreaker,
    CompactEvent,
    ContentReplacementState,
    RecoveryState,
    append_replacement_records,
    apply_tool_result_budget,
    auto_compact,
    create_replacement_state,
    ensure_session_dir,
)
from lumecode.conversation import ConversationManager, ToolResultBlock, ToolUseBlock
from lumecode.conversation import ThinkingBlock as ConvThinkingBlock
from lumecode.harness.store import HarnessStore
from lumecode.hooks import HookContext, HookEngine
from lumecode.memory.auto_memory import MemoryManager
from lumecode.permissions import (
    PermissionChecker,
    PermissionMode,
)
from lumecode.prompts import (
    build_environment_context,
    build_plan_mode_reminder,
    build_system_prompt,
)
from lumecode.tasks import spawn_background
from lumecode.telemetry import SpanKind, SpanStatus, TelemetryRecorder, Usage
from lumecode.tools import ToolRegistry
from lumecode.tools.base import (
    MAX_OUTPUT_CHARS,
    StreamEnd,
    StreamEvent,
    TextDelta,
    ThinkingComplete,
    ThinkingDelta,
    ToolCallComplete,
    ToolCallDelta,
    ToolCallStart,
    ToolResult,
)

log = logging.getLogger(__name__)

MEMORY_EXTRACTION_INTERVAL = 4
# 两次提取之间至少要有这么多条新消息，避免短会话/空转时反复付一次 LLM 调用。
MEMORY_EXTRACTION_MIN_MESSAGES = 4
MAX_TOKENS_CEILING = 64000
MAX_OUTPUT_TOKENS_RECOVERIES = 3


# ---------------------------------------------------------------------------
# AgentEvent 事件类型
# ---------------------------------------------------------------------------


@dataclass
class StreamText:
    text: str


@dataclass
class ThinkingText:
    text: str


@dataclass
class RetryEvent:
    reason: str
    wait: float = 0.0


@dataclass
class ToolUseEvent:
    tool_name: str
    tool_id: str
    arguments: dict[str, Any]


@dataclass
class ToolResultEvent:
    tool_id: str
    tool_name: str
    output: str
    is_error: bool
    elapsed: float


@dataclass
class TurnComplete:
    turn: int


@dataclass
class LoopComplete:
    total_turns: int


@dataclass
class UsageEvent:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


@dataclass
class ErrorEvent:
    message: str


@dataclass
class CompactNotification:
    before_tokens: int
    message: str
    # 结构化 boundary（摘要 + 原文保留尾部），UI/session 层用它持久化 compact_boundary 记录。
    # 失败路径下为 None。
    boundary: CompactBoundary | None = None


@dataclass
class HookEvent:
    hook_id: str
    event: str
    output: str
    success: bool


class PermissionResponse(Enum):
    ALLOW = "allow"
    DENY = "deny"
    ALLOW_ALWAYS = "allow_always"


@dataclass
class PermissionRequest:
    tool_name: str
    description: str
    future: asyncio.Future[PermissionResponse]


AgentEvent = (
    StreamText
    | ThinkingText
    | RetryEvent
    | ToolUseEvent
    | ToolResultEvent
    | TurnComplete
    | LoopComplete
    | UsageEvent
    | ErrorEvent
    | PermissionRequest
    | CompactNotification
    | HookEvent
)


# ---------------------------------------------------------------------------
# LLM 响应收集器
# ---------------------------------------------------------------------------


@dataclass
class ThinkingBlock:
    thinking: str
    signature: str


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCallComplete] = field(default_factory=list)
    thinking_blocks: list[ThinkingBlock] = field(default_factory=list)
    stop_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_creation: int = 0


class StreamCollector:
    def __init__(self) -> None:
        self.response = LLMResponse()

    async def consume(self, stream: AsyncIterator[StreamEvent]) -> AsyncIterator[AgentEvent]:
        async for event in stream:
            if isinstance(event, TextDelta):
                self.response.text += event.text
                yield StreamText(text=event.text)
            elif isinstance(event, ThinkingDelta):
                yield ThinkingText(text=event.text)
            elif isinstance(event, ThinkingComplete):
                self.response.thinking_blocks.append(
                    ThinkingBlock(thinking=event.thinking, signature=event.signature)
                )
            elif isinstance(event, (ToolCallStart, ToolCallDelta)):
                pass
            elif isinstance(event, ToolCallComplete):
                self.response.tool_calls.append(event)
                yield ToolUseEvent(
                    tool_name=event.tool_name,
                    tool_id=event.tool_id,
                    arguments=event.arguments,
                )
            elif isinstance(event, StreamEnd):
                self.response.stop_reason = event.stop_reason
                self.response.input_tokens = event.input_tokens
                self.response.output_tokens = event.output_tokens
                self.response.cache_read = event.cache_read
                self.response.cache_creation = event.cache_creation


# ---------------------------------------------------------------------------
# tool 批量执行
# ---------------------------------------------------------------------------


@dataclass
class ToolBatch:
    concurrent: bool
    calls: list[ToolCallComplete]


def partition_tool_calls(
    tool_calls: list[ToolCallComplete],
    registry: ToolRegistry,
) -> list[ToolBatch]:
    batches: list[ToolBatch] = []
    for tc in tool_calls:
        tool = registry.get(tc.tool_name)
        safe = tool is not None and tool.is_concurrency_safe and registry.is_enabled(tc.tool_name)

        if safe and batches and batches[-1].concurrent:
            batches[-1].calls.append(tc)
        else:
            batches.append(ToolBatch(concurrent=safe, calls=[tc]))
    return batches


# ---------------------------------------------------------------------------
# streaming 执行器 — 在 LLM streaming 期间启动 tool 执行
# ---------------------------------------------------------------------------


@dataclass
class _ToolExecResult:
    tool_id: str
    tool_name: str
    result: ToolResult
    elapsed: float
    is_unknown: bool


class StreamingExecutor:
    def __init__(self) -> None:
        self._tasks: list[tuple[int, asyncio.Task[_ToolExecResult]]] = []
        self._order = 0

    def submit(
        self,
        coro: Any,
    ) -> None:
        task = asyncio.create_task(coro)
        self._tasks.append((self._order, task))
        self._order += 1

    async def collect_results(self) -> list[_ToolExecResult]:
        if not self._tasks:
            return []
        tasks = [t for _, t in sorted(self._tasks, key=lambda x: x[0])]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: list[_ToolExecResult] = []
        for r in results:
            if isinstance(r, BaseException):
                out.append(
                    _ToolExecResult(
                        tool_id="",
                        tool_name="",
                        result=ToolResult(output=f"Tool execution error: {r}", is_error=True),
                        elapsed=0.0,
                        is_unknown=False,
                    )
                )
            else:
                out.append(r)
        return out


# ---------------------------------------------------------------------------
# 工具执行用的读写锁
# ---------------------------------------------------------------------------


class _AsyncRWLock:
    """FIFO 读写锁，用于让 `is_concurrency_safe` 真正生效。

    并发安全的工具（ReadFile/Grep/Glob 等）获取共享的**读锁**，可以同时执行；
    非并发安全的工具（WriteFile/EditFile/Bash/Agent 等）获取独占的**写锁**。

    获取顺序严格 FIFO，因此一个写请求会挡住它之后到达的读请求。这条顺序保证
    正是"流式执行"能保持正确性的关键：模型若先发 WriteFile 再发 ReadFile，
    后者会排队等前者写完，而不是并发读到旧内容。等价于 codex 中
    `ToolCallRuntime` 用 RwLock 做并行闸门的做法。
    """

    def __init__(self) -> None:
        self._readers = 0
        self._writer = False
        self._waiters: deque[tuple[bool, asyncio.Future[None]]] = deque()

    @asynccontextmanager
    async def read(self) -> AsyncIterator[None]:
        await self._acquire(exclusive=False)
        try:
            yield
        finally:
            self._release(exclusive=False)

    @asynccontextmanager
    async def write(self) -> AsyncIterator[None]:
        await self._acquire(exclusive=True)
        try:
            yield
        finally:
            self._release(exclusive=True)

    async def _acquire(self, exclusive: bool) -> None:
        # 快路径：没有排队者时直接获取，避免创建 Future。
        if not self._waiters:
            if exclusive:
                if not self._writer and self._readers == 0:
                    self._writer = True
                    return
            elif not self._writer:
                self._readers += 1
                return
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        self._waiters.append((exclusive, fut))
        await fut

    def _release(self, exclusive: bool) -> None:
        if exclusive:
            self._writer = False
        else:
            self._readers -= 1
        self._wake()

    def _wake(self) -> None:
        while self._waiters:
            exclusive, fut = self._waiters[0]
            if fut.done():
                # 等待者已被取消，丢弃后继续。
                self._waiters.popleft()
                continue
            if exclusive:
                if self._writer or self._readers > 0:
                    return
                self._writer = True
                self._waiters.popleft()
                fut.set_result(None)
                return
            if self._writer:
                return
            self._readers += 1
            self._waiters.popleft()
            fut.set_result(None)
            # 继续循环，把排在这个读者之后、写者之前的所有读者一并放行。


# ---------------------------------------------------------------------------
# Agent 主循环
# ---------------------------------------------------------------------------


class Agent:
    def __init__(
        self,
        client: LLMClient,
        registry: ToolRegistry,
        protocol: str,
        work_dir: str = ".",
        max_iterations: int = 0,
        permission_checker: PermissionChecker | None = None,
        context_window: int = 200_000,
        instructions_content: str = "",
        memory_manager: MemoryManager | None = None,
        hook_engine: HookEngine | None = None,
        telemetry: TelemetryRecorder | None = None,
        memory_extraction_interval: int | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        self.protocol = protocol
        self.work_dir = work_dir
        self.max_iterations = max_iterations
        self.permission_checker = permission_checker
        self.permission_mode: PermissionMode = (
            permission_checker.mode if permission_checker else PermissionMode.DEFAULT
        )
        self.context_window = context_window
        self.session_dir = ensure_session_dir(work_dir)
        self.compact_breaker = CompactCircuitBreaker()
        self.replacement_state: ContentReplacementState = create_replacement_state()
        # 流式工具执行的并发闸门：并发安全的工具走读锁共享执行，
        # 非并发安全的工具走写锁独占执行（见 _AsyncRWLock 说明）。
        self._tool_lock = _AsyncRWLock()
        # 保存重建工作上下文所需的快照，在 Layer 2 压缩对话后使用：
        # 最近的文件读取和 skill 调用。每次 ReadFile / skill 调用时记录，
        # auto_compact 触发阈值时消费。
        self.recovery_state: RecoveryState = RecoveryState()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_creation_tokens = 0
        self.instructions_content = instructions_content
        self.memory_manager = memory_manager
        # harness 自我改进：基础提示词分节与经验库都从这里读写。只读加载，
        # 没有 .lumecode/harness/ 目录时提示词输出与旧版本逐字相同。
        self.harness = HarnessStore(work_dir)
        # 每 N 轮对话触发一次记忆提取。1 = 每轮都提（旧行为）；0 = 只在
        # 会话结束/退出时提取。
        self.memory_extraction_interval = (
            MEMORY_EXTRACTION_INTERVAL
            if memory_extraction_interval is None
            else max(0, memory_extraction_interval)
        )
        self.hook_engine = hook_engine
        # 本地 telemetry。为空时所有埋点都是 no-op，不影响现有行为。
        self.telemetry = telemetry
        # 当前 run 的根 span；供子 Agent / 工具显式挂载父节点。
        self._run_span: Any | None = None
        self._loop_count = 0
        # 记忆提取合并策略（对齐 Go 版 inProgress + pendingContext）：
        # _extracting: 标记是否有提取正在进行
        # _pending_extraction: 提取期间又触发了新请求，标记需要尾随提取
        self._extracting = False
        self._pending_extraction = False
        self._consolidator: MemoryConsolidator | None = None
        if memory_manager is not None:
            from lumecode.memory.consolidation import MemoryConsolidator

            self._consolidator = MemoryConsolidator(work_dir)
        self.session_id: str = ""
        # 子 agent 不写主会话文件（见 _start_session_writer 的说明）。
        self.write_session: bool = True
        self.active_skills: dict[str, str] = {}
        self._skill_catalog: str = ""
        self._agent_catalog: str = ""
        self._agent_catalog_list: list[tuple[str, str]] = []
        self.agent_id: str = uuid.uuid4().hex[:12]
        self.parent_id: str | None = None
        self.trace_id: str | None = None
        self.coordinator_mode: bool = False
        self.team_name: str = ""
        self._team_manager: Any = None
        # Coordinator 模式下被替换前的全量工具注册表快照（见 team_create）。
        self._full_registry: ToolRegistry | None = None
        # fork 子 Agent 启动时传入的父对话上下文（见 agent_tool 的 is_fork）。
        self._fork_conversation: ConversationManager | None = None
        self.notification_fn: Callable[[], list[str]] | None = None
        self.file_history: Any = None

        # 非阻塞 memory recall：prefetch task 与主 LLM 调用并行。
        # recall_first_wait 是首轮模型调用前的最长等待（秒）；0 表示完全不等。
        self.memory_recall_task: Any | None = None
        self._memory_recall_consumed: bool = False
        self.recall_first_wait: float = 0.0
        # 本次 run 已注入的记忆文件路径，跨轮去重。
        self._surfaced_memory_paths: set[str] = set()

    @property
    def _transcript_path(self) -> str:
        if self.session_id:
            return str(Path(self.work_dir) / ".lumecode" / "sessions" / f"{self.session_id}.jsonl")
        return ""

    @property
    def plan_mode(self) -> bool:
        return self.permission_mode == PermissionMode.PLAN

    @property
    def current_conversation(self) -> ConversationManager | None:
        """当前正在运行的对话（fork 型 skill 需要读取它来构造上下文）。"""
        return getattr(self, "_current_conversation", None)

    _plan_path_cache: Path | None = None

    def _get_plan_path(self) -> Path:
        if self._plan_path_cache is not None:
            return self._plan_path_cache
        import datetime
        import random

        _ADJECTIVES = [
            "bold",
            "bright",
            "calm",
            "cool",
            "deep",
            "fair",
            "fast",
            "fine",
            "glad",
            "keen",
            "kind",
            "lean",
            "mild",
            "neat",
            "pure",
            "safe",
            "slim",
            "soft",
            "tall",
            "warm",
            "wise",
            "grand",
            "swift",
            "vivid",
        ]
        _NOUNS = [
            "sketch",
            "draft",
            "spark",
            "bloom",
            "trail",
            "ridge",
            "creek",
            "grove",
            "cliff",
            "cloud",
            "field",
            "forge",
            "frost",
            "haven",
            "pearl",
            "stone",
            "storm",
            "river",
            "tower",
            "delta",
            "flame",
            "orbit",
            "pulse",
            "shore",
        ]
        plans_dir = Path(self.work_dir) / ".lumecode" / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%m%d-%H%M")
        slug = f"{random.choice(_ADJECTIVES)}-{random.choice(_NOUNS)}-{ts}"
        self._plan_path_cache = plans_dir / f"{slug}.md"
        return self._plan_path_cache

    def set_permission_mode(self, mode: PermissionMode) -> None:
        self.permission_mode = mode
        if self.permission_checker:
            self.permission_checker.mode = mode

    def activate_skill(self, name: str, prompt_body: str) -> None:
        self.active_skills[name] = prompt_body

    def clear_active_skills(self) -> None:
        self.active_skills.clear()

    def set_skill_catalog(self, catalog: str) -> None:
        self._skill_catalog = catalog

    def set_agent_catalog(
        self, catalog: str, catalog_list: list[tuple[str, str]] | None = None
    ) -> None:
        self._agent_catalog = catalog
        if catalog_list is not None:
            self._agent_catalog_list = catalog_list

    def _build_hook_context(self, event: str, **kwargs: Any) -> HookContext:
        return HookContext(
            event_name=event,
            tool_name=str(kwargs.get("tool_name", "")),
            tool_args=kwargs.get("tool_args", {}),
            file_path=str(kwargs.get("file_path", "")),
            message=str(kwargs.get("message", "")),
            error=str(kwargs.get("error", "")),
        )

    def _infer_file_path(self, args: dict) -> str:
        return str(args.get("file_path", args.get("path", "")))

    def _drain_hook_events(self) -> list[HookEvent]:
        if not self.hook_engine:
            return []
        return [
            HookEvent(
                hook_id=n.hook_id,
                event=n.event,
                output=n.output,
                success=n.success,
            )
            for n in self.hook_engine.drain_notifications()
        ]

    # ------------------------------------------------------------------
    # telemetry helpers
    # ------------------------------------------------------------------

    def _model_name(self) -> str:
        model = getattr(self.client, "model", "")
        return str(model) if model else "unknown"

    def _accumulate_usage(self, response: LLMResponse) -> Usage:
        """累计 token 用量，并返回本次调用（而非累计值）的 Usage。"""
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.total_cache_read_tokens += response.cache_read
        self.total_cache_creation_tokens += response.cache_creation
        return Usage(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cache_read_tokens=response.cache_read,
            cache_creation_tokens=response.cache_creation,
        )

    def _start_generation_span(self, conversation: ConversationManager) -> Any | None:
        if self.telemetry is None:
            return None
        model = self._model_name()
        last = conversation.history[-1].content if conversation.history else None
        return self.telemetry.start_span(
            SpanKind.GENERATION,
            "llm",
            parent=self._run_span,
            model=model,
            attributes={
                "gen_ai.system": self.protocol,
                "gen_ai.operation.name": "chat",
            },
            input=last,
            set_current=False,
        )

    def _finish_generation_span(
        self, span: Any | None, response: LLMResponse, usage: Usage
    ) -> None:
        if span is None:
            return
        span.end(
            usage=usage,
            output=response.text,
            attributes={"gen_ai.response.stop_reason": response.stop_reason},
        )

    def _start_tool_span(self, tc: ToolCallComplete) -> Any | None:
        if self.telemetry is None:
            return None
        return self.telemetry.start_span(
            SpanKind.TOOL,
            tc.tool_name,
            parent=self._run_span,
            attributes={
                "gen_ai.tool.name": tc.tool_name,
                "gen_ai.tool.call.id": tc.tool_id,
            },
            input=tc.arguments,
            set_current=False,
        )

    def _end_tool_span(
        self, span: Any | None, result: ToolResult, elapsed: float | None = None
    ) -> None:
        if span is None:
            return
        attrs: dict[str, Any] = {"tool.is_error": result.is_error}
        if elapsed is not None:
            attrs["tool.elapsed_ms"] = round(elapsed * 1000, 3)
        span.end(
            status=SpanStatus.ERROR if result.is_error else SpanStatus.OK,
            attributes=attrs,
            output=result.output,
            error=result.output if result.is_error else None,
        )

    async def run(self, conversation: ConversationManager) -> AsyncIterator[AgentEvent]:
        """运行 Agent，并为整轮执行建立一个根 span。

        埋点关闭时直接走 ``_run_impl``，不引入额外对象。
        """
        # 每次请求重新开始一轮记忆去重。跨请求的 task 状态在首轮 drain 时
        # 清理——如果上一轮的召回没赶上，本轮首轮的 block 等待会消费它。
        self._surfaced_memory_paths = set()
        self._memory_recall_consumed = False

        if self.telemetry is None:
            async for event in self._run_impl(conversation):
                yield event
            return

        root = self.telemetry.start_span(
            SpanKind.AGENT,
            f"agent:{self.agent_id[:8]}",
            parent=self._run_span,
            attributes={"agent.id": self.agent_id, "agent.type": "lead"},
            input=conversation.history[-1].content if conversation.history else None,
        )
        self._run_span = root
        try:
            async for event in self._run_impl(conversation):
                yield event
        except GeneratorExit:
            root.end(status=SpanStatus.CANCELLED)
            raise
        except BaseException as e:
            root.end(status=SpanStatus.ERROR, error=str(e))
            raise
        finally:
            root.end()
            self._run_span = None

    def _start_session_writer(self, conversation: ConversationManager) -> None:
        """无 UI 路径（remote / headless）的写穿钩子。

        交互式路径由 App 在会话切换时安装 sink；remote 和后台子 agent 不经过
        App，需要在这里把当前 conversation 接到会话文件上。子 agent 故意不写：
        它们的对话进入主 agent 的 fork/subagent 工具结果，独立落盘会产生
        无人恢复的孤儿会话。
        """
        if not self.write_session or not self.session_id or self.parent_id:
            return
        if conversation.has_sink:
            # App / remote 已经装好 sink，重复安装会为同一会话文件再开一个句柄。
            return
        try:
            from lumecode.memory.session import SessionManager

            manager = SessionManager(self.work_dir)
            session = manager.resume(self.session_id)
            if session is None:
                return
            conversation.attach_sink(session.session.on_message, session.session.on_replace)
        except Exception:
            log.debug("session writer attach failed", exc_info=True)

    async def _run_impl(self, conversation: ConversationManager) -> AsyncIterator[AgentEvent]:
        self._current_conversation = conversation
        if not conversation.turn_id:
            conversation.turn_id = f"turn-{uuid.uuid4().hex[:12]}"
        self._start_session_writer(conversation)
        env_context = build_environment_context(
            self.work_dir, self.active_skills, self._skill_catalog, self._agent_catalog
        )
        conversation.inject_environment(env_context)

        memory_content = self.memory_manager.load() if self.memory_manager else ""
        conversation.inject_long_term_memory(self.instructions_content, memory_content)

        if self.hook_engine:
            ctx = self._build_hook_context("session_start")
            await self.hook_engine.run_hooks("session_start", ctx)
            for he in self._drain_hook_events():
                yield he

        iteration = 0
        consecutive_unknown = 0
        max_tokens_escalated = False
        output_recoveries = 0

        while True:
            iteration += 1

            if self.max_iterations > 0 and iteration > self.max_iterations:
                yield ErrorEvent(
                    message=f"Agent reached maximum iterations ({self.max_iterations})"
                )
                break

            if self.hook_engine:
                ctx = self._build_hook_context("turn_start")
                await self.hook_engine.run_hooks("turn_start", ctx)
                for he in self._drain_hook_events():
                    yield he

            self._consume_mailbox(conversation)
            await self._drain_memory_recall(
                conversation,
                block=(iteration == 1 and self.recall_first_wait > 0),
            )
            if self.notification_fn:
                for note in self.notification_fn():
                    conversation.add_system_reminder(note)

            if self.hook_engine:
                ctx = self._build_hook_context("pre_send")
                await self.hook_engine.run_hooks("pre_send", ctx)
                for he in self._drain_hook_events():
                    yield he

            hook_prompts = self.hook_engine.get_prompt_messages() if self.hook_engine else None
            system = build_system_prompt(
                hook_prompts=hook_prompts,
                coordinator_mode=self.coordinator_mode,
                agent_catalog=self._agent_catalog_list or None,
                work_dir=self.work_dir,
                harness=self.harness,
            )

            if self.plan_mode:
                plan_path = str(self._get_plan_path())
                if self.permission_checker:
                    self.permission_checker.plan_file_path = plan_path
                plan_exists = self._get_plan_path().exists()
                plan_reminder = build_plan_mode_reminder(plan_path, plan_exists, iteration)
                conversation.add_system_reminder(plan_reminder)

            if self.hook_engine:
                for hook_note in self.hook_engine.drain_notifications():
                    conversation.add_system_reminder(
                        f"Hook [{hook_note.hook_id}] {hook_note.event}: {hook_note.output}"
                    )

            deferred_names = self.registry.get_deferred_tool_names()
            if deferred_names:
                conversation.add_system_reminder(
                    "The following deferred tools are available via ToolSearch. "
                    "Their schemas are NOT loaded - use ToolSearch with "
                    'query "select:<name>[,<name>...]" to load tool schemas before calling them:\n'
                    + "\n".join(deferred_names)
                )

            tools = self.registry.get_all_schemas(self.protocol)

            # Layer 1: apply tool-result budget（就地修改 conversation）
            new_records = apply_tool_result_budget(
                conversation, self.session_dir, self.replacement_state
            )
            if new_records:
                append_replacement_records(self.session_dir, new_records)

            # Layer 2: 接近 context window 上限时自动 compact
            # tool-result budget 已就地修改 conversation，直接用 conversation.history 估算
            compact_result = await auto_compact(
                conversation,
                self.client,
                self.context_window,
                self.session_dir,
                protocol=self.protocol,
                breaker=self.compact_breaker,
                recovery=self.recovery_state,
                tool_schemas=self.registry.get_all_schemas(self.protocol),
                transcript_path=self._transcript_path,
                telemetry=self.telemetry,
            )
            if isinstance(compact_result, CompactEvent):
                yield CompactNotification(
                    before_tokens=compact_result.before_tokens,
                    message=f"上下文已压缩（压缩前 {compact_result.before_tokens:,} tokens）",
                    boundary=compact_result.boundary,
                )
                conversation.inject_environment(env_context)
                mem = self.memory_manager.load() if self.memory_manager else ""
                conversation.inject_long_term_memory(self.instructions_content, mem)
                # 压缩后重新应用 budget（就地修改）
                apply_tool_result_budget(conversation, self.session_dir, self.replacement_state)
            elif isinstance(compact_result, str):
                yield ErrorEvent(message=compact_result)

            collector = StreamCollector()
            executor = StreamingExecutor()
            deferred_tool_calls: list[ToolCallComplete] = []
            generation_span = self._start_generation_span(conversation)
            try:
                llm_stream = self.client.stream(conversation, system=system, tools=tools)
                async for event in collector.consume(llm_stream):
                    # 流式工具执行：收到完整 tool_use 就立刻提交执行，不等整个响应结束
                    if isinstance(event, ToolUseEvent):
                        tc = collector.response.tool_calls[-1]
                        # 需要交互式权限确认的工具延迟到流结束后顺序执行
                        tool = self.registry.get(tc.tool_name)
                        needs_ask = False
                        if tool and self.permission_checker:
                            decision = self.permission_checker.check(tool, tc.arguments)
                            needs_ask = decision.effect == "ask"
                        if needs_ask:
                            deferred_tool_calls.append(tc)
                        else:
                            executor.submit(self._execute_streaming_tool(tc))
                    yield event
            except GeneratorExit:
                if generation_span is not None:
                    generation_span.end(status=SpanStatus.CANCELLED)
                raise
            except BaseException as e:
                if generation_span is not None:
                    generation_span.end(status=SpanStatus.ERROR, error=str(e))
                raise

            response = collector.response
            call_usage = self._accumulate_usage(response)
            self._finish_generation_span(generation_span, response, call_usage)

            if self.hook_engine:
                ctx = self._build_hook_context("post_receive", message=response.text)
                await self.hook_engine.run_hooks("post_receive", ctx)
                for he in self._drain_hook_events():
                    yield he

            yield UsageEvent(
                input_tokens=self.total_input_tokens,
                output_tokens=self.total_output_tokens,
                cache_read_tokens=self.total_cache_read_tokens,
                cache_creation_tokens=self.total_cache_creation_tokens,
            )

            conv_thinking = [
                ConvThinkingBlock(thinking=tb.thinking, signature=tb.signature)
                for tb in response.thinking_blocks
            ]

            if response.stop_reason == "max_tokens":
                if not max_tokens_escalated:
                    self.client.set_max_output_tokens(MAX_TOKENS_CEILING)
                    max_tokens_escalated = True
                    if response.text:
                        conversation.add_assistant_message(
                            response.text, thinking_blocks=conv_thinking
                        )
                        conversation.add_user_message(
                            "Output token limit hit. Resume directly from where you stopped. "
                            "Do not apologize or repeat previous content. Pick up mid-thought if needed."
                        )
                    yield RetryEvent(reason="max_tokens escalation")
                    continue
                elif output_recoveries < MAX_OUTPUT_TOKENS_RECOVERIES:
                    output_recoveries += 1
                    conversation.add_assistant_message(response.text, thinking_blocks=conv_thinking)
                    conversation.add_user_message(
                        "Output token limit hit. Resume directly from where you stopped. "
                        "Break remaining work into smaller pieces."
                    )
                    yield RetryEvent(
                        reason=f"max_tokens recovery {output_recoveries}/{MAX_OUTPUT_TOKENS_RECOVERIES}"
                    )
                    continue
            else:
                output_recoveries = 0

            if not response.tool_calls:
                conversation.add_assistant_message(response.text, thinking_blocks=conv_thinking)
                self._loop_count += 1
                self._maybe_trigger_extraction(conversation)
                if self._consolidator is not None:
                    spawn_background(
                        self._consolidator.maybe_run(
                            self.client,
                            conversation,
                            self.protocol,
                            telemetry=self.telemetry,
                        )
                    )
                if self.hook_engine:
                    ctx = self._build_hook_context("turn_end")
                    await self.hook_engine.run_hooks("turn_end", ctx)
                    ctx = self._build_hook_context("session_end")
                    await self.hook_engine.run_hooks("session_end", ctx)
                    for he in self._drain_hook_events():
                        yield he
                if self.file_history is not None:
                    summary = (
                        response.text[:60] + "..." if len(response.text) > 60 else response.text
                    )
                    self.file_history.make_snapshot(len(conversation.history), summary)
                yield LoopComplete(total_turns=iteration)
                break

            tool_uses = [
                ToolUseBlock(
                    tool_use_id=tc.tool_id,
                    tool_name=tc.tool_name,
                    arguments=tc.arguments,
                )
                for tc in response.tool_calls
            ]
            conversation.add_assistant_message(
                response.text, tool_uses, thinking_blocks=conv_thinking
            )
            # 在 assistant 回复加入历史后锚定实际用量：基线（input + cache + output）
            # 覆盖到当前位置，因此下一轮迭代顶部的 auto-compact 检查只需对
            # 接下来追加的 tool results 做字符估算。
            conversation.record_usage_anchor(
                response.input_tokens,
                response.output_tokens,
                response.cache_read,
                response.cache_creation,
            )

            # 收集流式执行器中已提交的工具结果（工具在 LLM 流式输出期间已开始执行）
            tool_results: list[ToolResultBlock] = []
            streaming_results = await executor.collect_results()

            for br in streaming_results:
                if br.is_unknown:
                    consecutive_unknown += 1
                else:
                    consecutive_unknown = 0
                content = self._maybe_persist_or_truncate(br.tool_id, br.result.output)
                tool_results.append(
                    ToolResultBlock(
                        tool_use_id=br.tool_id,
                        content=content,
                        is_error=br.result.is_error,
                    )
                )
                yield ToolResultEvent(
                    tool_id=br.tool_id,
                    tool_name=br.tool_name,
                    output=br.result.output,
                    is_error=br.result.is_error,
                    elapsed=br.elapsed,
                )

            # 需要交互式权限确认的工具，在流结束后顺序执行
            for tc in deferred_tool_calls:
                result: ToolResult | None = None
                elapsed = 0.0
                is_unknown = False

                async for item in self._execute_tool(tc):
                    if isinstance(item, PermissionRequest):
                        yield item
                    else:
                        result, elapsed, is_unknown = item

                if result is None:
                    result = ToolResult(output="Error: no result from tool", is_error=True)

                if is_unknown:
                    consecutive_unknown += 1
                else:
                    consecutive_unknown = 0

                content = self._maybe_persist_or_truncate(tc.tool_id, result.output)
                tool_results.append(
                    ToolResultBlock(
                        tool_use_id=tc.tool_id,
                        content=content,
                        is_error=result.is_error,
                    )
                )
                yield ToolResultEvent(
                    tool_id=tc.tool_id,
                    tool_name=tc.tool_name,
                    output=result.output,
                    is_error=result.is_error,
                    elapsed=elapsed,
                )

            if consecutive_unknown >= 3:
                yield ErrorEvent(
                    message="Agent terminated: too many consecutive unknown tool calls"
                )
                break

            exit_plan_called = any(tc.tool_name == "ExitPlanMode" for tc in response.tool_calls)
            conversation.add_tool_results_message(tool_results)

            # 非阻塞 memory recall：工具执行完后检查 prefetch 是否就绪
            await self._drain_memory_recall(conversation, block=False)

            if exit_plan_called:
                yield TurnComplete(turn=iteration)
                yield LoopComplete(total_turns=iteration)
                break

            if self.hook_engine:
                ctx = self._build_hook_context("turn_end")
                await self.hook_engine.run_hooks("turn_end", ctx)
                for he in self._drain_hook_events():
                    yield he
            yield TurnComplete(turn=iteration)

    async def _drain_memory_recall(
        self, conversation: ConversationManager, *, block: bool = False
    ) -> None:
        """在模型调用前把就绪的召回结果注入对话。

        block=True 时最多等待 recall_first_wait 秒（仅首轮使用）。等待超时
        不取消后台 task：它会在本轮后续的 drain 或下一次请求的第一轮补注入。
        普通 drain（block=False）只消费已完成的 task，不阻塞。
        """
        task = self.memory_recall_task
        if task is None or self._memory_recall_consumed:
            return

        if not task.done():
            if not block or self.recall_first_wait <= 0:
                return
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=self.recall_first_wait)
            except TimeoutError:
                return
            except Exception as e:
                log.debug("memory recall prefetch failed: %s", e)
                self._memory_recall_consumed = True
                self.memory_recall_task = None
                return

        try:
            result = task.result()
        except Exception as e:
            log.debug("memory recall prefetch failed: %s", e)
            result = None
        self._memory_recall_consumed = True
        self.memory_recall_task = None

        reminder = getattr(result, "reminder", "") or ""
        paths = list(getattr(result, "paths", []) or [])
        if reminder:
            conversation.add_system_reminder(reminder)
            self._surfaced_memory_paths.update(paths)

    def _consume_mailbox(self, conversation: ConversationManager) -> None:
        if not self.team_name or not self._team_manager:
            return
        try:
            mailbox = self._team_manager.get_mailbox(self.team_name)
            if mailbox is None:
                return
            messages = mailbox.consume(self.agent_id)
            for msg in messages:
                prefix = f"[Message from {msg.from_agent}]"
                if msg.message_type != "text":
                    prefix = f"[{msg.message_type} from {msg.from_agent}]"
                content = f"{prefix} {msg.content}"
                conversation.add_user_message(content)
        except Exception as e:
            log.debug("Mailbox consumption failed: %s", e)

    def _build_permission_description(self, tc: ToolCallComplete) -> str:
        """为 HITL 权限确认生成人类可读的操作描述。"""
        return PermissionChecker.describe_tool_action(tc.tool_name, tc.arguments)

    async def _execute_streaming_tool(self, tc: ToolCallComplete) -> _ToolExecResult:
        """流式路径的工具执行入口：按 `is_concurrency_safe` 选择读写锁。

        并发安全的工具（ReadFile/Grep/Glob 等）拿读锁，可以与其他安全工具
        并行；其余工具拿写锁独占，既不会被并发调用，也会挡住排在它之后的
        读请求。这样模型发出的工具调用顺序在语义上得到保持。
        """
        tool = self.registry.get(tc.tool_name)
        safe = (
            tool is not None and tool.is_concurrency_safe and self.registry.is_enabled(tc.tool_name)
        )
        if safe:
            async with self._tool_lock.read():
                return await self._execute_single_tool_direct(tc)
        async with self._tool_lock.write():
            return await self._execute_single_tool_direct(tc)

    async def _execute_single_tool_direct(self, tc: ToolCallComplete) -> _ToolExecResult:
        """执行单个工具，并记录一个 tool span。"""
        span = self._start_tool_span(tc)
        try:
            exec_result = await self._execute_single_tool_direct_inner(tc)
        except BaseException as e:
            if span is not None:
                span.end(status=SpanStatus.ERROR, error=str(e))
            raise
        self._end_tool_span(span, exec_result.result, exec_result.elapsed)
        return exec_result

    async def _execute_single_tool_direct_inner(self, tc: ToolCallComplete) -> _ToolExecResult:
        tool = self.registry.get(tc.tool_name)
        start = time.monotonic()

        if tool is None:
            return _ToolExecResult(
                tool_id=tc.tool_id,
                tool_name=tc.tool_name,
                result=ToolResult(output=f"Error: unknown tool '{tc.tool_name}'", is_error=True),
                elapsed=time.monotonic() - start,
                is_unknown=True,
            )

        if not self.registry.is_enabled(tc.tool_name):
            return _ToolExecResult(
                tool_id=tc.tool_id,
                tool_name=tc.tool_name,
                result=ToolResult(
                    output=f"Error: tool '{tc.tool_name}' is disabled", is_error=True
                ),
                elapsed=time.monotonic() - start,
                is_unknown=False,
            )

        if self.permission_checker:
            decision = self.permission_checker.check(tool, tc.arguments)
            if decision.effect == "deny":
                return _ToolExecResult(
                    tool_id=tc.tool_id,
                    tool_name=tc.tool_name,
                    result=ToolResult(
                        output=f"Permission denied: {decision.reason}", is_error=True
                    ),
                    elapsed=time.monotonic() - start,
                    is_unknown=False,
                )

        # pre_tool_use hooks：交互式主循环同样要跑，否则用户配置的拦截型
        # hook（安全策略）在日常使用中形同虚设。
        if self.hook_engine:
            hook_ctx = self._build_hook_context(
                "pre_tool_use",
                tool_name=tc.tool_name,
                tool_args=tc.arguments,
                file_path=self._infer_file_path(tc.arguments),
            )
            rejection = await self.hook_engine.run_pre_tool_hooks(hook_ctx)
            if rejection is not None:
                return _ToolExecResult(
                    tool_id=tc.tool_id,
                    tool_name=tc.tool_name,
                    result=ToolResult(output=f"Hook rejected: {rejection.reason}", is_error=True),
                    elapsed=time.monotonic() - start,
                    is_unknown=False,
                )

        try:
            params = tool.params_model.model_validate(tc.arguments)
            result = await tool.execute(params)
        except ValidationError as e:
            result = ToolResult(output=f"Parameter validation error: {e}", is_error=True)
        except Exception as e:
            result = ToolResult(output=f"Tool execution error: {e}", is_error=True)

        if self.hook_engine:
            hook_ctx = self._build_hook_context(
                "post_tool_use",
                tool_name=tc.tool_name,
                tool_args=tc.arguments,
                file_path=self._infer_file_path(tc.arguments),
            )
            await self.hook_engine.run_hooks("post_tool_use", hook_ctx)

        self._snapshot_for_recovery(tc, result)

        return _ToolExecResult(
            tool_id=tc.tool_id,
            tool_name=tc.tool_name,
            result=result,
            elapsed=time.monotonic() - start,
            is_unknown=False,
        )

    async def _execute_batch_parallel(self, calls: list[ToolCallComplete]) -> list[_ToolExecResult]:
        tasks = [self._execute_single_tool_direct(tc) for tc in calls]
        return list(await asyncio.gather(*tasks))

    async def _execute_tool(
        self, tc: ToolCallComplete
    ) -> AsyncIterator[tuple[ToolResult, float, bool] | PermissionRequest]:
        """交互式工具路径（可能等待权限确认），并记录 tool span。"""
        span = self._start_tool_span(tc)
        try:
            async for item in self._execute_tool_inner(tc):
                if not isinstance(item, PermissionRequest):
                    result, elapsed, _ = item
                    self._end_tool_span(span, result, elapsed)
                    span = None
                yield item
        except BaseException as e:
            if span is not None:
                span.end(status=SpanStatus.ERROR, error=str(e))
            raise
        finally:
            if span is not None:
                span.end()

    async def _execute_tool_inner(
        self, tc: ToolCallComplete
    ) -> AsyncIterator[tuple[ToolResult, float, bool] | PermissionRequest]:
        tool = self.registry.get(tc.tool_name)
        start = time.monotonic()
        is_unknown = False

        if tool is None:
            result = ToolResult(output=f"Error: unknown tool '{tc.tool_name}'", is_error=True)
            is_unknown = True
            elapsed = time.monotonic() - start
            yield result, elapsed, is_unknown
            return

        if not self.registry.is_enabled(tc.tool_name):
            result = ToolResult(
                output=f"Error: tool '{tc.tool_name}' is disabled in current mode",
                is_error=True,
            )
            elapsed = time.monotonic() - start
            yield result, elapsed, is_unknown
            return

        # 权限检查
        if self.permission_checker:
            decision = self.permission_checker.check(tool, tc.arguments)

            if decision.effect == "deny":
                result = ToolResult(
                    output=f"Permission denied: {decision.reason}",
                    is_error=True,
                )
                elapsed = time.monotonic() - start
                yield result, elapsed, is_unknown
                return

            if decision.effect == "ask":
                loop = asyncio.get_running_loop()
                future: asyncio.Future[PermissionResponse] = loop.create_future()
                desc = self._build_permission_description(tc)
                # 向调用方 yield 权限请求事件，由调用方处理
                yield PermissionRequest(
                    tool_name=tc.tool_name,
                    description=desc,
                    future=future,
                )
                response = await future

                if response == PermissionResponse.DENY:
                    result = ToolResult(
                        output="Permission denied: 用户拒绝了此操作",
                        is_error=True,
                    )
                    elapsed = time.monotonic() - start
                    yield result, elapsed, is_unknown
                    return

                if response == PermissionResponse.ALLOW_ALWAYS:
                    from lumecode.permissions.rules import Rule, extract_content

                    content = extract_content(tc.tool_name, tc.arguments)
                    pattern = f"{content[:60]}*" if len(content) > 60 else f"{content}*"
                    # 持久化规则写入本地文件
                    rule = Rule(tool_name=tc.tool_name, pattern=pattern, effect="allow")
                    self.permission_checker.rule_engine.append_local_rule(rule)
                    # 同时加入会话级放行集合，本轮立即生效无需磁盘读取
                    self.permission_checker.add_session_allow(tc.tool_name, content)

        try:
            params = tool.params_model.model_validate(tc.arguments)
            result = await tool.execute(params)
        except ValidationError as e:
            result = ToolResult(output=f"Parameter validation error: {e}", is_error=True)
        except Exception as e:
            result = ToolResult(output=f"Tool execution error: {e}", is_error=True)

        self._snapshot_for_recovery(tc, result)

        elapsed = time.monotonic() - start
        yield result, elapsed, is_unknown

    def _snapshot_for_recovery(self, tc: ToolCallComplete, result: ToolResult) -> None:
        """捕获 ReadFile 刚交给模型的内容，以便 Layer 2 压缩对话后
        auto_compact 能重新附加这些数据。每次 ReadFile 多一次磁盘读取，
        比从 tool 输出中反向解析行号要划算。
        """
        if result.is_error or tc.tool_name != "ReadFile":
            return
        path = tc.arguments.get("file_path") if isinstance(tc.arguments, dict) else None
        if not path:
            return
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except OSError:
            return
        self.recovery_state.record_file_read(path, content)

    def _should_extract(self, conversation: ConversationManager) -> bool:
        """是否在轮末触发一次记忆提取。

        条件：轮数达到间隔（interval=0 表示不在轮末提取）、且自上轮提取以来
        新增消息数达到下限——短会话和纯工具空转不付这次 LLM 调用。
        """
        interval = self.memory_extraction_interval
        if not self.memory_manager or interval <= 0:
            return False
        if self._loop_count % interval != 0:
            return False
        unseen = len(conversation.history) - self.memory_manager.extracted_message_count
        return unseen >= MEMORY_EXTRACTION_MIN_MESSAGES

    def _maybe_trigger_extraction(self, conversation: ConversationManager) -> None:
        if self._should_extract(conversation):
            spawn_background(self._extract_memories(conversation))

    async def flush_memories(self, conversation: ConversationManager) -> None:
        """会话结束/退出时提取一次，并在 1.5s 内等待完成（尽力而为）。

        interval=0 时这是唯一的常规提取时机；其它情况下它兜住"最后几轮
        还没到间隔"的尾巴。没有未提取的新消息时是 no-op。
        """
        if not self.memory_manager:
            return
        if len(conversation.history) <= self.memory_manager.extracted_message_count:
            return
        if not self._extracting and self.memory_extraction_interval == 0:
            try:
                await asyncio.wait_for(self._extract_memories(conversation), timeout=1.5)
                return
            except TimeoutError:
                log.debug(
                    "[extractMemories] final extraction exceeded 1.5s, continuing in background"
                )
        spawn_background(self._extract_memories(conversation))

    async def _extract_memories(self, conversation: ConversationManager) -> None:
        """触发记忆提取，对齐 Go 版 inProgress + pendingContext 合并策略。

        当提取正在进行时，新的触发不会启动并发提取，而是标记 _pending_extraction。
        当前提取完成后检查该标志，如果有 pending 则立即执行一次尾随提取，
        防止多个触发器同时执行导致重复提取。
        """
        if not self.memory_manager:
            return

        # 合并策略：正在提取时暂存新请求，等当前提取完成后尾随执行
        if self._extracting:
            log.debug("[extractMemories] extraction in progress — stashing for trailing run")
            self._pending_extraction = True
            return

        self._extracting = True
        try:
            await self.memory_manager.extract(
                self.client,
                conversation,
                self.protocol,
                telemetry=self.telemetry,
                session_id=self.session_id,
            )
        except Exception as e:
            log.debug("Memory extraction failed: %s", e)
        finally:
            self._extracting = False
            # 检查是否有尾随提取请求
            if self._pending_extraction:
                self._pending_extraction = False
                log.debug("[extractMemories] running trailing extraction for stashed context")
                # 递归调用自身处理尾随请求
                await self._extract_memories(conversation)

    async def manual_compact(
        self, conversation: ConversationManager
    ) -> CompactNotification | ErrorEvent:
        # auto_compact 会用摘要替换 conversation.history，所有 tool-result 内容
        # （原始或已替换的）都将被丢弃。这里跳过 apply_tool_result_budget —
        # 它在主循环中的唯一目的是为 LLM 调用生成 api_conv，而本路径不需要
        # 发起看到替换结果的 LLM 调用（auto_compact 内部的摘要调用操作的是原始对话）。
        result = await auto_compact(
            conversation,
            self.client,
            self.context_window,
            self.session_dir,
            protocol=self.protocol,
            manual=True,
            breaker=self.compact_breaker,
            recovery=self.recovery_state,
            tool_schemas=self.registry.get_all_schemas(self.protocol),
            transcript_path=self._transcript_path,
            telemetry=self.telemetry,
        )
        if isinstance(result, CompactEvent):
            env_context = build_environment_context(
                self.work_dir, self.active_skills, self._skill_catalog, self._agent_catalog
            )
            conversation.inject_environment(env_context)
            memory_content = self.memory_manager.load() if self.memory_manager else ""
            conversation.inject_long_term_memory(self.instructions_content, memory_content)
            return CompactNotification(
                before_tokens=result.before_tokens,
                message=f"上下文已压缩（压缩前 {result.before_tokens:,} tokens）",
                boundary=result.boundary,
            )
        return ErrorEvent(message=result or "压缩失败：对话历史为空或未达到压缩条件")

    async def run_to_completion(
        self,
        task: str,
        conversation: ConversationManager | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        """非交互式运行，并为整轮执行建立一个根 span。"""
        if self.telemetry is None:
            return await self._run_to_completion_impl(task, conversation, event_callback)

        root = self.telemetry.start_span(
            SpanKind.AGENT,
            f"agent:{self.agent_id[:8]}",
            parent=self._run_span,
            attributes={"agent.id": self.agent_id, "agent.type": "subagent"},
            input=task,
        )
        self._run_span = root
        try:
            return await self._run_to_completion_impl(task, conversation, event_callback)
        except BaseException as e:
            root.end(status=SpanStatus.ERROR, error=str(e))
            raise
        finally:
            root.end()
            self._run_span = None

    async def _run_to_completion_impl(
        self,
        task: str,
        conversation: ConversationManager | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        if conversation is None:
            conversation = ConversationManager()

            env_context = build_environment_context(
                self.work_dir, self.active_skills, self._skill_catalog, self._agent_catalog
            )
            conversation.inject_environment(env_context)

            if self.instructions_content:
                memory_content = self.memory_manager.load() if self.memory_manager else ""
                conversation.inject_long_term_memory(self.instructions_content, memory_content)

        if not conversation.turn_id:
            conversation.turn_id = f"turn-{uuid.uuid4().hex[:12]}"
        self._start_session_writer(conversation)
        if task:
            conversation.add_user_message(task)

        hook_prompts = self.hook_engine.get_prompt_messages() if self.hook_engine else None
        system = build_system_prompt(
            hook_prompts=hook_prompts,
            coordinator_mode=self.coordinator_mode,
            work_dir=self.work_dir,
            harness=self.harness,
        )

        tools = self.registry.get_all_schemas(self.protocol)

        log.info(
            "[run_to_completion] agent=%s tools=%d names=%s coordinator=%s",
            self.agent_id,
            len(tools),
            [t["name"] for t in tools][:10],
            self.coordinator_mode,
        )

        last_text = ""

        iteration = 0
        while True:
            iteration += 1
            if self.max_iterations > 0 and iteration > self.max_iterations:
                break
            if self.hook_engine:
                ctx = self._build_hook_context("turn_start")
                await self.hook_engine.run_hooks("turn_start", ctx)

            self._consume_mailbox(conversation)
            if self.notification_fn:
                for note in self.notification_fn():
                    conversation.add_system_reminder(note)

            # 先应用 tool-result budget（就地修改），再做 auto-compact，确保预算内的结果不会被误压缩
            pre_compact_records = apply_tool_result_budget(
                conversation, self.session_dir, self.replacement_state
            )
            if pre_compact_records:
                append_replacement_records(self.session_dir, pre_compact_records)

            compact_result = await auto_compact(
                conversation,
                self.client,
                self.context_window,
                self.session_dir,
                protocol=self.protocol,
                breaker=self.compact_breaker,
                recovery=self.recovery_state,
                tool_schemas=self.registry.get_all_schemas(self.protocol),
                transcript_path=self._transcript_path,
                telemetry=self.telemetry,
            )
            if isinstance(compact_result, CompactEvent):
                conversation.inject_environment(env_context)

            deferred_names = self.registry.get_deferred_tool_names()
            if deferred_names:
                conversation.add_system_reminder(
                    "The following deferred tools are available via ToolSearch. "
                    "Their schemas are NOT loaded - use ToolSearch with "
                    'query "select:<name>[,<name>...]" to load tool schemas before calling them:\n'
                    + "\n".join(deferred_names)
                )

            # 压缩后或追加 deferred 提示后重新应用 budget（就地修改）
            _new_records = apply_tool_result_budget(
                conversation, self.session_dir, self.replacement_state
            )
            if _new_records:
                append_replacement_records(self.session_dir, _new_records)

            collector = StreamCollector()
            generation_span = self._start_generation_span(conversation)
            try:
                llm_stream = self.client.stream(conversation, system=system, tools=tools)
                async for _event in collector.consume(llm_stream):
                    pass
            except BaseException as e:
                if generation_span is not None:
                    generation_span.end(status=SpanStatus.ERROR, error=str(e))
                raise

            response = collector.response
            call_usage = self._accumulate_usage(response)
            self._finish_generation_span(generation_span, response, call_usage)

            if event_callback:
                event_callback(
                    {
                        "type": "usage",
                        "usage": {
                            "inputTokens": self.total_input_tokens,
                            "outputTokens": self.total_output_tokens,
                            "cacheReadTokens": self.total_cache_read_tokens,
                            "cacheCreationTokens": self.total_cache_creation_tokens,
                        },
                    }
                )

            if response.text:
                last_text = response.text
                if event_callback:
                    event_callback(
                        {
                            "type": "stream_text",
                            "text": response.text,
                        }
                    )

            log.info(
                "[run_to_completion] agent=%s iter=%d tool_calls=%d text_len=%d stop=%s",
                self.agent_id,
                iteration,
                len(response.tool_calls),
                len(response.text),
                response.stop_reason,
            )

            if not response.tool_calls:
                conversation.add_assistant_message(response.text)
                if self.file_history is not None:
                    summary = (
                        response.text[:60] + "..." if len(response.text) > 60 else response.text
                    )
                    self.file_history.make_snapshot(len(conversation.history), summary)
                break

            tool_uses = [
                ToolUseBlock(
                    tool_use_id=tc.tool_id,
                    tool_name=tc.tool_name,
                    arguments=tc.arguments,
                )
                for tc in response.tool_calls
            ]
            conversation.add_assistant_message(response.text, tool_uses)
            # assistant 回复已在历史中，锚定实际用量；下一轮迭代只需对
            # 下方追加的 tool results 做字符估算。
            conversation.record_usage_anchor(
                response.input_tokens,
                response.output_tokens,
                response.cache_read,
                response.cache_creation,
            )

            tool_results: list[ToolResultBlock] = []
            for tc in response.tool_calls:
                if event_callback:
                    event_callback(
                        {
                            "type": "tool_use",
                            "toolName": tc.tool_name,
                            "args": tc.arguments,
                        }
                    )
                result = await self._execute_tool_noninteractive(tc)
                content = self._maybe_persist_or_truncate(tc.tool_id, result.output)
                tool_results.append(
                    ToolResultBlock(
                        tool_use_id=tc.tool_id,
                        content=content,
                        is_error=result.is_error,
                    )
                )

            conversation.add_tool_results_message(tool_results)

            if self.hook_engine:
                ctx = self._build_hook_context("turn_end")
                await self.hook_engine.run_hooks("turn_end", ctx)

        return last_text

    async def _execute_tool_noninteractive(self, tc: ToolCallComplete) -> ToolResult:
        """非交互路径的工具执行，并记录一个 tool span。"""
        span = self._start_tool_span(tc)
        try:
            result = await self._execute_tool_noninteractive_inner(tc)
        except BaseException as e:
            if span is not None:
                span.end(status=SpanStatus.ERROR, error=str(e))
            raise
        self._end_tool_span(span, result)
        return result

    async def _execute_tool_noninteractive_inner(self, tc: ToolCallComplete) -> ToolResult:
        tool = self.registry.get(tc.tool_name)

        if tool is None:
            return ToolResult(output=f"Error: unknown tool '{tc.tool_name}'", is_error=True)

        if not self.registry.is_enabled(tc.tool_name):
            return ToolResult(
                output=f"Error: tool '{tc.tool_name}' is disabled",
                is_error=True,
            )

        if self.hook_engine:
            file_path = self._infer_file_path(tc.arguments)
            hook_ctx = self._build_hook_context(
                "pre_tool_use",
                tool_name=tc.tool_name,
                tool_args=tc.arguments,
                file_path=file_path,
            )
            rejection = await self.hook_engine.run_pre_tool_hooks(hook_ctx)
            if rejection is not None:
                return ToolResult(
                    output=f"Hook rejected: {rejection.reason}",
                    is_error=True,
                )

        if self.permission_checker:
            decision = self.permission_checker.check(tool, tc.arguments)
            if decision.effect == "deny":
                return ToolResult(
                    output=f"Permission denied: {decision.reason}",
                    is_error=True,
                )
            if decision.effect == "ask":
                if self.permission_mode == PermissionMode.BYPASS:
                    pass  # BYPASS 模式自动批准
                else:
                    return ToolResult(
                        output="Permission denied: non-interactive agent cannot prompt user",
                        is_error=True,
                    )

        try:
            params = tool.params_model.model_validate(tc.arguments)
            result = await tool.execute(params)
        except ValidationError as e:
            result = ToolResult(output=f"Parameter validation error: {e}", is_error=True)
        except Exception as e:
            result = ToolResult(output=f"Tool execution error: {e}", is_error=True)

        if self.hook_engine:
            file_path = self._infer_file_path(tc.arguments)
            hook_ctx = self._build_hook_context(
                "post_tool_use",
                tool_name=tc.tool_name,
                tool_args=tc.arguments,
                file_path=file_path,
            )
            await self.hook_engine.run_hooks("post_tool_use", hook_ctx)

        return result

    def _maybe_persist_or_truncate(self, tool_use_id: str, text: str) -> str:
        from lumecode.context.manager import (
            SINGLE_RESULT_CHAR_LIMIT,
            make_persisted_preview,
            persist_tool_result,
        )

        if len(text) > SINGLE_RESULT_CHAR_LIMIT:
            fp = persist_tool_result(tool_use_id, text, self.session_dir)
            return make_persisted_preview(text, fp)
        if len(text) > MAX_OUTPUT_CHARS:
            return text[:MAX_OUTPUT_CHARS] + "\n… (output truncated)"
        return text
