"""针对一批 bug 修复的回归测试。

每个测试类对应一个具体缺陷，注释里写明"修复前会怎样失败"，
便于日后回归时理解这些断言在保护什么。
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from lumecode.agent import _AsyncRWLock
from lumecode.conversation import ConversationManager
from lumecode.hooks.executors import execute_command
from lumecode.hooks.models import Action, HookContext
from lumecode.memory.session import SessionManager
from lumecode.skills.executor import SkillExecutor
from lumecode.teams.mailbox import Mailbox, create_message
from lumecode.teams.shared_task import SharedTaskStore
from lumecode.tools.bash import Bash
from lumecode.tools.bash import Params as BashParams
from lumecode.tools.edit_file import EditFile
from lumecode.tools.edit_file import Params as EditParams
from lumecode.tools.file_io import read_text_file, write_text_file
from lumecode.tools.file_state_cache import FileStateCache
from lumecode.tools.read_file import Params as ReadParams
from lumecode.tools.read_file import ReadFile
from lumecode.tools.write_file import Params as WriteParams
from lumecode.tools.write_file import WriteFile

# ===========================================================================
# is_concurrency_safe 的并发闸门（_AsyncRWLock）
# ===========================================================================


class TestAsyncRWLock:
    """修复前：流式主循环无视 is_concurrency_safe，所有工具一律并发，
    导致 WriteFile/EditFile 会与其它工具同时跑，竞争同一份文件状态。
    """

    @pytest.mark.asyncio
    async def test_readers_run_concurrently(self) -> None:
        lock = _AsyncRWLock()
        active = 0
        max_active = 0

        async def reader() -> None:
            nonlocal active, max_active
            async with lock.read():
                active += 1
                max_active = max(max_active, active)
                await asyncio.sleep(0.02)
                active -= 1

        await asyncio.gather(*(reader() for _ in range(4)))
        assert max_active == 4

    @pytest.mark.asyncio
    async def test_writer_is_exclusive(self) -> None:
        lock = _AsyncRWLock()
        active = 0
        max_active = 0

        async def writer() -> None:
            nonlocal active, max_active
            async with lock.write():
                active += 1
                max_active = max(max_active, active)
                await asyncio.sleep(0.02)
                active -= 1

        await asyncio.gather(*(writer() for _ in range(4)))
        assert max_active == 1

    @pytest.mark.asyncio
    async def test_writer_waits_for_inflight_readers(self) -> None:
        lock = _AsyncRWLock()
        events: list[str] = []

        async def reader() -> None:
            async with lock.read():
                events.append("r-in")
                await asyncio.sleep(0.05)
                events.append("r-out")

        async def writer() -> None:
            await asyncio.sleep(0.01)
            async with lock.write():
                events.append("w-in")

        await asyncio.gather(reader(), writer())
        assert events == ["r-in", "r-out", "w-in"]

    @pytest.mark.asyncio
    async def test_writer_blocks_later_readers_fifo(self) -> None:
        """写者会挡住排在它之后的读者——这是"先写后读"保持正确的关键。"""
        lock = _AsyncRWLock()
        events: list[str] = []
        release = asyncio.Event()

        async def first_writer() -> None:
            async with lock.write():
                events.append("w-in")
                await release.wait()
                events.append("w-out")

        async def later_reader() -> None:
            await asyncio.sleep(0.01)
            async with lock.read():
                events.append("r-in")

        t1 = asyncio.create_task(first_writer())
        await asyncio.sleep(0.01)
        t2 = asyncio.create_task(later_reader())
        await asyncio.sleep(0.05)
        # 写者尚未释放，读者必须还在排队。
        assert events == ["w-in"]

        release.set()
        await asyncio.gather(t1, t2)
        assert events == ["w-in", "w-out", "r-in"]


# ===========================================================================
# 文件写入：原子替换 + 行尾保持
# ===========================================================================


class TestAtomicFileWrites:
    """修复前：Path.write_text 截断后重写（崩溃即损坏），且文本模式会把
    LF 文件整体改写成 CRLF。
    """

    def test_write_text_file_restores_crlf(self, tmp_path: Path) -> None:
        p = tmp_path / "f.txt"
        write_text_file(p, "a\nb\n", newline="\r\n")
        assert p.read_bytes() == b"a\r\nb\r\n"

    def test_write_text_file_leaves_no_temp_files(self, tmp_path: Path) -> None:
        p = tmp_path / "f.txt"
        write_text_file(p, "hello\n")
        leftovers = [f for f in tmp_path.iterdir() if f.name != "f.txt"]
        assert leftovers == []

    def test_read_text_file_normalizes_and_reports_newline(self, tmp_path: Path) -> None:
        p = tmp_path / "f.txt"
        p.write_bytes(b"a\r\nb\r\n")
        text, newline, truncated = read_text_file(p)
        assert text == "a\nb\n"
        assert newline == "\r\n"
        assert truncated is False

    def test_read_text_file_detects_binary(self, tmp_path: Path) -> None:
        p = tmp_path / "f.bin"
        p.write_bytes(b"\x00\x01\x02")
        from lumecode.tools.file_io import BinaryFileError

        with pytest.raises(BinaryFileError):
            read_text_file(p)

    @pytest.mark.asyncio
    async def test_write_file_preserves_crlf(self, tmp_path: Path) -> None:
        p = tmp_path / "crlf.txt"
        p.write_bytes(b"old1\r\nold2\r\n")
        state = FileStateCache()

        await ReadFile(file_state_cache=state).execute(ReadParams(file_path=str(p)))
        await WriteFile(file_state_cache=state).execute(
            WriteParams(file_path=str(p), content="new1\nnew2\n")
        )
        assert p.read_bytes() == b"new1\r\nnew2\r\n"

    @pytest.mark.asyncio
    async def test_edit_file_preserves_crlf(self, tmp_path: Path) -> None:
        p = tmp_path / "crlf.txt"
        p.write_bytes(b"alpha\r\nbeta\r\n")
        state = FileStateCache()

        await ReadFile(file_state_cache=state).execute(ReadParams(file_path=str(p)))
        await EditFile(file_state_cache=state).execute(
            EditParams(file_path=str(p), old_string="beta", new_string="gamma")
        )
        assert p.read_bytes() == b"alpha\r\ngamma\r\n"


# ===========================================================================
# ReadFile 缓存陈旧
# ===========================================================================


class TestReadFileCacheFreshness:
    """修复前：FileCache 命中即返回旧内容，不校验 mtime；紧接着又用当前
    磁盘 mtime 写入 FileStateCache，喂给 read-before-edit 门槛一对错配数据。
    """

    @pytest.mark.asyncio
    async def test_externally_modified_file_is_reread(self, tmp_path: Path) -> None:
        from lumecode.cache import FileCache

        p = tmp_path / "f.txt"
        p.write_text("original\n", encoding="utf-8")

        cache = FileCache()
        state = FileStateCache()
        reader = ReadFile(file_cache=cache, file_state_cache=state)

        first = await reader.execute(ReadParams(file_path=str(p)))
        assert "original" in first.output

        # 外部修改文件（模拟用户/其它进程改动），并确保 mtime 前进。
        import os
        import time

        await asyncio.sleep(0.01)
        p.write_text("changed\n", encoding="utf-8")
        os.utime(p, (time.time() + 1, time.time() + 1))

        second = await reader.execute(ReadParams(file_path=str(p)))
        assert "changed" in second.output
        assert "original" not in second.output


# ===========================================================================
# session id 路径穿越
# ===========================================================================


class TestSessionIdTraversal:
    """修复前：/session delete ../../x 会用用户传入的 id 直接拼路径删文件。"""

    def test_resume_rejects_traversal(self, tmp_path: Path) -> None:
        mgr = SessionManager(str(tmp_path))
        assert mgr.resume("../../etc/passwd") is None
        assert mgr.resume("..\\..\\windows") is None
        assert mgr.resume("") is None

    def test_delete_rejects_traversal(self, tmp_path: Path) -> None:
        victim = tmp_path / "victim.txt"
        victim.write_text("keep me", encoding="utf-8")

        mgr = SessionManager(str(tmp_path / "work"))
        assert mgr.delete("../victim") is False
        assert victim.exists()

    def test_valid_id_still_works(self, tmp_path: Path) -> None:
        mgr = SessionManager(str(tmp_path))
        session = mgr.create()
        session.close()  # Windows 下句柄不关就删不掉文件
        assert mgr.resume(session.session_id) is not None
        assert mgr.delete(session.session_id) is True


# ===========================================================================
# Mailbox 锁
# ===========================================================================


class TestMailboxLock:
    """修复前：10 次抢锁都失败后仍无锁继续写，并在 finally 里把别人
    持有的锁文件删掉。
    """

    def test_does_not_proceed_when_lock_held(self, tmp_path: Path) -> None:
        mb = Mailbox(tmp_path)
        lock_file = mb._lock_path("a1")
        # 模拟另一个进程正持有锁（新鲜，不会被判为 stale）。
        lock_file.write_text("", encoding="utf-8")

        with pytest.raises(TimeoutError):
            mb.write("a1", create_message("sender", "a1", "hi"))

        # 别人的锁必须原封不动，不能被我们删掉。
        assert lock_file.exists()

    def test_write_then_consume_roundtrip(self, tmp_path: Path) -> None:
        mb = Mailbox(tmp_path)
        mb.write("a1", create_message("sender", "a1", "hello"))
        messages = mb.consume("a1")
        assert len(messages) == 1
        assert messages[0].content == "hello"
        # 已消费，再次读取应为空。
        assert mb.consume("a1") == []

    def test_inbox_is_never_partially_written(self, tmp_path: Path) -> None:
        """原子写入：读者要么看到完整的旧内容，要么看到完整的新内容。"""
        import json

        mb = Mailbox(tmp_path)
        mb.write("a1", create_message("s", "a1", "one"))
        # 落盘后应是合法 JSON（旧的 write_text 截断重写会留下半截内容）。
        data = json.loads(mb._inbox_path("a1").read_text(encoding="utf-8"))
        assert isinstance(data, list) and len(data) == 1


# ===========================================================================
# Bash：进程组终止 + 输出上限
# ===========================================================================


class TestBashHardening:
    @pytest.mark.asyncio
    async def test_large_output_is_capped(self) -> None:
        from lumecode.tools import bash as bash_mod

        cmd = f'"{sys.executable}" -c "print(\'x\' * 3000000)"'
        result = await Bash().execute(BashParams(command=cmd, timeout=60))
        assert len(result.output) <= bash_mod.MAX_OUTPUT_BYTES + 200
        assert "truncated" in result.output

    @pytest.mark.asyncio
    async def test_timeout_kills_process_tree(self) -> None:
        """修复前：proc.kill() 只杀 sh，后台子进程会变成孤儿继续存活。"""
        marker = Path(__file__).parent / "_orphan_marker.txt"
        marker.unlink(missing_ok=True)

        cmd = f"sleep 5 && \"{sys.executable}\" -c \"open(r'{marker}', 'w').write('x')\" & sleep 30"
        result = await Bash().execute(BashParams(command=cmd, timeout=1))
        assert result.is_error
        assert "timed out" in result.output

        # 子进程若存活，5 秒后会写出 marker 文件。
        await asyncio.sleep(6)
        assert not marker.exists()


# ===========================================================================
# fork 型 skill 的上下文
# ===========================================================================


class TestSkillForkContext:
    """修复前：读的是 self.agent._conversation（不存在），hasattr 恒为 False，
    fork 型 skill 的 context: full/recent 静默失效、永远拿到空上下文。
    """

    class _StubAgent:
        """模拟 Agent 暴露的 public 属性（Agent 上为 property）。"""

        def __init__(self, conversation: ConversationManager) -> None:
            self._current_conversation = conversation

        @property
        def current_conversation(self) -> ConversationManager:
            return self._current_conversation

    def _executor(self, conversation: ConversationManager) -> SkillExecutor:
        return SkillExecutor(self._StubAgent(conversation), client=None, protocol="anthropic")

    def test_recent_context_is_populated(self) -> None:
        conv = ConversationManager()
        conv.add_user_message("hello there")
        conv.add_assistant_message("hi, how can I help?")

        ex = self._executor(conv)
        ctx = ex._build_fork_context("recent")
        assert len(ctx) == 2
        assert "hello there" in ctx[0].content

    def test_full_context_is_populated(self) -> None:
        conv = ConversationManager()
        conv.add_user_message("first question")
        conv.add_assistant_message("first answer")

        ex = self._executor(conv)
        ctx = ex._build_fork_context("full")
        assert len(ctx) == 1
        assert "first question" in ctx[0].content

    def test_none_context_is_empty(self) -> None:
        conv = ConversationManager()
        conv.add_user_message("ignored")
        ex = self._executor(conv)
        assert ex._build_fork_context("none") == []


# ===========================================================================
# hook command 动作的命令注入
# ===========================================================================


class TestHookCommandInjection:
    """修复前：ctx.expand() 把工具参数原样拼进 shell 字符串再交给
    create_subprocess_shell，模型只要构造一个含分隔符的参数就能执行任意命令。
    """

    def test_expand_quotes_substituted_values(self) -> None:
        ctx = HookContext(
            event_name="pre_tool_use",
            tool_name="Bash",
            tool_args={"file_path": "a; rm -rf ~"},
        )
        raw = ctx.expand("echo $TOOL_ARGS.file_path")
        quoted = ctx.expand("echo $TOOL_ARGS.file_path", shell_quote=True)

        # 未转义：分隔符裸露在命令串里，会被 shell 当作命令分隔符。
        assert raw == "echo a; rm -rf ~"
        assert quoted != raw

        if os.name != "nt":
            # POSIX 下的真正不变量：整段值成为**单个**参数。
            import shlex

            assert shlex.split(quoted) == ["echo", "a; rm -rf ~"]

    def test_expand_without_quote_leaves_values_verbatim(self) -> None:
        """prompt / http 类动作不做 shell 转义，值应原样保留。"""
        ctx = HookContext(tool_name="Bash", message="a; rm -rf ~")
        assert ctx.expand("$MESSAGE", shell_quote=False) == "a; rm -rf ~"

    def test_quoted_template_literals_are_preserved(self) -> None:
        """作者自己写的管道/重定向不该被转义掉。"""
        ctx = HookContext(tool_args={"v": "x"})
        out = ctx.expand("echo $TOOL_ARGS.v | wc -l", shell_quote=True)
        assert "|" in out

    @pytest.mark.asyncio
    async def test_injected_argument_is_not_executed(self, tmp_path: Path) -> None:
        marker = tmp_path / "pwned.txt"
        sep = "&" if os.name == "nt" else ";"
        payload = f"x {sep} echo pwned > {marker}"

        ctx = HookContext(
            event_name="pre_tool_use",
            tool_name="Bash",
            tool_args={"file_path": payload},
        )
        action = Action(type="command", command="echo $TOOL_ARGS.file_path")
        result = await execute_command(action, ctx)

        assert not marker.exists(), f"command injection succeeded: {result.output!r}"


# ===========================================================================
# SharedTaskStore 跨进程一致性
# ===========================================================================


class TestSharedTaskStoreLocking:
    """修复前：create/update 是无锁的 load-modify-save。两个进程（或两个
    store 实例）并发写入会丢更新并产生重复的 next_id。
    """

    def test_two_stores_do_not_collide_or_lose_tasks(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.json"
        a = SharedTaskStore(path)
        b = SharedTaskStore(path)  # 构造时文件还不存在

        t1 = a.create(title="one")
        t2 = b.create(title="two")

        assert t1.id != t2.id, "两个实例分配了相同的任务 id"

        reloaded = SharedTaskStore(path)
        ids = {t.id for t in reloaded.list_tasks()}
        assert ids == {t1.id, t2.id}, "先写入的任务被后写入的覆盖了"

    def test_update_is_visible_to_other_store(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.json"
        a = SharedTaskStore(path)
        task = a.create(title="one")

        b = SharedTaskStore(path)
        b.update(task.id, status="completed")

        assert SharedTaskStore(path).get(task.id).status == "completed"

    def test_corrupt_file_degrades_to_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.json"
        path.write_text("{ not json", encoding="utf-8")
        store = SharedTaskStore(path)
        assert store.list_tasks() == []


# ===========================================================================
# bwrap 挂载不存在的路径
# ===========================================================================


class TestBwrapPathHandling:
    """修复前：对不存在的 deny_write / allow_write 路径也照发 --ro-bind /
    --bind，bwrap 会因源路径不存在而让整条沙箱命令失败——一个尚未创建的
    配置文件就能让所有 Bash 命令跑不起来。
    """

    def test_missing_paths_are_skipped(self, tmp_path: Path) -> None:
        from lumecode.sandbox import SandboxConfig
        from lumecode.sandbox.bwrap import BwrapSandbox

        missing = tmp_path / "not-created.conf"
        existing = tmp_path / "exists.conf"
        existing.write_text("x", encoding="utf-8")

        cfg = SandboxConfig(
            allow_write=[str(missing)],
            deny_write=[str(missing), str(existing)],
        )
        cmd = BwrapSandbox().wrap("echo hi", cfg)

        assert str(missing.resolve()) not in cmd
        assert str(existing.resolve()) in cmd

    def test_die_with_parent_is_set(self) -> None:
        from lumecode.sandbox import SandboxConfig
        from lumecode.sandbox.bwrap import BwrapSandbox

        cmd = BwrapSandbox().wrap("echo hi", SandboxConfig())
        assert "--die-with-parent" in cmd


# ===========================================================================
# 沙箱策略：可写根内的只读例外（.git / .lumecode）
# ===========================================================================


class TestSandboxReadonlyCarveouts:
    """修复前：可写根内没有任何例外，Agent 能直接改写 .git 与自身的
    .lumecode 配置/技能/会话元数据。现由 SandboxPolicy.readonly_paths 剥离。
    """

    def test_git_and_lumecode_are_readonly(self, tmp_path: Path) -> None:
        from lumecode.permissions.sandbox import PathSandbox

        sandbox = PathSandbox(str(tmp_path))
        for rel in (".git/config", ".lumecode/config.yaml", ".lumecode/memory/a.md"):
            target = tmp_path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x", encoding="utf-8")
            ok, reason = sandbox.check(str(target))
            assert not ok, f"{rel} 应为只读"
            assert "只读" in reason

    def test_normal_project_file_still_writable(self, tmp_path: Path) -> None:
        from lumecode.permissions.sandbox import PathSandbox

        sandbox = PathSandbox(str(tmp_path))
        f = tmp_path / "src" / "main.py"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x", encoding="utf-8")
        ok, _ = sandbox.check(str(f))
        assert ok

    def test_policy_is_exposed_as_object(self, tmp_path: Path) -> None:
        from lumecode.permissions.sandbox import PathSandbox, SandboxPolicy

        sandbox = PathSandbox(str(tmp_path))
        assert isinstance(sandbox.policy, SandboxPolicy)
        assert sandbox.policy.writable_roots[0] == tmp_path.resolve()
