from __future__ import annotations

import re
import shlex

_DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bmkfs(\.[a-z0-9]+)?\b"), "格式化磁盘"),
    (re.compile(r"\bdd\b[^\n]*\bof=/dev/"), "直接写磁盘设备"),
    (re.compile(r"\bchmod\s+-R\s+777\s+/(?:\s|$)"), "递归修改根目录权限"),
    (re.compile(r":\(\)\s*\{\s*:\|:&\s*\}\s*;:"), "fork bomb"),
    (re.compile(r"\b(curl|wget)\b[^\n]*\|\s*(sudo\s+)?(ba|z|k)?sh\b"), "管道执行远程脚本"),
    (re.compile(r">\s*/dev/(sd|nvme|hd|vd)"), "覆盖磁盘设备"),
    (re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"), "关机/重启系统"),
    (re.compile(r"\b(chown|chmod)\b[^\n]*\s-R\b[^\n]*\s/(?:\s|$)"), "递归修改根目录属主/权限"),
]


# 自动放行的"只读"命令白名单。
#
# 收录标准：该命令**无法**写文件、执行子进程或产生网络副作用。
# 因此刻意排除了以下几类图灵完备/可写命令，即使它们常被当作只读工具：
#   awk / sed   —— 可用 system()/e 命令执行任意 shell
#   xargs       —— 可执行任意命令
#   tee         —— 可写任意文件
#   find        —— -delete / -exec
#   npx         —— 可下载并执行任意包
#   sort / uniq —— -o / 第二参数可写文件
#   env         —— `env <cmd>` 会执行 <cmd>
# 这些命令不再自动放行，但仍可通过权限规则或人工确认执行。
_SAFE_COMMANDS = frozenset(
    {
        "ls",
        "dir",
        "pwd",
        "echo",
        "cat",
        "head",
        "tail",
        "wc",
        "which",
        "whereis",
        "whoami",
        "hostname",
        "uname",
        "date",
        "cal",
        "uptime",
        "df",
        "du",
        "free",
        "printenv",
        "file",
        "stat",
        "readlink",
        "realpath",
        "basename",
        "dirname",
        "tr",
        "cut",
        "grep",
        "egrep",
        "fgrep",
        "diff",
        "comm",
        "true",
        "false",
        "test",
        "git status",
        "git log",
        "git diff",
        "git show",
        "git branch",
        "git tag",
        "git remote",
        "git rev-parse",
        "git ls-files",
        "git blame",
        "git stash list",
        "go version",
        "go env",
        "node -v",
        "npm -v",
        "python --version",
        "pip list",
        "cargo --version",
        "rustc --version",
        "java -version",
        "java --version",
    }
)


# 命令分隔/替换符：出现任意一个就说明这不是一条"简单只读命令"。
# 注意必须覆盖换行符和单独的 & —— 旧实现只检查 "|;&&>$( `"，
# 导致 `echo hi\nrm -rf /` 和 `echo x & rm -rf ~` 被误判为安全命令，
# 在危险命令检测器之前就被自动放行。
_UNSAFE_SHELL_CHARS = ("|", ";", "&", ">", "<", "$(", "`", "\n", "\r", "{", "}")


def is_safe_command(command: str) -> bool:
    trimmed = command.strip()
    if not trimmed:
        return False
    for ch in _UNSAFE_SHELL_CHARS:
        if ch in trimmed:
            return False
    return any(trimmed == safe or trimmed.startswith(safe + " ") for safe in _SAFE_COMMANDS)


# 递归强制删除的"关键目录"——命中即硬拒绝。
_CRITICAL_RM_TARGETS = frozenset(
    {
        "/",
        "/*",
        "/.",
        "~",
        "~/",
        "~/*",
        "$HOME",
        "${HOME}",
        "/home",
        "/usr",
        "/etc",
        "/var",
        "/bin",
        "/boot",
        "/dev",
        "/lib",
        "/lib64",
        "/opt",
        "/root",
        "/sbin",
        "/srv",
        "/sys",
        "/tmp",
        "/proc",
        "*",
    }
)


def _detect_destructive_rm(command: str) -> str:
    """识别 `rm` 同时带递归 + 强制标志且目标为关键目录的情况。

    正则难以覆盖 `rm -fr /`、`rm -r -f ~`、`sudo rm --recursive --force /*`
    这些等价的写法，因此改为按 token 解析：只要命令中出现 rm、同时含递归
    (-r/-R/--recursive) 与强制 (-f/--force/--no-preserve-root) 标志，且某个
    目标落在关键目录集合里，就判为灾难性删除。
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    for i, token in enumerate(tokens):
        if token.rsplit("/", 1)[-1] != "rm":
            continue
        recursive = False
        force = False
        targets: list[str] = []
        for arg in tokens[i + 1 :]:
            if arg in ("&&", "||", ";", "|"):
                break
            if arg.startswith("--"):
                if arg == "--recursive":
                    recursive = True
                elif arg in ("--force", "--no-preserve-root"):
                    force = True
            elif arg.startswith("-") and len(arg) > 1:
                flags = arg[1:]
                if "r" in flags or "R" in flags:
                    recursive = True
                if "f" in flags:
                    force = True
            else:
                targets.append(arg)
        if recursive and force:
            for target in targets:
                if target in _CRITICAL_RM_TARGETS:
                    return f"递归强制删除关键目录: rm ... {target}"
    return ""


class DangerousCommandDetector:
    def __init__(self, extra_patterns: list[tuple[str, str]] | None = None) -> None:
        self._patterns = list(_DANGEROUS_PATTERNS)
        if extra_patterns:
            for regex_str, reason in extra_patterns:
                self._patterns.append((re.compile(regex_str), reason))

    def detect(self, command: str) -> tuple[bool, str]:
        for pattern, reason in self._patterns:
            if pattern.search(command):
                return True, reason
        rm_reason = _detect_destructive_rm(command)
        if rm_reason:
            return True, rm_reason
        return False, ""
