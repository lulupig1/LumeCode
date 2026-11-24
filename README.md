# LumeCode

> 一个用 Python 编写的终端 AI 编程助手（Terminal AI Coding Assistant），带有 Textual 全屏 TUI、子 Agent、工具调用、权限/沙箱、MCP、记忆与可观测性。

LumeCode 的目标是把「终端里的 AI 结对编程」做成一套可读、可改、可扩展的 Python 实现：一个 agent loop + 一组工具 + 一套权限/沙箱策略 + 记忆与上下文管理 + 本地 telemetry，外加一个能「从失败里学经验」的 harness 自我改进闭环。

---

## 设计借鉴（灵感来源）

LumeCode 不是从零发明的，它在架构上明确对齐了若干成熟的开源/公开设计：

| 方面 | 借鉴来源 | 说明 |
|---|---|---|
| **核心 agent 架构 / 权限与沙箱双轴** | **[OpenAI Codex CLI](https://github.com/openai/codex)**（开源） | `permissions/checker.py`、`permissions/sandbox.py` 明确「对齐 Codex」：把 **SandboxPolicy（能碰什么）** 与 **approval policy（要不要问）** 拆成两条正交的轴；worktree / 文件锁等实现也参照 Codex 的做法。 |
| **产品形态与交互范式** | **Claude Code** 这类终端编码助手 | 斜杠命令（`/help`、`/compact`、`/plan`…）、子 Agent、Skills、Hooks、MCP、记忆、`-p` 非交互模式等，都是对 Claude Code 交互范式的开源复刻。 |
| **Harness 自我改进闭环** | **Google「Harness²」** 的三原则 | 见 `docs/harness.md`：**基线只读、经验可写、写入前快照**。不改模型权重，只改模型外面那层「外壳」。 |
| **Telemetry 字段命名** | **OpenTelemetry / Langfuse** | 本地 JSONL trace 的字段命名对齐 OTel/Langfuse，便于日后接 OTLP/Langfuse exporter 而无需改埋点。 |

> 一句话：**内核 agent / 权限 / 沙箱对齐 OpenAI 开源的 Codex CLI，交互形态对标 Claude Code，harness 进化参考 Google Harness²。**

---

## 主要特性

- **终端 TUI**：基于 [Textual](https://github.com/Textualize/textual)，全屏交互、流式输出、计划/权限/会话等弹窗。
- **多模型 Provider**：`anthropic`、`openai`、`openai-compat`（可接 DeepSeek / 通义千问 / 智谱 / Moonshot 等），支持自动拉取 context window。
- **工具集**：`bash`、`read_file`、`write_file`、`edit_file`、`glob`、`grep`、`diff`、`ask_user`、任务管理（`task_*`）、团队（`team_*`）、worktree、技能加载等。
- **子 Agent 与团队**：`agents/` 内置 `explore` / `general-purpose` / `plan` / `verification`，支持 fork、任务分派与协作。
- **权限与沙箱**：四种权限模式（见下），macOS 走 `sandbox-exec`（Seatbelt）、Linux 走 `bubblewrap`（bwrap）。
- **上下文管理**：token 估算、自动压缩（compact）、摘要锚点重建。
- **记忆系统**：自动记忆、embedding 混合召回、textmatch、后台整理（consolidation）、指令文件。
- **MCP**：stdio / http 两种 MCP server 接入，工具自动包装。
- **Hooks**：事件 + 条件 + 执行器的钩子引擎。
- **Worktree**：git worktree 隔离开发。
- **本地 Telemetry**：零依赖 JSONL trace，`/trace` 查看调用树与成本。
- **Harness 自我改进**：`/distill` 复盘失败轨迹 → 生成经验提议 → 人工评审采纳 → 注入 system prompt，可回滚。

---

## 目录结构

```
lumecode/
├── __main__.py          # CLI 入口（argparse、-p 非交互、--remote）
├── app.py               # Textual TUI 应用
├── agent.py             # agent 主循环
├── client.py            # LLM 客户端（anthropic / openai / openai-compat）
├── config.py            # 配置加载（config.yaml + 环境变量）
├── validator.py         # 配置校验、context window 映射
├── conversation.py      # 会话与消息、token 估算
├── context/             # 上下文管理与压缩
├── agents/              # 子 Agent、fork、任务管理、内置 agent 定义
├── tools/               # 内置工具实现
├── permissions/         # 权限判定（双轴）、模式、沙箱策略
├── sandbox/             # Seatbelt(macOS) / bwrap(Linux) 沙箱
├── hooks/               # Hook 引擎
├── mcp/                 # MCP 客户端与工具包装
├── memory/              # 自动记忆、embedding 召回、整理
├── skills/              # 技能
├── teams/               # 多 agent 团队
├── telemetry/           # 本地 trace / 成本
├── harness/             # harness 自我改进闭环
├── commands/            # 斜杠命令与处理器
├── filehistory/         # 文件历史/回滚
├── worktree/            # git worktree
└── prompts.py           # 内置 system prompt 常量
```

---

## 安装

要求 **Python ≥ 3.11**。

```bash
# 方式一：uv（推荐）
uv sync

# 方式二：pip 可编辑安装
pip install -e .
```

安装后命令为 `lumecode`。

---

## 配置

在项目根目录放一个 `config.yaml`（已在 `.gitignore` 中，不会被提交），至少配置一个 provider：

```yaml
providers:
  - name: anthropic
    protocol: anthropic          # anthropic | openai | openai-compat
    base_url: https://api.anthropic.com
    model: claude-sonnet-5-5
    # api_key: 留空则从环境变量读取

  - name: openai
    protocol: openai
    base_url: https://api.openai.com/v1
    model: gpt-4o

  - name: deepseek
    protocol: openai-compat
    base_url: https://api.deepseek.com/v1
    model: deepseek-chat
    thinking: false
    # context_window / max_output_tokens 可选

permission_mode: default         # default | acceptEdits | plan | bypassPermissions

# 可选：MCP servers（stdio 或 http）
mcp_servers:
  - name: github
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_TOKEN: "${GITHUB_TOKEN}"
```

API Key 通过环境变量提供（也可在 provider 里写 `api_key`）：

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
# openai-compat 还会按 base_url 回退读取 DEEPSEEK_API_KEY / DASHCOPE_API_KEY / ZHIPU_API_KEY / MOONSHOT_API_KEY
```

### 权限模式

| 模式 | 含义 |
|---|---|
| `default` | 读操作直接放行，写操作与命令执行需确认 |
| `acceptEdits` | 自动接受文件编辑，命令仍按策略确认 |
| `plan` | 只读探索，产出计划，不落地修改 |
| `bypassPermissions` | 跳过确认（谨慎使用） |

---

## 使用

```bash
# 交互式 TUI
lumecode

# 指定权限模式
lumecode --mode acceptEdits

# 非交互执行（打印结果）
lumecode -p "为这个仓库生成一个 README"

# 非交互 + NDJSON 流式事件
lumecode -p "解释 lumecode/agent.py" --output-format stream-json

# 远程模式：在 0.0.0.0:18888 起 WebSocket + 浏览器 UI
lumecode --remote
```

### 常用斜杠命令

| 命令 | 作用 |
|---|---|
| `/help [命令名]` | 帮助 |
| `/clear` | 清空会话 |
| `/compact` | 压缩上下文 |
| `/status` | 会话状态 |
| `/permission` | 查看/切换权限模式 |
| `/plan` | 计划模式 |
| `/review` | 代码评审 |
| `/rewind` | 回退到历史节点 |
| `/sandbox` | 沙箱相关 |
| `/session` | 会话管理 |
| `/skill` | 技能 |
| `/tasks` | 任务 |
| `/trace` | 本地 telemetry 调用树/成本 |
| `/worktree` | git worktree |
| `/mcp` | MCP 管理 |
| `/memory` | 记忆 |
| `/harness` / `/distill` / `/evolve` | harness 自我改进闭环 |

更多细节见 [`docs/harness.md`](docs/harness.md) 与 [`docs/telemetry.md`](docs/telemetry.md)。

---

## 开发与测试

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run mypy
```

CI 见 `.github/workflows/ci.yml`。

---

## 状态

版本 `0.2.0`，持续开发中。欢迎以学习/研究为目的阅读与试用。
