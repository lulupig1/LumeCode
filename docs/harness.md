# Harness 自我改进

LumeCode 的「进化」不改模型权重，只改模型外面那层外壳（harness）：基础提示词分节
+ 一小撮从失败中提炼出来的经验。设计参考 Google Harness² 的「基线只读、经验可写、
写入前快照」三原则。

## 目录

```
.lumecode/harness/
  prompt/<key>.md        # 基础提示词分节（人改，Agent 不可写）
  experience/INDEX.md    # 经验索引，自动重建
  experience/<name>.md   # 单条经验
  proposals/<id>.md      # 待评审的提议
  versions.jsonl         # 改动历史
  .history/<stamp>/      # 每次自动写入前的整目录快照
```

没有 `prompt/` 目录时，`build_system_prompt()` 回退到 `lumecode/prompts.py` 里的
内置常量——**输出与旧版本逐字相同**，这是硬约束（有测试守着）。

## 命令

| 命令 | 作用 |
|---|---|
| `/harness` | 查看版本、经验数、注入预算、待评审条数 |
| `/harness base` | 列出已文件化的基础分节 |
| `/harness edit <key> <file>` | 用文件内容覆盖某个基础分节 |
| `/distill [--since 7d] [--dry-run]` | 复盘失败轨迹，生成提议（只提议，不写入） |
| `/evolve status` | 四类指标 + 每条经验的命中与复发 |
| `/evolve review` | 查看待评审提议 |
| `/evolve accept <id>` / `reject <id>` | 采纳 / 拒绝 |
| `/evolve log` | 改动历史 |
| `/evolve retire <name>` | 停用一条经验（保留文件） |
| `/evolve rollback` | 回退到上一个快照 |

## 闭环

1. **记录**：`agent.py` 已经往 `.lumecode/traces/*.jsonl` 写 span。
2. **归因**（`attribution.py`）：把工具失败抽成 `FailureCase`，自愈型（错→报错→
   换办法→成功）优先，签名稳定为 `tool:bash:<命令头>:<错误类别>`。
3. **提议**（`distill.py`）：全新上下文的 editor 子 Agent 读失败案例与当前 harness，
   产出 YAML 提议。**必须引用 `session_id:span_id`**，引用不到的丢弃；触及权限/
   沙箱的丢弃；单轮最多 3 条。
4. **评审**（`proposals.py` + `/evolve accept`）：人工确认后才写。写入前一定
   `snapshot()`，并追加一条版本记录。
5. **注入**（`store.render_injection()`）：经验以 `# Learned conventions for this
   project` 段落进 system prompt，排在 Skills 之后、Memory 之前，总量受
   `MAX_INJECT_CHARS` 硬上限约束。
6. **回滚**：`/evolve rollback` 恢复最近快照；回滚前同样先快照，所以回滚本身可撤销。

## 硬约束（代码层，不依赖提示词）

- 经验条数上限 `MAX_EXPERIENCES = 30`，注入总字符上限 `MAX_INJECT_CHARS = 6000`。
- 经验名必须是 `^[a-z0-9][a-z0-9._-]*$`。
- 基础分节只能通过 `/harness edit`（人）写入；Agent 的提议路径写不到 `prompt/`。
- 触及 `permission|sandbox|approval|bypass|disable checks|allow all` 的提议直接丢弃。

## 判断是否真的进化

只看三个动作：

1. `/evolve log` 里每条经验能指到具体一次失败（证据 span 是否存在）。
2. `/evolve status` 的「创建后仍复发」归零，同时 `token/任务`、`工具调用/任务` 不涨。
3. 用一条**没进过语料库**的同类任务验证：还会不会犯。

前两条过、第三条不过 = 只是记住了案例，不是学会规则。

## 当前已知边界

- **阈值自动触发**（同类失败累计 N 次后后台自动 distill）尚未接入，目前只有手动
  `/distill`。骨架可复用 `memory/consolidation.py` 的时间门 + 锁 + 节流。
- **自动回滚**未接：回滚目前是手动命令。
- **回归测试集**未接：`/evolve accept` 不做自动验证，v1 依赖人工确认。
- 该目录被 `.gitignore` 忽略（`.lumecode/`）。如需跨机器同步经验，把
  `.lumecode/harness/` 单独 whitelist。
