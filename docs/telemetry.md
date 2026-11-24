# 本地 Telemetry

LumeCode 内置一套零依赖的本地 trace，用于回答三个问题：

- 一次任务里，钱花在哪些 LLM 调用和工具上？
- 时间花在哪一步？压缩、记忆召回、子 Agent 各自占多少？
- 为什么这次跑了 N 轮 / 为什么压缩后失忆？

这一版**不接任何外部服务**，所有数据只写本地 JSONL。字段命名对齐
OpenTelemetry / Langfuse，后续加 OTLP 或 Langfuse exporter 时不需要改埋点。

## 存储位置

```
<project>/.lumecode/traces/<session_id>.jsonl
```

- 每行一个已经结束的 span，带 `schema_version: 1`。
- append-only；崩溃最多丢掉正在写的最后一行。
- `.lumecode/` 已在 `.gitignore` 中，不会被提交。
- 默认保留 30 天，可用 `/trace cleanup` 手动清理。

## 记录哪些 span

| kind | 说明 |
|---|---|
| `agent` | 一次 `agent.run` / `run_to_completion`，根 span |
| `generation` | 一次 LLM 调用：模型、input/output/cache token、成本、stop_reason |
| `tool` | 一次工具执行：工具名、参数、输出、耗时、是否报错 |
| `compact` | 上下文压缩：压缩前后 token、保留消息数、摘要长度 |
| `memory_recall` | 记忆召回 selector：query、选中文件、注入字符数 |
| `memory_extract` | 记忆抽取：写入文件数、token、输出 |
| `memory_consolidation` | 后台记忆整理：会话数、子 Agent 开销 |
| `session_summary` | 会话标题/摘要生成 |

其中后四类以前是“隐形 LLM 调用”，不计入 `total_input_tokens`；现在会作为
独立 span 出现在 `/trace cost` 里。

## 命令

```text
/trace                  当前会话的完整调用树
/trace cost             当前会话的成本报告（按类型/模型/工具耗时）
/trace list             最近的 trace 会话
/trace <session_id>     指定会话的调用树
/trace cleanup          清理超过 30 天的 trace
```

当前会话还没有落盘记录时，`/trace` 会退回旧的实时 Agent 树视图。

## 开关

```bash
# 关闭 telemetry（不创建后台写入线程，span 全部 no-op）
LUMECODE_TELEMETRY=0 lumecode
```

程序化使用时：

```python
from lumecode.telemetry import TelemetryRecorder

recorder = TelemetryRecorder(
    "session-id",
    base_dir=".lumecode/traces",
    capture_content=False,  # 不落 prompt / 工具输出
    max_content_chars=200_000,
)
```

`enabled=None`（默认）时读取 `LUMECODE_TELEMETRY`；显式传 `True/False` 可覆盖。

## 隐私

- 本地默认记录 prompt、工具参数/输出，便于调试。
- 未来接入远程 exporter 时，应默认只发送模型、token、耗时、工具名和错误标志；
  内容需要显式开启并脱敏。
- trace 与 session JSONL 一样可能包含源码和路径，不要提交到仓库或共享。

## 成本估算

`lumecode/telemetry/pricing.py` 内置了一张近似价格表，按模型名子串匹配。
未知模型会显示“无法匹配价格表”，而不是误报为免费。价格会随 provider 调价
过时，`PricingTable(overrides=...)` 可以覆盖任意条目。

成本公式：

```
input_tokens * input_price
+ output_tokens * output_price
+ cache_read_tokens * cache_read_price
+ cache_creation_tokens * cache_creation_price
```

所有价格均为每百万 token 的美元价格。

## 排错：token 全是 0

先看 `/trace` 里的 generation span 有没有 `↑x ↓y`。如果全是 0，说明 provider
没有把 usage 传给上层，问题在 `lumecode/client.py` 而不是 telemetry：

- **OpenAI 协议**：`stream_options: {"include_usage": true}`，usage 在
  `choices` 为空的独立 chunk 里。
- **DeepSeek 协议**：不接收 `stream_options`，但会把 usage 挂在**最后一个
  内容 chunk** 上（`choices` 非空）。只读独立 chunk 会永远拿不到。
- 缓存命中字段也不统一：OpenAI 是 `prompt_tokens_details.cached_tokens`，
  DeepSeek 是 `prompt_cache_hit_tokens`。两者都会读。

`tests/test_client_usage.py` 覆盖了这几种情况。

## 设计约束

- **零新依赖**：不引入 `opentelemetry-sdk` 或 Langfuse SDK，CLI 启动开销不变。
- **best-effort**：后台线程写盘；队列满、写盘失败、内容不可序列化都不会影响主流程。
- **不阻塞**：退出时最多等待 2 秒刷盘，超时直接放弃。
- **可移植**：`kind`、`usage`、`gen_ai.*` 属性可直接映射到 OTLP。
