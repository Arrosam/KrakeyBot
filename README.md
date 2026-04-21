# KrakeyBot

> 以持续心跳维持"存在"的自主认知 Agent。
> 沙盒 VM 配置见 [`SANDBOX.md`](SANDBOX.md)（启用 coding/GUI/file/browser 任一 tentacle 前必读）。

---

## 环境要求

- **Python**: 3.12（也兼容 3.11）
- **LLM Server (chat)**: OpenAI 兼容 `/v1/chat/completions`
  - 本地 `llama-server` / `llama.cpp` / `vllm` / `lmstudio` / `ollama(openai 兼容模式)`
  - 云端 DashScope / Anthropic Claude / One API 聚合代理等
- **Embedding Server**: OpenAI 兼容 `/v1/embeddings`（Phase 1 起必需；推荐 `bge-m3`）
- **Reranker Server (可选)**: OpenAI 兼容 `/v1/rerank`（无则自动降级脚本排序；推荐 `bge-reranker-v2-m3`）
- `sqlite-vec` 会随 `pip install` 自动装好（已含预编译轮子，Windows/Linux/macOS 均可）

---

## 安装

**你只需要系统里有 Python 3.11+，然后直接跑 `main.py`：**

```bash
git clone <repo>
cd KrakeyBot
python main.py
```

首次启动时 `main.py` 会自动：
1. 在项目根下创建 `.venv/` 虚拟环境
2. `pip install -r requirements.txt` 安装运行依赖
3. 在 `.venv/.installed-hash` 记录 requirements 哈希
4. 切到 `.venv` 中的解释器重新启动程序

之后再运行时，只要 `requirements.txt` 没变就直接启动，不再重装。

> **想跑测试** 需要额外的 dev 依赖（pytest 等），手动装一次：
> ```bash
> .venv/Scripts/pip install -r requirements-dev.txt   # Windows
> .venv/bin/pip     install -r requirements-dev.txt   # Linux/macOS
> ```

---

## 配置

编辑 [`config.yaml`](config.yaml)：

1. **指向你的 LLM 服务**
   ```yaml
   llm:
     providers:
       local_main:
         type: "openai_compatible"
         base_url: "http://localhost:8080"    # 改成你的 llama-server 地址
         api_key: null
         models:
           - name: "your-model-name"
             capabilities: ["chat"]
     roles:
       self:             { provider: "local_main", model: "your-model-name" }
       hypothalamus:     { provider: "local_main", model: "your-model-name" }
       tentacle_default: { provider: "local_main", model: "your-model-name" }
   ```

2. **使用云端 API**（可选）——在环境变量中提供 key：
   ```bash
   # Windows (PowerShell)
   $env:DASHSCOPE_API_KEY = "sk-..."
   $env:ANTHROPIC_API_KEY = "sk-ant-..."
   # Linux/macOS
   export DASHSCOPE_API_KEY=sk-...
   export ANTHROPIC_API_KEY=sk-ant-...
   ```
   `config.yaml` 中的 `${DASHSCOPE_API_KEY}` / `${ANTHROPIC_API_KEY}` 会在启动时替换。

3. **心跳与疲惫度参数**（可选微调，默认值基本够用）：
   ```yaml
   hibernate:
     default_interval: 30   # 无输入时默认睡多久（秒）
     min_interval: 2
     max_interval: 300
   ```

---

## 运行

```bash
python main.py
```

启动后，程序进入心跳循环：
- 无输入时按 `default_interval` 秒数空转（默认 30s）
- 终端输入文本回车 → 标记为 adrenalin stimulus → **立刻打断 hibernate 唤醒 Self**
- Self 通过 Hypothalamus 翻译决策 → 调用 `Action` Tentacle
- Tentacle 回复通过 `[action] ...` 前缀打印到终端
- `Ctrl+C` 终止

### 示例会话

```
$ python main.py
[bootstrap] creating venv at ...\.venv              # 仅首次
[bootstrap] installing deps with ...\python.exe     # 仅首次
[bootstrap] relaunching in ...\.venv                # 仅首次
[HB #1] stimuli=0 (thinking...)
[HB #1] decision: (none)
[HB #1] hibernate 10s
hello                                              # ← 你输入（随时可敲，会立即唤醒）
[HB #2] stimuli=1 (thinking...)
[HB #2] decision: Use action tentacle to greet user.
[dispatch] action ← 'Greet the user' (adrenalin)
[action] Hi! How can I help you?
[HB #2] hibernate 10s
```

每次心跳都会打印三行：`stimuli=K (thinking...)` → `decision: ...` → `hibernate Ns`。
这样即使 Self 选择 "No action"，也能直观看到程序还活着。

> **注意**：实际效果依赖你接的 LLM 模型。小模型可能不严格按 `[THINKING]` / `[DECISION]` 格式输出——解析器有 fallback（无标记时整段视作 THINKING+DECISION），但强 instruction-following 模型（Qwen 2.5+、Claude、GPT-4 级）体验会好很多。

---

## Phase 0 验收标准

| # | 检查项 | 怎么验证 |
|---|--------|----------|
| 1 | 程序启动，心跳循环打印 | `python main.py` 后能看到 `[HB #N] ...` 日志 |
| 2 | 终端输入 `hello` → 唤醒 → Self → Hypothalamus → Action 回复 | 输入 `hello` 并回车，几秒内看到 `[action] <回复>` |
| 3 | 无输入时按 `default_interval` 空转 | 不输入，观察到 `hibernate Ns` 按配置秒数等待 |
| 4 | `Ctrl+C` 终止 | 按下 Ctrl+C 程序干净退出 |

## 怎么确认记忆在工作

### 实时（终端）

每次心跳都会打印一行 GM 状态：
```
[HB #5] gm: nodes=3 (+1), edges=2 (+2), fatigue=2%
```
- `(+N)` 表示本次心跳新增。一直是 `(+0)` 说明 `auto_ingest` / `explicit_write` / `compact` 都没触发。
- 节点数 = 总记忆条目数。fatigue% = `node_count / soft_limit × 100`。

### 离线（SQLite）

GM 数据库默认在 `workspace/data/graph_memory.sqlite`。任何时候都能用 `sqlite3` CLI 查（运行中查也安全，sqlite WAL 不会打架）：

```bash
# 节点数
sqlite3 workspace/data/graph_memory.sqlite "SELECT COUNT(*) FROM gm_nodes"

# 最近 20 个节点（按时间倒序）
sqlite3 workspace/data/graph_memory.sqlite \
  "SELECT id, category, source_type, name FROM gm_nodes ORDER BY id DESC LIMIT 20"

# 按类别统计
sqlite3 workspace/data/graph_memory.sqlite \
  "SELECT category, COUNT(*) FROM gm_nodes GROUP BY category"

# 看边
sqlite3 workspace/data/graph_memory.sqlite \
  "SELECT na.name, e.predicate, nb.name FROM gm_edges e
   JOIN gm_nodes na ON na.id=e.node_a
   JOIN gm_nodes nb ON nb.id=e.node_b LIMIT 20"

# 看哪些 auto 节点已被异步分类
sqlite3 workspace/data/graph_memory.sqlite \
  "SELECT id, category, name, json_extract(metadata,'\$.classified') AS classified
   FROM gm_nodes WHERE source_type='auto' ORDER BY id DESC LIMIT 20"
```

`source_type` 含义：`auto` = `auto_ingest` 写入的、还没经 LLM 分类；`explicit` = `explicit_write` (Self 说"记住...")；`compact` = 滑动窗口压缩出来的。

### 确认链路在跑

健康对话过几轮后应该能观察到：
1. 终端 `gm: nodes=N (+1)` 增量出现 → tentacle_feedback 在 auto_ingest
2. SQLite 里 `auto` 节点逐步出现 `classified=1` → 异步分类在跑
3. 长对话后 `compact` 节点出现 → 滑动窗口超限被压缩
4. `explicit` 节点出现 → Self 说过"记住..."

如果第 1 项一直 `+0`：tentacle 没回馈 / embedder 报错。看终端有无 `[runtime] auto_ingest error:`。


## 运行测试

```bash
pytest -q
```

全部单元+集成测试使用 **Mock LLM**，不会发起真实网络请求；因此测试即使在无 LLM 服务的环境下也能通过。

```bash
# 只跑某一模块
pytest tests/test_hypothalamus.py -q

# 详细输出
pytest -v
```

当前覆盖（截至 Phase 0 完成）：**68 tests / 约 4 秒**。

---

## 项目结构（简要）

```
KrakeyBot/
├── main.py                 # 根启动器
├── config.yaml             # 运行配置
├── src/
│   ├── main.py             # Runtime + 主循环
│   ├── self_agent.py       # Self 输出解析
│   ├── hypothalamus.py     # 决策→结构化翻译
│   ├── llm/client.py       # 统一 LLM Client
│   ├── models/             # config / stimulus / self_model
│   ├── prompt/             # DNA + builder
│   ├── runtime/            # stimulus_buffer / hibernate
│   ├── interfaces/         # Tentacle / Sensory ABC + Registry
│   ├── sensories/cli_input.py
│   └── tentacles/action.py
├── tests/                  # pytest，全 Mock，无网络依赖
└── workspace/              # 运行时数据（gitignored）
    ├── data/
    └── logs/
```
## 常见问题

**Q: 启动后没反应？**
- 确认 `config.yaml` 中的 `base_url` 指向**正在运行**的 LLM 服务
- `curl http://localhost:8080/v1/models` 能否拿到响应？
- 看看终端是否在等 hibernate（Self 可能选择 30s+ 间隔）——输入点东西敲回车试试

**Q: Self 不回复用户？**
- Phase 0 没有记忆，Self 每次心跳上下文都是空的
- 模型太小可能输出格式乱；换个稍大一点的模型（Qwen 2.5 7B+ / Claude Haiku+）
- 把 `[DECISION]` 内容打印出来看 Self 到底决定做什么

**Q: 我的输入打在屏幕上和程序 log 搅在一起？**
- 是的，Phase 0 没有 UI 层，stdin 和 stdout 混用。后续 Phase 会考虑。

---

## License
MIT
