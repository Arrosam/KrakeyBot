# KrakeyBot

> 以持续心跳维持"存在"的自主认知 Agent。
> 完整设计见 [`KrakeyBot_DevSpec_v2.1.md`](KrakeyBot_DevSpec_v2.1.md)，实现路线见 [`KrakeyBot_Checklist.md`](KrakeyBot_Checklist.md)。
> 沙盒 VM 配置见 [`SANDBOX.md`](SANDBOX.md)（启用 coding/GUI/file/browser 任一 tentacle 前必读）。

---

## 当前阶段: **Phase 3 — 扩展能力**

### 已实现

**Phase 0（基础）**
- `Self` 心跳循环 + 4 分区输出解析 (`[THINKING]` / `[DECISION]` / `[NOTE]` / `[HIBERNATE]`)
- `Hypothalamus`（下丘脑）自然语言 → 结构化 JSON 翻译层（Sleep vs Hibernate 明确区分）
- `Tentacle` / `Sensory` 注册器 + `Action` Tentacle + `CLI Input` Sensory
- `Stimulus Buffer` 含 adrenalin 打断 + peek 预加载接口
- Adrenalin 继承（Hypothalamus → Tentacle 回馈 stimulus）
- 统一 `LLMClient`（OpenAI-compatible + Anthropic）
- Self-Model YAML 读写
- 配置加载（`${VAR}` 环境变量替换 + fatigue 阈值校验）
- 心跳可见日志 + Self/Hypothalamus 错误兜底

**Phase 1（记忆系统）**
- **Graph Memory** — SQLite + `sqlite-vec` + FTS5。5 类节点（FACT/RELATION/KNOWLEDGE/TARGET/FOCUS），9 类边，**无环约束**（插入时 CTE 连通检测，必要时创建 RELATION 桥节点）
- **两种写入路径**
  - `auto_ingest`：零 LLM，embedding 相似度 ≥ 0.92 → 节点 importance +0.5 去重；否则新建 FACT
  - `explicit_write`：LLM 提取 nodes + edges（Self 说"记住..." 触发）
- **异步分类**：`classify_and_link_pending()`，每心跳后后台 LLM 对未分类 auto 节点重新归类 + 建边
- **IncrementalRecall** — per-stimulus 向量搜索，adrenalin 权重 ×10，三层排序：
  - Layer 1：cosine top-K
  - Layer 2：Reranker 精排（URL 调用，失败自动降级）
  - Layer 3：脚本加权（向量 × 0.3 × 时间衰减 × 0.2 × log(访问) × 0.5 × importance × 0.5 × 类型权重）
  - FTS5 降级：embedding endpoint 不可用时自动启用
- **滑动窗口 + Compact** — 按 token 动态窗口，超限自动 LLM 压缩旧轮次为 GM 节点/边；单轮过大拆 chunks
- **疲惫度** — `gm_node_count / soft_limit × 100%`，Status 中渲染分级提示
- **Batch Tracker Sensory** — 派出去的 tentacle 全部回馈 → adrenalin 唤醒；支持 `extend_batch`
- **hibernate_with_recall** — hibernate 期间 `peek_unrecalled` 喂给 IncrementalRecall 预加载
- **主循环** — §6.4 完整顺序：drain → incremental add → compact → finalize → uncovered 退回 → Self → auto_ingest feedback → Hypothalamus → classify (async) → hibernate with recall

**Phase 2（完整生命周期）**
- **Bootstrap** — 首次启动读 GENESIS.md + 4 阶段 Self-Model 填充：
  - `<self-model>{...}</self-model>` 标签（在 [NOTE] 中）→ runtime 深度合并到 self_model.yaml
  - Bootstrap 期间心跳固定 10s
  - Self 在 [NOTE] 写 `bootstrap complete` → 进入正常运行
- **Knowledge Base 集群** — 每个 KB = 独立 SQLite (`workspace/data/knowledge_bases/*.sqlite`)，含 entries + edges + FTS5；GM 中的 `kb_registry` 表追踪所有 KB
- **memory_recall 跟随 KB 索引** — Self 调 `memory_recall` 命中 KB 索引节点 (metadata `is_kb_index=true`) → 自动从该 KB 拉相关 entries；也支持 `params={"kb_id": "X"}` 直查 KB
- **7-phase Sleep** — 完整生命周期转换：
  1. 暂停非紧急 Sensory（保留 adrenalin 监听）
  2. **Leiden 聚类** (igraph + leidenalg) → 社区摘要 + embedding
  3. FACT/RELATION/KNOWLEDGE 节点按社区迁入 KB（含同社区边）
  4. 已完成 TARGET → FACT → 已迁；未完成 TARGET 留在 GM
  5. FOCUS 全部清理
  6. 重建 GM Index Graph：每 KB 一个 KNOWLEDGE 索引节点 + LLM 提取跨 KB 边
  7. 写每日 sleep 日志 (`workspace/logs/sleep-YYYY-MM-DD.jsonl`)
- **Force Sleep** — fatigue ≥ `force_sleep_threshold` 自动触发 sleep + 醒来推 "之前因过于疲劳昏睡过去了" stimulus
- **Voluntary Sleep** — Self 说"进入睡眠模式" → Hypothalamus `sleep:true` → 同样 7 phase + 醒来推清爽 stimulus
- **共享持久化层** — `src/memory/_db.py` 提取 GM/KB 共用的 sqlite-vec / cosine / FTS5 / embedding 编解码

**Phase 3（扩展能力）**
- **Override 命令** — `/status` `/memory_stats` `/sleep` `/wake` `/kill` 在 CLI 输入时被 runtime 拦截, 不经 Self
- **Search Tentacle** — DuckDuckGo (ddgs)，无需 API key；内部输出 (Self 决定是否转告用户)
- **Telegram Sensory + ReplyTentacle** — 双向：getUpdates 长轮询拉消息推 buffer; sendMessage 出站；同一 HttpTelegramClient 共用
- **Coding Tentacle** — Python / shell 子进程 + 超时 (默认禁用; 不是真沙盒)
- **GUI Control Tentacle** — PyAutoGUI 鼠标 / 键盘 / 截图 / 屏幕尺寸 (默认禁用; 鼠标拖到屏幕角可中断)
- **GM 性能跑分** — `python scripts/bench_gm.py` 测各规模延迟, 推荐 `gm_node_soft_limit` 值
- **Tentacle.is_internal 标志** — 区分"对外说话" (绿) vs "Self 内部参考" (紫); memory_recall / search / coding / gui_control 都是 internal

### 尚未实现（留给后续 Phase）

| 功能 | 说明 |
|------|------|
| 可视化 Dashboard | 浏览器看 GM 图 + 心跳流; 设计待讨论 |

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

---

## Phase 3 验收标准

| # | 检查项 | 怎么验证 |
|---|---|---|
| 1 | `/status` 返回 runtime + GM 状态 | CLI 输入 `/status` → 终端打印 `name=Krakey heartbeats=N gm_nodes=M ...` |
| 2 | `/memory_stats` 返回分类计数 + KB 列表 | CLI 输入 `/memory_stats` |
| 3 | `/sleep` 立即触发 7-phase 睡眠 | CLI 输入 `/sleep` → 终端打 `sleep started` → GM 节点重组 |
| 4 | `/kill` 优雅退出 | CLI 输入 `/kill` → runtime 退出 |
| 5 | Search Tentacle 真返回结果 | Self 派 `tentacle:search` → 紫色 `[search] Search results for ...` |
| 6 | Telegram 双向 | 配 `bot_token` + `default_chat_id` → 在 Telegram 给 Krakey 发消息 → 终端 stimulus 显示 → Self 用 `telegram_reply` 回复, 你在 Telegram 收到 |
| 7 | Coding Tentacle 执行 | `enabled: true` 后, Self 派 coding tentacle 跑 `print('hello')` → 紫色 `[coding] exit=0 stdout: hello` |
| 8 | GUI Tentacle 真控屏 (有风险) | `enabled: true` + 派 `screenshot` → `workspace/screenshots/*.png` 出现 |
| 9 | 性能跑分 | `python scripts/bench_gm.py --sizes 100 200 500 --target-ms 100` → 输出推荐 soft_limit |

---

## Phase 2 验收标准

| # | 检查项 | 怎么验证 |
|---|--------|----------|
| 1 | 首次启动进入 Bootstrap | 删 `workspace/self_model.yaml` → `python main.py` → Self 读 GENESIS → `<self-model>` 更新 → 写 "bootstrap complete" |
| 2 | Bootstrap 完成后 self_model.yaml 非空 | `cat workspace/self_model.yaml` → identity / goals 已填 |
| 3 | 长对话触发 Sleep（自愿） | Self 累积疲惫后说 "进入睡眠模式" → 终端打 `sleep started` → `workspace/data/knowledge_bases/` 出现 KB 文件 |
| 4 | Sleep 后 GM 主要剩 TARGET + KB 索引节点 | `sqlite3 workspace/data/graph_memory.sqlite "SELECT category, COUNT(*) FROM gm_nodes GROUP BY category"` → FACT/RELATION 大幅减少，KNOWLEDGE 含 KB 索引 |
| 5 | 醒来后能 recall 到 KB | 提某个迁移过的话题 → `[GRAPH MEMORY]` 含 KB 索引节点 + memory_recall 拉出 KB entries |
| 6 | 强制 Sleep | 改 `config.yaml` 把 `gm_node_soft_limit` 设小 (如 5) → 几次心跳后达 120% → 自动 sleep + 醒来含 "昏睡" stimulus |
| 7 | 主动 KB 浏览 | Self 说 "回忆 KB 'community_X'" → tentacle dispatch memory_recall with kb_id → 返回该 KB entries |
| 8 | sleep 日志 | `cat workspace/logs/sleep-*.jsonl` 看见迁移 / 聚类 / 索引数据 |

---

## Phase 1 验收标准

| # | 检查项 | 怎么验证 |
|---|--------|----------|
| 1 | 对话写入 GM（auto_ingest） | `sqlite3 workspace/data/graph_memory.sqlite "SELECT count(*) FROM gm_nodes"` → 对话后非零 |
| 2 | 后续对话 recall 返回相关节点 | 先让 bot 记录 "A 喜欢吃苹果"，再提"苹果"，观察 [GRAPH MEMORY] 区段出现 A 节点 |
| 3 | 长对话触发 compact → GM 节点增长 | 连续多轮对话后节点数明显增长 |
| 4 | 异步分类：auto FACT 节点被重新分类 | `SELECT category, source_type FROM gm_nodes` → auto 节点类别逐渐从 FACT 变为 RELATION/KNOWLEDGE |
| 5 | Reranker 不可用自动降级 | 关闭 reranker server，观察程序继续运行（tests 锁死此行为） |

---

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

完整目录规划见 DevSpec §16。

---

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

见 [`LICENSE`](LICENSE)。
