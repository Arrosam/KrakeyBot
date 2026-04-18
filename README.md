# CogniBot (KrakeyBot)

> 以持续心跳维持"存在"的自主认知 Agent。
> 完整设计见 [`CogniBot_DevSpec_v2.1.md`](CogniBot_DevSpec_v2.1.md)，实现路线见 [`CogniBot_Checklist.md`](CogniBot_Checklist.md)。

---

## 当前阶段: **Phase 1 — 记忆系统**

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

### 尚未实现（留给后续 Phase）

| 功能 | Phase |
|------|-------|
| Bootstrap 首次启动 / Knowledge Base 集群 / 7-phase Sleep | 2 |
| Telegram Sensory / 更多 Tentacles / Override 命令 / Dashboard | 3 |

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
