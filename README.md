# KrakeyBot

> 以持续心跳维持"存在"的自主认知 Agent。
> 沙盒 VM 配置见 [`SANDBOX.md`](SANDBOX.md)（启用 coding/GUI/file/browser 任一 tool 前必读）。
> 自定义 tool / sensory 见 [`PLUGINS.md`](PLUGINS.md)（把插件项目丢进 `workspace/plugins/<project>/` 自动加载）。

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

### 用户 — 从 PyPI 装

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install krakey
```

装完 `krakey` 命令直接可用：

```bash
krakey --version          # krakey 0.1.0
krakey                    # 显示 help
krakey onboard            # 走配置向导
krakey run                # 启动心跳
```

升级：`pip install -U krakey`。

### 开发者 — 从源码 editable 装

```bash
git clone https://github.com/Arrosam/KrakeyBot.git
cd KrakeyBot
python -m venv .venv && source .venv/bin/activate   # mac/linux
pip install -e ".[dev]"                              # 含 pytest 等
pytest -q                                            # 560+ 测试，应该全绿
```

editable 模式下 `krakey update` / `krakey repair` 会基于 git tag 操作仓库；
PyPI 用户跑 `krakey update` 会得到提示，让用 `pip install -U krakey` 替代。

---

## 配置

**首次安装：跑一遍 onboarding 向导**，自动生成 `config.yaml`：

```bash
krakey onboard
```

向导会引导你三步走：
1. 选一个 chat LLM provider（label / base URL / API key / 模型名），自动绑到 `self_thinking` + `compact` + `classifier` 核心用途
2. 可选：再配一个 embedding provider/模型（recall + KB 索引需要它）
3. 勾选要启用的插件（**dashboard 默认勾选并强烈推荐**——不然没有任何 in-app 方式查看 Krakey 的状态）

向导可以反复跑：已存在的 `config.yaml` 会先备份到 `workspace/backups/` 再覆盖。

**云端 API**：把 key 写进环境变量，然后在 `config.yaml` 里用 `${ENV_VAR}` 占位符引用：
```bash
# Windows (PowerShell)
$env:DASHSCOPE_API_KEY = "sk-..."
# Linux/macOS
export DASHSCOPE_API_KEY=sk-...
```
`config.yaml` 里写 `api_key: ${DASHSCOPE_API_KEY}`，加载时自动替换。

**心跳 / 疲惫度参数**（可选微调，默认值基本够用）：
```yaml
hibernate:
  default_interval: 30   # 无输入时默认睡多久（秒）
  min_interval: 2
  max_interval: 300
```

---

## 运行

`krakey` CLI 提供两种运行模式：

```bash
krakey run        # 前台运行（终端附着，Ctrl+C 退出）
krakey start      # 后台守护进程（detach，写 pidfile + 日志）
krakey stop       # 停掉后台进程
krakey status     # 查询当前状态（运行中 / 已停止 / 版本号 / 日志路径）
```

后台模式下：
- pidfile：`workspace/.krakey.pid`
- 日志：`workspace/logs/daemon.log`
- 守护进程接 `SIGTERM` 优雅退出，10s 超时则强杀

启动后程序进入心跳循环：
- 无输入时按 `default_interval` 秒数空转（默认 30s）
- 终端输入文本回车（仅 `krakey run` 模式）→ 标记为 adrenalin stimulus → **立刻打断 hibernate 唤醒 Self**
- Self 通过 Hypothalamus 翻译决策 → 调用 `Action` Tool
- Tool 回复通过 `[action] ...` 前缀打印到终端
- `Ctrl+C` (或后台模式下 `krakey stop`) 终止

### 示例会话

```
$ krakey run
[HB #1] stimuli=0 (thinking...)
[HB #1] decision: (none)
[HB #1] hibernate 10s
hello                                              # ← 你输入（随时可敲，会立即唤醒）
[HB #2] stimuli=1 (thinking...)
[HB #2] decision: Use action tool to greet user.
[dispatch] action ← 'Greet the user' (adrenalin)
[action] Hi! How can I help you?
[HB #2] hibernate 10s
```

每次心跳都会打印三行：`stimuli=K (thinking...)` → `decision: ...` → `hibernate Ns`。
这样即使 Self 选择 "No action"，也能直观看到程序还活着。

> **注意**：实际效果依赖你接的 LLM 模型。小模型可能不严格按 `[THINKING]` / `[DECISION]` 格式输出——解析器有 fallback（无标记时整段视作 THINKING+DECISION），但强 instruction-following 模型（Qwen 2.5+、Claude、GPT-4 级）体验会好很多。

---

## 升级 / 修复 / 卸载

```bash
krakey update      # 拉取 origin 最新 release tag (vX.Y.Z) 并重装
krakey repair      # 强制 checkout 当前版本的 release tag (会丢弃本地未提交改动，会先确认)
krakey uninstall   # pip uninstall krakey（保留 repo / config / workspace）
krakey uninstall --full   # 同时删掉整个 repo 目录（config + workspace + .venv 全删，会先确认）
```

`update` 要求工作树干净（无未提交改动），否则会要求你先 commit/stash。
`repair` 用于 repo 文件被改坏后回到某个 release 版本。
版本号即 `pyproject.toml` 里的 `[project] version`，git tag `vX.Y.Z` 必须与之对齐。

---

## Phase 0 验收标准

| # | 检查项 | 怎么验证 |
|---|--------|----------|
| 1 | 程序启动，心跳循环打印 | `krakey run` 后能看到 `[HB #N] ...` 日志 |
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
1. 终端 `gm: nodes=N (+1)` 增量出现 → tool_feedback 在 auto_ingest
2. SQLite 里 `auto` 节点逐步出现 `classified=1` → 异步分类在跑
3. 长对话后 `compact` 节点出现 → 滑动窗口超限被压缩
4. `explicit` 节点出现 → Self 说过"记住..."

如果第 1 项一直 `+0`：tool 没回馈 / embedder 报错。看终端有无 `[runtime] auto_ingest error:`。


## 运行测试

```bash
pip install -e ".[dev]"
pytest -q
```

全部单元+集成测试使用 **Mock LLM**，不会发起真实网络请求；因此测试即使在无 LLM 服务的环境下也能通过。

```bash
# 只跑某一模块
pytest tests/test_hypothalamus.py -q

# 详细输出
pytest -v
```

---

## 项目结构（简要）

```
KrakeyBot/
├── pyproject.toml          # 安装 / 依赖 / krakey 入口配置
├── config.yaml             # 运行配置（首次 onboard 后生成）
├── src/
│   ├── cli/                # `krakey` 命令行 (run/start/stop/onboard/update/...)
│   ├── main.py             # Runtime + 主循环
│   ├── self_agent.py       # Self 输出解析
│   ├── hypothalamus.py     # 决策→结构化翻译
│   ├── llm/client.py       # 统一 LLM Client
│   ├── models/             # config / stimulus / self_model
│   ├── prompt/             # DNA + builder
│   ├── runtime/            # stimulus_buffer / hibernate / fatigue
│   ├── memory/             # GraphMemory / KnowledgeBase / recall
│   ├── sleep/              # 7 阶段 Sleep 管线
│   ├── dashboard/          # FastAPI + WS + web chat history
│   ├── sandbox/            # SubprocessRunner + guest VM backend
│   ├── interfaces/         # Tool / Sensory ABC + Registry
│   └── plugins/
│       ├── loader.py       # 插件发现 + 安全 import
│       └── builtin/        # 内置插件项目（search / coding / ...）
├── tests/                  # pytest，全 Mock，无网络依赖
└── workspace/              # 运行时数据（gitignored）
    ├── data/
    ├── logs/               # daemon.log 在这里
    ├── .krakey.pid         # daemon 模式下的 pidfile
    └── plugins/            # 用户自定义插件（可选）
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

**Q: `krakey start` 启动失败但 `krakey run` 能跑？**
- 看 `workspace/logs/daemon.log` —— 后台模式下 stdout/stderr 都被重定向过去
- 检查 pidfile 是否陈旧：`krakey status` 会自动清理无效 pidfile

---

## License
MIT
