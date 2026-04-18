# CogniBot (KlarkyBot)

> 以持续心跳维持"存在"的自主认知 Agent。
> 完整设计见 [`CogniBot_DevSpec_v2.1.md`](CogniBot_DevSpec_v2.1.md)，实现路线见 [`CogniBot_Checklist.md`](CogniBot_Checklist.md)。

---

## 当前阶段: **Phase 0 — 最小心跳循环**

### 已实现

- `Self` 心跳循环 + 4 分区输出解析 (`[THINKING]` / `[DECISION]` / `[NOTE]` / `[HIBERNATE]`)
- `Hypothalamus`（下丘脑）自然语言 → 结构化 JSON 翻译层
- `Tentacle` / `Sensory` 注册器 + `Action` Tentacle + `CLI Input` Sensory
- `Stimulus Buffer` 含 adrenalin 打断 + peek 预加载接口
- `hibernate` 等待 + adrenalin 唤醒 + 间隔 clamp
- Adrenalin 继承 (Hypothalamus → Tentacle 回馈 stimulus)
- 统一 `LLMClient`（OpenAI-compatible + Anthropic）
- Self-Model YAML 读写
- 配置加载（`${VAR}` 环境变量替换 + fatigue 阈值校验）

### 尚未实现（留给后续 Phase）

| 功能 | Phase |
|------|-------|
| Graph Memory / Recall / Compact | 1 |
| 疲惫度 / Batch Tracker / 异步分类建边 | 1 |
| Bootstrap / Knowledge Base / 7-phase Sleep | 2 |
| Telegram / 更多 Tentacles / Dashboard | 3 |

Phase 0 的 `Self` 每次心跳看不到历史记忆，滑动窗口仅用于当前会话短期上下文；长期记忆能力在 Phase 1 接入。

---

## 环境要求

- **Python**: 3.12（也兼容 3.11）
- **LLM Server**: OpenAI 兼容的 `/v1/chat/completions` endpoint
  - 推荐本地 `llama-server` / `llama.cpp` / `vllm` / `lmstudio` / `ollama(openai兼容模式)`
  - 也可用云端服务：DashScope、Anthropic Claude（配置中已预置示例）
- Embedding / Reranker endpoint 在 Phase 0 **尚未用到**，可以暂不启动

---

## 安装

```bash
git clone <repo>
cd KlarkyBot

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements-dev.txt
```

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
# (Self 第 1 次心跳，无 stimulus，进入 hibernate 30s)
hello                                # ← 你输入
[action] Hi! How can I help you?     # ← Action Tentacle 回复
# (Self 继续心跳，等待下一个输入)
```

> **注意**：实际效果依赖你接的 LLM 模型。小模型可能不严格按 `[THINKING]` / `[DECISION]` 格式输出——解析器有 fallback（无标记时整段视作 THINKING+DECISION），但强 instruction-following 模型（Qwen 2.5+、Claude、GPT-4 级）体验会好很多。

---

## Phase 0 验收标准

| # | 检查项 | 怎么验证 |
|---|--------|----------|
| 1 | 程序启动，心跳循环打印 | `python main.py` 后能看到 `[action] ...` 或 hibernate 过程 |
| 2 | 终端输入 `hello` → 唤醒 → Self → Hypothalamus → Action 回复 | 输入 `hello` 并回车，几秒内看到 `[action] <回复>` |
| 3 | 无输入时按 `default_interval` 空转 | 不输入，观察大约 30 秒（默认）才开始下一次心跳 |
| 4 | `Ctrl+C` 终止 | 按下 Ctrl+C 程序干净退出 |

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
KlarkyBot/
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
