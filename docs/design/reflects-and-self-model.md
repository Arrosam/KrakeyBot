# Reflects + Self-model 瘦身 + 回忆层 LLM — 设计草案

> 状态：**草案 / 讨论中**。本文只是 Samuel 口述的需求固化，未开始实现。
> 修改或补充请直接编辑本文件；实现前会开对应的 PR 讨论贴。
> 记录日期：2026-04-25。

---

## 🔒 核心设计原则（载重不变量, 2026-04-25 / 2026-04-26）

> **删除 / 关闭任何插件 (Reflects, tentacles, sensories) 都不应该
> 影响本体 (runtime) 的运行。**

> **插件代码在用户启用前不会被加载。** 启用前唯一允许的影响是
> 把"配置选项"以纯文本格式暴露给 Web UI（Web UI 读 ``meta.yaml`` 得到
> 元数据，**不会** import 任何 plugin 代码）。

> **插件无法访问中央 ``config.yaml`` (含 API key + provider 配置)。**
> 插件 Python 代码看到的只有：(1) 自己 folder 下的 ``config.yaml``
> 纯文本设置（通过 ``ctx.config`` 获取）；(2) ``ctx.get_llm_for_tag(tag_name)``
> 返回的 ``LLMClient`` 实例。插件读自己的 config 拿到 ``llm_purposes``
> 里绑定的 tag 名，再向 runtime 要对应的 client。LLMClient 内部封装
> provider 信息，但插件**拿不到底层 API key**。

## 🏷️ Tag-based LLM 系统（2026-04-26）

三级抽象，每一级单一职责：

```
1. providers (中央 config.yaml)
   API 连接 + 密钥。runtime 唯一可见, 插件永远看不到。

2. tags (中央 config.yaml)
   语义命名 → (provider, model, params) 三元组。
   provider 字段 = "<provider_name>/<model_name>" 紧凑字符串
   (按第一个 / 切; provider 名不能含 /; model 名可以)。

3. assignments (purpose → tag)
   - core 用途 (Self/compact/classifier ...): 中央
     llm.core_purposes:
   - 插件用途: 该插件 folder 下的 config.yaml
     llm_purposes:
       <purpose>: <tag>
```

**特殊 model-type slot**（不是 purpose, 是 capability):

```yaml
llm:
  embedding: <tag_name>     # 必填用于 GM auto-recall
  reranker: <tag_name>      # 可选, 用于 recall 重排
```

**完整中央配置示例**:

```yaml
llm:
  providers:
    "One API": {type: openai_compatible, base_url: ..., api_key: ...}
    SiliconFlow: {type: openai_compatible, ..., api_key: ...}
  tags:
    fast_generation:
      provider: "One API/qwen3.6-9b"
      params: {max_output_tokens: 512, temperature: 0.3}
    high_performance:
      provider: "One API/astron"
      params: {max_output_tokens: 8192, reasoning_mode: medium}
    text_compression:
      provider: "SiliconFlow/qwen3.6-27b"
      params: {max_output_tokens: 4096}
    embed_tag:
      provider: "SiliconFlow/BAAI/bge-m3"
    rerank_tag:
      provider: "SiliconFlow/BAAI/bge-reranker-v2-m3"
  core_purposes:
    self_thinking: high_performance
    compact: text_compression
    classifier: text_compression
  embedding: embed_tag
  reranker: rerank_tag
```

**插件端**（``workspace/reflects/<name>/config.yaml``）:

```yaml
llm_purposes:
  <plugin's purpose name>: <tag name>
```

**插件 ``meta.yaml`` 端**:

```yaml
llm_purposes:
  - name: <purpose name>
    description: ...
    suggested_tag: <hint, optional>     # 不会自动绑定
```

**Plugin factory 签名**:

```python
def build_reflect(ctx: PluginContext) -> Reflect | None:
    # 插件读自己的 config 拿到绑定的 tag 名，
    # 再向 runtime 要对应的 LLMClient。
    purposes = ctx.config.get("llm_purposes") or {}
    tag = purposes.get("translator") if isinstance(purposes, dict) else None
    llm = ctx.get_llm_for_tag(tag)
    if llm is None:
        return None    # 没绑定 → 跳过自己注册 (additive 原则)
    return MyReflect(llm)
```

**LLMClient 共享**: 多个 purpose 映射到同一个 tag → 共用一个客户端
实例 (省连接 + 一致的 rate-limit 计数)。共享缓存 ``deps.llm_clients_by_tag``
跨 core 和插件路径共用。

**迁移**: 旧 ``llm.roles:`` shape 已删除。loader 检测到则**loud 报错并
退出**，附详细迁移说明。配置无变化时 runtime 无法启动 — 强制用户显式
迁移。



含义:
- runtime 的核心 heartbeat 循环必须能在**零插件**状态下跑起来。
  Self 拿到没有 recall 节点的空 `[GRAPH MEMORY]`、没有外部 stimuli、
  没有 tentacles 可调用的 prompt, 仍然能完成完整 heartbeat。
- 所有插件**严格加性**。任何一个被禁用 / 卸载, runtime 不抛异常、
  不挂 phase、不进入坏状态。
- 落实方式: 每个插件在 runtime 调用站点都**有 fallback** ——
  null-object (e.g. `NoopRecall`) 或者 soft-fail (e.g. tentacle 派发
  时找不到名字 → 推 `Unknown tentacle: X` 系统事件给 Self 看, 不抛)。
- 测试上要求: 无任何 Reflect 注册时 runtime 仍能跑完整 heartbeat。
  这是 regression 防线, 任何破坏这个不变量的改动会被测试拒绝。

这条原则**优先级高于**"默认行为"——本节决定了**结构上的可靠性**,
"默认开启什么 Reflect"只是 UX 选择。

---

## 背景 / 动机

讨论 Self-model 字段时发现几个长期问题：

1. **`statistics` 字段** 大部分不被写入，却占 `[SELF-MODEL]` prompt 层的大段 token。
2. **`relationships.users`** 把"和谁有关系"显式结构化 —— 违反"重要记忆应通过 GM 召回自然涌现"的设计哲学。
3. **`is_sleeping`** Self 永远看不到 `true`（sleep 期间 Self 不跑），是个死字段。
4. **`mood_baseline`** 简单字段无法承载情绪这种复杂涌现现象。
5. **`focus_topic` / `goals.active`** 与 GM 里的 `FOCUS` / `TARGET` 节点是双重实现；GM 版本更强（边、importance、sleep migration），self-model 版本是冗余。

与此同时，Samuel 提出了三个更激进的想法：

- **回忆层 LLM（recall-layer LLM）**：在处理 stimulus 前加一层专门的 LLM，从 stimulus + 历史提取"回忆特征点"，用这些特征去驱动 GM 召回。比现在"stimulus 原文向量搜索"精确得多，尤其能主动召回发言人印象、相关情景等。
- **Reflect 插件类型**：比 tentacle / sensory 更深的扩展点，监听心跳开始/结束事件，可以接管或替换 runtime 核心机制。
- **默认机制 = 默认 Reflect**：现有 Hypothalamus 和 auto-recall 都改写成内置 Reflect，用户可选择替换或关闭。

---

## Part 1 — Self-model 瘦身

### 最终保留的 schema

```yaml
identity:
  name: ""
  persona: ""
state:
  bootstrap_complete: false
```

**理由**：
- `identity` 是 Self 的不变核心，Bootstrap 后由 Self 通过专门路径更新（不再用 `<self-model>` tag 黑魔法，见 Reflect #3 `in_mind`）。
- `bootstrap_complete` 是 runtime 必需的开关。

### 被删除的字段 + 替代方案

| 字段 | 替代 |
|---|---|
| `statistics.*` | 搬到 `workspace/data/runtime_stats.json`（或一个 `system:*` 节点），仅 dashboard 使用，不进 prompt |
| `relationships.users` | 通过 GM 的 RELATION 节点 + 回忆层 LLM 自然涌现（Part 2） |
| `state.mood_baseline` | 删除。真正的情绪通过 `in_mind` Reflect 的短语描述承载（Part 3） |
| `state.is_sleeping` | 删除。runtime in-memory flag 即可，不需要落盘 |
| `state.focus_topic` | 用 GM 的 `category=FOCUS` 节点替代（已存在机制） |
| `state.energy_level` | 删除。如果需要"累"的概念，用 fatigue% 就够了 |
| `goals.active` | 用 GM 的 `category=TARGET` 节点替代 |
| `goals.completed` | 不单独存；TARGET 完成后 sleep 会降级为 FACT，这就是"已完成的目标"的自然表达 |

### 影响面

- `src/models/self_model.py`: 简化 `default_self_model()`
- `src/main.py`: 删除 sleep 后的 `total_sleep_cycles` 更新逻辑（搬到别处）
- `src/prompt/builder.py`: `[SELF-MODEL]` 层变短，prompt 缓存更稳
- 所有 `test_*.py` 里依赖这些字段的测试需要更新

---

## Part 2 — 回忆层 LLM

### 思路

在 `_phase_drain_and_seed_recall` 之后、Self LLM 调用之前插入一个**回忆层 LLM**，它读当前所有 stimulus + 滚动历史，输出"要用来召回的特征点列表"。

**现在**：
```
stimulus.content → embed → vec_search → ranked merge
```
纯原文向量搜索，精度受限：
- 用户说"hi"召不到"Samuel 聊过 Agent 架构"的记忆（语义不够近）
- 无法抽离"要召回哪些人 / 哪些事件 / 哪些相关主题"

**新方案**：
```
stimuli + window → 回忆层 LLM → [
  {anchor: "user:Samuel", reason: "消息来自 Samuel"},
  {anchor: "topic:Agent architecture", reason: "话题延续自上下文"},
  {anchor: "event:刚结束 Bootstrap", reason: "时间性关联"},
] → 每个 anchor 分别触发召回（向量 / FTS / name 精确匹配）→ 合并权重
```

LLM 的 prompt 模板（finalized 2026-04-25）：

```
# Memory Recall Guide

为下一轮决策挑选回忆关键词。读输入, 输出关键词列表, 用于检索图记忆。

## 输入

[CURRENT_STIMULI]
{stimuli}

[RECENT_HISTORY]
{history}

## 提取标准

✅ 选作 anchor:
- 具体人名
- 具体话题或概念
- 具体事件或里程碑
- 领域术语
- 时间关联指代

❌ 不选:
- 功能词、问候语
- 宽泛词
- 刺激原文里的整句
- 长句子

最多 8 个 anchor。无明显可回忆点时输出空列表。

## 输出 (严格 JSON, 无其他文字, 无 markdown 代码块)

{"anchors": ["..."]}

## 示例

示例 1
[CURRENT_STIMULI]
[1] user_message from sensory:chat:
    "Alex: 那个优化方案最后跑出来速度怎么样？"
[RECENT_HISTORY]
Heartbeat #N-1: Decision: "用 Cython 重写 hot loop"
预期输出: {"anchors": ["Alex", "Cython optimization", "performance benchmark"]}

示例 2
[CURRENT_STIMULI]
[1] user_message from sensory:chat: "Bob: 哦。"
[RECENT_HISTORY]
(空)
预期输出: {"anchors": []}

示例 3
[CURRENT_STIMULI]
[1] tentacle_feedback from tentacle:weather_check: "Sunny, 22°C"
[2] batch_complete from sensory:batch_tracker: "All dispatched."
[RECENT_HISTORY]
Heartbeat #N-1: Decision: "Check the weather to plan tomorrow's hike"
预期输出: {"anchors": ["weather check", "hiking plan"]}
```

**模板设计决策**：

- **没有 role 介绍 / 不解释 KrakeyBot 是什么** —— 任务一行话讲清楚。
  少说一句话就少一个出错点（不需要再交代"你不是 KrakeyBot"）。
- **没有 ``<thinking>`` 块或任何 CoT 脚手架** —— 让用户自己挑模型。
  推理模型自己 reason；非推理模型表现差是模型选择的代价，不在核心
  机制里烘焙拐杖（和 Reflect #1 默认关闭的哲学一致）。
- **anchors 是扁平字符串列表**，没有 ``reason`` / ``kind`` /
  ``rationale`` 等字段，因为下游不消费，传输纯浪费。
- **不传 GM 现状** —— 黑箱。anchor 来自情境语义而非"已存在性"，让
  vec_search / FTS / name 精确匹配自己处理"找不到"的情况。
- **不传 agent identity slot** —— in_mind 通过历史层最近端注入
  （见 Reflect #3 设计），其他自我相关的上下文从历史 / stimulus 里
  天然可推。
- **示例用通用人名 / 话题（Alex / Bob / Cython / weather）** —— 不
  污染 prompt 模板自身的语义中性，避免 LLM 把示例里的 "Samuel" /
  "ReAct" 这种实际对话内容当成隐含上下文。

### 与 Reflect #2 的关系

这个功能就是 **Reflect #2 — 回忆特征提取器**。详见 Part 3。

### 开关语义

- **开启**：上述 LLM-driven 的 anchor 抽取 → 多路召回
- **关闭**：走现在的脚本化纯向量召回（不变）

两种模式都共享 recall 的排序、token-budget 裁剪等后处理 —— 只是**召回候选从哪来**的差异。

---

## Part 3 — Reflect 插件系统

### 定位

Reflect 是**深度插件** —— 不同于 tentacle（outbound 肢体）和 sensory（inbound 感官），它听取 **每个心跳的开始 / 结束** 事件，可以替换或拦截 runtime 核心机制。

**核心约束**：Reflect 不是任意 monkey-patching，而是在 runtime 明确的 hook 点上插入。所有 hook 点都应该能被 grep 找到。

### 基础协议（草案）

```python
class Reflect(Protocol):
    name: str
    kind: str  # "hypothalamus" | "recall_anchor" | "in_mind" | ...
    enabled: bool

    async def on_heartbeat_start(self, ctx: HeartbeatContext) -> None: ...
    async def on_heartbeat_end(self, ctx: HeartbeatContext) -> None: ...

    # kind-specific hooks，各种子类自定义
```

具体 hook 点按 kind 定义。例如 `kind="hypothalamus"` 的 Reflect 覆盖"翻译 Self Decision → tentacle calls"这一步。

### 三个要做的 Reflect

#### Reflect #1 — `hypothalamus`（可关闭的 Hypothalamus）

**重要前提（2026-04-25 澄清）**：默认机制和 Hypothalamus Reflect 是
**通过 prompt 层互斥的**，不是"core 配合 Reflect 并行运行"。

- **默认状态（Reflect #1 未激活 = 不在 reflects 列表）**：
  - Self 的 prompt 里有一层 `[ACTION FORMAT]` 块教它如何用结构化 tag
    输出 tentacle 调用
  - Self 的输出里包含 `[ACTION]...[/ACTION]` 块写 JSONL 调用
  - 内置 **action executor** (脚本, 非 LLM) 扫描该块, 直接 parse 成
    tentacle calls 派发
  - 路径短: 1 次 Self LLM 调用 → 解析 → 派发

- **激活状态（Reflect #1 在 reflects 列表里）**：
  - Reflect #1 激活时**抑制 `[ACTION FORMAT]` 块从 prompt 里消失**
  - Self 看不到结构化调用的指令, 自然回到自然语言 decision 风格
  - Hypothalamus LLM 翻译 Self 的自然语言决策为 tentacle calls
  - 路径长: 2 次 LLM 调用（Self + Hypothalamus）→ 派发

**为什么这样设计**:

把"教 Self 写 ACTION tag"和"让 Hypothalamus 翻译"放在同一个 prompt
里**会互相干扰** —— Self 看到 ACTION 教程会输出 tag, Hypothalamus 又
会试图翻译这段 tag, 双解释 = 双错乱。所以激活 Reflect #1 = 一刀切移除
ACTION 教程, Self 不知道有这种格式, Hypothalamus 唯一翻译者。

**默认**: **关闭**（强模型不需要 Hypothalamus 这层翻译, 直接发 ACTION
JSONL 比走第二次 LLM 调用快也便宜）。

**保留 Reflect #1 是为了让小模型也能跑** —— 小模型可能无法稳定生成
合法 JSONL, 这时打开 Reflect #1 让 Hypothalamus 帮它兜底。

**Action tag 格式（finalized 2026-04-25）：OpenAI tool_calls 风格的
JSONL，每行一个 JSON 对象，外层包 `[ACTION]...[/ACTION]` 划界**。

```
[ACTION]
{"name": "web_chat_reply", "arguments": {"text": "Hi Alex!"}}
{"name": "search", "arguments": {"query": "Cython optimization"}}
[/ACTION]
```

字段：
- ``name`` (str, required): tentacle 名字
- ``arguments`` (object, optional): 参数字典；未传时 = 空字典
- ``adrenalin`` (bool, optional): 紧迫信号；未传时 = false

**为什么选这个**：
- OpenAI function calling 是行业标准（2023-至今），DeepSeek / Mistral
  / Qwen / Gemini 全跟这个 schema —— LLM 训练数据覆盖最广。
- 字段名 ``name`` + ``arguments`` 一比一对应 OpenAI tool 定义，便于
  复用现有的 tool schema 描述。
- JSONL 一行一个对象 —— Python `for line in block: json.loads(line)`
  解析最简单，不需要嵌套 XML 解析器。
- 单行错不影响其它行（比 XML 鲁棒：XML 一个 tag 错就整段废）。
- 中文 / 转义 / 多行参数等边缘情况靠 JSON 标准处理。

XML-ish `<use>...</use>` 和函数风格 `@tentacle(...)` 都被拒绝：前者啰嗦
且 parser 需要更多状态机；后者紧凑但 LLM 在尾随逗号 / 引号转义上常出错。

**问题点**：
- 默认关闭意味着小模型用户需要手动打开这个 Reflect。OK，可以接受。
- 现有的 Hypothalamus 代码会被搬进 `src/reflects/builtin/hypothalamus/` 之类的位置。

#### Reflect #2 — `recall_anchor`（回忆特征提取器）

**默认**：？（待 Samuel 定，建议**开启**因为质量提升明显）

**开启时**：Part 2 描述的 LLM-driven anchor 抽取 + 多路召回。

**关闭时**：现有的脚本化纯向量召回（`IncrementalRecall` 当前行为）。

**这个 Reflect 的存在形式**：
- 现有的脚本化召回被抽成"默认内置 Reflect"—— 即使用户"关闭"它，runtime 也会 fallback 到这个默认。
- LLM-anchor Reflect 是另一个可选 Reflect，打开后**替换**默认。
- 也可以允许"同时开两个"—— 先 LLM anchor，再脚本兜底。设计时再决定。

#### Reflect #3 — `in_mind`（心智状态自述）

**默认**：？（建议**开启**，因为替代掉了删除的 focus/mood/goals）

**功能**：允许 Self 在运行过程中持续记录：
- **当前最重要的思绪**（一句话，什么事占据心头）
- **情绪状态**（短段文字 + 简短原因）
- **正在专注的事**（一句话）

这些内容作为**固定位置的 prompt 层**（e.g. `[IN MIND]`），每跳显式可见，不随着滑动窗口消失。

**Self 怎么更新它**：
- 通过新的 tentacle `update_in_mind`（由 Reflect 在 `attach(runtime)` 时
  注册到 `runtime.tentacles`, Self 通过 [ACTION] 直接派发或 Hypothalamus
  翻译路径都支持）, 传 3 个字段（全可选, 部分更新）
- 状态文件：`workspace/data/in_mind.json`，由 Reflect #3 自己读写

**实现 2026-04-26 落地** (`src/plugins/in_mind_note/`):
- `meta.yaml` 静态元数据声明, kind="in_mind"
- `state.py`: `InMindState` dataclass + 原子 load/save (tempfile + replace)
- `tentacle.py`: `UpdateInMindTentacle` (is_internal=True, 直接调 Reflect.update)
- `prompt.py`: `IN_MIND_INSTRUCTIONS_LAYER` 常量 + 虚拟 round 渲染
- `reflect.py`: `InMindReflectImpl` 主体 + `attach(runtime)` 注册 tentacle
- 协议新增 `InMindReflect` Protocol + Reflect 通用 `attach(runtime)` 钩子
- Registry 新增 `in_mind_state()` (零插件返回 None) + `attach_all(runtime)`
- Builder 新增 `in_mind` / `in_mind_instructions` 参数, `_layer_history`
  按 in_mind 状态在头部插虚拟 "Heartbeat #now (in mind)" round
- `RuntimeDeps.in_mind_state_path` 用于测试 state 隔离

**状态 vs 配置分离**：``workspace/data/`` 是运行时状态的家（已有
``graph_memory.sqlite`` / ``web_chat.jsonl`` / ``knowledge_bases/``），
``workspace/plugin-configs/`` 和 ``workspace/reflects/<name>/config.yaml``
存用户配置。in_mind 是 Self 写、runtime 读的运行时状态，所以归 `data/`，
和 Reflect 自己的 config（如果有）分开存。

**注入到 prompt 的方式：通过历史层最近端虚拟 round**（finalized 2026-04-25）

不开新的 prompt layer，而是让 prompt builder 在渲染 ``[HISTORY]`` 层时
**虚拟一条最新的 round 插在头部**，内容就是当前的 in_mind 三个字段。例如：

```
[HISTORY]
--- Heartbeat #now (in mind) ---
Thoughts: 正在思考如何回应 Alex 的优化问题
Mood: 略微紧张, 因为 benchmark 数字还没跑完
Focus: 把 hot loop 从 Python 换到 Cython
--- Heartbeat #N-1 ---
Stimulus: ...
Decision: ...
...
```

**为什么这样而不是开独立 layer**：

1. **零新 slot**：所有 prompt 消费者（Self LLM、回忆 LLM、未来其他
   Reflect）只要读历史层就自动拿到 in_mind，不需要每个消费者各自支持。
2. **时间感正确**：in_mind 是"现在心里的事"，放在历史最近端语义自然。
3. **单一真源**：in_mind 渲染逻辑只在 prompt builder 一处，其他地方
   零感知。
4. **缓存友好**：in_mind 变化频率介于 history 层（每跳变）和
   self-model 层（几乎不变）之间，正好处在 history 层的位置不会破坏
   更高层的稳定缓存。
5. **回忆 LLM 自动受益**：in_mind 里的"focus" / "thoughts"成为下跳
   anchor 抽取的天然种子。Self 上跳记下"在思考 Cython 优化"，下跳
   回忆 LLM 看到"Cython optimization"作为最近 history，自然就会去
   GM 召回相关节点。**自我引导回忆**这个机制免费拿到。

**与删除的 self_model 字段的关系**：
- 替换了 `state.focus_topic`、`state.mood_baseline`
- 替换了 `goals.active` 的"当下心头目标"这部分（完整的目标体系走
  GM TARGET 节点，in_mind 只是"此刻关注的一两件"）

---

## Part 4 — 实现顺序建议

Samuel 最终拍板，这里是我的推荐：

1. **先修 bug：stimulus 截断**（2026-04-25 同步修掉，不等设计讨论）
2. **Self-model 瘦身** — 独立 commit，小、干净、不涉及任何新机制
3. **Reflect 协议 + 默认内置骨架** — 不做任何新功能，只是把现有 Hypothalamus 和 recall 从主流程抽出变成默认 Reflect，保持行为不变。这是最容易失败的重构，先打好基础。
4. **Reflect #1 `hypothalamus` 可关闭 + action executor engine** — 最贴近 Samuel 的"强模型省 LLM"诉求，先落地获得信号。
5. **Reflect #3 `in_mind`** — 替代掉删除的 focus/mood/goals 字段，给 Self 主动记录心智状态的能力。
6. **Reflect #2 `recall_anchor`** — 最复杂（多了一次 LLM 调用，可能影响心跳延迟），最后做；落地时注意测延迟影响。

---

## Part 5 — 待 Samuel 决策的开放问题

- [x] ~~Reflect #1 action tag 格式~~ — 2026-04-25：**OpenAI tool_calls
      风格 JSONL** (`{"name": "...", "arguments": {...}}` 每行一个),
      外层 `[ACTION]...[/ACTION]` 包裹。理由记录在 Part 3 Reflect #1
      段。
- [x] ~~Reflect #2 默认开关~~ — 2026-04-25：**默认关**。每跳多一次
      LLM 调用是延迟 / 成本代价；用户主动开就是接受这个代价。和
      Reflect #1 默认关同哲学（不为弱模型烘焙拐杖）。
- [x] ~~Reflect #3 `in_mind` 状态文件存哪~~ — 2026-04-25：
      `workspace/data/in_mind.json`。运行时状态（Self 写 / runtime 读）
      归 ``workspace/data/`` —— 和已有 ``graph_memory.sqlite`` /
      ``web_chat.jsonl`` 同住；用户配置归 ``workspace/plugin-configs/``
      或 ``workspace/reflects/<name>/config.yaml``。state ≠ config,
      分开存。注入路径同日已定：通过历史层最近端虚拟 round。
- [x] ~~Reflect 启停 / UI~~ — 2026-04-25：**config.yaml 是单一真源**，
      dashboard 仅显示运行状态 + 提供 config 编辑界面。Reflects 各自的
      详细配置在各自文件夹（`workspace/reflects/<name>/config.yaml`,
      沿用现有 plugin 模式）。改 config 要重启。
      **实现 2026-04-25 落地**: `config.reflects: list[str] | None`
      字段；列表元素是 `BUILTIN_FACTORIES` 里的 name；按列表顺序注册
      （= 链执行顺序）。`None` (字段缺失) → 兼容老 config, 注册旧默认
      + loud deprecation。`[]` → 显式零插件, 静默执行。未知 name → log
      + skip, 不阻塞启动（strictly additive 原则）。
- [x] ~~多 Reflect 同 kind 共存~~ — 2026-04-25：**允许，顺序执行**。
      `config.yaml` 里有一个 reflects 列表，注册顺序 = 执行顺序。同
      kind 的 Reflects 链式调用：前一个的输出（e.g. anchors 列表 / 翻译
      结果）作为后一个的输入。允许后续 Reflect 增补 / 改写 / 否决前一个
      的输出，组合行为通过链式表达，不开"并行 + 合并"的复杂语义。
- [x] ~~`statistics` 搬到哪里？~~ 2026-04-25 已落地：sleep_cycles 改成
      `Runtime._sleep_cycles` 内存计数器，其他字段直接删除（自 commit
      `ce59ab4`，self-model slim 重构）。
- [x] ~~回忆 LLM 输出结构里要不要 `reason` 字段？~~ 2026-04-25 决定：
      不要。**纯 JSON 输出，连 ``<thinking>`` 块都不加** —— 让用户自己
      选模型，弱模型表现差是用户选择的代价，不在核心机制里烘焙拐杖
      （和 Reflect #1 默认关闭的哲学一致）。详见 Part 2。

---

## 附录 — 已同步修复的 bug

- **Stimulus 截断 bug**（2026-04-25）：`_summarize_stimuli` 把每条 stimulus 的 content 砍到 60 字符后才写进 sliding window。这导致 `[HISTORY]` 层里 Self 回忆自己接收过的用户消息时只看到开头，下游多个机制（recall 特征提取、compact 提炼、bootstrap 指令识别）都受影响。已在这次 bug 修复中移除截断。
