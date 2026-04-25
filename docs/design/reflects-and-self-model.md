# Reflects + Self-model 瘦身 + 回忆层 LLM — 设计草案

> 状态：**草案 / 讨论中**。本文只是 Samuel 口述的需求固化，未开始实现。
> 修改或补充请直接编辑本文件；实现前会开对应的 PR 讨论贴。
> 记录日期：2026-04-25。

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

**默认**：**关闭**（强模型不需要）

**开启时**：行为同现在的 Hypothalamus —— 一个专门的 LLM 把 Self 的自然语言 Decision 翻译成结构化的 tentacle calls。保留是为了**让小模型也能跑 Krakey**（小模型靠自己结构化输出 tentacle call 可能不稳）。

**关闭时**：
- Self prompt 里会注入新的"action 格式约定"指令，告诉 Self 用结构化 tag（类似 tool_use 的样式）直接声明 tentacle 调用
- 新增 **"action executor engine"**：扫描 Self 的 `[DECISION]` / `[ACTION]` 区域，正则抠出 tentacle-call tag，直接派发
- 这条路径是强模型的默认 —— 省一次 LLM 调用、省延迟、省 token

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
- 通过新的 tentacle `update_in_mind`（走 Hypothalamus 或 direct action path），传 3 个字段（部分或全部）
- 状态文件：`workspace/reflects/in_mind/state.json`，由 Reflect #3 自己读写

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
      `workspace/reflects/in_mind/state.json`，跟着 Reflect 自己的文件夹
      走。与"plugin 配置 / 状态在各自文件夹"哲学一致。注入路径同日已
      定：通过历史层最近端虚拟 round。
- [x] ~~Reflect 启停 / UI~~ — 2026-04-25：**config.yaml 是单一真源**，
      dashboard 仅显示运行状态 + 提供 config 编辑界面。Reflects 各自的
      详细配置在各自文件夹（`workspace/reflects/<name>/config.yaml`,
      沿用现有 plugin 模式）。改 config 要重启。
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
