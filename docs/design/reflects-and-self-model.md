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

LLM 的 prompt 模板（待细化，示例）：

```
这是你的当前刺激和近期历史：
<stimuli>...</stimuli>
<history>...</history>

作为回忆引导员，你的任务是提取出需要召回的特征点，帮助意识层（Self）
更好地回应。输出 JSON：
{
  "anchors": [
    {"kind": "user"|"topic"|"event"|"emotion"|"fact",
     "query": "召回关键词或人名",
     "rationale": "为什么要召回它"}
  ]
}
```

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

**tag 格式建议**（待定）：

```
[ACTION]
<use tentacle="web_chat_reply" adrenalin="false">
  {"text": "Hi Samuel!"}
</use>
<use tentacle="search">
  {"query": "agent memory architectures 2026"}
</use>
[/ACTION]
```

或更简短的：

```
[ACTION]
@web_chat_reply(text="Hi Samuel!")
@search(query="agent memory architectures 2026")
[/ACTION]
```

哪种格式更适合 LLM 生成需要测试。

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
- 通过新的 tentacle `update_in_mind`（走 Hypothalamus 或 direct action path），传 3 个字段
- Reflect 存储在 `workspace/data/in_mind.json`（或 self_model.yaml 里留一块？待定）
- 每跳 `on_heartbeat_start` 时把内容注入到 prompt layer

**与删除的 self_model 字段的关系**：
- 替换了 `state.focus_topic`、`state.mood_baseline`
- 替换了 `goals.active` 的"当下心头目标"这部分（完整的目标体系走 GM TARGET 节点，in_mind 只是"此刻关注的一两件"）

**Prompt layer 位置**：
- 比较稳定（几跳才变一次），放在 `[SELF-MODEL]` 附近更利于缓存
- 建议顺序：DNA → SELF-MODEL → IN MIND → CAPABILITIES → STIMULUS → ...

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

- [ ] Reflect #1 的 action tag 用哪种格式？XML-ish `<use>` 还是函数调用风格 `@tentacle(...)`？
- [ ] Reflect #2 默认开还是默认关？LLM-anchor 模式会让每跳多一次 LLM 往返，对延迟敏感。
- [ ] Reflect #3 `in_mind` 存在哪里？独立文件 / self_model 子块 / GM 节点？
- [ ] Reflect 可以禁用/启用吗？通过 dashboard UI 开关？还是只能在 config.yaml？
- [ ] 多个 Reflect 同 kind 能共存吗？(e.g. 两个 `recall_anchor` 同时跑、结果合并？)
- [ ] `statistics` 搬到哪里？runtime_stats.json / GM `system:*` 节点 / dashboard-only？

---

## 附录 — 已同步修复的 bug

- **Stimulus 截断 bug**（2026-04-25）：`_summarize_stimuli` 把每条 stimulus 的 content 砍到 60 字符后才写进 sliding window。这导致 `[HISTORY]` 层里 Self 回忆自己接收过的用户消息时只看到开头，下游多个机制（recall 特征提取、compact 提炼、bootstrap 指令识别）都受影响。已在这次 bug 修复中移除截断。
