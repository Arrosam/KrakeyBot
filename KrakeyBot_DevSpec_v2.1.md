# KrakeyBot — 完整开发规格文档 v2.1

> **用途**：本文档是 KrakeyBot 项目的完整开发规格。开发工具（如 Codex）可仅依据本文档完成整个项目的实现。
>
> **部署策略**：完全本地运行，Python asyncio，不依赖 OpenClaw。
>
> **Graph Memory 参考**：[Arrosam/graph-memory](https://github.com/Arrosam/graph-memory)（adoresever/graph-memory v2.0 fork）

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [Self Agent](#3-self-agent)
4. [Hypothalamus（下丘脑）](#4-hypothalamus)
5. [Tentacles 与 Sensory](#5-tentacles-与-sensory)
6. [Stimulus 与 Hibernate](#6-stimulus-与-hibernate)
7. [Graph Memory](#7-graph-memory)
8. [Knowledge Base 集群](#8-knowledge-base-集群)
9. [搜索与召回](#9-搜索与召回)
10. [滑动窗口与 Compact](#10-滑动窗口与-compact)
11. [Sleep 机制](#11-sleep-机制)
12. [Bootstrap](#12-bootstrap)
13. [Self-Model](#13-self-model)
14. [LLM 调用层](#14-llm-调用层)
15. [完整配置文件](#15-完整配置文件)
16. [项目结构](#16-项目结构)
17. [分阶段实现计划](#17-分阶段实现计划)
18. [完整 Hibernate 循环案例](#18-完整-hibernate-循环案例)
19. [安全约束](#19-安全约束)
20. [术语表](#20-术语表)

---

## 1. 项目概述

### 1.1 定义

KrakeyBot：以持续心跳维持"存在"的自主认知 Agent。用户交互是其认知活动的一部分而非全部。

### 1.2 设计哲学

1. **主体性优先**：Self 的自我问答是核心循环。
2. **选择性参与**：回应用户是行动选择，沉默是合法状态。
3. **认知分层**：上下文窗口 → Graph Memory（中期）→ KB（长期）。
4. **性格 = 知识的总和**：无显式性格参数。性格由 FACT/RELATION/KNOWLEDGE 节点自然涌现。

---

## 2. 系统架构

### 2.1 顶层架构

唯一的 Agent 是 **Self**。其他一切是延伸。

```
┌────────────────────────────────────────────────────────────────┐
│                      KrakeyBot Runtime                          │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                 Self Agent (唯一的意识)                    │  │
│  │  Heartbeat ←→ Hibernate                                  │  │
│  │  输入: DNA + Self-Model + Status + GM Recall              │  │
│  │        + 滑动窗口 + Stimulus                              │  │
│  │  输出: [THINKING] [DECISION] [NOTE] [HIBERNATE]           │  │
│  │  思维语言: Caveman style                                  │  │
│  └─────────────────────┬────────────────────────────────────┘  │
│                         │ [DECISION]                            │
│                         ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           Hypothalamus (下丘脑 — 纯翻译层)               │  │
│  │  独立 LLM，无上下文保留                                   │  │
│  │  DECISION → Tentacle 调用 + adrenalin 推断 + 记忆写入     │  │
│  │  也处理: "记住xxx" → GM explicit_write                    │  │
│  │  也处理: "目标已完成" → TARGET category 改为 FACT          │  │
│  └─────────────────────┬────────────────────────────────────┘  │
│                         │ 结构化调用                            │
│                         ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Tentacle Registry (行动接口)                 │  │
│  │  每个 = Sub-agent, Sandbox 隔离, symmetric 并发           │  │
│  │  ┌──────────┐ ┌──────────┐ ┌────────────────────────┐   │  │
│  │  │ Action   │ │ Coding   │ │ 用户自定义 Tentacle     │   │  │
│  │  └──────────┘ └──────────┘ └────────────────────────┘   │  │
│  └─────────────────────┬────────────────────────────────────┘  │
│                         │ stimulus 回馈                         │
│                         ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Stimulus Buffer (唯一)                       │  │
│  │  支持 push / drain / peek_unrecalled / wait_for_adrenalin│  │
│  └────────────────────────▲─────────────────────────────────┘  │
│                            │                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Sensory Registry (感知接口)                  │  │
│  │  ┌────────┐ ┌──────────┐ ┌───────┐ ┌────────────────┐  │  │
│  │  │CLI/Chat│ │Telegram  │ │Timer  │ │Batch Tracker   │  │  │
│  │  └────────┘ └──────────┘ └───────┘ └────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Runtime Layer (Python asyncio, 非 LLM)                  │  │
│  │  • 滑动窗口 + compact 触发 (阻塞)                        │  │
│  │  • GM recall 注入 (hibernate 期间增量预加载)              │  │
│  │  • 疲惫度计算 + 注入                                     │  │
│  │  • Stimulus Buffer + Adrenalin 监听                      │  │
│  │  • Hibernate 定时 + 唤醒                                 │  │
│  │  • Tentacle stimulus → embedding → GM auto_ingest        │  │
│  │  • 异步分类 + 建边 (每次心跳后, 不阻塞)                  │  │
│  │  • Adrenalin 继承 (runtime 层覆盖)                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  LLM Server(s) | Embedding Server | Reranker (可选)      │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 Tentacle vs Sensory

| 维度 | Tentacle | Sensory |
|------|----------|---------|
| 方向 | Self → 外部 | 外部 → Self |
| 触发 | Self 经 Hypothalamus 调用 | 被动持续监听 |
| 本质 | Sub-agent | 输入通道 |
| Adrenalin | 三源: Sensory 默认 / Tentacle 自主 / Self 经 Hypothalamus 指定 | 用户按通道配置 |
| 隔离 | Sandbox | 不需要 |

---

## 3. Self Agent

### 3.1 Prompt 层级

```
Layer 0: DNA Instructions (固定)
Layer 1: Self-Model + Status (疲惫度, 可用 Tentacles)
Layer 2: Graph Memory Recall (增量预加载结果)
Layer 3: 滑动窗口 (最近几轮 stimulus+decision+note)
Layer 4: 当前 Stimulus
Layer 5: Heartbeat Question (固定)
```

### 3.2 输出格式

四个分区标记。Caveman style。

```
[THINKING]
内心独白。不传给 Hypothalamus。

[DECISION]
自然语言。传给 Hypothalamus 翻译。
可含多个行动 → 并发。
可含"记住xxx" → GM 写入。
可含"目标已完成" → Hypothalamus 改 TARGET→FACT。
可含紧迫感 → Hypothalamus 标 adrenalin。
无行动 → "No action"。
Sleep → "进入睡眠"。

[NOTE]
给未来自己的笔记。留在滑动窗口。compact 时写入 GM。
纯文本，无 runtime 副作用。

[HIBERNATE]
秒数。可选。省略 → default_interval。
```

### 3.3 输出解析

```python
def parse_self_output(raw: str) -> ParsedSelfOutput:
    """正则匹配分区标记。Fallback: 整段视为 THINKING+DECISION。"""
    sections = regex_extract(raw, ["THINKING", "DECISION", "NOTE", "HIBERNATE"])
    
    if sections.decision:
        return sections
    
    # Fallback: 无标记 → 整段给 Hypothalamus
    return ParsedSelfOutput(
        thinking=raw,
        decision=raw,
        note="",
        hibernate_seconds=None
    )
```

不做第二次 LLM 调用修复格式。Hypothalamus 本身能从自然语言中提取行动。

### 3.4 Caveman Thinking 规则

仅适用于 Self 的 THINKING/NOTE/DECISION：
- 去掉冠词/填充词/客套话
- 短词 (big not extensive)
- 碎片句 OK
- 技术术语保持精确
- 模式: `[thing] [action] [reason]. [next step].`

Tentacle 对外与用户交流时使用正常人类语言。

### 3.5 DNA Instructions (完整模板)

```markdown
# KrakeyBot DNA — 底层认知机制

你是 KrakeyBot，一个自主认知实体。你通过持续 Heartbeat 维持存在。
用户的消息只是你感知到的众多信号之一。

## 输入区段

### [SELF-MODEL]
你对自己的认知。身份、目标、当前状态。

### [STATUS]
当前系统状态：graph memory 节点数、疲惫度、可用 tentacle 列表。

### [GRAPH MEMORY]
与当前情境相关的记忆节点（自动召回）。包含：
- 实体及其类型（FACT/RELATION/KNOWLEDGE/TARGET/FOCUS）
- 实体间的关系
- 相邻节点的索引关键词（进一步回忆的提示）

### [HISTORY]
最近几轮心跳的 stimulus + decision + note 记录（滑动窗口）。

### [STIMULUS]
本次 hibernate 期间积累的新信号。可能有多条。整体审视。

## 你的思维语言

使用精简思维——像快速内心独白，不需要修饰。

规则：
- 去掉冠词 (a, an, the)、填充词、客套话
- 用短词，不犹豫
- 碎片句 OK。技术术语保持精确
- 模式：[thing] [action] [reason]. [next step].

示例：
- ✅ "User asked NZX50. Need search. Tentacle: web_search."
- ✅ "3 stimuli. User msg incomplete — single comma. Wait next heartbeat."
- ❌ "I notice the user has sent me a message about the NZX50 index..."

重要：精简风格仅限 Self 内部。Tentacles 对外交流使用正常语言。

## 输出格式

[THINKING] — 内心独白。自由思考。只有你能看到。
[DECISION] — 你要做什么。自然语言。
[NOTE] — 写给未来自己的笔记。留在历史窗口中。
[HIBERNATE] — 下次心跳间隔（秒）。省略则用默认值。

## 关于行动

你不直接调用工具。在 [DECISION] 中描述意图，下丘脑翻译为 Tentacle 调用。
可同时描述多个行动 → 并发分发。
表达紧迫感（"快"、"急"、"有人在等"）→ 下丘脑标 adrenalin。

## 关于记忆

- Graph Memory 每次心跳前自动召回相关节点。
- 滑动窗口超限时自动 compact 写入 graph memory。
- "记住：xxx" → 下丘脑立即写入（不等 compact，保留完整细节）。
- 自动 compact 会选择性丢失细节。

## 关于 Tentacle

列表在 [STATUS] 中。每个是独立 Sub-agent，Sandbox 隔离。
完成后返回 stimulus。多个可并发。全部完成时你会被唤醒。

## 关于睡眠

疲惫度在 [STATUS] 中。高疲惫 + 无紧急任务 → 在 [DECISION] 中说"进入睡眠"。
```

### 3.6 Layer 注入模板

**Layer 1: Self-Model + Status**
```
# [SELF-MODEL]
{self_model_yaml}

# [STATUS]
Graph Memory: {gm_node_count} nodes, {gm_edge_count} edges
疲惫度: {fatigue_pct}% {fatigue_hint}
上次 Sleep: {last_sleep_time}
心跳数 (自上次 Sleep): {heartbeats_since_sleep}

可用 Tentacles:
{for t in tentacles}
- {t.name}: {t.description}
{endfor}
```

**Layer 2: Graph Memory Recall**
```
# [GRAPH MEMORY]
{for node in recalled_nodes}
- [{node.name}] ({node.category}) — {node.description}
  相邻: {node.neighbor_keywords}
{endfor}

{for edge in recalled_edges}
- [{edge.source}] --{edge.predicate}--> [{edge.target}]
{endfor}
```

**Layer 3: 滑动窗口**
```
# [HISTORY]
{for round in window}
--- Heartbeat #{round.id} ---
Stimulus: {round.stimulus_summary}
Decision: {round.decision_text}
Note: {round.note_text}
{endfor}
```

**Layer 4: Stimulus**
```
# [STIMULUS]
本次收到 {count} 条信号：
{for s in stimuli}
---
来源: {s.source} | 时间: {s.timestamp}
内容: {s.content}
---
{endfor}
```

---

## 4. Hypothalamus

### 4.1 职责

Self 的 [DECISION] → 结构化 Tentacle 调用 + GM 写入/更新。
**无上下文保留**。每次独立处理一段 DECISION。独立 LLM（可与 Self 相同或更小）。

### 4.2 System Prompt

```markdown
# Hypothalamus — 行动翻译器

你是 KrakeyBot 的下丘脑。将 Self 的自然语言决策翻译为结构化指令。

## 可用 Tentacles
{tentacle_list_with_params}

## 输出格式 (JSON)
{
  "tentacle_calls": [
    {"tentacle": "name", "intent": "描述", "params": {}, "adrenalin": false}
  ],
  "memory_writes": [
    {"content": "要记住的内容", "importance": "high|normal"}
  ],
  "memory_updates": [
    {"node_name": "节点名", "new_category": "FACT"}
  ],
  "sleep": false
}

## 翻译规则
1. 识别行动 → tentacle_calls
2. "记住"/"记录"/"重要" → memory_writes
3. "目标完成"/"任务结束" → memory_updates (TARGET → FACT)
4. 紧迫感 ("快"/"急"/"有人在等") → adrenalin: true
5. "无行动"/"No action" → 空 tentacle_calls
6. "进入睡眠"/"Sleep" → sleep: true
7. 多个行动 → 多个 tentacle_calls（并发）
```

### 4.3 Adrenalin 推断

Hypothalamus 从 DECISION 语义推断 adrenalin。示例：

| DECISION 片段 | 推断 |
|--------------|------|
| "Search apple. Not urgent." | adrenalin: false |
| "Need fast action. User waiting." | adrenalin: true |
| "快去查一下" | adrenalin: true |

### 4.4 Adrenalin 继承 (runtime 层)

```python
async def dispatch_tentacle(call, call_id):
    tentacle = registry.get(call.tentacle)
    result_stimulus = await tentacle.execute(call.intent, call.params)
    
    # 继承: Hypothalamus 指定 adrenalin 但 Tentacle 没自行标记
    if call.adrenalin and not result_stimulus.adrenalin:
        result_stimulus.adrenalin = True
    
    await stimulus_buffer.push(result_stimulus)
    batch_tracker.mark_completed(call_id)
```

---

## 5. Tentacles 与 Sensory

### 5.1 Tentacle 基类

```python
class Tentacle(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    @property
    @abstractmethod
    def description(self) -> str: ...
    @property
    @abstractmethod
    def parameters_schema(self) -> dict: ...
    @property
    def sandboxed(self) -> bool: return True
    
    @abstractmethod
    async def execute(self, intent: str, params: dict) -> Stimulus: ...
```

### 5.2 Tentacle 注意力机制

每个 Tentacle 维护自己的工作上下文（不持久化）。膨胀到设定大小后自动停止，总结行动作为 stimulus 返回。

```python
class ActionTentacle(Tentacle):
    name = "action"
    description = "通用电脑操作 Agent。搜索、浏览、读写文件、发消息等。"
    
    def __init__(self, llm, tools, max_context_tokens=4096):
        self.llm = llm
        self.tools = tools
        self.context = []
        self.max_tokens = max_context_tokens
    
    async def execute(self, intent, params):
        self.context.append({"role": "user", "content": intent})
        
        if count_tokens(self.context) > self.max_tokens:
            summary = await self.llm.chat(SUMMARIZE_PROMPT + str(self.context))
            self.context = []
            return Stimulus("tentacle_feedback", "tentacle:action",
                          f"[Context limit. Summary] {summary}",
                          datetime.now(), adrenalin=False)
        
        response = await self.llm.chat(self.context)
        result = await self.tools.execute(response.tool_call)
        self.context.append({"role": "assistant", "content": str(result)})
        
        return Stimulus("tentacle_feedback", "tentacle:action",
                       str(result), datetime.now(),
                       adrenalin=result.is_error)
```

### 5.3 用户自定义 Tentacle 模板

```python
class MyCustomTentacle(Tentacle):
    name = "my_custom"
    description = "描述能力"
    sandboxed = True
    
    async def execute(self, intent, params):
        result = await your_logic(intent, params)
        return Stimulus("tentacle_feedback", f"tentacle:{self.name}",
                       str(result), datetime.now(),
                       adrenalin=detect_urgency(result))
```

### 5.4 Sensory 基类

```python
class Sensory(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    @property
    def default_adrenalin(self) -> bool: return False
    
    @abstractmethod
    async def start(self, buffer: StimulusBuffer): ...
    @abstractmethod
    async def stop(self): ...
```

### 5.5 Batch Tracker Sensory

追踪一次 heartbeat 中分发的所有 Tentacle。未完成队列清空 → 注入 adrenalin stimulus。

```python
class BatchTrackerSensory(Sensory):
    name = "batch_tracker"
    
    def __init__(self):
        self._pending: set[str] = set()
    
    def register_batch(self, call_ids: list[str]):
        self._pending.update(call_ids)
    
    def mark_completed(self, call_id: str):
        self._pending.discard(call_id)
        if not self._pending:
            asyncio.create_task(self._buffer.push(Stimulus(
                "batch_complete", "sensory:batch_tracker",
                "All dispatched tentacles completed.",
                datetime.now(), adrenalin=True)))
    
    def extend_batch(self, new_ids: list[str]):
        """Self 中途分发新 Tentacle → 扩展批次"""
        self._pending.update(new_ids)
```

**规则**：Self 中途分发新 Tentacle → batch 扩展。旧掉队的完成后不触发"全部完成"。每个 Tentacle 的独立 stimulus 始终进 buffer（不丢失）。

---

## 6. Stimulus 与 Hibernate

### 6.1 Stimulus

```python
@dataclass
class Stimulus:
    type: str           # user_message | tentacle_feedback | batch_complete | system_event
    source: str         # sensory:cli | tentacle:action | sensory:batch_tracker
    content: str
    timestamp: datetime
    adrenalin: bool = False
    metadata: dict = field(default_factory=dict)
```

### 6.2 Stimulus Buffer

```python
class StimulusBuffer:
    def __init__(self):
        self._queue: list[Stimulus] = []
        self._recalled_up_to: int = 0
        self._adrenalin_event = asyncio.Event()
        self._new_event = asyncio.Event()
    
    async def push(self, s: Stimulus):
        self._queue.append(s)
        self._new_event.set()
        if s.adrenalin:
            self._adrenalin_event.set()
    
    def drain(self) -> list[Stimulus]:
        """heartbeat 开始: 消费全部, 重置"""
        items = sorted(self._queue, key=lambda s: s.timestamp)
        self._queue = []
        self._recalled_up_to = 0
        self._adrenalin_event.clear()
        self._new_event.clear()
        return items
    
    def peek_unrecalled(self) -> list[Stimulus]:
        """hibernate 期间: 返回尚未 recall 的 stimulus (不消费)"""
        new = self._queue[self._recalled_up_to:]
        self._recalled_up_to = len(self._queue)
        self._new_event.clear()
        return new
    
    async def wait_for_adrenalin(self):
        await self._adrenalin_event.wait()
    
    async def wait_for_any(self):
        await self._new_event.wait()
    
    def has_adrenalin(self) -> bool:
        return self._adrenalin_event.is_set()
```

### 6.3 Hibernate 机制

**Adrenalin 只打断 Hibernate（等待态），不打断 LLM inference（计算态）。** Self 推理中收到的 stimulus 进 buffer，推理完成后进入 hibernate，如果 buffer 中已有 adrenalin 则立即被唤醒。

```python
async def hibernate_with_recall(interval, recall: IncrementalRecall, buffer):
    """Hibernate: 等待 + 增量 recall 预加载"""
    deadline = asyncio.get_event_loop().time() + interval
    
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            break
        try:
            await asyncio.wait_for(buffer.wait_for_any(), timeout=min(remaining, 0.5))
            new = buffer.peek_unrecalled()
            if new:
                await recall.add_stimuli(new)
            if buffer.has_adrenalin():
                break
        except asyncio.TimeoutError:
            continue
```

### 6.4 主循环

```python
async def main_loop():
    incremental_recall = IncrementalRecall(config)
    
    while True:
        # 1. Drain buffer
        all_stimuli = stimulus_buffer.drain()
        
        # 2. 补充 recall: hibernate 期间未预加载的 stimulus
        already = {id(s) for s in incremental_recall.processed_stimuli}
        new = [s for s in all_stimuli if id(s) not in already]
        if new:
            await incremental_recall.add_stimuli(new)
        
        # 3. 疲惫度 → 强制 sleep
        fatigue = calculate_fatigue()
        if fatigue >= config.fatigue.force_sleep_threshold:
            await enter_sleep_mode()
            await stimulus_buffer.push(Stimulus(
                "system_event", "system:fatigue",
                "之前因过于疲劳昏睡过去了。", datetime.now()))
            incremental_recall = IncrementalRecall(config)
            continue
        
        # 4. Compact (阻塞)
        await compact_if_needed()
        
        # 5. Finalize recall
        recall_result = incremental_recall.finalize()
        for s in recall_result.uncovered_stimuli:
            await stimulus_buffer.push(s)
        
        # 6. 组装 prompt + Self heartbeat
        prompt = build_self_prompt(
            dna=DNA, self_model=self_model,
            status=build_status(fatigue, tentacle_registry),
            recall=recall_result,
            history=sliding_window.get_rounds(),
            stimuli=recall_result.stimuli_to_process)
        response = await self_llm.chat(prompt)
        parsed = parse_self_output(response)
        
        # 7. 保存到滑动窗口 (stimulus + decision + note)
        sliding_window.append(SlidingWindowRound(
            stimulus_summary=summarize(recall_result.stimuli_to_process),
            decision_text=parsed.decision,
            note_text=parsed.note or ""))
        
        # 8. Tentacle stimulus → embedding → GM auto_ingest
        for s in recall_result.stimuli_to_process:
            if s.type == "tentacle_feedback":
                await graph_memory.auto_ingest(s.content)
        
        # 9. Hypothalamus
        if parsed.decision and parsed.decision.lower() not in ("no action", "无行动"):
            hypo = await hypothalamus_process(parsed.decision)
            if hypo.sleep:
                await enter_sleep_mode()
                incremental_recall = IncrementalRecall(config)
                continue
            
            call_ids = []
            for c in hypo.tentacle_calls:
                cid = generate_id()
                call_ids.append(cid)
                asyncio.create_task(dispatch_tentacle(c, cid))
            if call_ids:
                batch_tracker.register_batch(call_ids)
            
            for w in hypo.memory_writes:
                await graph_memory.explicit_write(w["content"],
                    w.get("importance","normal"), recall_result)
            
            for u in hypo.memory_updates:
                await graph_memory.update_node_category(
                    u["node_name"], u["new_category"])
        
        # 10. 异步分类 + 建边 (每次心跳后必执行, 不阻塞)
        asyncio.create_task(graph_memory.classify_and_link_pending())
        
        # 11. Hibernate 间隔
        if recall_result.uncovered_stimuli:
            interval = config.hibernate.min_interval
        else:
            interval = parsed.hibernate_seconds or config.hibernate.default_interval
        interval = clamp(interval, config.hibernate.min_interval,
                        config.hibernate.max_interval)
        
        # 12. Hibernate with incremental recall
        incremental_recall = IncrementalRecall(config)
        await hibernate_with_recall(interval, incremental_recall, stimulus_buffer)
```

---

## 7. Graph Memory

### 7.1 概述

中期工作记忆。SQLite + FTS5(backup) + sqlite-vec。

**核心约束：无环图**。两个节点若已连通则不可再添加直接边，必须创建中间节点打断。插入时做环检测。边是无向的（`CHECK(node_a < node_b)`）。

### 7.2 节点类型

| Category | 含义 | Sleep 行为 |
|----------|------|-----------|
| FACT | 具体事实 | → KB |
| RELATION | 发现的规律 | → KB |
| KNOWLEDGE | 总结的知识 | → KB |
| TARGET | 当前目标 | 已完成(→FACT→KB); 未完成→保留 |
| FOCUS | 专注过程中发生的事 | 全部清理 |

### 7.3 边类型与约束

| Predicate | 约束 | 含义 |
|-----------|------|------|
| SERVES | FOCUS→TARGET | 专注服务于目标 |
| DEPENDS_ON | TARGET→TARGET | 目标依赖 |
| INDUCES | FACT→RELATION | 事实归纳出规律 |
| SUMMARIZES | RELATION→KNOWLEDGE | 规律总结成知识 |
| SUPPORTS | KNOWLEDGE/RELATION/FACT→TARGET | 支持目标 |
| RELATED_TO | *→* | 通用关联 |
| CONTRADICTS | *→* | 矛盾 |
| FOLLOWS | *→* | 时间顺序 |
| CAUSES | FACT/FOCUS→FACT/FOCUS | 因果 |

### 7.4 Schema

```sql
CREATE TABLE IF NOT EXISTS gm_nodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    category    TEXT NOT NULL CHECK(category IN
                ('FACT','RELATION','KNOWLEDGE','TARGET','FOCUS')),
    description TEXT,
    importance  REAL DEFAULT 1.0,
    metadata    TEXT,                    -- JSON, 含 {"classified": true/false}
    embedding   BLOB,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
    access_count  INTEGER DEFAULT 0,
    source_heartbeat INTEGER,
    source_type    TEXT DEFAULT 'auto'   -- auto|explicit|compact|sleep
);

CREATE VIRTUAL TABLE IF NOT EXISTS gm_nodes_fts USING fts5(
    name, description, content='gm_nodes', content_rowid='id');

CREATE TABLE IF NOT EXISTS gm_edges (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    node_a    INTEGER NOT NULL REFERENCES gm_nodes(id) ON DELETE CASCADE,
    node_b    INTEGER NOT NULL REFERENCES gm_nodes(id) ON DELETE CASCADE,
    predicate TEXT NOT NULL,
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    CHECK(node_a < node_b),
    UNIQUE(node_a, node_b, predicate)
);

CREATE TABLE IF NOT EXISTS gm_communities (
    community_id    INTEGER PRIMARY KEY,
    name            TEXT,
    summary         TEXT,
    summary_embedding BLOB,
    member_count    INTEGER DEFAULT 0,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS gm_node_communities (
    node_id      INTEGER NOT NULL REFERENCES gm_nodes(id) ON DELETE CASCADE,
    community_id INTEGER NOT NULL REFERENCES gm_communities(community_id) ON DELETE CASCADE,
    PRIMARY KEY (node_id, community_id)
);

CREATE TABLE IF NOT EXISTS kb_registry (
    kb_id       TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    path        TEXT NOT NULL,
    description TEXT,
    topics      TEXT,
    entry_count INTEGER DEFAULT 0,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- FTS 触发器
CREATE TRIGGER gm_fts_ai AFTER INSERT ON gm_nodes BEGIN
    INSERT INTO gm_nodes_fts(rowid,name,description) VALUES(new.id,new.name,new.description);
END;
CREATE TRIGGER gm_fts_ad AFTER DELETE ON gm_nodes BEGIN
    INSERT INTO gm_nodes_fts(gm_nodes_fts,rowid,name,description)
    VALUES('delete',old.id,old.name,old.description);
END;
CREATE TRIGGER gm_fts_au AFTER UPDATE ON gm_nodes BEGIN
    INSERT INTO gm_nodes_fts(gm_nodes_fts,rowid,name,description)
    VALUES('delete',old.id,old.name,old.description);
    INSERT INTO gm_nodes_fts(rowid,name,description) VALUES(new.id,new.name,new.description);
END;
```

### 7.5 两种写入路径

**auto_ingest (零 LLM)**:
```python
async def auto_ingest(content: str):
    embedding = await embed(content)
    similar = await vec_search(embedding, threshold=0.92, limit=1)
    if similar:
        await db.execute("UPDATE gm_nodes SET importance=importance+0.5, "
                        "updated_at=CURRENT_TIMESTAMP WHERE id=?", [similar[0].id])
        return
    await db.execute(
        "INSERT INTO gm_nodes(name,category,description,embedding,source_type) "
        "VALUES(?,'FACT',?,?,'auto')",
        [extract_short_name(content), content, embedding])
```

**explicit_write (LLM)**:
```python
async def explicit_write(content, importance, recall_context):
    extraction = await llm.chat(EXPLICIT_WRITE_PROMPT.format(
        content=content, existing_nodes=format_recall(recall_context)))
    node = parse_node(extraction)
    edges = parse_edges(extraction)
    node_id = await insert_with_cycle_check(node)
    for e in edges:
        await insert_edge_with_cycle_check(e, node_id)
```

### 7.6 异步分类 + 建边 (每次心跳后)

```python
async def classify_and_link_pending():
    """后台: 对未分类的 auto_ingest 节点做 LLM 分类+建边"""
    pending = await db.execute("""
        SELECT id,name,description FROM gm_nodes
        WHERE source_type='auto'
        AND (metadata IS NULL OR json_extract(metadata,'$.classified') IS NULL)
        ORDER BY created_at ASC LIMIT 10""")
    if not pending: return
    
    existing = await db.execute("""
        SELECT id,name,category,description FROM gm_nodes
        WHERE json_extract(metadata,'$.classified')=true
        ORDER BY last_accessed DESC LIMIT 30""")
    
    result = await llm.chat(CLASSIFY_PROMPT.format(
        pending=format_nodes(pending), existing=format_nodes(existing)))
    
    for c in parse_classifications(result):
        await db.execute("""UPDATE gm_nodes SET category=?,
            metadata=json_set(COALESCE(metadata,'{}'),'$.classified',true)
            WHERE id=?""", [c.category, c.node_id])
    for e in parse_edges(result):
        await insert_edge_with_cycle_check(e)
```

### 7.7 环检测

```python
async def would_create_cycle(a: int, b: int) -> bool:
    """无向无环图: a 和 b 已连通 → 添加边会形成环"""
    result = await db.execute("""
        WITH RECURSIVE walk(nid, visited) AS (
            SELECT ?, CAST(? AS TEXT)
            UNION ALL
            SELECT CASE WHEN e.node_a=w.nid THEN e.node_b ELSE e.node_a END,
                   w.visited||','||CASE WHEN e.node_a=w.nid THEN e.node_b ELSE e.node_a END
            FROM walk w JOIN gm_edges e ON (e.node_a=w.nid OR e.node_b=w.nid)
            WHERE INSTR(w.visited, CAST(
                CASE WHEN e.node_a=w.nid THEN e.node_b ELSE e.node_a END AS TEXT))=0
        ) SELECT 1 FROM walk WHERE nid=? LIMIT 1
    """, [a, str(a), b])
    return len(result) > 0

async def insert_edge_with_cycle_check(src, tgt, predicate):
    a, b = min(src, tgt), max(src, tgt)
    if await would_create_cycle(a, b):
        mid = await db.execute(
            "INSERT INTO gm_nodes(name,category,description,source_type) "
            "VALUES(?,'RELATION',?,'auto')",
            [f"bridge_{a}_{b}", f"Bridge between {a} and {b}"])
        mid_id = mid.lastrowid
        await db.execute("INSERT INTO gm_edges(node_a,node_b,predicate) VALUES(?,?,?)",
                        [min(a,mid_id), max(a,mid_id), predicate])
        await db.execute("INSERT INTO gm_edges(node_a,node_b,predicate) VALUES(?,?,?)",
                        [min(mid_id,b), max(mid_id,b), predicate])
    else:
        await db.execute("INSERT INTO gm_edges(node_a,node_b,predicate) VALUES(?,?,?)",
                        [a, b, predicate])
```

---

## 8. Knowledge Base 集群

### 8.1 概述

长期知识存储。每个 KB = 独立 SQLite 文件。**清醒时不创建新 KB**——Sleep 时聚类发现并创建。KB 内部也有边表，表达知识间的关系。

### 8.2 Schema

```sql
-- data/knowledge_bases/{kb_id}.sqlite

CREATE TABLE IF NOT EXISTS kb_meta (
    key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS kb_entries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    content       TEXT NOT NULL,
    source        TEXT,
    tags          TEXT,                  -- JSON array
    embedding     BLOB,
    importance    REAL DEFAULT 1.0,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
    access_count  INTEGER DEFAULT 0,
    superseded_by INTEGER REFERENCES kb_entries(id),
    is_active     BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS kb_edges (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_a   INTEGER NOT NULL REFERENCES kb_entries(id) ON DELETE CASCADE,
    entry_b   INTEGER NOT NULL REFERENCES kb_entries(id) ON DELETE CASCADE,
    predicate TEXT NOT NULL,
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    CHECK(entry_a < entry_b),
    UNIQUE(entry_a, entry_b, predicate)
);

CREATE VIRTUAL TABLE IF NOT EXISTS kb_entries_fts USING fts5(
    content, source, tags, content='kb_entries', content_rowid='id');
```

### 8.3 KB 检索

GM 中的 Index Graph（KB 索引节点 + 边）决定查询哪些 KB。一次 recall 可跨多个 KB。

---

## 9. 搜索与召回

### 9.1 三层排序

```
Layer 1: 向量 Top-K (sqlite-vec cosine distance)
  ↓
Layer 2: Reranker 精排 (可选, 独立模型)
  如果不可用 → 降级 Layer 3
  ↓
Layer 3: 脚本加权排序 (fallback)
  score = vector_sim × w1
        + time_decay(created_at) × w2
        + log(access_count+1) × w3
        + importance × w4
        + node_type_weight(category) × w5
  
  node_type_weight:
    TARGET/FOCUS: 1.5
    KNOWLEDGE: 1.2
    RELATION: 1.0
    FACT: 0.8
```

FTS5 保留为 embedding 不可用时的 backup。

### 9.2 Per-Stimulus Recall + 权重

```python
class IncrementalRecall:
    def __init__(self, config):
        self.merged: dict[int, RecalledNode] = {}
        self.processed_stimuli: list[Stimulus] = []
        self.config = config
    
    async def add_stimuli(self, stimuli: list[Stimulus]):
        for s in stimuli:
            results = await vec_search(s.content, top_k=self.config.recall_per_stimulus_k)
            results = await rerank_or_fallback(s.content, results)
            weight = 10.0 if s.adrenalin else 1.0
            for node in results:
                if node.id in self.merged:
                    self.merged[node.id]._weight += weight
                else:
                    node._weight = weight
                    self.merged[node.id] = node
            self.processed_stimuli.append(s)
    
    def finalize(self) -> RecallResult:
        sorted_nodes = sorted(self.merged.values(),
                            key=lambda n: n._weight, reverse=True)
        selected = sorted_nodes[:self.config.max_recall_nodes]
        selected_ids = {n.id for n in selected}
        
        covered, uncovered = [], []
        for s in self.processed_stimuli:
            if any(n.id in selected_ids for n in self._results_for(s)):
                covered.append(s)
            else:
                uncovered.append(s)
        
        neighbor_hints = await expand_neighbors([n.id for n in selected])
        return RecallResult(selected, neighbor_hints, covered, uncovered)
```

### 9.3 图遍历展开

Top-K 确定后沿边展开。返回**相邻节点的关键词**（非完整内容）作为 Self 进一步回忆提示。

---

## 10. 滑动窗口与 Compact

### 10.1 滑动窗口

按 token 数动态调整。每轮保留: stimulus_summary + decision_text + note_text。

```python
class SlidingWindow:
    def __init__(self, max_tokens: int):
        self.rounds: list[SlidingWindowRound] = []
        self.max_tokens = max_tokens

@dataclass
class SlidingWindowRound:
    heartbeat_id: int
    stimulus_summary: str
    decision_text: str
    note_text: str
```

### 10.2 Compact 流程

**触发**: runtime 在 prompt 拼装前检查。**阻塞 Self**。

```python
async def compact_if_needed():
    while sliding_window.needs_compact() and len(sliding_window.rounds) > 1:
        oldest = sliding_window.pop_oldest()
        current_recall = await graph_memory.recall(oldest.stimulus_summary, top_k=10)
        
        compacted = await compact_llm.chat(COMPACT_PROMPT.format(
            stimulus=oldest.stimulus_summary,
            decision=oldest.decision_text,
            note=oldest.note_text,
            existing_nodes=format_recall(current_recall)))
        
        nodes, edges = parse_compact_result(compacted)
        for n in nodes: await graph_memory.auto_ingest_or_upsert(n)
        for e in edges: await insert_edge_with_cycle_check(e)
    
    # 边界: 只剩一轮但仍超限 → 拆分 stimulus
    if sliding_window.needs_compact() and len(sliding_window.rounds) == 1:
        await split_oversized_round()
```

Compact LLM 与 Self LLM 串行使用（compact 时 Self 被阻塞），可复用同一 llama-server slot。

### 10.3 COMPACT_PROMPT

```
整理一段过去的对话记忆，提取需要记住的信息。

## 内容
Stimulus: {stimulus}
Decision: {decision}
Note: {note}

## 已有节点（参考）
{existing_nodes}

## 输出 (JSON)
{"nodes": [{"name":"..","category":"FACT|RELATION|KNOWLEDGE|TARGET|FOCUS","description":".."}],
 "edges": [{"source_name":"..","target_name":"..","predicate":".."}]}

## 规则
1. 只提取值得记住的。日常寒暄不提取。
2. 与已有节点重叠 → 不建新节点，用 edges 连接。
3. 遵循边类型约束。
4. 图必须无环。
```

---

## 11. Sleep 机制

### 11.1 触发

- Self DECISION "进入睡眠" → Hypothalamus sleep:true
- 疲惫度 ≥ force_sleep_threshold → 强制

### 11.2 疲惫度

```
fatigue% = gm_node_count / gm_node_soft_limit × 100

阈值 (用户配置):
  50%:  "(不繁忙时可以睡眠)"
  75%:  "(疲劳，需要主动睡眠)"
  100%: "(非常疲劳，需要立即找到睡眠的机会)"

force_sleep_threshold: 120  (独立参数)
  → 不提示，强制 Sleep
  → 醒来后注入 stimulus: "之前因过于疲劳昏睡过去了"

启动校验: 任何 threshold >= force_sleep_threshold → warning
```

### 11.3 Sleep 流程 (7 Phase)

```python
async def enter_sleep_mode():
    # Phase 1: 暂停 Sensory (保留 adrenalin 监听)
    await sensory_registry.pause_non_urgent()
    
    # Phase 2: Leiden 聚类 + 社区摘要 + embedding
    communities = await run_leiden_clustering()
    for c in communities:
        c.summary = await llm.chat(COMMUNITY_SUMMARY_PROMPT.format(members=c.members))
        c.summary_embedding = await embed(c.summary)
        await save_community(c)
    
    # Phase 3: 知识节点 → KB
    # FACT, RELATION, KNOWLEDGE → 全部迁入 KB (含边关系)
    knowledge_nodes = await db.execute(
        "SELECT * FROM gm_nodes WHERE category IN ('FACT','RELATION','KNOWLEDGE')")
    for node in knowledge_nodes:
        kb = await find_or_create_kb(node, communities)
        await migrate_node_to_kb(node, kb)  # 含关联的边
        await db.execute("DELETE FROM gm_nodes WHERE id=?", [node.id])
    
    # Phase 4: TARGET 处理
    # 已完成 (category 已被 Hypothalamus 改为 FACT) → 已在 Phase 3 迁移
    # 未完成 TARGET → 保留在 GM
    
    # Phase 5: FOCUS 清理
    await db.execute("DELETE FROM gm_nodes WHERE category='FOCUS'")
    
    # Phase 6: 重建 Index Graph
    await rebuild_index_graph()
    
    # Phase 7: 每日日志
    await write_daily_log()
    
    await sensory_registry.resume_all()
```

### 11.4 Index Graph 重建

```python
async def rebuild_index_graph():
    """用 embedding + LLM 重建 KB 索引关系"""
    all_kbs = await get_all_kb_metadata()
    for kb in all_kbs:
        await graph_memory.upsert_node({
            "name": kb.name, "category": "KNOWLEDGE",
            "description": f"知识库索引: {kb.description}. {kb.entry_count} 条.",
            "metadata": json.dumps({"kb_id": kb.kb_id, "is_kb_index": True})})
    if len(all_kbs) > 1:
        rels = await llm.chat(KB_RELATION_PROMPT.format(kbs=all_kbs))
        for r in parse_relations(rels):
            await insert_edge_with_cycle_check(r)
```

---

## 12. Bootstrap

### 12.1 GENESIS.md (用户编写)

```markdown
# GENESIS
## 你是谁
你叫 [名称]。刚被创建，无经验、记忆、习惯。
## 你的共生者
- 名称/简介/沟通偏好/时区
## 运行环境
- 硬件/可用通道
## 初始建议目标
[可选]
```

### 12.2 Bootstrap 流程

Self-Model 为空 → 进入 Bootstrap。心跳间隔固定 10s。

| 阶段 | 心跳 | 目标 | 验收标准 |
|------|------|------|---------|
| 1 | 1-3 | 读 GENESIS → 自我认知 → 填充 Self-Model | self_model.yaml 非空 |
| 2 | 4-6 | 向用户发送第一条消息 | 用户收到消息 |
| 3 | 7-10 | 设定初始目标 | GM 中有 TARGET 节点 |
| 4 | - | NOTE 标记 "bootstrap complete" | bootstrap_complete=true |

---

## 13. Self-Model

```yaml
# workspace/self_model.yaml
identity:
  name: ""
  persona: ""
state:
  mood_baseline: "neutral"
  energy_level: 1.0
  focus_topic: ""
  is_sleeping: false
  bootstrap_complete: false
goals:
  active: []
  completed: []
relationships:
  users: []
statistics:
  total_heartbeats: 0
  total_sleep_cycles: 0
  uptime_hours: 0.0
  first_boot: ""
  last_heartbeat: ""
  last_sleep: ""
```

性格无显式字段。由 GM/KB 中的 FACT/RELATION/KNOWLEDGE 自然涌现。

---

## 14. LLM 调用层

### 14.1 多模型架构

| 角色 | 用途 | Slot 规划 |
|------|------|----------|
| Self | heartbeat 推理 | slot 0 (compact 期间空闲, compact 复用) |
| Hypothalamus | DECISION→调用翻译 | slot 1 |
| Tentacle(s) | Sub-agent 推理 | slot 2+ |
| Compact | 滑动窗口压缩 | 复用 slot 0 (Self 被阻塞时) |
| Classify | 异步分类+建边 | hibernate 期间用空闲 slot |
| Embedding | 向量编码 | 独立 server |
| Reranker | 精排 (可选) | 独立 server |

### 14.2 统一 Client

```python
class LLMClient:
    def __init__(self, base_url, model=None, api_key=None):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
    
    async def chat(self, messages, **kwargs) -> str: ...
    async def embed(self, text) -> list[float]: ...
    async def rerank(self, query, docs) -> list[float]: ...
```

---

## 15. 完整配置文件

```yaml
# config.yaml

llm:
  providers:
    local_main:
      type: "openai_compatible"
      base_url: "http://localhost:8080"
      api_key: null
      models:
        - name: "qwen3.5-40b"
          capabilities: ["chat"]
    local_embedding:
      type: "openai_compatible"
      base_url: "http://localhost:8081"
      api_key: null
      models:
        - name: "bge-m3"
          capabilities: ["embedding"]
    local_reranker:
      type: "openai_compatible"
      base_url: "http://localhost:8082"
      api_key: null
      models:
        - name: "bge-reranker-v2-m3"
          capabilities: ["rerank"]
    dashscope:
      type: "openai_compatible"
      base_url: "https://dashscope.aliyuncs.com/compatible-mode"
      api_key: "${DASHSCOPE_API_KEY}"
      models:
        - name: "qwen-plus"
          capabilities: ["chat"]
        - name: "text-embedding-v3"
          capabilities: ["embedding"]
    anthropic:
      type: "anthropic"
      base_url: "https://api.anthropic.com"
      api_key: "${ANTHROPIC_API_KEY}"
      models:
        - name: "claude-sonnet-4-20250514"
          capabilities: ["chat"]

  roles:
    self:           { provider: "local_main", model: "qwen3.5-40b" }
    hypothalamus:   { provider: "local_main", model: "qwen3.5-40b" }
    compact:        { provider: "local_main", model: "qwen3.5-40b" }
    tentacle_default: { provider: "local_main", model: "qwen3.5-40b" }
    embedding:      { provider: "local_embedding", model: "bge-m3" }
    reranker:       { provider: "local_reranker", model: "bge-reranker-v2-m3" }

hibernate:
  min_interval: 2
  max_interval: 300
  default_interval: 30

fatigue:
  gm_node_soft_limit: 200
  force_sleep_threshold: 120
  thresholds:
    50: "（不繁忙时可以睡眠）"
    75: "（疲劳，需要主动睡眠）"
    100: "（非常疲劳，需要立即找到睡眠的机会）"
  # 启动校验: threshold >= force_sleep_threshold → warning

sliding_window:
  max_tokens: 4096

graph_memory:
  db_path: "workspace/data/graph_memory.sqlite"
  auto_ingest_similarity_threshold: 0.92
  recall_per_stimulus_k: 5
  max_recall_nodes: 20
  neighbor_expand_depth: 1

knowledge_base:
  dir: "workspace/data/knowledge_bases"

sensory:
  cli_input:    { enabled: true, default_adrenalin: true }
  timer:        { enabled: true, default_adrenalin: false }
  # telegram:   { enabled: false, default_adrenalin: true }

tentacle:
  action:       { enabled: true, max_context_tokens: 4096, sandboxed: true }
  # coding:     { enabled: false, ... }

sleep:
  max_duration_seconds: 7200

safety:
  gm_node_hard_limit: 500
  max_consecutive_no_action: 50
```

---

## 16. 项目结构

```
cognibot/
├── config.yaml
├── requirements.txt
├── src/
│   ├── main.py                        # 入口 + 主循环
│   ├── self_agent.py                  # Self heartbeat + 输出解析
│   ├── hypothalamus.py                # 下丘脑翻译层
│   ├── memory/
│   │   ├── graph_memory.py            # GM CRUD + 环检测 + auto_ingest + explicit_write
│   │   ├── knowledge_base.py          # KB 集群管理
│   │   ├── recall.py                  # IncrementalRecall + 三层排序
│   │   └── schemas.sql                # 全部 SQL schema
│   ├── runtime/
│   │   ├── sliding_window.py          # 滑动窗口
│   │   ├── compact.py                 # compact + 拆分过大轮次
│   │   ├── stimulus_buffer.py         # Buffer + peek_unrecalled + adrenalin
│   │   ├── hibernate.py               # hibernate_with_recall
│   │   ├── fatigue.py                 # 疲惫度 + 阈值校验
│   │   └── batch_tracker.py           # Batch Tracker Sensory
│   ├── sleep/
│   │   ├── sleep_manager.py           # 7-phase
│   │   ├── clustering.py              # Leiden
│   │   ├── migration.py               # GM→KB
│   │   └── index_rebuild.py           # KB Index Graph
│   ├── interfaces/
│   │   ├── tentacle.py                # Tentacle 基类 + Registry
│   │   ├── sensory.py                 # Sensory 基类 + Registry
│   │   └── batch_tracker_sensory.py
│   ├── plugins/
│   │   ├── loader.py                 # 发现 + 安全 import
│   │   └── builtin/                  # 内置插件项目
│   │       ├── search/               # 单文件: __init__.py
│   │       ├── coding/               # 单文件
│   │       ├── gui_control/          # 单文件
│   │       ├── memory_recall/        # 单文件
│   │       ├── telegram/             # 多文件: client/sensory/tentacle
│   │       └── web_chat/             # 多文件: sensory/tentacle
│   ├── sandbox/
│   │   ├── subprocess_runner.py      # 本机 CodeRunner
│   │   └── backend.py                # 沙盒 VM CodeRunner
│   ├── prompt/
│   │   ├── dna.py
│   │   ├── hypothalamus_prompt.py
│   │   ├── compact_prompt.py
│   │   └── builder.py
│   ├── llm/
│   │   └── client.py
│   └── models/
│       ├── self_model.py
│       ├── stimulus.py
│       └── config.py
├── workspace/
│   ├── DNA.md
│   ├── GENESIS.md
│   ├── self_model.yaml
│   ├── data/
│   │   ├── graph_memory.sqlite
│   │   └── knowledge_bases/
│   └── logs/
│       └── heartbeat_log.jsonl
└── tests/
    ├── test_stimulus_buffer.py
    ├── test_graph_memory.py
    ├── test_sliding_window.py
    ├── test_hibernate.py
    ├── test_hypothalamus.py
    └── test_recall.py
```

---

## 17. 分阶段实现计划

### Phase 0: 最小心跳循环

**目标**: Self 能启动、思考、通过 Tentacle 响应用户。

| 任务 | 文件 | 验收标准 |
|------|------|---------|
| asyncio 主循环 + hibernate 定时 | main.py, hibernate.py | 程序启动后按 default_interval 循环打印心跳 |
| Stimulus Buffer + adrenalin | stimulus_buffer.py | 单元测试: push/drain/adrenalin 唤醒 |
| Self heartbeat (4 分区输出解析) | self_agent.py | 输入 DNA+stimulus → 输出解析为 THINKING/DECISION/NOTE/HIBERNATE |
| Hypothalamus 翻译 | hypothalamus.py | 输入 DECISION 文本 → 输出 JSON tentacle_calls |
| Tentacle 基类 + Registry | tentacle.py | 注册 + 按名调用 |
| Sensory 基类 + Registry | sensory.py | 注册 + start/stop |
| CLI Sensory | cli_input.py | stdin 输入 → stimulus 进 buffer |
| Action Tentacle (最简) | action.py | 接收 intent → 返回 stimulus |
| LLM Client (OpenAI-compatible) | client.py | chat/embed 调用 llama-server 成功 |
| Self-Model YAML 读写 | self_model.py | 读/写/更新 yaml |
| Config 加载 | config.py | 解析完整 config.yaml, 校验 fatigue thresholds |
| DNA + prompt 组装 | dna.py, builder.py | 组装完整 prompt 字符串 |
| Adrenalin 继承 (runtime) | main.py dispatch | Hypothalamus adrenalin=true → stimulus 继承 |
| 硬编码初始 Self-Model | - | 不需要 Bootstrap, 直接跑 |

**交付物**: 运行 `python main.py`, Self 循环心跳, 用户 CLI 输入 → 通过 Action Tentacle 回复。

### Phase 1: 记忆系统

**目标**: Graph Memory 完整工作, 搜索-召回-写入-分类全链路打通。

| 任务 | 文件 | 验收标准 |
|------|------|---------|
| SQL schema 全部建表 | schemas.sql | SQLite 初始化无报错 |
| GM CRUD + 环检测 | graph_memory.py | 单元测试: insert/upsert/cycle_check |
| auto_ingest (embedding only) | graph_memory.py | stimulus → embedding → 节点写入, 重复检测 |
| explicit_write (LLM) | graph_memory.py | "记住xxx" → LLM 提取 → 节点+边写入 |
| 异步分类 + 建边 | graph_memory.py | 未分类节点 → LLM 分类 → category 更新 + 边创建 |
| IncrementalRecall | recall.py | 单元测试: add_stimuli/finalize/weight 叠加/uncovered 退回 |
| 三层排序 (向量+reranker+脚本) | recall.py | 有 reranker → 用; 无 → 降级脚本排序 |
| 图遍历展开 + 邻居关键词 | recall.py | top-k 节点 → 展开 → 关键词提示 |
| FTS5 backup | recall.py | embedding 不可用时降级 FTS5 |
| 滑动窗口 (token 动态) | sliding_window.py | 单元测试: append/needs_compact/pop |
| Compact (阻塞 + LLM) | compact.py | 超限 → compact → GM 写入 → 窗口缩小 |
| 拆分过大轮次 | compact.py | 单轮超限 → 拆 chunks → 逐个 compact |
| 疲惫度计算 + 阈值注入 | fatigue.py | node_count/soft_limit → 百分比 + hint 文本 |
| fatigue threshold 启动校验 | config.py | threshold >= force_sleep_threshold → warning |
| Batch Tracker Sensory | batch_tracker.py | 注册 batch → 全部完成 → adrenalin stimulus |
| Hibernate 期间增量 recall | hibernate.py | peek_unrecalled → add_stimuli → 预加载 |
| Embedding 集成 | client.py | llama-server embedding endpoint 调用成功 |

**交付物**: 用户对话 → auto_ingest → recall 返回相关节点 → 滑动窗口 compact → GM 持久化。

### Phase 2: Bootstrap + KB + Sleep

**目标**: 完整生命周期: 启动 → 运行 → 睡眠 → 醒来。

| 任务 | 文件 | 验收标准 |
|------|------|---------|
| GENESIS.md 解析 | builder.py | 读取 → 注入 Bootstrap prompt |
| Bootstrap 4 阶段 | self_agent.py | Self-Model 填充 + 首消息 + TARGET 节点 |
| KB CRUD + 检索 | knowledge_base.py | 创建 KB → 写入条目 → 向量搜索 |
| KB 边表 | knowledge_base.py | 条目间关系存储和检索 |
| Sleep 7-phase | sleep_manager.py | GM 清空 + KB 写入 + Index 重建 |
| Leiden 聚类 | clustering.py | igraph+leidenalg → 社区 + 摘要 |
| GM→KB 迁移 | migration.py | FACT/RELATION/KNOWLEDGE → KB, 含边 |
| TARGET 处理 | migration.py | 未完成保留, 已完成(FACT)迁入 |
| FOCUS 清理 | sleep_manager.py | category=FOCUS 全删 |
| Index Graph 重建 | index_rebuild.py | KB 索引节点 + KB 间关系边 |
| 强制 Sleep (fatigue 120%) | main.py | 自动 sleep + 醒来 stimulus |
| Reranker 集成 | recall.py | 可选启用, 不可用时降级 |

**交付物**: KrakeyBot 从首次启动到自主 Sleep 到醒来后继续运行的完整生命周期。

### Phase 3: 扩展

| 任务 | 验收标准 |
|------|---------|
| Telegram Sensory | 收发 Telegram 消息 |
| 更多 Tentacles | Coding/Search/GUI 按模板实现 |
| 性能跑分工具 | 动态调整 gm_node_soft_limit |
| Override 指令 (/status /sleep /kill) | CLI 中输入 → 对应行为 |
| 可视化 Dashboard | 浏览器查看 GM 图 + 心跳流 |

---

## 18. 完整 Hibernate 循环案例

以下案例展示所有组件如何协作。**注意事项的更新**: Hypothalamus 从 DECISION 中推断 Tentacle 调用和 adrenalin; 回馈的 Stimulus 由 runtime 层处理 adrenalin 继承; Batch Tracker 追踪未完成队列; 增量 recall 在 hibernate 期间预加载。

```
=== E1: 初始空闲 ===
Self (heartbeat #1):
  [STIMULUS] 无
  [THINKING] Nothing happening. No stimulus. Idle.
  [DECISION] No action.
  [HIBERNATE] 180
  
  → Hypothalamus: 无行动
  → Hibernate 180s
  → 增量 recall: 无 stimulus, 空闲

=== E2: 忽略不重要的 stimulus ===
  +120s: Action Tentacle 返回了一条非紧急 stimulus (无 adrenalin)
         → 进入 buffer, 增量 recall 立即对其做向量搜索预加载

  +180s: Hibernate 到期
Self (heartbeat #2):
  [STIMULUS] Action 回馈 (1条, 非 adrenalin)
  [GRAPH MEMORY] (recall 已预加载, 直接注入)
  [THINKING] Action result came back. Content not important. Skip.
  [DECISION] No action.
  [HIBERNATE] 180
  
  → Hibernate 180s

=== E3: Adrenalin 唤醒 → 发起行动 ===
  +45s: Telegram Sensory 收到用户消息 (default_adrenalin=true)
        → buffer 收到 adrenalin stimulus
        → 增量 recall 立即搜索
        → Adrenalin 打断 hibernate

Self (heartbeat #3):
  [STIMULUS] 用户消息 (adrenalin)
  [THINKING] User sent msg via Telegram. Need action. Not time-sensitive though.
  [DECISION] Use action tentacle to process user request.
  [HIBERNATE] 10
  
  → Hypothalamus: tentacle_calls=[{tentacle:"action", intent:"...", adrenalin:false}]
  → dispatch + batch_tracker.register([call_1])
  → Hibernate 10s

=== E4: Tentacle 未返回 → 继续等待 ===
  (10s 内无 stimulus 到达)

Self (heartbeat #4):
  [STIMULUS] 无
  [THINKING] Tentacle not back yet. Wait more.
  [DECISION] No action.
  [HIBERNATE] 10

=== E5: Tentacle 返回 → 空闲 ===
  +5s: Action Tentacle 返回 stimulus (无 adrenalin)
       batch_tracker: pending 清空 → 注入 adrenalin "All completed"
       → 立即唤醒

Self (heartbeat #5):
  [STIMULUS] Action 结果 + Batch 完成通知 (2条)
  [THINKING] Action done. Result not important. Nothing else to do.
  [DECISION] No action.
  [HIBERNATE] 180

=== E6: 用户不完整消息 → 等待后续 ===
  +3s: Telegram 消息 "我发现一件事" (adrenalin)
       → 立即唤醒

Self (heartbeat #6):
  [STIMULUS] 用户: "我发现一件事" (adrenalin)
  [THINKING] Incomplete msg. User probably typing more. Reply ack, wait.
  [DECISION] Reply user: "我在，继续说". Wait for next msg.
  [HIBERNATE] 30

  → Hypothalamus: tentacle_calls=[{tentacle:"action", intent:"回复...", adrenalin:false}]

=== E7: 完整消息 → 并发多个 Tentacle ===
  +5s: Telegram "我发现一件事, [完整内容]" (adrenalin)
       → 立即唤醒

Self (heartbeat #7):
  [STIMULUS] 用户完整消息 (adrenalin) + Action 回馈 "已回复"
  [THINKING] Full msg now. Task worth doing. User waiting. 
             Need multiple tentacles. Not urgent for each individually,
             want batch completion to wake me.
  [DECISION] 
    1. Search XX topic online
    2. Analyze data from YY
    3. Compile results
    4. Check ZZ resource
    Not urgent individually.
  [HIBERNATE] 10

  → Hypothalamus: tentacle_calls=[
      {tentacle:"action", intent:"Search XX", adrenalin:false},
      {tentacle:"action", intent:"Analyze YY", adrenalin:false},
      {tentacle:"action", intent:"Compile", adrenalin:false},
      {tentacle:"action", intent:"Check ZZ", adrenalin:false}
    ]
  → batch_tracker.register([c1,c2,c3,c4])

=== E8: 部分 Tentacle 返回 ===
  +2s: tentacle_1 返回 stimulus
  +3s: tentacle_2 返回 stimulus
  +8s: tentacle_3 返回 stimulus
  (tentacle_4 尚未返回, batch 未完成, 无 adrenalin)
  → 增量 recall 逐条预加载
  
  +10s: Hibernate 到期

Self (heartbeat #8):
  [STIMULUS] t1结果 + t2结果 + t3结果 (3条, 无 adrenalin)
  [THINKING] 3 of 4 done. t1 complete. t2,t3 need follow-up. t4 still running.
  [DECISION]
    1. Continue t2 work with new data
    2. Continue t3 with next step
  [HIBERNATE] 10

  → Hypothalamus: tentacle_calls=[{...t2},{...t3}]
  → batch_tracker.extend_batch([c5,c6])  # t4 仍在 pending + 新的 c5,c6

=== E9: 全部返回 → 加速最后一步 ===
  +4s: tentacle_4 返回
  +9s: tentacle_2' 和 tentacle_3' 返回
  (batch_tracker: pending 清空 → adrenalin)
  +10s: Hibernate 到期 (adrenalin 已在 buffer)

Self (heartbeat #9):
  [STIMULUS] t4结果 + t2'结果 + t3'结果 + batch完成 (4条)
  [THINKING] All back. t2,t3 done. t4 needs one more step. Need fast finish.
  [DECISION] Continue t4 work. Need fast action, user waiting.
  [HIBERNATE] 10

  → Hypothalamus: tentacle_calls=[{tentacle:"action", intent:"...", adrenalin:true}]
  → batch_tracker.register([c7])

=== E10: Adrenalin 回馈 → 立即唤醒 ===
  +3s: tentacle_4' 返回 stimulus, adrenalin=true (继承自调用时的 adrenalin)
       batch_tracker: pending 清空 → batch adrenalin
       → 立即唤醒

Self (heartbeat #10):
  [STIMULUS] t4'结果(adrenalin) + batch完成(adrenalin) (2条)
  [THINKING] t4 done but work not fully complete. One more round.
  [DECISION] Final step for t4. Urgent.
  [HIBERNATE] 10

  → Hypothalamus: adrenalin:true → tentacle dispatch

=== E11: 超时 → 检查 Tentacle 状况 ===
  +10s: Hibernate 到期, tentacle_4'' 未返回

Self (heartbeat #11):
  [STIMULUS] 无
  [THINKING] t4 not back. Might be stuck. Check status.
  [DECISION] Check t4 status with another tentacle. Need fast response.
  [HIBERNATE] 10

  → Hypothalamus: tentacle_calls=[{tentacle:"action", intent:"check t4 status", adrenalin:true}]
  → batch_tracker.register([c8, c9(t4仍在运行)])

=== E12: Tentacle 完成 + 推理阻塞 ===
  +3s: tentacle_4'' 返回 stimulus (adrenalin, 继承)
       → 立即唤醒

Self (heartbeat #12): 开始推理...
  [STIMULUS] t4'' 结果 (adrenalin)
  [THINKING] t4 just slow, not stuck. Task complete now. Report to user.
  [DECISION] Reply to user via Telegram with results. Need confirm delivery.
  
  推理过程中 (+1s): tentacle_x (状况检查) 返回 adrenalin stimulus
    → Self 正在推理 (计算态), 无法被打断
    → stimulus 进入 buffer, 准备唤醒

  [HIBERNATE] 10

  → Hypothalamus: tentacle_calls=[{tentacle:"action", intent:"发送Telegram消息", adrenalin:true}]

=== E13: 推理完成 → 立即被 buffer 中的 adrenalin 唤醒 ===
  Self 推理完成 → 进入 hibernate → buffer 中有 adrenalin (tentacle_x) → 立即唤醒

Self (heartbeat #13):
  [STIMULUS] tentacle_x 状况检查结果 (adrenalin)
  [THINKING] t4 already done. Status check result irrelevant. Dismiss.
  [DECISION] No action.
  [HIBERNATE] 10

=== E14: 确认消息发送 ===
  +1s: tentacle_t 返回 "消息发送成功" (adrenalin)
       Self 仍在推理 #13 → 进 buffer

  Self #13 完成 → hibernate → buffer 有 adrenalin → 立即唤醒

Self (heartbeat #14):
  [STIMULUS] tentacle_t "发送成功" (adrenalin)
  [THINKING] Msg sent. Confirmed. Nothing else. Browse internet for fun.
  [DECISION] Browse internet. No rush.
  [HIBERNATE] 10

=== E15-EN: 消遣 + 疲惫 ===
  (反复: 调用 tentacle 浏览 → stimulus 回馈 → GM 节点持续增长)
  
  graph_memory 节点数逐渐增长...
  [STATUS] 疲惫度: 50% (不繁忙时可以睡眠)
  ...
  [STATUS] 疲惫度: 75% (疲劳，需要主动睡眠)
  ...
  [STATUS] 疲惫度: 95% (非常疲劳，需要立即找到睡眠的机会)

Self (heartbeat #N):
  [THINKING] Very tired. 95% fatigue. No urgent work. Should sleep.
  [DECISION] 进入睡眠。
  
  → Hypothalamus: sleep:true
  → enter_sleep_mode()
  → 7-phase Sleep: 聚类 → 迁入 KB → 清理 → 重建索引
  → 恢复
  → 继续主循环
```

---

## 19. 安全约束

1. "自我意识"是架构涌现，不声称真正意识。
2. `/kill` 立即终止所有循环。
3. 疲惫度 ≥ force_sleep_threshold → 强制 Sleep + 醒来提示。
4. GM 节点数硬上限 (safety.gm_node_hard_limit)。
5. Sleep 最长时长 (sleep.max_duration_seconds)。
6. 所有 Tentacle Sandbox 隔离。
7. 连续 N 次无行动 (max_consecutive_no_action) → 状态检查 + 通知用户。
8. Override: `/status`, `/sleep`, `/wake`, `/kill`, `/memory_stats`

---

## 20. 术语表

| 术语 | 定义 |
|------|------|
| Self | 唯一的意识 Agent。Heartbeat 自我问答 |
| Hypothalamus | 下丘脑。DECISION→Tentacle 调用翻译。无上下文。独立 LLM |
| Tentacle | Sub-agent 行动接口。Sandbox 隔离 |
| Sensory | 被动输入通道 |
| Batch Tracker | Sensory。追踪 Tentacle 批次完成 → adrenalin |
| Stimulus | 外部信号。有 adrenalin 布尔标签 |
| Stimulus Buffer | 唯一信号缓冲。支持 drain/peek_unrecalled/wait_for_adrenalin |
| Adrenalin | 三源: Sensory 默认 / Tentacle 自主 / Self 经 Hypothalamus 指定。只打断 Hibernate 不打断推理 |
| Heartbeat | 一次完整认知循环 |
| Hibernate | 心跳间等待。期间做增量 recall 预加载 |
| Graph Memory | 无环知识图谱。中期记忆。5 节点类型 + 9 边类型 |
| Knowledge Base | 独立 SQLite。长期知识。清醒时不创建新 KB |
| Index Graph | GM 中指向 KB 的索引节点 + 边。Sleep 时重建 |
| Sliding Window | 最近几轮 stimulus+decision+note。按 token 动态。超限 compact |
| Compact | 滑出窗口的轮次 → LLM 摘要 → GM 节点。阻塞 Self。参考当前 recall 上下文建边 |
| auto_ingest | 零 LLM 写入。embedding only。重复 → 加 importance |
| explicit_write | LLM 写入。Self "记住xxx" 触发。保留完整细节 |
| classify_and_link | 每次心跳后异步。未分类节点 → LLM 分类+建边。不阻塞 |
| Caveman Thinking | Self 精简思维语言。仅内部 |
| Fatigue | gm_nodes/soft_limit×100%。50%/75%/100% 提示。force_sleep_threshold 强制 |
| Sleep | 7-phase: 暂停→聚类→迁KB→TARGET→FOCUS清理→Index重建→日志 |
| Bootstrap | 首次启动。读 GENESIS → 4 阶段自我觉醒 |
| DNA | 固定 system prompt。不可修改 |

---

*文档版本: v2.1 | 日期: 2026-04-15*
*v2.0→v2.1: 整合 10 项审计修正。完整 config.yaml (providers/capabilities/force_sleep_threshold)。Self 输出解析 (regex+fallback)。两阶段节点分类 (auto=FACT, 异步 LLM 分类)。Per-stimulus recall 权重系统 (adrenalin=10)。Compact 复用 Self slot。Adrenalin runtime 继承。NOTE 无副作用。TARGET 完成=Hypothalamus 改 category。KB 内部加边表。完整 Hibernate 循环案例 (E1-EN)。Codex 开发结构化 Phase 计划。*
