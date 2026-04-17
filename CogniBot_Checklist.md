# CogniBot — Implementation Checklist

> **配套文档**: `CogniBot_DevSpec_v2.1.md`
> **用法**: 按 Phase 顺序执行。每个任务完成后勾选。每个 Phase 结束时运行验收测试。

---

## Phase 0: 最小心跳循环

**目标**: Self 启动、思考、通过 Tentacle 响应用户 CLI 输入。

### 0.1 基础设施

- [ ] **config.py** — 解析 `config.yaml`（参考 DevSpec §15）
  - [ ] 加载 providers / roles / hibernate / fatigue / sensory / tentacle
  - [ ] 启动校验: 任何 fatigue.threshold >= force_sleep_threshold → 打印 warning
  - [ ] 环境变量替换: `${VAR_NAME}` → `os.environ`
  - 验收: `python -c "from src.models.config import load_config; c=load_config(); print(c.hibernate.default_interval)"`

- [ ] **client.py** — 统一 LLM Client（参考 DevSpec §14）
  - [ ] `chat(messages) -> str` — OpenAI-compatible `/v1/chat/completions`
  - [ ] `embed(text) -> list[float]` — `/v1/embeddings`
  - [ ] `rerank(query, docs) -> list[float]` — reranker endpoint
  - [ ] 支持 `type: "anthropic"` 的 provider（不同 header/body 格式）
  - 验收: 对本地 llama-server 调用 chat + embed 成功返回

- [ ] **self_model.py** — Self-Model YAML 读写（参考 DevSpec §13）
  - [ ] `load() -> dict`, `save(dict)`, `update(delta)`
  - 验收: 读/写/更新 `self_model.yaml` 字段

### 0.2 Stimulus 系统

- [ ] **stimulus.py** — Stimulus 数据类（参考 DevSpec §6.1）
  - [ ] `@dataclass Stimulus: type, source, content, timestamp, adrenalin, metadata`

- [ ] **stimulus_buffer.py** — Stimulus Buffer（参考 DevSpec §6.2）
  - [ ] `push(stimulus)` — 追加 + 设置 adrenalin/new events
  - [ ] `drain() -> list` — 消费全部, 重置索引和 events
  - [ ] `peek_unrecalled() -> list` — 返回未 recall 的 (不消费)
  - [ ] `wait_for_adrenalin()`, `wait_for_any()`, `has_adrenalin()`
  - 验收: 单元测试 — push 3 条 (1 条 adrenalin) → drain 返回 3 条按时间排序 → adrenalin event 正确触发

### 0.3 Tentacle + Sensory 框架

- [ ] **tentacle.py** — Tentacle ABC + TentacleRegistry（参考 DevSpec §5.1）
  - [ ] `register(tentacle)`, `get(name)`, `list_descriptions()`
  - 验收: 注册 mock tentacle → 按名获取 → 调用 execute 返回 Stimulus

- [ ] **sensory.py** — Sensory ABC + SensoryRegistry（参考 DevSpec §5.4）
  - [ ] `register()`, `start_all()`, `pause_non_urgent()`, `resume_all()`
  - 验收: 注册 mock sensory → start → push stimulus 到 buffer

- [ ] **cli_input.py** — CLI Sensory（参考 DevSpec §5.4）
  - [ ] stdin 监听 → Stimulus(type="user_message", adrenalin=config 值)
  - 验收: 终端输入文字 → buffer 中出现对应 stimulus

- [ ] **action.py** — Action Tentacle 最简实现（参考 DevSpec §5.2）
  - [ ] 接收 intent → LLM chat → 返回 Stimulus
  - [ ] 工作上下文膨胀检测 → 超限自动总结返回
  - 验收: 传入 "search hello world" → 返回 stimulus 含结果

### 0.4 Hypothalamus

- [ ] **hypothalamus.py** — 下丘脑翻译层（参考 DevSpec §4）
  - [ ] 输入: DECISION 文本 + 可用 tentacle 列表
  - [ ] 输出: JSON `{tentacle_calls, memory_writes, memory_updates, sleep}`
  - [ ] System prompt 中写死翻译规则（参考 DevSpec §4.2）
  - [ ] 无上下文保留 — 每次独立调用
  - 验收: 输入 "Search apple online. Not urgent." → 输出 `{tentacle_calls:[{tentacle:"action", adrenalin:false}]}`
  - 验收: 输入 "快去查一下, 有人在等" → adrenalin:true
  - 验收: 输入 "记住: 用户不喜欢夸奖" → memory_writes 非空
  - 验收: 输入 "苹果搜索任务已完成" → memory_updates 含 new_category:"FACT"

### 0.5 Self Agent

- [ ] **dna.py** — DNA Instructions 完整模板（参考 DevSpec §3.5）
  - 验收: 渲染出的 prompt 包含所有区段说明

- [ ] **builder.py** — Prompt 组装器（参考 DevSpec §3.6）
  - [ ] 组装 Layer 0-5: DNA + Self-Model + Status + Recall + History + Stimulus
  - 验收: 给定 mock 数据 → 输出完整 prompt 字符串

- [ ] **self_agent.py** — Self heartbeat + 输出解析（参考 DevSpec §3.2-3.3）
  - [ ] `parse_self_output(raw)` — regex 匹配 `[THINKING]`/`[DECISION]`/`[NOTE]`/`[HIBERNATE]`
  - [ ] Fallback: 无标记 → 整段视为 THINKING+DECISION
  - 验收: 输入含标记文本 → 正确拆分四个字段
  - 验收: 输入无标记文本 → fallback 生效, decision 非空

### 0.6 主循环 + Hibernate

- [ ] **hibernate.py** — Hibernate（参考 DevSpec §6.3，暂不含增量 recall）
  - [ ] `hibernate(interval)` — 等待 + adrenalin 打断
  - [ ] clamp(interval, min, max)
  - 验收: hibernate(5) → 5s 后返回; hibernate(60) + push adrenalin stimulus → 立即返回

- [ ] **main.py** — 主循环（参考 DevSpec §6.4，简化版）
  - [ ] drain → Self heartbeat → Hypothalamus → dispatch tentacle → hibernate
  - [ ] adrenalin 继承: Hypothalamus 指定 + Tentacle 未自行标记 → runtime 覆盖
  - [ ] uncovered stimulus → 退回 buffer → min_interval
  - [ ] 硬编码初始 Self-Model（跳过 Bootstrap）
  - 验收: `python main.py` → Self 循环心跳 → CLI 输入 → Action Tentacle 处理 → 回复显示

### Phase 0 验收

```bash
python main.py
# 1. 程序启动, 打印 heartbeat 循环
# 2. 终端输入 "hello" → adrenalin 唤醒 → Self 思考 → Hypothalamus 翻译 → Action 回复
# 3. 无输入时按 default_interval 空转
# 4. Ctrl+C 终止
```

---

## Phase 1: 记忆系统

**目标**: Graph Memory 完整工作。搜索-召回-写入-分类-compact 全链路。

### 1.1 数据库

- [ ] **schemas.sql** — 全部 SQL schema（参考 DevSpec §7.4 + §8.2）
  - [ ] gm_nodes + gm_edges + gm_communities + gm_node_communities + kb_registry
  - [ ] FTS5 虚拟表 + 触发器
  - [ ] KB schema: kb_meta + kb_entries + kb_edges + kb_entries_fts
  - 验收: `sqlite3 test.db < schemas.sql` 无错误

### 1.2 Graph Memory 核心

- [ ] **graph_memory.py** — CRUD（参考 DevSpec §7.4-7.7）
  - [ ] `auto_ingest(content)` — embedding only, 重复检测(阈值), importance 递增
  - [ ] `explicit_write(content, importance, recall_context)` — LLM 提取节点+边
  - [ ] `update_node_category(node_name, new_category)` — Hypothalamus 调用
  - [ ] `upsert_node(node)` — 同名同类 → 更新; 否则 → 新建
  - [ ] `would_create_cycle(a, b)` — 递归 CTE 连通性检测
  - [ ] `insert_edge_with_cycle_check(src, tgt, predicate)` — 环 → 创建中间 RELATION 节点
  - [ ] `classify_and_link_pending()` — 异步: 未分类 auto 节点 → LLM 分类+建边, limit 10
  - 验收: 
    - auto_ingest 两次相同内容 → importance 增加, 节点数不变
    - explicit_write → 节点+边写入
    - 环检测: A-B 已连通 → 插入 A-B 边 → 创建中间节点
    - classify: auto FACT 节点 → 分类为 RELATION/KNOWLEDGE

### 1.3 搜索与召回

- [ ] **recall.py** — IncrementalRecall（参考 DevSpec §9）
  - [ ] `add_stimuli(stimuli)` — per-stimulus 向量搜索, adrenalin 权重=10
  - [ ] `finalize()` — 去重, 权重排序, top max_recall_nodes, covered/uncovered 分离
  - [ ] 三层排序: 向量 top-k → reranker (可选) → 脚本 fallback
  - [ ] 脚本排序: vector_sim × w1 + time_decay × w2 + log(access+1) × w3 + importance × w4 + type_weight × w5
  - [ ] 图遍历展开: 相邻节点关键词作为提示
  - [ ] FTS5 backup: embedding 不可用时降级
  - 验收:
    - 5 条 stimulus → 5 次独立搜索 → 去重后 ≤ max_recall_nodes
    - adrenalin stimulus 的结果权重更高
    - uncovered stimulus 正确识别

### 1.4 滑动窗口 + Compact

- [ ] **sliding_window.py** — 按 token 动态窗口（参考 DevSpec §10.1）
  - [ ] `append(round)`, `get_rounds()`, `needs_compact()`, `pop_oldest()`
  - [ ] 每轮保存: stimulus_summary + decision_text + note_text
  - 验收: 追加到超限 → needs_compact()=True → pop 后 needs_compact()=False

- [ ] **compact.py** — Compact 流程（参考 DevSpec §10.2-10.3）
  - [ ] `compact_if_needed()` — 逐轮 compact 直到窗口满足
  - [ ] Compact LLM: 参考当前 recall 上下文建边
  - [ ] `split_oversized_round()` — 单轮超限 → 拆 chunks
  - [ ] COMPACT_PROMPT 模板
  - 验收: 窗口超限 → compact → GM 中出现新节点+边 → 窗口缩小

### 1.5 疲惫度 + Batch Tracker

- [ ] **fatigue.py** — 疲惫度（参考 DevSpec §11.2）
  - [ ] `calculate_fatigue() -> (pct, hint)`
  - [ ] force_sleep_threshold 检查
  - 验收: 200 nodes, soft_limit=200 → 100%, hint="非常疲劳"

- [ ] **batch_tracker.py** — Batch Tracker Sensory（参考 DevSpec §5.5）
  - [ ] `register_batch()`, `mark_completed()`, `extend_batch()`
  - [ ] 队列清空 → push adrenalin stimulus
  - 验收: register 3 → complete 2 → 无触发 → complete 1 → adrenalin 触发
  - 验收: register 3 → complete 2 → extend 1 → complete 剩余旧的 → 无触发 → complete 新的 → 触发

### 1.6 Hibernate 增量 Recall

- [ ] **hibernate.py** 升级 — hibernate_with_recall（参考 DevSpec §6.3）
  - [ ] 期间用 `peek_unrecalled()` 获取新 stimulus
  - [ ] 调用 `incremental_recall.add_stimuli()` 预加载
  - [ ] adrenalin 打断
  - 验收: hibernate 期间 push stimulus → peek 返回 → recall 结果非空 → adrenalin push → 立即返回

### 1.7 主循环升级

- [ ] **main.py** 升级 — 集成记忆系统（参考 DevSpec §6.4）
  - [ ] 步骤 3: 增量 recall (含 hibernate 预加载结果)
  - [ ] 步骤 4: compact (阻塞)
  - [ ] 步骤 5: finalize recall + uncovered 退回
  - [ ] 步骤 8: auto_ingest tentacle stimulus
  - [ ] 步骤 10: classify_and_link_pending (异步, 每心跳后)
  - [ ] 步骤 12: hibernate_with_recall
  - 验收: 对话 → auto_ingest → recall 返回相关节点 → compact → GM 持久化

### Phase 1 验收

```bash
python main.py
# 1. 对话内容写入 GM (auto_ingest)
# 2. 后续对话 recall 返回之前的相关节点
# 3. 长对话触发 compact → GM 节点增长
# 4. sqlite3 workspace/data/graph_memory.sqlite "SELECT count(*) FROM gm_nodes;" → 非零
# 5. 异步分类: auto FACT 节点被重新分类
```

---

## Phase 2: Bootstrap + KB + Sleep

**目标**: 完整生命周期 — 启动 → 运行 → 睡眠 → 醒来。

### 2.1 Bootstrap

- [ ] **builder.py** 扩展 — GENESIS.md 解析 + Bootstrap prompt 注入（参考 DevSpec §12）
  - [ ] 检测 self_model.yaml 为空 → Bootstrap 模式
  - [ ] Bootstrap 心跳间隔固定 10s
  - 验收: 空 self_model → Bootstrap → 4 阶段后 bootstrap_complete=true

### 2.2 Knowledge Base

- [ ] **knowledge_base.py** — KB 集群管理（参考 DevSpec §8）
  - [ ] `create_kb(kb_id, name, description)` → 创建独立 SQLite
  - [ ] `write_entry(kb_id, content, tags, embedding)` + `write_edge(kb_id, a, b, predicate)`
  - [ ] `search_kb(kb_id, query, top_k)` → 向量搜索 + FTS5 backup
  - [ ] `get_all_kb_metadata() -> list`
  - 验收: 创建 KB → 写入条目+边 → 搜索返回相关条目

### 2.3 Sleep

- [ ] **sleep_manager.py** — 7-Phase Sleep（参考 DevSpec §11.3）
  - [ ] Phase 1: sensory_registry.pause_non_urgent()
  - [ ] Phase 2: Leiden 聚类 + 社区摘要 + embedding
  - [ ] Phase 3: FACT/RELATION/KNOWLEDGE → 迁入 KB (含边)
  - [ ] Phase 4: TARGET (已完成=已改为FACT→Phase3已迁; 未完成→保留)
  - [ ] Phase 5: FOCUS 全删
  - [ ] Phase 6: rebuild_index_graph()
  - [ ] Phase 7: write_daily_log()
  - [ ] sensory_registry.resume_all()
  - 验收: Sleep 前 GM 有 50 节点 → Sleep 后 GM 只剩未完成 TARGET + KB 索引节点

- [ ] **clustering.py** — Leiden 聚类（参考 DevSpec §11.3）
  - [ ] igraph + leidenalg → 社区 → 摘要 → embedding
  - 验收: 30 节点 → 聚类 → ≥2 社区 → 各有摘要

- [ ] **migration.py** — GM→KB 迁移（参考 DevSpec §11.3）
  - [ ] 按社区匹配已有 KB 或创建新 KB
  - [ ] 迁移节点 + 关联边 → KB entries + kb_edges
  - [ ] 删除 GM 中已迁移的节点
  - 验收: GM FACT 节点 → KB entry + 对应 edges

- [ ] **index_rebuild.py** — Index Graph 重建（参考 DevSpec §11.4）
  - [ ] 为每个 KB 创建/更新 GM 中的 KNOWLEDGE 索引节点
  - [ ] LLM 判断 KB 间关系 → 创建边
  - 验收: 3 个 KB → GM 中 3 个索引节点 + 关系边

### 2.4 强制 Sleep

- [ ] **main.py** 扩展 — fatigue ≥ force_sleep_threshold（参考 DevSpec §11.2）
  - [ ] 自动 enter_sleep_mode()
  - [ ] 醒来后 push stimulus "之前因过于疲劳昏睡过去了"
  - 验收: 手动设 gm_node_soft_limit=5 → 快速达到 120% → 强制 sleep → 醒来有 stimulus

### 2.5 Reranker 集成

- [ ] **recall.py** 扩展 — reranker（参考 DevSpec §9.1）
  - [ ] config.roles.reranker.enabled → 调用 reranker endpoint
  - [ ] 不可用 → 降级脚本排序
  - 验收: 有 reranker → Layer 2 精排; 关掉 → 降级 Layer 3

### Phase 2 验收

```bash
# 1. 删除 self_model.yaml → 重启 → Bootstrap 完成 → self_model 非空
# 2. 长时间对话 → GM 增长 → fatigue 提示 → Self 决定 sleep
# 3. Sleep 后: GM 只剩 TARGET + 索引节点; KB 目录下有 .sqlite 文件
# 4. 醒来后继续运行, recall 能通过 Index Graph 查到 KB 中的知识
# 5. 强制 sleep 测试: 低 soft_limit → 自动 sleep → 醒来 stimulus
```

---

## Phase 3: 扩展

按需实现，无硬性顺序。

- [ ] **Telegram Sensory** — 收发 Telegram 消息
  - 验收: Telegram 发消息 → CogniBot 回复

- [ ] **更多 Tentacles** — Coding / Search / GUI Control
  - 验收: 按自定义模板实现 → 注册 → Self 可调用

- [ ] **Override 指令** — `/status`, `/sleep`, `/wake`, `/kill`, `/memory_stats`
  - 验收: CLI 输入 `/status` → 打印当前状态

- [ ] **性能跑分工具** — 动态调整 gm_node_soft_limit
  - 验收: 跑分 → 输出建议 soft_limit 值

- [ ] **可视化 Dashboard** — 浏览器查看 GM 图 + 心跳流
  - 验收: 打开 localhost:port → 看到 GM 图

---

## 跨 Phase 注意事项

1. **每个文件写完后立即写对应的测试**。测试文件命名 `test_{module}.py`。
2. **所有 LLM 调用都经过 `LLMClient`**，不直接 `aiohttp.post`。
3. **所有 SQLite 操作用 `aiosqlite`**，不用同步 sqlite3。
4. **环检测在每次 `insert_edge` 时必须执行**，无例外。
5. **config.yaml 中的 `${VAR}` 在 load_config 时替换**，不在运行时。
6. **Caveman style 仅在 DNA prompt 中规定**，代码中不强制检测输出风格。
7. **NOTE 无 runtime 副作用**——不解析、不更新 self_model、不触发写入。只存窗口。
8. **compact 阻塞 Self**——compact_if_needed() 必须 await 完成后才开始 Self heartbeat。
9. **classify_and_link_pending() 不阻塞**——asyncio.create_task, 后台运行。
10. **Adrenalin 继承在 dispatch_tentacle 中实现**，不修改 Tentacle 基类。
