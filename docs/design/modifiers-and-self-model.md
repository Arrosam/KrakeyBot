# Modifiers + Self-model slimdown + Recall-layer LLM — design draft

> Status: **draft / under discussion**. This document is just a written
> snapshot of Samuel's stated requirements; implementation has not
> started.
> Edit this file directly to change or extend it; a PR thread will be
> opened before implementation.
> Last recorded: 2026-04-25.

---

## 🔒 Core design principle (load-bearing invariant, 2026-04-25 / 2026-04-26)

> **Removing or disabling any plugin (Modifiers, tools, channels) must
> NOT affect the runtime's core loop.**
>
> **Plugin code is not loaded until the user enables it.** The only
> influence allowed before enablement is exposing "configuration
> options" in plain text to the Web UI (the Web UI reads `meta.yaml`
> for the metadata; it does **not** import any plugin code).
>
> **Plugins cannot access the central `config.yaml`** (which contains
> API keys + provider config). Plugin Python code only sees:
> (1) plain-text settings from its own folder's `config.yaml`
> (via `ctx.config`); (2) the `LLMClient` instance returned by
> `ctx.get_llm_for_tag(tag_name)`. Plugins read their own config to
> get the tag name bound under `llm_purposes`, then ask the runtime
> for the corresponding client. The `LLMClient` encapsulates provider
> info internally, but plugins **never see the underlying API key**.

## 🏷️ Tag-based LLM system (2026-04-26)

Three layers of abstraction, each with a single responsibility:

```
1. providers (central config.yaml)
   API connections + secrets. Visible only to runtime; plugins never see them.

2. tags (central config.yaml)
   Semantic name → (provider, model, params) triple.
   The provider field uses the compact "<provider_name>/<model_name>" form
   (split on the first "/"; provider names must not contain "/", model
   names may).

3. purposes — bind a use case to a tag
   - core purposes (Self/compact/classifier ...): central config
   - plugin purposes: that plugin's folder config.yaml
```

**Special model-type slots** (not purposes — capabilities):

```yaml
llm:
  embedding: <tag_name>     # required for GM auto-recall
  reranker: <tag_name>      # optional, for recall reranking
```

**Full central-config example**:

```yaml
llm:
  providers:
    "One API":
      base_url: https://api.example/v1
      api_key: ${ONEAPI_KEY}
  tags:
    qwen_self:
      provider: "One API/qwen3.6-9b"
      params: {temperature: 0.7, max_input_tokens: 32768}
    bge_embed:
      provider: "Local/bge-m3"
      params: {}
  core_purposes:
    self_thinking: qwen_self
    compact: qwen_self
    classifier: qwen_self
  embedding: bge_embed
  # reranker: ... (optional)
```

**Plugin side** (`workspace/modifiers/<name>/config.yaml`):

```yaml
llm_purposes:
  translator: qwen_self     # bind plugin purpose to a central tag
```

**Plugin `meta.yaml` side**:

```yaml
llm_purposes:
  - name: translator
    description: "..."
    suggested_tag: <hint, optional>     # not auto-bound
```

**Plugin factory signature**:

```python
def build_modifier(ctx: PluginContext) -> Modifier | None:
    # The plugin reads its own config to get the bound tag name,
    # then asks the runtime for the corresponding LLMClient.
    purposes = ctx.config.get("llm_purposes") or {}
    tag = purposes.get("translator") if isinstance(purposes, dict) else None
    llm = ctx.get_llm_for_tag(tag)
    if llm is None:
        return None    # not bound → skip self-registration (additive principle)
    return MyModifier(llm)
```

**LLMClient sharing**: multiple purposes mapped to the same tag share
one client instance (saves connections + a single rate-limit counter).
The shared cache `deps.llm_clients_by_tag` is reused across the core
and plugin paths.

**Migration**: the old `llm.roles:` shape has been removed. The loader
detects it and **fails loudly with a detailed migration message**, then
exits. The runtime will not start until the config is migrated — users
are forced to migrate explicitly.

---

Implications:
- The runtime's core heartbeat loop must run with **zero plugins**.
  Self can complete a full heartbeat with an empty `[GRAPH MEMORY]`
  layer, no external stimuli, and no callable tools.
- All plugins are **strictly additive**. Disabling or removing any one
  must not raise, hang a phase, or put the runtime in a bad state.
- Implementation rule: every plugin call site in the runtime has a
  **fallback** — a null-object (e.g. `NoopRecall`) or a soft-fail
  (e.g. tool dispatch with an unknown name → push an
  `Unknown tool: X` system event to Self instead of raising).
- Test coverage: with no Modifier registered, the runtime must
  complete a full heartbeat. This is the regression line; any change
  that breaks the invariant is rejected by the test suite.

This principle has **higher priority than "default behavior"**: this
section governs **structural reliability**; "which Modifiers are on by
default" is just a UX choice.

---

## Background / motivation

Discussing Self-model fields surfaced a few long-standing problems:

1. **`statistics` field** is mostly unwritten yet eats a large chunk
   of `[SELF-MODEL]` prompt tokens.
2. **`relationships.users`** explicitly structures "who you have a
   relationship with" — violating the design philosophy that important
   memories should naturally emerge through GM recall.
3. **`is_sleeping`** is never seen as `true` by Self (Self does not
   run during sleep), so it is a dead field.
4. **`mood_baseline`** is too simple a field to carry something as
   complex as emergent emotion.
5. **`focus_topic` / `goals.active`** duplicate GM's `FOCUS` /
   `TARGET` nodes; the GM versions are stronger (edges, importance,
   sleep migration), making the self-model versions redundant.

In parallel, Samuel proposed three more aggressive ideas:

- **Recall-layer LLM**: insert a dedicated LLM before stimulus
  processing that extracts "recall feature points" from stimulus +
  history and uses them to drive GM recall. Far more accurate than the
  current "vector-search the raw stimulus" approach — it can actively
  recall a speaker's profile, related events, and so on.
- **Modifier plugin type**: a deeper extension point than tools or
  channels. Modifiers listen to heartbeat-start / -end events and can
  intercept or replace runtime core mechanisms.
- **Default mechanisms = default Modifiers**: rewrite the existing
  Hypothalamus and auto-recall as built-in Modifiers so users can
  swap them out or disable them.

---

## Part 1 — Self-model slimdown

### Final retained schema

```yaml
identity:
  name: ""
  persona: ""
state:
  bootstrap_complete: false
```

**Reasoning**:
- `identity` is Self's invariant core. After Bootstrap, Self updates
  it through a dedicated path (no more `<self-model>` tag black
  magic — see Modifier #3 `in_mind`).
- `bootstrap_complete` is the runtime's required switch.

### Removed fields and their replacements

| Field | Replacement |
|---|---|
| `statistics.*` | Moved to `workspace/data/runtime_stats.json` (or a `system:*` node) — used only by the dashboard, not in the prompt. |
| `relationships.users` | Naturally emerges via GM RELATION nodes + the recall-layer LLM (Part 2). |
| `state.mood_baseline` | Removed. Real emotion is carried by the `in_mind` Modifier's free-text descriptions (Part 3). |
| `state.is_sleeping` | Removed. An in-memory runtime flag is enough; no need to persist. |
| `state.focus_topic` | Replaced by GM's `category=FOCUS` nodes (existing mechanism). |
| `state.energy_level` | Removed. If you need a "tired" concept, fatigue% suffices. |
| `goals.active` | Replaced by GM's `category=TARGET` nodes. |
| `goals.completed` | Not stored separately; once a TARGET completes, Sleep demotes it to FACT — that is the natural representation of a "completed goal". |

### Affected files

- `src/models/self_model.py`: simplify `default_self_model()`.
- `src/main.py`: remove the post-sleep `total_sleep_cycles` update
  (move it elsewhere).
- `src/prompt/builder.py`: the `[SELF-MODEL]` layer becomes shorter,
  improving prompt-cache stability.
- All `test_*.py` tests that depend on these fields need updating.

---

## Part 2 — Recall-layer LLM

### Idea

Insert a **recall-layer LLM** between `_phase_drain_and_seed_recall`
and the Self LLM call. It reads all current stimuli + the rolling
window and outputs a list of "feature points to recall on".

**Today**:
```
stimulus.content → embed → vec_search → ranked merge
```
Pure raw-text vector search, with limited precision:
- A user saying "hi" cannot recall the memory "Samuel discussed agent
  architecture" (semantically too far).
- It cannot extract "which people / which events / which related
  topics to recall".

**New approach**:
```
stimuli + window → recall-layer LLM → [
  {anchor: "user:Samuel", reason: "message came from Samuel"},
  {anchor: "topic:Agent architecture", reason: "topic continues from context"},
  {anchor: "event:just finished Bootstrap", reason: "temporal association"},
] → each anchor triggers its own recall (vector / FTS / exact-name match) → merge weights
```

LLM prompt template (finalized 2026-04-25):

```
# Memory Recall Guide

Pick recall keywords for the next decision. Read the input and output
a keyword list to retrieve from the graph memory.

## Input

[CURRENT_STIMULI]
{stimuli}

[RECENT_HISTORY]
{history}

## Selection criteria

✅ Choose as anchor:
- Concrete person names
- Concrete topics or concepts
- Concrete events or milestones
- Domain-specific terms
- Temporal references

❌ Do NOT choose:
- Function words / greetings
- Broad words
- Whole sentences from the stimulus
- Long sentences

At most 8 anchors. Output an empty list when no obvious recall points exist.

## Output (strict JSON; no other text, no markdown code fence)

{"anchors": ["..."]}

## Examples

Example 1
[CURRENT_STIMULI]
[1] user_message from channel:chat:
    "Alex: how did the optimization plan turn out for speed?"
[RECENT_HISTORY]
Heartbeat #N-1: Decision: "rewrite the hot loop in Cython"
Expected output: {"anchors": ["Alex", "Cython optimization", "performance benchmark"]}

Example 2
[CURRENT_STIMULI]
[1] user_message from channel:chat: "Bob: oh."
[RECENT_HISTORY]
(empty)
Expected output: {"anchors": []}

Example 3
[CURRENT_STIMULI]
[1] tool_feedback from tool:weather_check: "Sunny, 22°C"
[2] batch_complete from channel:batch_tracker: "All dispatched."
[RECENT_HISTORY]
Heartbeat #N-1: Decision: "Check the weather to plan tomorrow's hike"
Expected output: {"anchors": ["weather check", "hiking plan"]}
```

**Template design decisions**:

- **No role intro / no explanation of what KrakeyBot is** — the task
  is stated in one line. Fewer sentences, fewer error surfaces (no
  need to disclaim "you are not KrakeyBot").
- **No `<thinking>` block or any CoT scaffolding** — let users pick
  their own model. Reasoning models reason on their own; non-reasoning
  models doing poorly is the cost of model choice and is not baked
  into the core mechanism (consistent with Modifier #1's
  default-disabled philosophy).
- **`anchors` is a flat string list** — no `reason` / `kind` /
  `rationale` fields, since downstream does not consume them and they
  waste tokens.
- **GM state is not passed in** — black-box. Anchors come from
  situational semantics rather than "exists in GM"; vec_search / FTS /
  exact-name match handle "not found" themselves.
- **Agent identity slot is not passed in** — `in_mind` is injected at
  the most-recent end of the history layer (see Modifier #3); other
  self-related context is naturally inferable from history / stimuli.
- **Examples use generic names / topics (Alex / Bob / Cython /
  weather)** — keeps the prompt template's own semantics neutral and
  prevents the LLM from treating "Samuel" / "ReAct" or other actual
  conversation content as implicit context.

### Relationship to Modifier #2

This feature is **Modifier #2 — recall-feature extractor**. See
Part 3.

### Toggle semantics

- **On**: LLM-driven anchor extraction → multi-route recall.
- **Off**: scripted pure-vector recall (current behavior).

Both modes share recall's ranking + token-budget trim post-processing —
they differ only in **where the recall candidates come from**.

---

## Part 3 — Modifier plugin system

### Position

A Modifier is a **deep plugin** — unlike a tool (outbound limb) or a
channel (inbound sensor), it listens to **each heartbeat's start /
end** events and can replace or intercept core runtime mechanisms.

**Core constraint**: Modifiers are not arbitrary monkey-patching;
they hook into runtime-defined points. Every hook point should be
findable via `grep`.

### Base protocol (draft)

```python
class Modifier(Protocol):
    name: str
    kind: str  # "hypothalamus" | "recall_anchor" | "in_mind" | ...
    enabled: bool

    async def on_heartbeat_start(self, ctx: HeartbeatContext) -> None: ...
    async def on_heartbeat_end(self, ctx: HeartbeatContext) -> None: ...

    # kind-specific hooks defined per subclass
```

Concrete hook points are defined per kind. For example, a Modifier
with `kind="hypothalamus"` overrides "translate Self's Decision into
tool calls".

### Three Modifiers to build

#### Modifier #1 — `hypothalamus` (toggleable Hypothalamus)

**Important precondition (clarified 2026-04-25)**: the default
mechanism and the Hypothalamus Modifier are **mutually exclusive via
the prompt layer** — they do NOT run in parallel.

- **Default state (Modifier #1 inactive — not in the modifiers list)**:
  - Self's prompt contains an `[ACTION FORMAT]` block teaching it the
    structured tag syntax for tool calls.
  - Self's output contains `[ACTION]...[/ACTION]` blocks with JSONL
    calls.
  - The built-in **action executor** (script, no LLM) scans that block
    and parses it directly into tool calls for dispatch.
  - Short path: 1 Self LLM call → parse → dispatch.

- **Active state (Modifier #1 in the modifiers list)**:
  - Activating Modifier #1 **suppresses the `[ACTION FORMAT]` block —
    it disappears from the prompt**.
  - Self no longer sees instructions about structured calls and falls
    back to natural-language decisions.
  - The Hypothalamus LLM translates Self's natural-language decision
    into tool calls.
  - Long path: 2 LLM calls (Self + Hypothalamus) → dispatch.

**Why this design**:

Putting "teach Self to write ACTION tags" and "let Hypothalamus
translate" in the same prompt **interferes** — Self emits tags after
seeing the tutorial, and Hypothalamus then tries to translate that
tag, leading to dual interpretation and dual confusion. So activating
Modifier #1 cleanly removes the ACTION tutorial — Self does not know
the format exists, and Hypothalamus is the only translator.

**Default**: **off** (strong models do not need the Hypothalamus
translation layer; emitting ACTION JSONL directly is faster and
cheaper than a second LLM call).

**Modifier #1 exists so smaller models can still run** — small models
may not stably produce valid JSONL, so turning Modifier #1 on lets
Hypothalamus cover for them.

**Action tag format (finalized 2026-04-25): OpenAI tool_calls-style
JSONL — one JSON object per line, wrapped in
`[ACTION]...[/ACTION]`**:

```
[ACTION]
{"name": "web_chat_reply", "arguments": {"text": "Hi Alex!"}}
{"name": "search", "arguments": {"query": "Cython optimization"}}
[/ACTION]
```

Fields:
- `name` (str, required): tool name.
- `arguments` (object, optional): parameter dict; omitted = empty
  dict.
- `adrenalin` (bool, optional): urgency flag; omitted = false.

**Why this format**:
- OpenAI function calling is the industry standard (2023 onward);
  DeepSeek / Mistral / Qwen / Gemini all follow the same schema —
  LLM training data covers it best.
- Field names `name` + `arguments` map 1:1 to OpenAI tool definitions,
  reusing existing tool schema descriptions.
- JSONL with one object per line — `for line in block:
  json.loads(line)` is the simplest possible parser; no nested XML
  state machine.
- A bad line does not corrupt the others (more robust than XML, where
  a single broken tag invalidates the whole block).
- Edge cases like Unicode / escapes / multi-line arguments are
  handled by the JSON standard.

XML-ish `<use>...</use>` and function-style `@tool(...)` were
rejected: the former is verbose and needs a more stateful parser;
the latter is compact but LLMs frequently miss trailing commas /
quote escapes.

**Open issue**:
- Default-off means small-model users have to enable this Modifier
  manually. OK, acceptable.
- The existing Hypothalamus code will move into a location like
  `src/modifiers/builtin/hypothalamus/`.

#### Modifier #2 — `recall_anchor` (recall-feature extractor)

**Default**: ? (Samuel to decide; recommendation **on**, the quality
gain is significant).

**On**: the LLM-driven anchor extraction + multi-route recall
described in Part 2.

**Off**: the existing scripted pure-vector recall
(`IncrementalRecall`'s current behavior).

**Existence form of this Modifier**:
- The existing scripted recall is extracted into a "default built-in
  Modifier" — even when the user "disables" it, runtime falls back
  to this default.
- The LLM-anchor Modifier is an alternative optional Modifier that
  **replaces** the default when active.
- Allowing both at once (LLM anchor first, scripted fallback after)
  is also possible — decide during design.

#### Modifier #3 — `in_mind` (mental-state self-report)

**Default**: ? (recommendation **on**, since it replaces the deleted
focus / mood / goals fields).

**Function**: lets Self continually record three things during its run:
- **The most important thought right now** (one sentence — what is
  on its mind).
- **Mood** (a short phrase + brief reason).
- **What it is concretely focused on** (one sentence).

These appear as a **fixed-position prompt layer** (e.g. `[IN MIND]`)
that is always visible per beat and does not slide out of the window.

**How Self updates it**:
- Via a new tool `update_in_mind` (registered to `runtime.tools` by
  the Modifier in `attach(runtime)`; Self can dispatch it directly
  via [ACTION] or via the Hypothalamus translation path), with three
  optional fields (partial updates allowed).
- State file: `workspace/data/in_mind.json`, read/written by
  Modifier #3 itself.

**Implementation landed 2026-04-26** (`src/plugins/in_mind_note/`):
- `meta.yaml` — static metadata declaration, `kind="in_mind"`.
- `state.py` — `InMindState` dataclass + atomic load/save (tempfile +
  replace).
- `tool.py` — `UpdateInMindTool` (`is_internal=True`, calls Modifier
  directly).
- `prompt.py` — `IN_MIND_INSTRUCTIONS_LAYER` constant + virtual-round
  rendering.
- `modifier.py` — `InMindModifierImpl` body + `attach(runtime)` to
  register the tool.
- Protocol additions: `InMindModifier` Protocol + a generic
  `attach(runtime)` Modifier hook.
- Registry additions: `in_mind_state()` (returns None when no plugin)
  + `attach_all(runtime)`.
- Builder additions: `in_mind` / `in_mind_instructions` parameters;
  `_layer_history` inserts a virtual "Heartbeat #now (in mind)" round
  at the head when `in_mind` state is non-empty.
- `RuntimeDeps.in_mind_state_path` for test state isolation.

**State vs configuration separation**: `workspace/data/` is the home
for runtime state (already houses `graph_memory.sqlite` /
`web_chat.jsonl` / `knowledge_bases/`); user configuration lives in
`workspace/plugin-configs/` and `workspace/modifiers/<name>/config.yaml`.
`in_mind` is runtime state (Self writes, runtime reads), so it
belongs under `data/`, kept separate from any Modifier config file.

**Prompt-injection method: a virtual round at the most-recent end of
the history layer** (finalized 2026-04-25).

Instead of opening a new prompt layer, the prompt builder **inserts a
virtual newest round at the head** when rendering `[HISTORY]`. The
content is the three current `in_mind` fields. For example:

```
[HISTORY]
--- Heartbeat #now (in mind) ---
Thoughts: thinking about how to answer Alex's optimization question
Mood: slightly tense, benchmark numbers haven't finished
Focus: porting the hot loop from Python to Cython
--- Heartbeat #N-1 ---
Stimulus: ...
Decision: ...
...
```

**Why this rather than a separate layer**:

1. **Zero new slot**: every prompt consumer (Self LLM, recall LLM,
   future Modifiers) reads the history layer and gets `in_mind`
   automatically, without per-consumer support.
2. **Correct sense of time**: `in_mind` is "what is on my mind right
   now"; placing it at the most-recent end of history is semantically
   natural.
3. **Single source of truth**: rendering logic lives only in the
   prompt builder; everywhere else stays unaware.
4. **Cache-friendly**: the change rate of `in_mind` sits between the
   history layer (changes every beat) and the self-model layer
   (rarely changes), exactly where the history layer lives — so it
   does not break the higher layers' stable cache.
5. **Recall LLM benefits for free**: `in_mind`'s "focus" / "thoughts"
   become natural seeds for the next beat's anchor extraction. Self
   notes "thinking about Cython optimization" on this beat; on the
   next beat the recall LLM sees "Cython optimization" in recent
   history and naturally pulls related GM nodes. **Self-guided
   recall** comes for free.

**Relation to deleted self_model fields**:
- Replaces `state.focus_topic` and `state.mood_baseline`.
- Replaces the "current mental goal" portion of `goals.active` (the
  full goal system goes through GM TARGET nodes; `in_mind` only holds
  "the one or two things in mind right now").

---

## Part 4 — Suggested implementation order

Samuel makes the final call; this is my recommendation:

1. **Bug fix first: stimulus truncation** (fixed in sync 2026-04-25;
   not waiting on this design discussion).
2. **Self-model slimdown** — independent commit, small, clean,
   touches no new mechanism.
3. **Modifier protocol + default built-in skeleton** — no new
   features. Just extract the existing Hypothalamus and recall paths
   from the main flow into default Modifiers without changing
   behavior. This is the highest-risk refactor — get the foundation
   right first.
4. **Modifier #1 `hypothalamus` toggleable + action executor engine**
   — closest to Samuel's "save LLM calls on strong models" desire;
   land it early to gather signal.
5. **Modifier #3 `in_mind`** — replaces the deleted focus / mood /
   goals fields and gives Self the ability to actively record its
   mental state.
6. **Modifier #2 `recall_anchor`** — most complex (an extra LLM call
   per beat, may add heartbeat latency); ship last and watch latency
   regressions.

---

## Part 5 — Open questions for Samuel

- [x] ~~Modifier #1 action-tag format~~ — 2026-04-25: **OpenAI
      tool_calls-style JSONL** (`{"name": "...", "arguments": {...}}`,
      one per line), wrapped in `[ACTION]...[/ACTION]`. Reasoning
      recorded in Part 3, Modifier #1.
- [x] ~~Modifier #2 default toggle~~ — 2026-04-25: **default off**.
      An extra LLM call per beat costs latency / money; users opt in
      to accept that cost. Same philosophy as Modifier #1's default
      off (do not bake crutches for weak models).
- [x] ~~Where does Modifier #3 `in_mind`'s state file live?~~ —
      2026-04-25: `workspace/data/in_mind.json`. Runtime state (Self
      writes / runtime reads) lives under `workspace/data/`,
      alongside `graph_memory.sqlite` and `web_chat.jsonl`; user
      config lives under `workspace/plugin-configs/` or
      `workspace/modifiers/<name>/config.yaml`. State ≠ config; keep
      them apart. Injection path settled the same day: virtual round
      at the most-recent end of the history layer.
- [x] ~~Modifier on/off + UI~~ — 2026-04-25: **`config.yaml` is the
      single source of truth**; the dashboard only displays runtime
      status and provides a config editor. Per-Modifier detailed
      configuration lives in each Modifier's own folder
      (`workspace/modifiers/<name>/config.yaml`, following the
      existing plugin pattern). Editing config requires a restart.
      **Implementation landed 2026-04-25**: `config.modifiers:
      list[str] | None`. List elements are names from
      `BUILTIN_FACTORIES`; registration follows list order (= chain
      execution order). `None` (field absent) → backward-compat,
      registers the legacy defaults plus a loud deprecation
      warning. `[]` → explicit zero plugins, executed silently.
      Unknown name → log + skip without blocking startup (strictly
      additive principle).
- [x] ~~Multiple Modifiers of the same kind coexisting?~~ — 2026-04-25:
      **allowed, executed in order**. There is one modifiers list in
      `config.yaml`; registration order = execution order. Modifiers
      of the same kind are chained: each one's output (anchor list,
      translation result, etc.) is the input of the next. Later
      Modifiers may augment / overwrite / veto the earlier output;
      composition is expressed via chaining, not via a complex
      "parallel + merge" semantics.
- [x] ~~Where does `statistics` move to?~~ Landed 2026-04-25:
      `sleep_cycles` becomes a `Runtime._sleep_cycles` in-memory
      counter; the other fields are removed (commit `ce59ab4`,
      self-model slim refactor).
- [x] ~~Should the recall LLM output schema include a `reason`
      field?~~ Decided 2026-04-25: no. **Pure JSON output, not even a
      `<thinking>` block** — let users pick their own model; weak-
      model behavior is the user's choice and is not crutched in the
      core mechanism (consistent with Modifier #1's default-off
      philosophy). See Part 2.

---

## Appendix — bugs already fixed in sync

- **Stimulus truncation bug** (2026-04-25): `_summarize_stimuli` was
  truncating each stimulus's content to 60 characters before writing
  it into the sliding window. As a result, when Self looked back at
  user messages it had already received, only the prefix remained in
  `[HISTORY]`; downstream mechanisms (recall feature extraction,
  compact summarization, bootstrap-instruction detection) were all
  affected. The truncation was removed as part of this bug fix.
