# Extending core: replacing built-in services with your own

KrakeyBot has two axes of pluggability:

1. **Additive plugins** (Tools, Channels, Modifiers) — you opt in via
   `config.plugins:`. Anything in the runtime should still work even if
   you disable every plugin (DevSpec invariant).
2. **Replaceable core services** — built-in things the runtime
   _depends_ on (memory, prompt builder, embedder, ...). You can keep
   the defaults, or swap any of them with your own implementation. This
   doc covers axis #2.

> The "additive plugin" path is described separately in
> [`PLUGINS.md`](../PLUGINS.md). Use a plugin if you're adding a new
> capability to Krakey. Use this slot mechanism if you're replacing a
> built-in service with your own backend.

---

## How a slot works

In your `config.yaml`:

```yaml
core_implementations:
  embedder: "my_pkg.embedders:OpenAICompatibleEmbedder"
  prompt_builder: "my_pkg.prompts:CustomBuilder"
  reranker: "my_pkg.rerankers:CohereReranker"
```

Each slot's value is a dotted path in entry-point style
(`module.path:ClassName`). At startup, KrakeyBot:

1. Imports `module.path`.
2. Reads the `ClassName` attribute.
3. Instantiates it with the runtime-supplied kwargs for that slot
   (most slots take no kwargs — see per-slot contracts below).
4. Verifies the instance satisfies the slot's Protocol via
   `isinstance()`.
5. If any of those steps fail, prints an annotated error and exits
   non-zero — fail-fast at startup, not 30 minutes into a session.

If a slot is missing or empty, KrakeyBot uses its built-in default.
You can override one slot, all slots, or none — they're independent.

---

## Currently shipping slots

| Slot | Protocol | Default | Construction kwargs |
|------|----------|---------|---------------------|
| `embedder` | [`AsyncEmbedder`](../krakey/interfaces/duck.py) | tag-resolved LLMClient wrapper | _(none)_ |
| `reranker` | [`RerankerEngine`](../krakey/interfaces/engines/reranker.py) | tag-resolved LLMClient wrapper | _(none)_ |
| `prompt_builder` | [`PromptBuilderLike`](../krakey/interfaces/services/prompt_builder.py) | `PromptBuilder` | _(none)_ |
| `llm_client_factory` | [`ChatLike`](../krakey/interfaces/duck.py) | `LLMClient` | `provider`, `model`, `params` |
| `memory` | [`MemoryService`](../krakey/interfaces/services/memory.py) | `GraphMemory` | `db_path`, `embedder`, `auto_ingest_threshold`, `extractor_llm`, `classifier_llm` |
| `kb_registry` | [`KBRegistryService`](../krakey/interfaces/services/memory.py) | `KBRegistry` | `gm`, `kb_dir`, `embedder` |

If you need configuration for a no-arg slot, read it from environment
variables or a separate config file your package maintains. Slots that
take kwargs receive them positionally from the runtime (the slot's
contract specifies which); your `__init__` must accept those exact
kwarg names.

> **Future phases** will add slots for `sliding_window` and
> `sleep_manager`. The config dataclass already reserves these field
> names so your config won't break when the wiring lands; today they're
> silently ignored.

---

## Writing a custom embedder

The minimum class is six lines:

```python
# my_pkg/embedders.py
class MyEmbedder:
    async def __call__(self, text: str) -> list[float]:
        # Call your embedding service here, return a list of floats.
        return await my_http_client.embed(text)
```

In `config.yaml`:

```yaml
core_implementations:
  embedder: "my_pkg.embedders:MyEmbedder"
```

That's it. See [`examples/custom_embedder/`](../examples/custom_embedder/)
for a runnable example.

### The Protocol contract

`AsyncEmbedder` is defined as:

```python
@runtime_checkable
class AsyncEmbedder(Protocol):
    async def __call__(self, text: str) -> list[float]: ...
```

Your class needs:
- `async def __call__(self, text: str) -> list[float]`
- An `__init__` that takes no kwargs

The vector dimension depends on your model — KrakeyBot doesn't enforce
a specific size, but be consistent: switching dimensions mid-session
will break the GraphMemory's vector-search index. Rebuild the index
(`workspace/data/graph_memory.sqlite`) after any dim change.

---

## Writing a custom prompt builder

`PromptBuilderLike` is the largest of the Phase 1 Protocols:

```python
@runtime_checkable
class PromptBuilderLike(Protocol):
    def build_default_elements(
        self,
        *,
        self_model: dict,
        capabilities: list[CapabilityView],
        status: StatusSnapshot,
        recall: RecallResult,
        window: list[SlidingWindowRound],
        stimuli: list[Stimulus],
        current_time: datetime | None = None,
    ) -> PromptElements: ...

    def render(self, elements: PromptElements) -> str: ...
```

Your `build_default_elements` must produce a fully-populated
`PromptElements` with all canonical layer keys (see
`DEFAULT_ELEMENT_KEYS` in [`krakey/prompt/builder.py`](../krakey/prompt/builder.py)).
Your `render` serializes it to a string.

### Layer-order warning

The default `PromptBuilder` orders layers for Anthropic prefix-cache
hit rate (most-stable prefix first, most-volatile last). If your
custom builder reorders layers it will work, but you'll pay the cost
of cache invalidation every beat. Document this trade-off if you ship
a custom builder.

---

## Writing a custom reranker

`Reranker`:

```python
@runtime_checkable
class Reranker(Protocol):
    async def rerank(self, query: str, docs: list[str]) -> list[float]: ...
```

Returns one float per doc, in the same order, where higher = better.
The recall pipeline normalizes these internally — you don't need to
return values in any specific range.

If you DON'T want a reranker (the recall pipeline falls back to
scripted scoring), just leave the slot empty AND don't bind the
`llm.reranker` tag. KrakeyBot will run with `reranker = None`.

---

## Writing a custom LLM client

`ChatLike` is the smallest of the Protocols:

```python
@runtime_checkable
class ChatLike(Protocol):
    async def chat(self, messages, **kwargs) -> str: ...
```

Your class needs:
- `__init__(self, *, provider: Provider, model: str, params: LLMParams | None = None)` —
  these kwargs come from the tag binding (the same data that built
  the default `LLMClient`)
- `async def chat(self, messages, **kwargs) -> str`

Use this slot when you want to replace the entire LLM transport — your
own HTTP library, streaming impl, vLLM-native client, locally-loaded
inference, or a request-routing layer. Use `llm.providers` + `llm.tags`
instead if you only want to point at a different endpoint.

### Composition with the embedder / reranker slots

When you override `llm_client_factory`, your class is used for **every**
tag — chat, embedding, and reranker. That means:

- For `llm.embedding: t` to work without overriding the embedder slot,
  your class also needs `async def embed(self, text: str) -> list[float]`.
- For `llm.reranker: t` to work without overriding the reranker slot,
  your class also needs `async def rerank(self, query, docs)`.

Easiest approach: implement `chat()` only on your custom client and use
the **embedder slot** to point at your dedicated embedding service. The
two slots compose cleanly — KrakeyBot will use your custom chat client
for chat tags and your custom embedder for embedding tags.

---

## Writing a custom memory backend

The `memory` and `kb_registry` slots together let you swap the entire
memory subsystem — graph memory + knowledge bases — for a different
backend (Postgres, Redis, Neo4j, ScyllaDB, an in-memory dict for
testing, ...).

The Protocols are in
[`krakey/interfaces/services/memory.py`](../krakey/interfaces/services/memory.py)
— `MemoryService` covers ~22 methods on `GraphMemory`,
`KBRegistryService` covers ~7 methods on `KBRegistry`, and a KB
instance returned by `open_kb` must satisfy `KnowledgeBaseLike`
(~9 methods).

For a minimal end-to-end example showing every method stubbed at
viable fidelity in pure Python dicts, see
[`tests/_fake_memory.py`](../tests/_fake_memory.py) — it's used as the
fixture in the e2e swap test and demonstrates that the Protocol
surface is implementable without any SQLite or external service. Treat
it as scaffolding to copy into your project, then replace each
method's body with calls to your real backend.

### Constructor kwargs

The `memory` slot's `__init__` receives:

```python
def __init__(
    self, *,
    db_path: str,                    # whatever DSN/path makes sense for you
    embedder,                        # AsyncEmbedder — call await embedder(text)
    auto_ingest_threshold: float,    # cosine cutoff for dedup
    extractor_llm,                   # ChatLike — for compact-style extraction
    classifier_llm,                  # ChatLike — for async category classification
):
    ...
```

The `kb_registry` slot's `__init__` receives:

```python
def __init__(
    self, *,
    gm,                              # MemoryService — your memory instance
    kb_dir: str,                     # base path / namespace for KBs
    embedder,
):
    ...
```

### LLM-driven facades — auto_ingest, explicit_write, classify_and_link_pending

Three of the `MemoryService` methods are LLM-driven:
`auto_ingest`, `explicit_write`, `classify_and_link_pending`. The
runtime calls these every heartbeat. Your implementation must produce
the same shape of return values (`{"node_id": int, ...}`,
`{"classified": int, "linked": int}`) but you decide internally
whether to actually invoke an LLM. A trivial impl that just records
the call and writes a degenerate node is fine for testing — see
`InMemoryMemoryService.auto_ingest` in `tests/_fake_memory.py`.

### Why you usually need to override BOTH memory and kb_registry

The default `KBRegistry` reaches into `GraphMemory`'s private aiosqlite
connection (`gm._require()`) to share the same DB file. That coupling
is intentional in the SQLite case — keeps a single transactional
backing — but it means a non-SQLite memory backend cannot pair with
the default KBRegistry. The runtime's startup will succeed (the
default `KBRegistry.__init__` is lazy), but the first KB operation
will fail with a `TypeError` or `AttributeError`.

If you override `memory`, you should override `kb_registry` too,
unless you happen to be writing a SQLite-compatible memory backend
that exposes the same `_require()` method.

---

## Failure modes

These all fire at runtime startup (not later):

| Symptom | Cause |
|---------|-------|
| `ValueError: ... entry-point style` | Missing `:ClassName` in the dotted path |
| `ImportError: cannot import module ...` | Module doesn't exist on `sys.path` |
| `ImportError: ... has no attribute ...` | Module exists but lacks the named class |
| `TypeError: ... could not be instantiated with kwargs ...` | Your `__init__` rejects the slot's kwargs |
| `TypeError: ... does not satisfy ...` | Instance is missing a Protocol method (the message names which) |

Loud-at-startup is intentional. A typo in your dotted path should fail
when you run `krakey run`, not when the heartbeat first asks for an
embedding.

---

## Distribution

The user code lives wherever Python's import system can find it:

- A package on PyPI (`pip install my-krakey-extras`)
- A directory in `PYTHONPATH`
- An editable install (`pip install -e ./my-extras/`)
- A module in your project root if you're running KrakeyBot directly

KrakeyBot doesn't impose a packaging style. The slot mechanism just
calls `importlib.import_module(...)` on the dotted path you supply.

---

## When NOT to use a slot

Don't override `embedder` if you just want to point at a different
embedding endpoint — that's what `llm.providers` + `llm.tags` are for.
Override the slot when you want to replace the entire transport (use
your own HTTP library, swap to a local model loaded with
`sentence-transformers`, route to a custom inference cluster, ...).

The same applies to `reranker`.

For `prompt_builder` there's no equivalent config-only path — overriding
the slot is the only way to customize prompt layout.
