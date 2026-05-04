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

## Currently shipping slots (Phase 1)

| Slot | Protocol | Default | Construction kwargs |
|------|----------|---------|---------------------|
| `embedder` | [`AsyncEmbedder`](../krakey/llm/resolve.py) | tag-resolved LLMClient wrapper | _(none)_ |
| `reranker` | [`Reranker`](../krakey/memory/recall/scoring.py) | tag-resolved LLMClient wrapper | _(none)_ |
| `prompt_builder` | [`PromptBuilderLike`](../krakey/interfaces/services/prompt_builder.py) | `PromptBuilder` | _(none)_ |

All three Phase 1 slots take **no kwargs at construction**. Your class
must have an `__init__` that accepts no positional or keyword args
(beyond `self`). If you need configuration, read it from environment
variables or a separate config file your package maintains.

> **Future phases** will add slots for `memory`, `kb_registry`,
> `sliding_window`, `sleep_manager`, and `llm_client_factory`. The
> config dataclass already reserves these field names so your config
> won't break when the wiring lands; today they're silently ignored.

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
