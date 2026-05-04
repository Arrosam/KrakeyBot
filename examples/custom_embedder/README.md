# Custom embedder example — `HashEmbedder`

Minimal end-to-end example showing how to plug a user-supplied embedder
into KrakeyBot via the `core_implementations.embedder` slot.

`HashEmbedder` (in [`hash_embedder.py`](hash_embedder.py)) is a
**deterministic SHA-256 based fake**. It produces fixed-dimension
vectors that are stable across runs but carry NO semantic meaning —
two similar sentences get totally different vectors. Useful for:

- Local development without burning API quota on every startup
- CI tests that exercise the embedding path without external services
- Scaffolding a new custom embedder (start by copying this, then swap
  the `__call__` body for your actual provider call)

**Do not ship this in production.** The vectors are noise, recall
quality will be terrible.

---

## Try it

1. Copy `hash_embedder.py` into your project (or anywhere on `PYTHONPATH`):
   ```bash
   cp hash_embedder.py /your/project/
   ```
2. Add the override to `config.yaml`:
   ```yaml
   core_implementations:
     embedder: "hash_embedder:HashEmbedder"
   ```
3. Run KrakeyBot:
   ```bash
   krakey run
   ```

KrakeyBot will:
1. Import `hash_embedder`
2. Instantiate `HashEmbedder()`
3. Verify it satisfies `AsyncEmbedder` (it does — has an async `__call__`)
4. Use it for every embedding call during the heartbeat

If anything goes wrong (bad path, missing method, ...) you'll see a
loud error at startup, not partway through the first beat.

---

## Anatomy

The whole class is ~15 lines:

```python
class HashEmbedder:
    DIM = 1024

    async def __call__(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        floats = [(b - 127.5) / 127.5 for b in h]
        return (floats * (self.DIM // len(floats)))[: self.DIM]
```

What KrakeyBot requires:
- An `__init__` taking no kwargs (the default constructor here)
- An `async def __call__(self, text: str) -> list[float]`
- That's it.

No KrakeyBot imports, no inheritance, no decorators. Pure duck typing
plus a `runtime_checkable` Protocol check at startup.

See [`docs/extending-core.md`](../../docs/extending-core.md) for the
full slot system, including `prompt_builder` and `reranker`.
