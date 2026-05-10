"""``EmbedderEngine`` — text → vector callable.

Default impl ``TagBoundEmbedderEngine`` walks
``cfg.llm.embedding`` → tag → LLMClient and calls its ``embed(text)``.
The Engine layer hides cfg details from consumers: callers (memory
ingestion, recall, sleep clustering) only see "give me a vector for
this string".

A user replacing the Engine wires up whatever embedding source they
want (local ONNX model, HuggingFace transformers, batch service,
random vectors for tests, etc.). The default impl uses the
``LLMClientFactoryEngine`` to fetch a client; a custom Engine can
ignore the factory entirely and produce vectors locally.

**Caveat on Protocol**: ``@runtime_checkable`` only verifies the
``__call__`` attribute exists, not its signature. Documentation
discipline beats type-system enforcement here — any callable
returning ``list[float]`` will pass isinstance, but only one with
the correct signature works in practice.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbedderEngine(Protocol):
    """Async-callable returning an embedding vector for one text."""

    async def __call__(self, text: str) -> list[float]: ...
