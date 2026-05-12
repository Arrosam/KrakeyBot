"""Structural ``__call__``/``chat`` Protocols the runtime uses to talk
about LLMs and embedders generically — no inheritance required.

Built-in clients (``LLMClient``, embedding wrappers) and test doubles
(``ScriptedLLM``, ``NullEmbedder``) both satisfy these structurally.
The Engine impl modules type-annotate against these so they stay
agnostic of which concrete class the user wired in via the
``llm_client_factory`` slot.

Lives at ``krakey.interfaces.duck`` because they're cross-cutting
typing primitives — every Engine slot that consumes an LLM client
references one of them. Putting them under ``krakey/llm/`` (as they
used to live) made ``krakey/llm/`` look like a public dependency
target when in fact only the Protocol contracts are public — the
concrete ``LLMClient`` lives privately under
``engines/llm_client_factory/``.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ChatLike(Protocol):
    """Minimal chat surface — what every consumer of an LLM Engine
    client needs. ``messages`` follows the OpenAI ``[{"role", "content"}]``
    shape; ``**kwargs`` accommodates per-call overrides like
    ``max_tokens`` / ``temperature`` that some impls accept.
    """
    async def chat(self, messages, **kwargs) -> str: ...


@runtime_checkable
class AsyncEmbedder(Protocol):
    """Async-callable returning an embedding vector for one text.

    ``@runtime_checkable`` so ``EngineRegistry`` can isinstance-check
    user-supplied embedder slots at startup. Caveat: Python's
    runtime-checkable Protocol with ``__call__`` only verifies the
    method exists — it can't check the signature, so any callable
    technically passes. Documentation discipline beats type-system
    enforcement here.
    """
    async def __call__(self, text: str) -> list[float]: ...
