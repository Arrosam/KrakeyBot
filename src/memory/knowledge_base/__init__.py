"""Long-term per-topic stores (DevSpec §8).

Two distinct concerns:

  * ``KnowledgeBase`` — one .sqlite file, one topic cluster's entries
    (``entry_store.py``).
  * ``KBRegistry``    — fleet manager talking to GM's ``kb_registry``
    table (``registry.py``).

Re-exports both at the package root so ``from src.memory.knowledge_base
import KBRegistry, KnowledgeBase`` keeps working unchanged.
"""
from src.memory.knowledge_base.entry_store import (  # noqa: F401
    AsyncEmbedder,
    KnowledgeBase,
)
from src.memory.knowledge_base.registry import KBRegistry  # noqa: F401
