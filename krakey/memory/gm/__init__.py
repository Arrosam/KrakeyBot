"""GraphMemory implementation halves — storage + query mixin.

The public API class lives in ``krakey/memory/graph_memory.py``
(``GraphMemory``) and is built by combining ``GMStorage`` +
``GMQueryMixin`` from this package + the LLM-driven write facades
that exist on the GraphMemory class itself. Keeping the package
separate from the facade module gives each half its own focused file
without disturbing the established ``from krakey.memory.graph_memory
import GraphMemory`` import path.
"""
