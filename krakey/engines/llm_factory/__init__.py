"""``llm_factory`` Engine — long-lived factory for LLM clients.

Default impl ``DefaultLLMClientFactoryEngine`` lives in ``default.py``;
the ``LLMClientFactoryEngine`` Protocol lives in
``krakey.interfaces.engines.llm_factory``.

The Engine is the only place that touches ``cfg.llm`` — every other
Engine and every plugin asks the factory for a client by tag name or
core-purpose name, never reads providers / api keys / models
themselves. This is the API-key isolation boundary.
"""
from krakey.engines.llm_factory.default import DefaultLLMClientFactoryEngine

__all__ = ["DefaultLLMClientFactoryEngine"]
