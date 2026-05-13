"""``llm_factory`` Engine — long-lived factory for LLM clients.

The slot's catalog of impls lives in ``meta.yaml`` next to this file.
The LLMClientFactoryEngine Protocol lives at
``krakey.interfaces.engines.llm_factory``.
"""
from krakey.engines.llm_factory.default import DefaultLLMClientFactoryEngine

__all__ = ["DefaultLLMClientFactoryEngine"]
