"""``llm_client_factory`` slot — per-tag LLMClient class substitution.

The slot's catalog of impls lives in ``meta.yaml`` next to this file.
This slot is resolved per-tag by
``krakey.engines.llm_factory._resolve.resolve_llm_for_tag`` rather
than once at startup — that's why it doesn't show up on Runtime as
``self.<slot>``.
"""
from krakey.engines.llm_client_factory._client import LLMClient

__all__ = ["LLMClient"]
