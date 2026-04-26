"""KrakeyBot process entry point.

The Runtime class itself lives in ``src/runtime/runtime.py`` — this
file now only owns:

  * ``build_runtime_from_config`` — parse config.yaml, resolve every
    tag-bound LLM, hand the assembled deps to ``Runtime``.
  * ``resolve_llm_for_tag`` — the shared cache-aware tag → LLMClient
    resolver used by both core-purpose loading here and per-plugin
    purpose loading inside Runtime.
  * The ``__main__`` block.

Re-exports ``Runtime`` and ``RuntimeDeps`` from runtime.py so existing
``from src.main import Runtime`` call sites in tests keep working
without churn.
"""
from __future__ import annotations

import asyncio
import sys

from src.llm.client import LLMClient
from src.models.config import load_config
# Public re-exports — callers used to do ``from src.main import Runtime``
# and we don't want to chase down every test importer just because the
# class lives in a tidier file now.
from src.runtime.runtime import (  # noqa: F401
    AsyncEmbedder, ChatLike, Runtime, RuntimeDeps,
    _summarize_stimuli, resolve_llm_for_tag,
)


def build_runtime_from_config(config_path: str = "config.yaml") -> Runtime:
    """Construct a production Runtime from ``config.yaml``.

    Tag-based resolution path (Samuel 2026-04-26 refactor):
      1. ``cfg.llm.tags`` → name → (provider, model, params)
      2. ``cfg.llm.core_purposes`` → core purpose name → tag name
      3. ``cfg.llm.embedding`` / ``cfg.llm.reranker`` → tag name
         (model-type slots, not "purposes")

    Each LLMClient is built once per tag name and reused across all
    purposes that share it.
    """
    cfg = load_config(config_path)

    # Build LLMClient cache keyed by tag name. Multiple purposes that
    # point at the same tag share a single client. The cache is also
    # passed through to plugin loaders via RuntimeDeps so plugin
    # purposes resolve from the same cache.
    client_cache: dict[str, LLMClient] = {}

    def _client_for_tag(tag_name: str | None) -> LLMClient | None:
        return resolve_llm_for_tag(cfg, tag_name, client_cache)

    # Core purposes (Self / compact / classifier ...). Self is required
    # for the heartbeat to fire; if missing, we DON'T raise — the
    # Runtime drops into "setup mode" (dashboard runs, heartbeat
    # skipped) so the user can complete the config via Web UI without
    # having to hand-edit YAML before they ever see Krakey's UI.
    self_llm = _client_for_tag(cfg.llm.core_purposes.get("self_thinking"))
    compact_llm = _client_for_tag(cfg.llm.core_purposes.get("compact"))
    classify_llm = (
        _client_for_tag(cfg.llm.core_purposes.get("classifier"))
        or compact_llm  # historical reuse
    )
    if compact_llm is None:
        compact_llm = self_llm  # last-resort fallback so sleep doesn't crash
        # (in setup mode self_llm is also None — that's fine, sleep
        # never runs without a heartbeat)

    # Embedding + reranker (model-type slots, not purposes)
    embed_client = _client_for_tag(cfg.llm.embedding)

    async def embedder(text: str) -> list[float]:
        if embed_client is None:
            raise RuntimeError(
                "no embedding tag bound — set llm.embedding to a tag name "
                "in config.yaml (or use the dashboard's LLM section)"
            )
        return await embed_client.embed(text)

    reranker = None
    reranker_client = _client_for_tag(cfg.llm.reranker)
    if reranker_client is not None:
        class _RerankerAdapter:
            async def rerank(self, query, docs):
                return await reranker_client.rerank(query, docs)
        reranker = _RerankerAdapter()

    # hypo_llm is no longer eagerly required at the core level — it's
    # bound through the per-plugin config of `default_hypothalamus`.
    # We still keep the field on RuntimeDeps for back-compat with
    # existing plugin factories that pull `deps.hypo_llm`. Resolve
    # from the dedicated `hypothalamus` core purpose if the user
    # mapped one (compat shim), else None.
    hypo_llm = _client_for_tag(cfg.llm.core_purposes.get("hypothalamus"))

    deps = RuntimeDeps(
        config=cfg, self_llm=self_llm, hypo_llm=hypo_llm,
        compact_llm=compact_llm,
        classify_llm=classify_llm, embedder=embedder, reranker=reranker,
        config_path=str(config_path),
        llm_clients_by_tag=client_cache,  # shared with plugin loader
    )
    return Runtime(deps)


async def _amain() -> None:
    runtime = build_runtime_from_config()
    try:
        await runtime.run()
    finally:
        await runtime.close()


if __name__ == "__main__":
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        pass
