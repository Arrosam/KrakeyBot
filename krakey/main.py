"""KrakeyBot process entry point.

The Runtime class itself lives in ``krakey/runtime/runtime.py`` — this
file now only owns:

  * ``build_runtime_from_config`` — parse config.yaml, resolve every
    tag-bound LLM, hand the assembled deps to ``Runtime``.
  * ``resolve_llm_for_tag`` — the shared cache-aware tag → LLMClient
    resolver used by both core-purpose loading here and per-plugin
    purpose loading inside Runtime.
  * The ``__main__`` block.

Re-exports ``Runtime`` and ``RuntimeDeps`` from runtime.py so existing
``from krakey.main import Runtime`` call sites in tests keep working
without churn.
"""
from __future__ import annotations

import asyncio
import sys

from krakey.engines.registry import EngineRegistry
from krakey.interfaces.engines import LLMClientFactoryEngine
from krakey.models.config import load_config
# Public re-exports — callers used to do ``from krakey.main import Runtime``
# and we don't want to chase down every test importer just because the
# class lives in a tidier file now.
from krakey.llm.resolve import (  # noqa: F401
    AsyncEmbedder, ChatLike, resolve_llm_for_tag,
)
from krakey.runtime.runtime import Runtime, RuntimeDeps  # noqa: F401
from krakey.runtime.service_resolver import ServiceResolver
from krakey.runtime.heartbeat.heartbeat_orchestrator import (  # noqa: F401
    _summarize_stimuli,
)


def build_runtime_from_config(config_path: str = "config.yaml") -> Runtime:
    """Construct a production Runtime from ``config.yaml``.

    LLM resolution goes through the ``llm_factory`` Engine slot — the
    composition root never reads ``cfg.llm`` directly. The default
    ``DefaultLLMClientFactoryEngine`` wraps the legacy
    ``resolve_llm_for_tag`` util (so the older ``llm_client_factory``
    slot that substitutes the ``LLMClient`` class per-tag still works);
    users override the whole factory by setting
    ``cfg.core_implementations.llm_factory`` to a dotted path of their
    own ``LLMClientFactoryEngine`` impl.

    Tag-based resolution layers:
      1. ``cfg.llm.tags`` → name → (provider, model, params)
      2. ``cfg.llm.core_purposes`` → core purpose name → tag name
      3. ``cfg.llm.embedding`` / ``cfg.llm.reranker`` → tag name
         (model-type slots, not "purposes")

    Each LLMClient is built once per tag and reused across every
    purpose that shares it (cache lives inside the factory Engine,
    mirrored into ``RuntimeDeps.llm_clients_by_tag`` for plugin
    back-compat during the migration window).
    """
    cfg = load_config(config_path)

    # ---- Engine layer: factory Engine drives all LLM resolution.
    # Hides ``cfg.llm`` from every other Engine + plugin.
    registry = EngineRegistry(cfg)
    llm_factory: LLMClientFactoryEngine = registry.resolve(
        "llm_factory",
        default_path=(
            "krakey.engines.llm_factory.default:"
            "DefaultLLMClientFactoryEngine"
        ),
        expected_protocol=LLMClientFactoryEngine,
        cfg=cfg,
    )

    # Core purposes (Self / compact / classifier ...). Self is what
    # makes the heartbeat fire; if it isn't bound the runtime starts
    # in **pause mode** — channels run (so the dashboard is reachable
    # for the user to fix providers), but the heartbeat doesn't tick
    # until the user restarts after configuring. Surfacing this as a
    # crash on startup (the old behavior) was hostile to first-run
    # users who just installed the package and skipped chat config.
    self_llm = llm_factory.client_for_core_purpose("self_thinking")
    compact_llm = llm_factory.client_for_core_purpose("compact")
    classify_llm = (
        llm_factory.client_for_core_purpose("classifier")
        or compact_llm  # historical reuse
    )
    if compact_llm is None:
        # Sleep + classify still need *something* callable; reuse Self
        # if available, else leave None — the pause-mode runtime never
        # invokes either path.
        compact_llm = self_llm

    # Embedding + reranker (model-type slots, not purposes)
    embed_client = llm_factory.embed_client()

    # Default embedder wraps the tag-resolved embedding client. Wrapped
    # in a class (rather than a closure) so it satisfies the
    # ``AsyncEmbedder`` runtime-checkable Protocol and can be replaced
    # 1:1 by a user override via ``core_implementations.embedder``.
    class _DefaultEmbedder:
        async def __call__(self, text: str) -> list[float]:
            if embed_client is None:
                raise RuntimeError(
                    "no embedding tag bound — set llm.embedding to a tag "
                    "name in config.yaml (or use the dashboard's LLM "
                    "section)"
                )
            return await embed_client.embed(text)

    service_resolver = ServiceResolver(cfg)
    embedder = service_resolver.resolve(
        "embedder",
        default_factory=_DefaultEmbedder,
        expected_protocol=AsyncEmbedder,
    )

    reranker_client = llm_factory.rerank_client()

    class _DefaultReranker:
        """Adapts the tag-resolved LLMClient to the Reranker Protocol."""
        async def rerank(self, query, docs):
            return await reranker_client.rerank(query, docs)

    # Three states for the reranker slot:
    #   * user override set        → resolver builds user impl
    #   * no override, tag bound   → resolver builds _DefaultReranker
    #   * no override, no tag      → reranker = None (recall paths fall
    #                                 back to scripted scoring; callers
    #                                 already handle ``reranker is None``)
    from krakey.memory.recall import Reranker
    if cfg.core_implementations.reranker or reranker_client is not None:
        reranker = service_resolver.resolve(
            "reranker",
            default_factory=_DefaultReranker,
            expected_protocol=Reranker,
        )
    else:
        reranker = None

    # hypo_llm is no longer eagerly required at the core level — it's
    # bound through the per-plugin config of `hypothalamus`.
    # We still keep the field on RuntimeDeps for back-compat with
    # existing plugin factories that pull `deps.hypo_llm`. Resolve
    # from the dedicated `hypothalamus` core purpose if the user
    # mapped one (compat shim), else None.
    hypo_llm = llm_factory.client_for_core_purpose("hypothalamus")

    # InstallService — composition-root choice. We inject the
    # DefaultInstallService (krakey.install.service) so the
    # InstallTool + the startup advisory have a working
    # implementation. Runtime + dashboard depend on the
    # InstallService Protocol, never on this concrete class.
    from krakey.install import DefaultInstallService
    install_service = DefaultInstallService()

    deps = RuntimeDeps(
        config=cfg, self_llm=self_llm, hypo_llm=hypo_llm,
        compact_llm=compact_llm,
        classify_llm=classify_llm, embedder=embedder, reranker=reranker,
        config_path=str(config_path),
        # Mirror the factory's internal cache onto deps for plugin
        # back-compat: ``PluginContext.get_llm_for_tag`` still reads
        # from ``deps.llm_clients_by_tag`` until plugins migrate to
        # use ``ctx.services["llm_factory"]`` directly.
        llm_clients_by_tag=llm_factory.client_cache,
        install_service=install_service,
        # Pull the sliding-window state path straight from
        # config.yaml so dashboard edits to ``sliding_window.state_path``
        # take effect on the next restart. Empty string in YAML →
        # in-memory only (RuntimeDeps respects "" as the opt-out
        # sentinel).
        sliding_window_state_path=cfg.sliding_window.state_path,
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
    except FileNotFoundError as e:
        # Missing config.yaml — load_config raises this with a hint
        # pointing at the onboarding wizard. Print just the message
        # (not a traceback) and exit non-zero.
        print(str(e), file=sys.stderr)
        sys.exit(2)
