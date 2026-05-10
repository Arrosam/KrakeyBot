"""KrakeyBot process entry point.

The Runtime class itself lives in ``krakey/runtime/runtime.py`` — this
file owns:

  * ``build_runtime_from_config`` — parse config.yaml, resolve the
    LLMClientFactoryEngine + Embedder + Reranker through
    EngineRegistry, hand the assembled deps to ``Runtime``. Other
    Engines (memory, context, decision, recall, dispatch, heartbeat,
    explicit_history) resolve inside ``Runtime.__init__``.
  * The ``__main__`` block.

Re-exports ``Runtime`` / ``RuntimeDeps`` / ``AsyncEmbedder`` /
``ChatLike`` / ``resolve_llm_for_tag`` so existing
``from krakey.main import …`` call sites in tests keep working
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
from krakey.engines.heartbeat.orchestrator import (  # noqa: F401
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

    # ---- Embedder Engine ---------------------------------------------
    # Default impl takes the factory Engine via constructor + uses it
    # to lazily reach the configured embedding client. Shared factory
    # injection means every engine that holds an LLM client points at
    # the same per-tag cache — no duplicate clients for the same tag
    # across engines.
    from krakey.interfaces.engines import EmbedderEngine

    embedder = registry.resolve(
        "embedder",
        default_path=(
            "krakey.engines.embedder.default:TagBoundEmbedderEngine"
        ),
        expected_protocol=EmbedderEngine,
        factory=llm_factory,
    )

    deps = RuntimeDeps(
        config=cfg, self_llm=self_llm,
        compact_llm=compact_llm,
        classify_llm=classify_llm, embedder=embedder,
        config_path=str(config_path),
        # Mirror the factory's internal cache onto deps so
        # ``PluginContext.get_llm_for_tag`` (which reads
        # ``deps.llm_clients_by_tag``) sees the same client
        # instances Engine consumers do. Both paths share the same
        # dict — no duplicate clients per tag.
        llm_clients_by_tag=llm_factory.client_cache,
        llm_factory=llm_factory,
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
