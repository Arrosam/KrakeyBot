"""``EngineRegistry`` — turn ``cfg.core_implementations.<slot>`` short
names (or built-in defaults, or dotted paths) into concrete Engine
instances.

Resolution order for ``core_implementations.<slot>``:

  1. Empty string / missing → use the slot's default impl declared by
     ``engines/<slot>/meta.yaml`` (``default: true`` flag), falling
     back to ``engine_system.defaults.FALLBACK_ENGINES[slot]`` when
     the meta is missing or malformed.
  2. Value contains ``:`` → treat as ``module.path:ClassName`` dotted
     path (power-user fallback for impls that aren't catalogued).
  3. Otherwise → look up the short name in
     ``engines/<slot>/meta.yaml``'s ``builtin_engines`` list, then in
     the plugin-engine catalog (plugins under ``workspace/plugins/``
     declaring ``kind: engine`` + ``slot: <slot>`` in their
     ``meta.yaml``).

Failure modes (all loud — DIP says fail-fast at startup beats failing
30 minutes into a session with a confusing AttributeError):

  * unknown short name → ``ValueError`` listing the slot's catalog
  * malformed dotted path → ``ValueError``
  * import fails → ``ImportError`` annotated with the path/short name
  * instantiation TypeError on kwargs mismatch → ``TypeError`` annotated
    with the slot name + the kwargs the caller supplied
  * resulting object doesn't satisfy the Protocol → ``TypeError``
    listing the missing attributes
"""
from __future__ import annotations

import importlib
from typing import Any, Callable, TypeVar

from krakey.engine_system.catalog import EngineImpl
from krakey.models.config import Config

T = TypeVar("T")
Importer = Callable[[str], Any]


def _default_importer(path: str) -> Any:
    """Resolve ``module.path:ClassName`` to the class object.

    Pure: never instantiates the class — the registry does that with
    runtime-supplied kwargs after the import. Splitting "import" from
    "instantiate" lets tests stub one without the other.
    """
    module_path, _, attr = path.partition(":")
    if not attr:
        raise ValueError(
            f"engine override path must be 'module.path:ClassName' "
            f"(entry-point style), got {path!r}"
        )
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"engines: cannot import module {module_path!r} "
            f"from override path {path!r}: {e}"
        ) from e
    if not hasattr(mod, attr):
        raise ImportError(
            f"engines: module {module_path!r} has no attribute "
            f"{attr!r} (referenced by override path {path!r})"
        )
    return getattr(mod, attr)


def _missing_protocol_attrs(instance: Any, protocol: type) -> list[str]:
    """List Protocol attributes the instance lacks.

    ``isinstance(obj, RuntimeCheckableProto)`` only tells you "no" —
    not WHICH methods are missing. This helper enumerates the public
    callable attributes of the Protocol and reports those absent on
    the instance, so the registry can surface a useful error.
    """
    expected = {
        a for a in dir(protocol)
        if not a.startswith("_") and callable(getattr(protocol, a, None))
    }
    actual = set(dir(instance))
    return sorted(expected - actual)


def _filter_kwargs(cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs the class's ``__init__`` doesn't accept.

    The runtime passes cross-cutting deps (cfg, factory, memory) through
    ``resolve`` kwargs so default impls can pick what they need. A user
    override that takes only a subset (or no kwargs at all) shouldn't
    have to declare every field the runtime threads through. Inspect
    the constructor and drop kwargs it can't accept — unless it has
    ``**kwargs``, in which case we pass everything through.
    """
    import inspect
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        return kwargs
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    if has_var_keyword:
        return kwargs
    accepted = {
        name for name, p in sig.parameters.items()
        if p.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    }
    return {k: v for k, v in kwargs.items() if k in accepted}


def _load_slot_catalog(slot: str) -> tuple[dict[str, EngineImpl], str]:
    """Return the ``({short_name: EngineImpl}, default_short_name)``
    pair for ``slot``. Three-tier resolution:

      1. **meta.yaml** — ``engines/<slot>/meta.yaml`` declares the
         catalog. Authoritative when present and well-formed.
      2. **defaults.py** — ``FALLBACK_ENGINES[slot]`` dotted-path is
         used when meta.yaml is missing OR malformed. We synthesise a
         single-entry catalog around it. Loud-warns to stderr so the
         operator knows fallback fired.
      3. **loud ImportError** — neither layer has anything for this
         slot. Means the slot is genuinely unknown; the user typo'd
         a slot name in cfg or somebody renamed a folder without
         updating defaults.py.

    The registry never imports ``krakey.engines.<X>`` directly — meta
    parsing returns ``EngineImpl`` objects carrying their own
    dotted-path factory references, and the fallback path uses
    ``_default_importer`` on the FALLBACK_ENGINES string.
    """
    import sys

    from krakey.engine_system.defaults import FALLBACK_ENGINES
    from krakey.engine_system.meta_loader import (
        MetaParseError, load_slot_meta,
    )

    try:
        return load_slot_meta(slot)
    except FileNotFoundError:
        # No meta.yaml — silently fall through to defaults. Common
        # during partial migrations or third-party engines that
        # haven't authored a meta yet.
        pass
    except MetaParseError as e:
        # meta.yaml is present but malformed. Warn loudly + try
        # fallback; the operator wants to know their meta is broken,
        # but a typo shouldn't take the runtime down.
        print(
            f"warning: engine slot {slot!r}: meta.yaml malformed "
            f"({e}); falling back to engine_system.defaults",
            file=sys.stderr,
        )

    fallback_path = FALLBACK_ENGINES.get(slot)
    if fallback_path is None:
        raise ImportError(
            f"engine slot {slot!r}: no meta.yaml found at "
            f"engines/{slot}/meta.yaml AND no entry in "
            "engine_system.defaults.FALLBACK_ENGINES. Either add the "
            "meta.yaml or register a fallback dotted path."
        )

    # Synthesize a one-entry catalog around the dotted path. The
    # ``LazyImpl`` from meta_loader does the same trick — defer
    # ``importlib.import_module`` to actual ``cls()`` time.
    from krakey.engine_system.meta_loader import _LazyImpl
    module, _, attr = fallback_path.partition(":")
    fallback_name = "fallback"
    return (
        {
            fallback_name: EngineImpl(
                cls=_LazyImpl(module, attr),  # type: ignore[arg-type]
                description=(
                    f"emergency fallback for slot {slot!r} "
                    f"({fallback_path})"
                ),
            ),
        },
        fallback_name,
    )


def _load_plugin_engine_catalog() -> dict[str, dict[str, dict[str, Any]]]:
    """Walk every plugin's ``meta.yaml`` for ``kind: engine`` components.

    Returns ``{slot: {plugin_name: {path, description, config_schema}}}``.
    The plugin's top-level ``description`` + ``config_schema`` carry
    over to the engine entry — that's how plugin-supplied engines
    plug into the same dashboard surface as built-in ones (slot
    dropdown + per-engine config form). Scanning is pure-text
    (``parse_meta`` doesn't import plugin code) so this is cheap to
    call even when many plugins are installed.
    """
    from krakey.plugin_system.catalogue import (
        list_available_plugins,
    )
    catalog: dict[str, dict[str, dict[str, Any]]] = {}
    for plugin_name, meta in list_available_plugins().items():
        for comp in meta.components:
            if comp.kind != "engine":
                continue
            slot = getattr(comp, "slot", None)
            if not slot:
                continue
            catalog.setdefault(slot, {})[plugin_name] = {
                "path": f"{comp.factory_module}:{comp.factory_attr}",
                "description": meta.description.strip()
                              or "(plugin-supplied engine)",
                "config_schema": list(meta.config_schema or []),
            }
    return catalog


class EngineRegistry:
    """Resolves Engine slots to concrete instances.

    Constructed once per Runtime from a parsed ``Config``. Callers
    use ``resolve(slot, expected_protocol=..., **kwargs)`` one slot
    at a time. The user override is read from
    ``cfg.core_implementations.<slot>`` (short name OR dotted path);
    empty string falls back to the slot's catalog ``DEFAULT_ENGINE``.
    """

    def __init__(
        self,
        cfg: Config,
        *,
        importer: Importer | None = None,
    ):
        self._cfg = cfg
        self._import = importer or _default_importer
        self._plugin_engine_catalog: dict[str, dict[str, str]] | None = None

    def _plugin_catalog(self) -> dict[str, dict[str, dict[str, Any]]]:
        if self._plugin_engine_catalog is None:
            self._plugin_engine_catalog = _load_plugin_engine_catalog()
        return self._plugin_engine_catalog

    def _resolve_class(self, slot: str, name_or_path: str) -> type:
        """Map an override value to a concrete class.

        ``:`` in value → dotted path. Otherwise short name → walk the
        built-in catalog, then the plugin catalog. Misses raise with a
        list of available short names so the user can fix the typo.

        When the chosen catalog entry holds a ``_LazyImpl`` (the meta-
        loader's deferred-import wrapper), resolve it here — the
        registry's downstream ``_filter_kwargs`` needs an introspectable
        class object, and lazy was only valuable across the slot's
        unchosen alternatives. The chosen impl always materialises.
        """
        from krakey.engine_system.meta_loader import _LazyImpl

        if ":" in name_or_path:
            return self._import(name_or_path)
        builtins, _ = _load_slot_catalog(slot)
        if name_or_path in builtins:
            cls = builtins[name_or_path].cls
            if isinstance(cls, _LazyImpl):
                cls = cls._resolve()
            return cls
        plugin_entries = self._plugin_catalog().get(slot, {})
        if name_or_path in plugin_entries:
            return self._import(plugin_entries[name_or_path]["path"])
        available = sorted(builtins) + sorted(plugin_entries)
        raise ValueError(
            f"engine slot {slot!r}: unknown impl name "
            f"{name_or_path!r}. Available: {available!r}. Use "
            "'module.path:ClassName' for an impl that isn't "
            "catalogued."
        )

    def _engine_config(self, slot: str, short_name: str) -> dict[str, Any]:
        """Return the user's persisted config dict for the given
        ``(slot, short_name)`` pair, or an empty dict when nothing is
        configured. Engines that don't take a ``config`` kwarg ignore
        whatever this returns via ``_filter_kwargs``."""
        slot_cfg = self._cfg.engine_configs.get(slot, {}) if hasattr(
            self._cfg, "engine_configs",
        ) else {}
        return dict(slot_cfg.get(short_name, {}))

    def resolve(
        self,
        slot: str,
        *,
        expected_protocol: type,
        **kwargs: Any,
    ) -> Any:
        """Return the configured impl, instantiated.

        ``kwargs`` are forwarded to the constructor — ``_filter_kwargs``
        drops ones the impl's ``__init__`` doesn't accept so user
        overrides with narrower signatures still work. The user's
        per-engine config dict (from ``cfg.engine_configs.<slot>.
        <short_name>``) is added as a ``config`` kwarg automatically;
        impls that don't declare it ignore it.
        """
        override = self._cfg.core_implementations.get(slot)
        if override:
            name_or_path = override
        else:
            _, default = _load_slot_catalog(slot)
            name_or_path = default

        cls = self._resolve_class(slot, name_or_path)
        # When the user picked a short name (not a dotted path),
        # surface their per-engine config as ``config=`` so the impl
        # can read its tunables. Dotted-path overrides bypass the
        # catalog; they don't have a registered short_name to key the
        # config dict by, so they get an empty dict.
        if ":" not in name_or_path and "config" not in kwargs:
            kwargs = dict(kwargs)
            kwargs["config"] = self._engine_config(slot, name_or_path)
        accepted_kwargs = _filter_kwargs(cls, kwargs)
        try:
            instance = cls(**accepted_kwargs)
        except TypeError as e:
            raise TypeError(
                f"engine slot {slot!r} = {name_or_path!r} could not be "
                f"instantiated with kwargs {sorted(accepted_kwargs)}: {e}. "
                f"Custom engines for slot {slot!r} must accept the "
                "same kwargs as the built-in default."
            ) from e

        if not isinstance(instance, expected_protocol):
            missing = _missing_protocol_attrs(instance, expected_protocol)
            if missing:
                detail = f"missing attributes: {missing}"
            else:
                detail = (
                    "(Protocol may have non-method requirements that "
                    "isinstance cannot verify)"
                )
            raise TypeError(
                f"engine slot {slot!r} = {name_or_path!r} does not satisfy "
                f"{expected_protocol.__name__}; {detail}"
            )
        return instance
