"""``EngineRegistry`` — turn ``cfg.core_implementations.<slot>`` short
names (or built-in defaults, or dotted paths) into concrete Engine
instances.

Resolution order for ``core_implementations.<slot>``:

  1. Empty string / missing → use the slot's ``DEFAULT_ENGINE`` (each
     ``engines/<slot>/__init__.py`` declares one).
  2. Value contains ``:`` → treat as ``module.path:ClassName`` dotted
     path (power-user fallback for impls that aren't catalogued).
  3. Otherwise → look up the short name in
     ``engines/<slot>/BUILTIN_ENGINES``, then in the plugin-engine
     catalog (plugins under ``workspace/plugins/`` declaring
     ``kind: engine`` + ``slot: <slot>`` in their ``meta.yaml``).

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

from krakey.engines.catalog import EngineImpl
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
    """Import ``engines/<slot>/__init__.py`` and return its
    ``(BUILTIN_ENGINES, DEFAULT_ENGINE)`` pair."""
    pkg = importlib.import_module(f"krakey.engines.{slot}")
    builtins = getattr(pkg, "BUILTIN_ENGINES", None)
    default = getattr(pkg, "DEFAULT_ENGINE", None)
    if builtins is None or default is None:
        raise ImportError(
            f"engines.{slot}.__init__ is missing BUILTIN_ENGINES or "
            "DEFAULT_ENGINE — every Engine slot package must declare "
            "both so the registry can resolve short names."
        )
    return builtins, default


def _load_plugin_engine_catalog() -> dict[str, dict[str, str]]:
    """Walk every plugin's ``meta.yaml`` for ``kind: engine`` components.

    Returns ``{slot: {plugin_name: 'module.path:ClassAttr'}}``. Plugins
    that don't declare any engine just don't show up here. Scanning is
    pure-text (``parse_meta`` doesn't import plugin code) so this is
    cheap to call even when many plugins are installed.
    """
    from krakey.plugin_system.catalogue import (
        list_available_plugins,
    )
    catalog: dict[str, dict[str, str]] = {}
    for plugin_name, meta in list_available_plugins().items():
        for comp in meta.components:
            if comp.kind != "engine":
                continue
            slot = getattr(comp, "slot", None)
            if not slot:
                continue
            path = f"{comp.factory_module}:{comp.factory_attr}"
            catalog.setdefault(slot, {})[plugin_name] = path
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

    def _plugin_catalog(self) -> dict[str, dict[str, str]]:
        if self._plugin_engine_catalog is None:
            self._plugin_engine_catalog = _load_plugin_engine_catalog()
        return self._plugin_engine_catalog

    def _resolve_class(self, slot: str, name_or_path: str) -> type:
        """Map an override value to a concrete class.

        ``:`` in value → dotted path. Otherwise short name → walk the
        built-in catalog, then the plugin catalog. Misses raise with a
        list of available short names so the user can fix the typo.
        """
        if ":" in name_or_path:
            return self._import(name_or_path)
        builtins, _ = _load_slot_catalog(slot)
        if name_or_path in builtins:
            return builtins[name_or_path].cls
        plugin_paths = self._plugin_catalog().get(slot, {})
        if name_or_path in plugin_paths:
            return self._import(plugin_paths[name_or_path])
        available = sorted(builtins) + sorted(plugin_paths)
        raise ValueError(
            f"engine slot {slot!r}: unknown impl name "
            f"{name_or_path!r}. Available: {available!r}. Use "
            "'module.path:ClassName' for an impl that isn't "
            "catalogued."
        )

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
        overrides with narrower signatures still work.
        """
        override = self._cfg.core_implementations.get(slot)
        if override:
            name_or_path = override
        else:
            _, default = _load_slot_catalog(slot)
            name_or_path = default

        cls = self._resolve_class(slot, name_or_path)
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
