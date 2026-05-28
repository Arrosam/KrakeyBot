"""Scan and parse ``krakey/engines/<slot>/meta.yaml`` files.

Each engine slot is a standalone folder under ``krakey/engines/`` with
its own ``meta.yaml`` declaring:

  - ``slot``: the slot name (must match the folder name)
  - ``description``: free-form
  - ``builtin_engines``: list of entries; per-entry keys:
      ``name``, ``factory_module``, ``factory_attr`` (required);
      ``description``, ``default``, ``config_schema``,
      ``dependencies``, ``post_install`` (optional)

Per-entry optional keys:

  - ``config_schema``: list of dashboard / config-form field
    descriptors (same shape plugins use)
  - ``dependencies``: list[str] of pip-installable spec strings
    (e.g. ``"some-package>=1.0"``) collected by ``krakey install``
  - ``post_install``: list of secondary install steps run after pip;
    each step is ``{args: list[str], description: str,
    optional: bool}`` — entries with missing/malformed ``args`` are
    warn-and-skipped at load time (tolerant, never raises)

This module is the **only** part of ``engine_system`` that knows the
on-disk layout of ``krakey/engines/``. It returns plain dataclass
objects (``EngineImpl`` from ``catalog.py``) keyed by engine name —
the registry then walks that dict like it walked the old
``BUILTIN_ENGINES`` dict, with one difference: each entry now carries
its own dotted-path factory location, so the registry never imports
``krakey.engines.<slot>`` at module level.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from krakey.engine_system.catalog import EngineImpl


_log = logging.getLogger(__name__)


# Path to ``krakey/engines/`` resolved at import time — the meta_loader
# is itself inside ``krakey/engine_system/``, so we step up one
# directory and over to ``engines/``. Using ``__file__`` rather than
# ``importlib.resources`` keeps this honest: if someone installs Krakey
# as a wheel + tries to relocate the engines folder, the loader will
# loudly fail rather than silently fall back to the package metadata
# path (which wouldn't include the per-engine meta.yaml files anyway).
_ENGINES_ROOT: Path = Path(__file__).parent.parent / "engines"


class MetaParseError(Exception):
    """Raised when a ``meta.yaml`` exists but is structurally invalid.

    Distinct from ``FileNotFoundError`` (yaml absent) so the registry
    can fall back to ``defaults.py`` on absence but loud-fail on
    structural corruption — silent fallback would let a typo in
    ``meta.yaml`` ship a stale impl with no diagnostic."""


def load_slot_meta(
    slot: str, *, engines_root: Path | None = None,
) -> tuple[dict[str, EngineImpl], str]:
    """Read ``engines/<slot>/meta.yaml`` and return ``({short_name:
    EngineImpl}, default_short_name)``.

    Raises ``FileNotFoundError`` when no ``meta.yaml`` exists for the
    slot (registry caller will try ``defaults.py``).
    Raises ``MetaParseError`` when the file is present but malformed —
    missing required fields, no entries, multiple ``default: true``,
    etc.

    Note: this does NOT import the factory module — it only records
    the dotted path. The registry's ``_resolve_class`` does the
    ``importlib.import_module`` lazily on first use, so a broken
    factory in one slot's meta won't break loading of another slot.
    """
    root = engines_root or _ENGINES_ROOT
    meta_path = root / slot / "meta.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"no meta.yaml for engine slot {slot!r} at {meta_path}"
        )

    try:
        raw = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise MetaParseError(
            f"engine slot {slot!r}: meta.yaml parse error: {e}"
        ) from e

    if not isinstance(raw, dict):
        raise MetaParseError(
            f"engine slot {slot!r}: meta.yaml top-level must be a "
            f"mapping; got {type(raw).__name__}"
        )

    declared_slot = raw.get("slot")
    if declared_slot and declared_slot != slot:
        # Tolerant: warn but continue. The folder name is authoritative
        # (registry indexes by folder name); a stale ``slot:`` field
        # is a typo, not an error worth failing startup over.
        _log.warning(
            "engine slot %r: meta.yaml declares slot=%r; using folder "
            "name as authoritative", slot, declared_slot,
        )

    entries_raw = raw.get("builtin_engines") or []
    if not isinstance(entries_raw, list) or not entries_raw:
        raise MetaParseError(
            f"engine slot {slot!r}: meta.yaml is missing a non-empty "
            "``builtin_engines:`` list"
        )

    catalog: dict[str, EngineImpl] = {}
    default_name: str | None = None
    for entry in entries_raw:
        if not isinstance(entry, dict):
            raise MetaParseError(
                f"engine slot {slot!r}: builtin_engines entries must "
                f"be mappings; got {type(entry).__name__}"
            )
        name = entry.get("name")
        module = entry.get("factory_module")
        attr = entry.get("factory_attr")
        if not name or not module or not attr:
            raise MetaParseError(
                f"engine slot {slot!r}: each builtin_engines entry "
                "must have ``name``, ``factory_module``, and "
                f"``factory_attr`` set; got {entry!r}"
            )
        is_default = bool(entry.get("default", False))
        if is_default:
            if default_name is not None:
                raise MetaParseError(
                    f"engine slot {slot!r}: multiple builtin_engines "
                    f"entries declare default=true ({default_name!r} "
                    f"and {name!r}); exactly one is allowed."
                )
            default_name = name
        catalog[name] = EngineImpl(
            cls=_LazyImpl(module, attr),  # type: ignore[arg-type]
            description=str(entry.get("description", "") or ""),
            config_schema=list(
                _coerce_config_schema(entry.get("config_schema"))
            ),
            dependencies=_coerce_dependencies(entry.get("dependencies")),
            post_install=_coerce_post_install(entry.get("post_install")),
        )

    if default_name is None:
        # Tolerate single-entry catalogs without ``default: true`` —
        # the lone entry is implicitly the default. With 2+ entries
        # the user MUST pick one or the registry has no way to choose.
        if len(catalog) == 1:
            default_name = next(iter(catalog))
        else:
            raise MetaParseError(
                f"engine slot {slot!r}: meta.yaml has multiple "
                "builtin_engines but none marked ``default: true``"
            )

    return catalog, default_name


def _coerce_config_schema(raw: Any) -> list[dict[str, Any]]:
    """Pass config_schema through verbatim if it's a list of dicts;
    drop silently otherwise so a typo doesn't crash startup. Same
    contract dashboard expects from plugin meta.yaml."""
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(dict(item))
    return out


def _coerce_dependencies(raw: Any) -> list[str]:
    """Pass dependencies through verbatim if it's a list of non-empty
    strings; coerce/filter otherwise so a typo doesn't crash startup.

    - Not a list -> [].
    - Per item: must be a str; .strip() must be non-empty; otherwise dropped.
    - Returns the stripped strings in input order.
    """
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
    return out


def _coerce_post_install(raw: Any) -> list[dict[str, Any]]:
    """Pass post_install through verbatim if each entry is a well-formed
    dict; warn-and-skip otherwise (do NOT raise — same tolerance as
    _coerce_config_schema). Entry shape:

      {args: list[non-empty str] (REQUIRED, non-empty),
       description: str (optional, default ""),
       optional: bool (optional, default False)}

    Entries with a missing/empty/non-list ``args``, OR any non-string /
    empty-string element in ``args``, OR that aren't a dict at all, are
    dropped with a _log.warning. Other unknown keys in the entry are
    ignored.
    """
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            _log.warning(
                "post_install: skipping non-dict entry %r", entry,
            )
            continue
        args = entry.get("args")
        if not isinstance(args, list) or not args:
            _log.warning(
                "post_install: entry missing or empty 'args': %r", entry,
            )
            continue
        if any(not isinstance(a, str) or not a for a in args):
            _log.warning(
                "post_install: entry has non-string or empty arg in "
                "args=%r; skipping", args,
            )
            continue
        out.append({
            "args": list(args),
            "description": str(entry.get("description", "") or ""),
            "optional": bool(entry.get("optional", False)),
        })
    return out


class _LazyImpl:
    """Holds a ``module:attr`` dotted reference; ``importlib`` resolves
    on first attribute access. The registry needs ``EngineImpl.cls``
    to look + act like a class (for ``issubclass`` /
    ``inspect.signature`` / ``cls(**kwargs)`` calls), so this wrapper
    delegates everything to the actual imported class once loaded.
    Cached after first call.

    Why lazy: ``meta.yaml`` lists every available engine for a slot
    (including ones the user may never select). Eager-importing all
    of them at registry construction time would pull every engine
    impl + its transitive deps into the process even when only one
    is used. Lazy-import keeps startup proportional to actually-used
    impls.
    """

    def __init__(self, module: str, attr: str):
        self._module = module
        self._attr = attr
        self._resolved: Any = None

    def _resolve(self) -> Any:
        if self._resolved is None:
            import importlib
            mod = importlib.import_module(self._module)
            if not hasattr(mod, self._attr):
                raise ImportError(
                    f"meta-declared engine: module {self._module!r} "
                    f"has no attribute {self._attr!r}"
                )
            self._resolved = getattr(mod, self._attr)
        return self._resolved

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._resolve()(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)

    def __repr__(self) -> str:
        if self._resolved is not None:
            return f"<LazyImpl resolved={self._resolved!r}>"
        return f"<LazyImpl {self._module}:{self._attr} (unresolved)>"
