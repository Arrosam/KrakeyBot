"""``engine_system`` — the loader infrastructure that turns yaml-described
Engine packages under ``krakey/engines/`` into concrete impl instances.

Why this package exists separately from ``krakey/engines/``:

The engines themselves are intended to be **external components** —
folders under ``krakey/engines/`` that the rest of the codebase
**never imports directly**. Adding, removing, or renaming an engine
must not touch any file outside its own directory; the only contract
between an engine and the runtime is the per-engine ``meta.yaml`` +
the Protocol declared in ``krakey/interfaces/engines/``. Putting the
loader (``EngineRegistry``) and its support classes (``EngineImpl``
catalog, meta parser, dotted-path fallback list) inside
``krakey.engines`` would create exactly the coupling we want to
avoid: ``engines/__init__.py`` would have to import + re-export
loader symbols, and the loader would have to walk its own host
package to find engines.

So:

  * ``catalog.py``       — ``EngineImpl`` dataclass (cls + description
                           + config_schema). Pure data, no imports of
                           engine impls.
  * ``meta_loader.py``   — read & parse ``engines/<slot>/meta.yaml``.
                           Returns ``{name: EngineImpl}`` per slot.
  * ``defaults.py``      — emergency fallback ``{slot: dotted_path}``
                           dict. Used **only** when a slot's
                           ``meta.yaml`` is missing / malformed /
                           points at an unimportable class. Pure
                           strings — engine_system never imports
                           ``krakey.engines.<X>`` itself.
  * ``registry.py``      — ``EngineRegistry`` class. Resolves a slot
                           short name to a concrete instance via
                           meta-first, defaults-second, loud-fail
                           otherwise.

Nothing in this package imports ``krakey.engines`` at module level.
The registry's ``importlib.import_module`` calls only fire on actual
``resolve(slot)`` requests, and they're driven by yaml content +
the user's config — never by hard-coded references in the loader.
"""
from krakey.engine_system.registry import EngineRegistry

__all__ = ["EngineRegistry"]
