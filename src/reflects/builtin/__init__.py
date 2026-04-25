"""Built-in Reflect plugins.

Each Reflect lives in its own subfolder with a pure-text
``meta.yaml`` + an empty ``__init__.py`` + a ``reflect.py``
containing the actual implementation. Discovery
(``src.reflects.discovery.discover_reflects``) walks the meta files
**without importing any reflect.py**; module imports happen lazily
via ``load_reflect(name, deps)`` only after the user enables a name
in ``config.yaml``'s ``reflects:`` list.

This file used to expose a ``BUILTIN_FACTORIES`` dict that imported
every Reflect at startup. That violated the 2026-04-25 architectural
invariant ("plugin code must not load before the user enables it"),
so it was removed. New code should always go through
``src.reflects.discovery``.
"""
