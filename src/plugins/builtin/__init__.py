"""Builtin plugin root — shipped with Krakey.

Loader scans this directory as if it were `workspace/plugins/`, so the
factory contract is identical. Each subdirectory is a *project* that
may expose one or more tentacles + sensories via `create_plugins` (or
the single-component `create_tentacle` / `create_sensory` shortcuts).
Nothing special lives at this level beyond the plugin packages
themselves.
"""
