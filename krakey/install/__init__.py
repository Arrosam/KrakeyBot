"""``krakey.install`` — pip + post_install + install_state.json bookkeeping.

Plain utility module imported by the CLI (``krakey install`` command)
and the dashboard plugin's deps panel. Has nothing to do with the
runtime's heartbeat — install is a pre-startup concern, not part of
the per-beat loop.
"""
from krakey.install.service import (
    BUILTIN_ROOT,
    DefaultInstallService,
    INSTALL_STATE_PATH,
    WORKSPACE_ROOT,
    collect_core_dependencies,
    collect_engine_dependencies,
    collect_engine_post_install,
    collect_plugin_dependencies,
    collect_plugin_post_install,
    deps_hash,
    expand_python_token,
    has_pending_deps,
    install,
    read_install_state,
    run_post_install_for_plugin,
    write_install_state,
)

__all__ = [
    "BUILTIN_ROOT",
    "DefaultInstallService",
    "INSTALL_STATE_PATH",
    "WORKSPACE_ROOT",
    "collect_core_dependencies",
    "collect_engine_dependencies",
    "collect_engine_post_install",
    "collect_plugin_dependencies",
    "collect_plugin_post_install",
    "deps_hash",
    "expand_python_token",
    "has_pending_deps",
    "install",
    "read_install_state",
    "run_post_install_for_plugin",
    "write_install_state",
]
