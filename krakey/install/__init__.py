"""``krakey.install`` — concrete implementation of the
``InstallService`` Protocol (``krakey/interfaces/install_service.py``).

Re-exports the module-level functions for back-compat with tests
and existing scripts that imported them from ``krakey.cli.install``;
new code should depend on the Protocol via DI:

    from krakey.interfaces.install_service import InstallService

and let the composition root inject the concrete implementation.
"""
from krakey.install.service import (
    BUILTIN_ROOT,
    DefaultInstallService,
    INSTALL_STATE_PATH,
    WORKSPACE_ROOT,
    collect_core_dependencies,
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
