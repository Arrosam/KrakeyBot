"""``krakey install`` CLI handler — thin wrapper around
``krakey.install`` (the concrete implementation of the
``InstallService`` Protocol).

The actual install logic lives in ``krakey/install/service.py``
so runtime + dashboard can depend on the same code via the
``InstallService`` Protocol (``krakey/interfaces/install_service.py``)
without ``runtime/`` having to import from ``cli/`` (which would
invert the dependency direction — the CLI is supposed to wrap
the runtime, not the reverse).

Module-level names (``install``, ``has_pending_deps``,
``INSTALL_STATE_PATH``, etc.) are re-exported for back-compat
with older tests that imported from ``krakey.cli.install``
directly.
"""
from __future__ import annotations

# Re-export everything from the impl so old call sites
# ``from krakey.cli import install as install_mod`` keep working.
# New code should depend on ``InstallService`` Protocol +
# ``DefaultInstallService`` from ``krakey.install``.
from krakey.install.service import (  # noqa: F401
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

# A few existing tests reach into this module's ``subprocess``
# binding via monkeypatch.setattr(install_mod.subprocess, "call",
# fake). Re-exporting the same module keeps those tests green.
import subprocess  # noqa: F401
import sys  # noqa: F401
