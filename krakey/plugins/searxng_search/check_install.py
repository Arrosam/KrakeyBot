"""Post-install probe for ``searxng_search``.

Why
---
Docker is a **system-level binary** (Docker Desktop on Windows /
macOS, the ``docker`` package + ``dockerd`` on Linux). It is NOT a
pip-installable Python dependency — ``krakey install`` runs
``pip install <deps>`` + ``post_install`` shell hooks, neither of
which can bootstrap a Docker installation.

This module is invoked as the plugin's ``post_install`` hook so the
operator gets a clear warning *at install time* instead of a
confusing "connection refused" *at runtime* when the plugin's
``auto_start: true`` mode tries to spawn a container that can never
start.

The probe is **non-fatal** (``optional: true`` in meta.yaml). The
plugin still loads when Docker is missing — operators who run
SearXNG themselves (separate compose stack, systemd unit, public
instance) just set ``auto_start: false`` and ``instance_url``
accordingly.

Run as: ``python -m krakey.plugins.searxng_search.check_install``
"""
from __future__ import annotations

import shutil
import sys


_MISSING_MSG = (
    "searxng_search: docker not found on PATH.\n"
    "    The plugin's auto_start mode needs Docker to run a "
    "local SearXNG container.\n"
    "    Either:\n"
    "      * install Docker Desktop (https://docs.docker.com/get-docker/) "
    "and ensure the daemon is running, OR\n"
    "      * set auto_start: false in workspace/plugins/searxng_search/"
    "config.yaml and point instance_url at a SearXNG you host yourself."
)


def main() -> int:
    """Return 0 when Docker is on PATH, 1 otherwise (with a
    multi-line warning to stderr). Marked ``optional`` in meta.yaml
    so a non-zero return doesn't abort the rest of ``krakey
    install``."""
    # ``shutil.which`` returns the resolved path or None per the
    # stdlib contract; treat any falsy value as missing so a
    # corrupted PATH that yields an empty string also surfaces
    # the warning.
    if not shutil.which("docker"):
        print(_MISSING_MSG, file=sys.stderr)
        return 1
    print("searxng_search: docker found on PATH (auto_start ready)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
