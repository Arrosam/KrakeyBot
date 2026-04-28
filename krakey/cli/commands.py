"""Sub-command handlers. Each takes an argparse Namespace, returns int exit code."""
from __future__ import annotations

import argparse


def run(args: argparse.Namespace) -> int:
    from . import _banner, lifecycle
    _banner.print_banner()
    return lifecycle.run_foreground()


def start(args: argparse.Namespace) -> int:
    from . import lifecycle
    return lifecycle.start_daemon()


def stop(args: argparse.Namespace) -> int:
    from . import lifecycle
    return lifecycle.stop_daemon()


def status(args: argparse.Namespace) -> int:
    from . import lifecycle
    return lifecycle.status()


def onboard(args: argparse.Namespace) -> int:
    from . import _banner, _meta
    import sys
    import subprocess

    _banner.print_banner()
    repo = _meta.repo_root()
    return subprocess.call(
        [sys.executable, "-m", "krakey.onboarding"],
        cwd=str(repo),
    )


def update(args: argparse.Namespace) -> int:
    from . import release
    return release.update()


def repair(args: argparse.Namespace) -> int:
    from . import release
    return release.repair()


def uninstall(args: argparse.Namespace) -> int:
    from . import release
    return release.uninstall(full=args.full)
