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
    import os

    _banner.print_banner()
    repo = _meta.repo_root()
    # Run inside the repo so ``config.yaml`` (relative path) lands at
    # the repo root, matching where the runtime looks for it.
    prev_cwd = os.getcwd()
    os.chdir(str(repo))
    try:
        from krakey.onboarding import run_wizard
        try:
            run_wizard()
        except KeyboardInterrupt:
            print("\nkrakey: onboarding cancelled.")
            return 130
        except EOFError:
            print("\nkrakey: onboarding ended (stdin closed).")
            return 1
    finally:
        os.chdir(prev_cwd)
    return 0


def update(args: argparse.Namespace) -> int:
    from . import release
    return release.update()


def repair(args: argparse.Namespace) -> int:
    from . import release
    return release.repair()


def uninstall(args: argparse.Namespace) -> int:
    from . import release
    return release.uninstall(full=args.full)
