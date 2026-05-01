"""Sub-command handlers. Each takes an argparse Namespace, returns int exit code.

Banner-printing rule: at most one KRAKEY banner per CLI invocation. The
wizard prints its own banner via ``_print_intro``; the runtime prints
its banner when starting the heartbeat. Handlers in this module never
print the banner themselves — that double-prints in flows where one
calls the other (e.g. ``krakey run`` auto-launching onboarding when
config is missing).
"""
from __future__ import annotations

import argparse


def run(args: argparse.Namespace) -> int:
    from . import lifecycle
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
    from . import _meta, lifecycle
    import os

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
    # Auto-continue into the heartbeat so the user doesn't have to
    # type a second command. `run_foreground` checks for an existing
    # daemon and pid-files itself, so this is safe even if the user
    # had a stray daemon running.
    return lifecycle.run_foreground()


def update(args: argparse.Namespace) -> int:
    from . import release
    return release.update()


def repair(args: argparse.Namespace) -> int:
    from . import release
    return release.repair()


def uninstall(args: argparse.Namespace) -> int:
    from . import release
    return release.uninstall(full=args.full)
