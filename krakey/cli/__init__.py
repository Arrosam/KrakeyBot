"""`krakey` CLI entry — argparse dispatcher for subcommands."""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

from . import _meta


class _BannerArgumentParser(argparse.ArgumentParser):
    """Argparse parser that prints the KRAKEY banner above the help text."""

    def print_help(self, file=None) -> None:
        from . import _banner
        _banner.print_banner(file=file)
        print(file=file)
        super().print_help(file)


def _build_parser() -> argparse.ArgumentParser:
    p = _BannerArgumentParser(
        prog="krakey",
        description="KrakeyBot CLI — manage the heartbeat agent.",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"krakey {_meta.version()}",
    )
    sub = p.add_subparsers(dest="cmd", metavar="<command>")

    run_p = sub.add_parser("run", help="run heartbeat in foreground (Ctrl+C to stop)")
    run_p.add_argument(
        "-p", "--pause",
        action="store_true",
        dest="start_paused",
        help="start with heartbeat paused (use `krakey resume` to unpause)",
    )

    start_p = sub.add_parser("start", help="start heartbeat as background daemon")
    start_p.add_argument(
        "-p", "--pause",
        action="store_true",
        dest="start_paused",
        help="start with heartbeat paused (use `krakey resume` to unpause)",
    )

    sub.add_parser("stop", help="stop the running daemon")
    sub.add_parser(
        "restart",
        help="stop + start the daemon (use after config / plugin edits "
             "that need a fresh process)",
    )
    sub.add_parser("status", help="show whether the daemon is running")

    pause_p = sub.add_parser(
        "pause",
        help="pause the running daemon's heartbeat (optional duration in seconds)",
    )
    pause_p.add_argument(
        "seconds",
        type=int,
        nargs="?",
        default=None,
        metavar="SECONDS",
        help="pause for this many seconds then auto-resume; omit for indefinite pause",
    )

    sub.add_parser("resume", help="resume a paused daemon")

    sub.add_parser("onboard", help="run the interactive onboarding wizard")

    inst = sub.add_parser(
        "install",
        help="pip-install main project deps + every plugin's "
             "declared dependencies",
    )
    inst.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="print what would be installed and exit, don't invoke pip",
    )
    inst.add_argument(
        "--upgrade",
        action="store_true",
        help="pass --upgrade to pip (re-resolve already-installed deps)",
    )

    sub.add_parser("update", help="fetch the newest release tag and reinstall")
    sub.add_parser(
        "repair",
        help="reinstall the currently-pinned release version (force)",
    )

    un = sub.add_parser("uninstall", help="uninstall the krakey CLI")
    un.add_argument(
        "--full",
        action="store_true",
        help="also delete the repo dir (config + workspace + venv)",
    )

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd is None:
        parser.print_help()
        return 0

    from . import commands

    handler = getattr(commands, args.cmd, None)
    if handler is None:
        parser.error(f"unknown command: {args.cmd}")
    return int(handler(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
