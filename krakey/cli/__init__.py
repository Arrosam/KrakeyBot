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

    sub.add_parser("run", help="run heartbeat in foreground (Ctrl+C to stop)")
    sub.add_parser("start", help="start heartbeat as background daemon")
    sub.add_parser("stop", help="stop the running daemon")
    sub.add_parser("status", help="show whether the daemon is running")
    sub.add_parser("onboard", help="run the interactive onboarding wizard")
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
