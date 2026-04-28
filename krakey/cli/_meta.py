"""Metadata helpers — version + repo path discovery via importlib.metadata."""
from __future__ import annotations

import json
import sys
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse

_DIST_NAME = "krakey"


def version() -> str:
    """Return the installed `krakey` package version."""
    try:
        return metadata.version(_DIST_NAME)
    except metadata.PackageNotFoundError:
        return "0.0.0+uninstalled"


def _find_direct_url() -> dict | None:
    """Locate `<dist-info>/direct_url.json` by scanning sys.path.

    `Distribution.read_text("direct_url.json")` returns None because the file
    is not listed in RECORD (PEP 610 writes it post-install). Scan the
    dist-info dirs directly.
    """
    for entry in sys.path:
        if not entry:
            continue
        p = Path(entry)
        if not p.is_dir():
            continue
        for di in p.glob(f"{_DIST_NAME}-*.dist-info"):
            f = di / "direct_url.json"
            if f.exists():
                try:
                    return json.loads(f.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
    return None


def repo_root() -> Path:
    """Resolve the source repo path from the editable-install metadata.

    Editable installs (`pip install -e .`) record the source dir in
    `direct_url.json` (PEP 610). Required for `update` / `repair` /
    `uninstall --full`, which all act on the on-disk repo.
    """
    try:
        metadata.distribution(_DIST_NAME)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "krakey is not installed; run `pip install -e .` from the repo first"
        ) from exc

    info = _find_direct_url()
    if info is None:
        raise RuntimeError(
            "krakey direct_url.json not found; reinstall with `pip install -e .`"
        )
    if not info.get("dir_info", {}).get("editable", False):
        raise RuntimeError(
            "krakey was installed non-editably; reinstall with `pip install -e .`"
        )

    url = info.get("url", "")
    parsed = urlparse(url)
    if parsed.scheme != "file":
        raise RuntimeError(f"unexpected install url scheme: {url!r}")

    path_str = unquote(parsed.path)
    # On Windows urlparse leaves a leading "/" before the drive letter.
    if len(path_str) > 2 and path_str[0] == "/" and path_str[2] == ":":
        path_str = path_str[1:]
    p = Path(path_str)
    if not p.exists():
        raise RuntimeError(f"recorded repo path does not exist: {p}")
    return p.resolve()
