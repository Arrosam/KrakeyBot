"""update / repair / uninstall — git-tag based release management.

Source-of-truth: git tags formatted `vX.Y.Z` matching the `pyproject.toml`
version. `update` jumps to the newest tag; `repair` re-checks-out the current
version's tag (force) to recover from local corruption.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

from . import _meta

_SEMVER_TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def _git(*args: str, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=check,
        text=True,
        capture_output=True,
    )


def _ensure_clean(repo: Path) -> bool:
    """Return True if working tree is clean (no uncommitted changes)."""
    res = _git("status", "--porcelain", cwd=repo, check=False)
    return res.returncode == 0 and not res.stdout.strip()


def _all_release_tags(repo: Path) -> list[tuple[tuple[int, int, int], str]]:
    res = _git("tag", "--list", "v*", cwd=repo, check=False)
    out = []
    for line in res.stdout.splitlines():
        m = _SEMVER_TAG.match(line.strip())
        if m:
            out.append(((int(m.group(1)), int(m.group(2)), int(m.group(3))), line.strip()))
    out.sort()
    return out


def _pip_reinstall(repo: Path) -> int:
    res = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(repo),
         "--upgrade", "--disable-pip-version-check"],
        check=False,
    )
    return res.returncode


def _confirm(prompt: str) -> bool:
    try:
        ans = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


# -------- update --------

def update() -> int:
    repo = _meta.repo_root()
    print(f"krakey: fetching tags from origin in {repo}")
    fetch = _git("fetch", "--tags", "--prune", "origin", cwd=repo, check=False)
    if fetch.returncode != 0:
        print(f"krakey: git fetch failed:\n{fetch.stderr}", file=sys.stderr)
        return 1

    tags = _all_release_tags(repo)
    if not tags:
        print("krakey: no release tags found on origin (expected `vX.Y.Z`)",
              file=sys.stderr)
        return 1

    _, newest = tags[-1]
    current = _meta.version()
    if f"v{current}" == newest:
        print(f"krakey: already at latest version ({current})")
        return 0

    print(f"krakey: updating  {current}  →  {newest.lstrip('v')}")
    if not _ensure_clean(repo):
        print("krakey: refusing to update — working tree has uncommitted changes.",
              file=sys.stderr)
        print("       commit or stash first, or use `krakey repair` to discard them.",
              file=sys.stderr)
        return 1

    co = _git("checkout", newest, cwd=repo, check=False)
    if co.returncode != 0:
        print(f"krakey: git checkout failed:\n{co.stderr}", file=sys.stderr)
        return 1

    rc = _pip_reinstall(repo)
    if rc != 0:
        print("krakey: pip install failed; repo is at new tag but deps may be stale.",
              file=sys.stderr)
        return rc

    print(f"krakey: updated to {newest}")
    return 0


# -------- repair --------

def repair() -> int:
    repo = _meta.repo_root()
    current = _meta.version()
    target = f"v{current}"

    print(f"krakey: repair will force-checkout {target} in {repo}")
    print("        local uncommitted changes will be DISCARDED.")
    if not _confirm("proceed?"):
        print("krakey: aborted")
        return 1

    fetch = _git("fetch", "--tags", "origin", cwd=repo, check=False)
    if fetch.returncode != 0:
        print(f"krakey: git fetch failed:\n{fetch.stderr}", file=sys.stderr)
        return 1

    tags = _all_release_tags(repo)
    if not any(name == target for _, name in tags):
        print(f"krakey: tag {target} not found in repo. cannot repair.",
              file=sys.stderr)
        return 1

    co = _git("checkout", "--force", target, cwd=repo, check=False)
    if co.returncode != 0:
        print(f"krakey: git checkout failed:\n{co.stderr}", file=sys.stderr)
        return 1

    rc = _pip_reinstall(repo)
    if rc != 0:
        print("krakey: pip install failed.", file=sys.stderr)
        return rc

    print(f"krakey: repaired to {target}")
    return 0


# -------- uninstall --------

def uninstall(*, full: bool) -> int:
    repo: Path | None = None
    try:
        repo = _meta.repo_root()
    except RuntimeError:
        repo = None

    if full and repo is not None:
        print("krakey: --full will permanently delete:")
        print(f"        {repo}")
        print("        (config.yaml, workspace/, .venv/, source — all gone)")
        if not _confirm("proceed?"):
            print("krakey: aborted")
            return 1

    print("krakey: uninstalling pip package…")
    rc = subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "krakey",
         "--disable-pip-version-check"],
        check=False,
    ).returncode
    if rc != 0:
        print("krakey: pip uninstall reported an error (continuing).",
              file=sys.stderr)

    if full and repo is not None:
        print(f"krakey: removing {repo}")
        shutil.rmtree(repo, ignore_errors=False)
        print("krakey: removed.")

    print("krakey: done.")
    return 0
