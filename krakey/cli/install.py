"""``krakey install`` — pip-install main project deps + every
plugin's declared deps in one shot.

Each plugin under ``BUILTIN_ROOT`` (krakey/plugins/) and
``WORKSPACE_ROOT`` (workspace/plugins/) declares its own
``dependencies:`` list in meta.yaml. This module walks them,
collects the union, optionally adds the main project's own
deps (when run from a checkout with pyproject.toml at repo root),
and dispatches a single ``pip install`` subprocess.

Side effect: writes ``workspace/data/install_state.json`` with a
hash of the declared-deps set after a successful install. The
runtime startup path (in ``krakey/cli/commands.py``'s ``run`` /
``start``) compares the live hash against the stored one and
prints a one-line warning if they differ — so the operator
sees "you've enabled / updated a plugin; run ``krakey install``"
without having to remember.

The hash is computed over the SORTED union of dep strings, so
adding a plugin with deps the user already had installed by
another plugin won't trigger a re-install prompt — only true
"new dep added" does.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from krakey.plugin_system.loader import (
    BUILTIN_ROOT,
    WORKSPACE_ROOT,
    parse_meta,
)


INSTALL_STATE_PATH = Path("workspace") / "data" / "install_state.json"
"""Where the post-install bookkeeping lands. Same data/ dir the
heartbeat uses for its own state, so a single ``rm -rf
workspace/data`` resets everything (useful for tests + reset
flows). Path is relative to cwd so the runtime + tests can swap
the workspace root via ``os.chdir``."""


# =====================================================================
# Discovery — walk plugin folders, parse meta, collect deps
# =====================================================================


def collect_plugin_dependencies() -> dict[str, list[str]]:
    """Walk every plugin folder under BUILTIN_ROOT + WORKSPACE_ROOT.
    Returns ``{plugin_name: [pip-spec-strings]}``.

    Workspace overrides built-in (matches ``load_plugin_meta``'s
    same-name semantics — but here we walk EVERY plugin, not just
    those in the user's enabled list, because a plugin's deps need
    to be installed before its meta.yaml is referenced from
    config.yaml).
    """
    out: dict[str, list[str]] = {}
    # Iterate built-in first, then workspace, so workspace wins on
    # name collisions (later writes override earlier).
    for root in (BUILTIN_ROOT, WORKSPACE_ROOT):
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            meta_path = d / "meta.yaml"
            if not meta_path.exists():
                continue
            try:
                meta = parse_meta(meta_path)
            except Exception as e:  # noqa: BLE001
                # Surface but don't abort — a malformed plugin
                # shouldn't block installing the others.
                print(
                    f"warning: skipping {d.name}: meta.yaml parse "
                    f"failed: {e}",
                    file=sys.stderr,
                )
                continue
            out[meta.name] = list(meta.dependencies)
    return out


def collect_core_dependencies() -> list[str]:
    """Read the main project's pyproject.toml (when in a checkout)
    and return its ``[project].dependencies`` list.

    When krakey was installed via wheel (no pyproject at repo
    root), returns ``[]`` — the wheel install already pulled the
    project's deps, so re-installing them via this path would be
    redundant.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.exists():
        return []
    # tomllib is stdlib on Python 3.11+ which is the project's
    # ``requires-python``; no extra dep needed.
    import tomllib
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        print(
            f"warning: could not read {pyproject} ({e}); skipping "
            "core deps",
            file=sys.stderr,
        )
        return []
    project = data.get("project") or {}
    deps = project.get("dependencies") or []
    if not isinstance(deps, list):
        return []
    return [str(d) for d in deps if isinstance(d, str) and d.strip()]


# =====================================================================
# Hash + state file
# =====================================================================


def deps_hash(plugin_deps: dict[str, list[str]]) -> str:
    """Stable sha256 over the SORTED union of all declared deps.

    Hashing the union (not per-plugin) means:
      - reordering plugins doesn't trip the hash
      - adding a plugin whose every dep is already declared by
        another plugin doesn't trigger a re-install warning
      - changing a version pin DOES (a new spec string is a
        different element of the set)
    """
    flat = sorted({d for deps in plugin_deps.values() for d in deps})
    return hashlib.sha256("\n".join(flat).encode("utf-8")).hexdigest()


def read_install_state() -> dict | None:
    if not INSTALL_STATE_PATH.exists():
        return None
    try:
        return json.loads(
            INSTALL_STATE_PATH.read_text(encoding="utf-8"),
        )
    except (OSError, json.JSONDecodeError):
        # Treat corrupt state as missing — next install rewrites it.
        return None


def write_install_state(state: dict) -> None:
    INSTALL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = INSTALL_STATE_PATH.with_suffix(
        INSTALL_STATE_PATH.suffix + ".tmp",
    )
    tmp.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(INSTALL_STATE_PATH)


def has_pending_deps() -> tuple[bool, dict[str, list[str]]]:
    """At startup, decide whether the user should run
    ``krakey install``. Returns ``(pending, plugin_deps)``.

    pending=True when:
      * no install_state.json exists yet (fresh checkout / first
        run), OR
      * the deps_hash on disk differs from the live one (a plugin
        was added / removed / changed its declared deps since last
        install).
    """
    plugin_deps = collect_plugin_dependencies()
    state = read_install_state()
    current = deps_hash(plugin_deps)
    if state is None:
        return True, plugin_deps
    if state.get("deps_hash") != current:
        return True, plugin_deps
    return False, plugin_deps


# =====================================================================
# Install entry point (called by ``krakey install``)
# =====================================================================


def install(args: argparse.Namespace) -> int:
    """``krakey install`` handler. Discovers, prints, dispatches.

    Returns the pip subprocess's exit code (0 on success). On
    pip failure the install_state.json is NOT updated, so the
    next startup still warns and the operator can re-run after
    fixing the underlying pip error.
    """
    plugin_deps = collect_plugin_dependencies()
    core_deps = collect_core_dependencies()

    print("krakey install: discovery")
    if core_deps:
        print(f"  core (pyproject.toml):")
        for d in core_deps:
            print(f"    - {d}")
    else:
        print("  core: (none — running from a wheel / no pyproject "
              "at repo root)")
    for name, deps in sorted(plugin_deps.items()):
        if deps:
            print(f"  plugin {name}:")
            for d in deps:
                print(f"    - {d}")
        else:
            print(f"  plugin {name}: (no third-party deps)")

    union = sorted(set(core_deps) | {
        d for deps in plugin_deps.values() for d in deps
    })

    if getattr(args, "dry_run", False):
        print("\n--dry-run: not invoking pip; would install:")
        for d in union:
            print(f"  - {d}")
        return 0

    if not union:
        print("\nNothing to install.")
        write_install_state({
            "deps_hash":    deps_hash(plugin_deps),
            "installed":    sorted(plugin_deps.keys()),
            "installed_at": datetime.now().isoformat(timespec="seconds"),
            "core_count":   len(core_deps),
        })
        return 0

    print(f"\nInstalling {len(union)} unique dep(s) via pip...")
    cmd = [sys.executable, "-m", "pip", "install", *union]
    if getattr(args, "upgrade", False):
        cmd.insert(4, "--upgrade")
    print(f"  $ {' '.join(cmd)}\n")

    rc = subprocess.call(cmd)
    if rc != 0:
        print(
            f"\npip exited with rc={rc}; install_state.json NOT "
            "updated. Fix the underlying error and re-run "
            "`krakey install`.",
            file=sys.stderr,
        )
        return rc

    write_install_state({
        "deps_hash":    deps_hash(plugin_deps),
        "installed":    sorted(plugin_deps.keys()),
        "installed_at": datetime.now().isoformat(timespec="seconds"),
        "core_count":   len(core_deps),
    })
    print("\nkrakey install: done.")
    return 0
