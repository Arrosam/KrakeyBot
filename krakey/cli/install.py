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


def _walk_plugin_metas():
    """Yield (plugin_name, parsed_meta) for every well-formed
    plugin under BUILTIN_ROOT + WORKSPACE_ROOT. Workspace wins on
    name collisions. Malformed metas log and skip."""
    seen: dict[str, Any] = {}
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
                print(
                    f"warning: skipping {d.name}: meta.yaml parse "
                    f"failed: {e}",
                    file=sys.stderr,
                )
                continue
            seen[meta.name] = meta
    for name, meta in seen.items():
        yield name, meta


def collect_plugin_dependencies() -> dict[str, list[str]]:
    """Walk every plugin folder under BUILTIN_ROOT + WORKSPACE_ROOT.
    Returns ``{plugin_name: [pip-spec-strings]}``."""
    return {n: list(m.dependencies) for n, m in _walk_plugin_metas()}


def collect_plugin_post_install() -> dict[str, list[dict[str, Any]]]:
    """Same walk, returns ``{plugin_name: [post_install entries]}``.
    Each entry is the validated dict from
    ``loader._parse_post_install``: ``{args, description, optional}``."""
    return {n: list(m.post_install) for n, m in _walk_plugin_metas()}


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


def deps_hash(
    plugin_deps: dict[str, list[str]],
    post_install: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    """Stable sha256 over the SORTED union of declared pip deps
    AND the per-plugin post_install commands (joined arg lists).

    Including post_install means a plugin that adds a new
    secondary install step (e.g. browser_exec adding firefox
    binaries) trips the startup advisory + dashboard
    "needs install" status, even if its pip deps haven't changed.

    ``post_install`` defaults to None for back-compat with
    callers that don't have it (existing tests). New code
    should pass both."""
    flat = sorted({d for deps in plugin_deps.values() for d in deps})
    parts = ["\n".join(flat)]
    if post_install:
        # Serialize each post_install entry deterministically:
        # plugin|args|optional. Description is documentation,
        # not part of the install behavior, so excluded from
        # the hash.
        post_lines = []
        for plugin in sorted(post_install.keys()):
            for entry in post_install[plugin]:
                args = json.dumps(entry.get("args") or [])
                opt = bool(entry.get("optional", False))
                post_lines.append(f"{plugin}|{args}|{opt}")
        parts.append("\n".join(sorted(post_lines)))
    return hashlib.sha256(
        "\n---\n".join(parts).encode("utf-8"),
    ).hexdigest()


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
        was added / removed / changed its declared deps OR its
        post_install commands since last install).
    """
    plugin_deps = collect_plugin_dependencies()
    post_install = collect_plugin_post_install()
    state = read_install_state()
    current = deps_hash(plugin_deps, post_install)
    if state is None:
        return True, plugin_deps
    if state.get("deps_hash") != current:
        return True, plugin_deps
    return False, plugin_deps


# =====================================================================
# Install entry point (called by ``krakey install``)
# =====================================================================


def expand_python_token(args: list[str]) -> list[str]:
    """Token-substitute ``{python}`` → ``sys.executable`` in argv
    so post_install commands ALWAYS hit the runtime's interpreter
    (same venv where pip just installed)."""
    return [
        sys.executable if a == "{python}" else a for a in args
    ]


def run_post_install_for_plugin(
    plugin_name: str,
    entries: list[dict[str, Any]],
) -> tuple[int, list[str]]:
    """Run a single plugin's post_install entries in declared
    order. Returns ``(rc, errors)`` where rc is 0 on full success.

    Per-entry behavior:
      - args list with ``{python}`` token substitution
      - description echoed to stdout before each command
      - ``optional=True`` failures are logged but don't abort
      - ``optional=False`` (default) failures abort the rest of
        THIS plugin's chain (other plugins still get to try)
    """
    errors: list[str] = []
    for i, entry in enumerate(entries):
        argv = expand_python_token(entry["args"])
        desc = entry.get("description") or ""
        optional = bool(entry.get("optional", False))
        label = f"  $ {' '.join(argv)}"
        if desc:
            label += f"     # {desc}"
        print(f"\n[{plugin_name} post_install #{i + 1}] " + (
            "(optional)" if optional else ""
        ))
        print(label)
        rc = subprocess.call(argv)
        if rc == 0:
            continue
        msg = (
            f"{plugin_name} post_install #{i + 1} "
            f"({argv[0]}...) returned rc={rc}"
        )
        if optional:
            print(f"  ⚠ optional step failed (continuing): {msg}",
                  file=sys.stderr)
            continue
        errors.append(msg)
        print(
            f"  ✗ aborting {plugin_name} post_install: {msg}",
            file=sys.stderr,
        )
        return rc, errors
    return 0, errors


def install(args: argparse.Namespace) -> int:
    """``krakey install`` handler. Discovers, prints, pip-installs,
    runs post_install hooks.

    Returns the pip subprocess's exit code (0 on success). On
    pip failure or any non-optional post_install failure, the
    install_state.json is NOT updated, so the next startup still
    warns and the operator can re-run after fixing the underlying
    error.
    """
    plugin_deps = collect_plugin_dependencies()
    plugin_post = collect_plugin_post_install()
    core_deps = collect_core_dependencies()

    print("krakey install: discovery")
    if core_deps:
        print(f"  core (pyproject.toml):")
        for d in core_deps:
            print(f"    - {d}")
    else:
        print("  core: (none — running from a wheel / no pyproject "
              "at repo root)")
    for name in sorted(set(plugin_deps) | set(plugin_post)):
        deps = plugin_deps.get(name) or []
        post = plugin_post.get(name) or []
        if not deps and not post:
            print(f"  plugin {name}: (no deps, no post_install)")
            continue
        print(f"  plugin {name}:")
        for d in deps:
            print(f"    - {d}")
        for entry in post:
            argv = " ".join(entry["args"])
            tag = " (optional)" if entry.get("optional") else ""
            desc = entry.get("description") or ""
            print(f"    + post_install: {argv}{tag}"
                  + (f" — {desc}" if desc else ""))

    union = sorted(set(core_deps) | {
        d for deps in plugin_deps.values() for d in deps
    })

    if getattr(args, "dry_run", False):
        print("\n--dry-run: not invoking pip; would install:")
        for d in union:
            print(f"  - {d}")
        for plugin_name in sorted(plugin_post):
            for entry in plugin_post[plugin_name]:
                argv = " ".join(expand_python_token(entry["args"]))
                tag = " (optional)" if entry.get("optional") else ""
                print(f"  + post_install ({plugin_name}): "
                      f"{argv}{tag}")
        return 0

    final_state = {
        "deps_hash":    deps_hash(plugin_deps, plugin_post),
        "installed":    sorted(plugin_deps.keys()),
        "installed_at": datetime.now().isoformat(timespec="seconds"),
        "core_count":   len(core_deps),
        "post_install_done": sorted(
            n for n, p in plugin_post.items() if p
        ),
    }

    if not union:
        print("\nNo pip deps to install — skipping pip step.")
    else:
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

    # ---- post_install hooks ----
    any_post_failure = False
    for plugin_name in sorted(plugin_post.keys()):
        entries = plugin_post[plugin_name]
        if not entries:
            continue
        rc, errs = run_post_install_for_plugin(plugin_name, entries)
        if rc != 0:
            any_post_failure = True
            # Don't break — other plugins' post_install still get
            # a chance, so a single broken plugin doesn't gate
            # ALL secondary install steps. The state is NOT
            # written below, so the operator re-runs after
            # fixing the issue.
            print(
                f"\nNote: {plugin_name} post_install failed; "
                "continuing with other plugins.",
                file=sys.stderr,
            )

    if any_post_failure:
        print(
            "\nkrakey install: pip succeeded but at least one "
            "non-optional post_install step failed. "
            "install_state.json NOT updated; re-run after fixing.",
            file=sys.stderr,
        )
        return 1

    write_install_state(final_state)
    print("\nkrakey install: done.")
    return 0
