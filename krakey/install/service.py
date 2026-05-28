"""``DefaultInstallService`` + module-level functions for the install
flow — pip + post_install + install_state.json bookkeeping.

Imported by the CLI (``krakey install`` subcommand) and the dashboard
plugin's deps panel. Not part of the heartbeat loop.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from krakey.plugin_system.loader import (
    BUILTIN_ROOT,
    WORKSPACE_ROOT,
    parse_meta,
)
from krakey.engine_system.meta_loader import (
    MetaParseError, load_slot_meta,
)
from krakey.engine_system.catalog import EngineImpl  # noqa: F401 — used for annotation
from krakey.models.config import Config, load_config


@dataclass
class InstallResult:
    """Outcome of a single install run."""
    rc: int
    stdout: str
    stderr: str


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
    """``{plugin_name: [pip-spec-strings]}``."""
    return {n: list(m.dependencies) for n, m in _walk_plugin_metas()}


def collect_plugin_post_install() -> dict[str, list[dict[str, Any]]]:
    """``{plugin_name: [{args, description, optional}]}``."""
    return {n: list(m.post_install) for n, m in _walk_plugin_metas()}


def collect_core_dependencies() -> list[str]:
    """Read the main project's pyproject.toml (when in a checkout)
    and return its ``[project].dependencies`` list. ``[]`` when
    running from a wheel install (no pyproject at repo root)."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.exists():
        return []
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
# Engine dep discovery — per-selected-engine, mirrors plugin pipeline
# =====================================================================


def _load_config_for_install() -> Config | None:
    """Tolerantly load ``config.yaml`` (relative to cwd).

    Returns a ``Config`` on success. On ``FileNotFoundError`` emits a
    stderr note and returns None. On any other exception emits a warning
    and returns None.
    """
    try:
        return load_config(Path("config.yaml"))
    except FileNotFoundError:
        print(
            "note: config.yaml not found; skipping engine dep collection "
            "(run krakey onboard first)",
            file=sys.stderr,
        )
        return None
    except Exception as e:  # noqa: BLE001
        print(
            f"warning: could not load config.yaml ({e}); skipping "
            "engine dep collection",
            file=sys.stderr,
        )
        return None


def _resolve_engine_slot(
    slot: str, selection: str,
) -> tuple[str, dict[str, "EngineImpl"]] | None:
    """Return ``(short_name, catalog)`` for a slot+selection pair, or
    None if the slot should be skipped.

    Rules:
    * ``selection`` contains ``":"`` → dotted-path override; skip (None).
    * ``selection`` is non-blank → use it as the short-name; load catalog.
    * ``selection`` is blank → load catalog, use the declared default.
    * ``FileNotFoundError`` / ``MetaParseError`` from ``load_slot_meta``
      → skip (None).
    """
    if ":" in selection:
        return None

    try:
        catalog, default_name = load_slot_meta(slot)
    except (FileNotFoundError, MetaParseError):
        return None

    short_name = selection if selection else default_name

    if short_name not in catalog:
        print(
            f"note: engine slot {slot!r}: short-name {short_name!r} "
            "not in catalog; skipping engine dep collection for this slot",
            file=sys.stderr,
        )
        return None

    return short_name, catalog


def collect_engine_dependencies(cfg: Config) -> dict[str, list[str]]:
    """``{"engine:slot:short_name": [pip-spec-strings]}`` for every
    resolved engine slot in the config.

    Slots where the user has set a dotted-path override (contains
    ``:``) are skipped — dotted-path engines may live outside the
    built-in catalog. Slots where the meta is absent or malformed are
    silently skipped. Unknown short-names are warned-and-skipped.
    """
    out: dict[str, list[str]] = {}
    for f in dataclasses.fields(cfg.core_implementations):
        slot = f.name
        selection = cfg.core_implementations.get(slot).strip()
        resolved = _resolve_engine_slot(slot, selection)
        if resolved is None:
            continue
        short_name, catalog = resolved
        impl = catalog[short_name]
        out[f"engine:{slot}:{short_name}"] = list(impl.dependencies)
    return out


def collect_engine_post_install(cfg: Config) -> dict[str, list[dict[str, Any]]]:
    """``{"engine:slot:short_name": [post_install entries]}`` for every
    resolved engine slot in the config. Parallel structure to
    ``collect_engine_dependencies``.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for f in dataclasses.fields(cfg.core_implementations):
        slot = f.name
        selection = cfg.core_implementations.get(slot).strip()
        resolved = _resolve_engine_slot(slot, selection)
        if resolved is None:
            continue
        short_name, catalog = resolved
        impl = catalog[short_name]
        out[f"engine:{slot}:{short_name}"] = list(impl.post_install)
    return out


# =====================================================================
# Hash + state file
# =====================================================================


def deps_hash(
    plugin_deps: dict[str, list[str]],
    post_install: dict[str, list[dict[str, Any]]] | None = None,
    engine_deps: dict[str, list[str]] | None = None,
    engine_post: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    """Stable sha256 over the SORTED union of declared pip deps
    AND the per-plugin/engine post_install commands (joined arg lists).
    Description is documentation, not part of install behaviour,
    so excluded from the hash.

    ``engine_deps`` and ``engine_post`` default to None / treated as
    ``{}`` so existing two-argument call sites keep producing the same
    hashes after the signature extension.
    """
    # Plugin pip specs.
    flat = sorted({d for deps in plugin_deps.values() for d in deps})
    # Engine pip specs folded into the same sorted set.
    if engine_deps:
        for deps in engine_deps.values():
            flat = sorted(set(flat) | set(deps))
    flat = sorted(set(flat))
    parts = ["\n".join(flat)]

    post_lines: list[str] = []
    if post_install:
        for plugin in sorted(post_install.keys()):
            for entry in post_install[plugin]:
                args = json.dumps(entry.get("args") or [])
                opt = bool(entry.get("optional", False))
                post_lines.append(f"{plugin}|{args}|{opt}")
    if engine_post:
        for key in sorted(engine_post.keys()):
            for entry in engine_post[key]:
                args = json.dumps(entry.get("args") or [])
                opt = bool(entry.get("optional", False))
                post_lines.append(f"{key}|{args}|{opt}")
    if post_lines:
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
    """``(pending, plugin_deps)``. Pending=True when no
    install_state.json yet OR the recorded deps_hash differs
    from the live one (including engine deps).

    The second element of the tuple stays ``plugin_deps`` to preserve
    the existing API; engine deps are folded only into the hash check.
    """
    plugin_deps = collect_plugin_dependencies()
    post_install = collect_plugin_post_install()

    cfg = _load_config_for_install()
    engine_deps: dict[str, list[str]] = {}
    engine_post: dict[str, list[dict[str, Any]]] = {}
    if cfg is not None:
        engine_deps = collect_engine_dependencies(cfg)
        engine_post = collect_engine_post_install(cfg)

    state = read_install_state()
    current = deps_hash(plugin_deps, post_install, engine_deps, engine_post)
    if state is None:
        return True, plugin_deps
    if state.get("deps_hash") != current:
        return True, plugin_deps
    return False, plugin_deps


# =====================================================================
# post_install dispatch
# =====================================================================


def expand_python_token(args: list[str]) -> list[str]:
    """Token-substitute ``{python}`` → ``sys.executable``."""
    return [sys.executable if a == "{python}" else a for a in args]


def run_post_install_for_plugin(
    plugin_name: str,
    entries: list[dict[str, Any]],
) -> tuple[int, list[str]]:
    """Run a single plugin's post_install entries. Returns
    ``(rc, errors)``. See InstallService docs for rules."""
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


# =====================================================================
# install() — core orchestration (used by CLI handler + InstallTool +
# dashboard endpoint via the DefaultInstallService Protocol wrapper)
# =====================================================================


def install(args: argparse.Namespace) -> int:
    """Discovery print → pip → post_install → state write.

    Returns the pip subprocess's exit code (0 on success). On pip
    failure or non-optional post_install failure, install_state.json
    is NOT updated. The CLI handler calls this directly; the
    dashboard goes through ``DefaultInstallService.install(...)``
    which wraps the result in a structured ``InstallResult``.
    """
    plugin_deps = collect_plugin_dependencies()
    plugin_post = collect_plugin_post_install()
    core_deps = collect_core_dependencies()

    cfg = _load_config_for_install()
    engine_deps: dict[str, list[str]] = {}
    engine_post: dict[str, list[dict[str, Any]]] = {}
    if cfg is not None:
        engine_deps = collect_engine_dependencies(cfg)
        engine_post = collect_engine_post_install(cfg)

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
    for key in sorted(set(engine_deps) | set(engine_post)):
        deps = engine_deps.get(key) or []
        post = engine_post.get(key) or []
        if not deps and not post:
            print(f"  engine {key}: (no deps, no post_install)")
            continue
        print(f"  engine {key}:")
        for d in deps:
            print(f"    - {d}")
        for entry in post:
            argv = " ".join(entry["args"])
            tag = " (optional)" if entry.get("optional") else ""
            desc = entry.get("description") or ""
            print(f"    + post_install: {argv}{tag}"
                  + (f" — {desc}" if desc else ""))

    union = sorted(
        set(core_deps)
        | {d for deps in plugin_deps.values() for d in deps}
        | {d for deps in engine_deps.values() for d in deps}
    )

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
        for key in sorted(engine_post):
            for entry in engine_post[key]:
                argv = " ".join(expand_python_token(entry["args"]))
                tag = " (optional)" if entry.get("optional") else ""
                print(f"  + post_install ({key}): {argv}{tag}")
        return 0

    final_state = {
        "deps_hash":    deps_hash(plugin_deps, plugin_post,
                                  engine_deps, engine_post),
        "installed":    sorted(plugin_deps.keys()),
        "engine_installed": sorted(engine_deps.keys()),
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

    any_post_failure = False
    for plugin_name in sorted(plugin_post.keys()):
        entries = plugin_post[plugin_name]
        if not entries:
            continue
        rc, errs = run_post_install_for_plugin(plugin_name, entries)
        if rc != 0:
            any_post_failure = True
            print(
                f"\nNote: {plugin_name} post_install failed; "
                "continuing with other plugins.",
                file=sys.stderr,
            )

    for key in sorted(engine_post.keys()):
        entries = engine_post[key]
        if not entries:
            continue
        rc, errs = run_post_install_for_plugin(key, entries)
        if rc != 0:
            any_post_failure = True
            print(
                f"\nNote: {key} post_install failed; "
                "continuing with other engines.",
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


# =====================================================================
# InstallService Protocol implementation
# =====================================================================


class DefaultInstallService:
    """Concrete implementation of ``InstallService`` Protocol.

    Methods are thin wrappers around the module-level functions
    above so existing tests + the CLI handler keep working. The
    Protocol surface is what runtime / dashboard / InstallTool
    consume; they never import this class directly.
    """

    # Constructor takes nothing today. Future variants might
    # accept an alternate workspace root or pip-cmd builder; the
    # Protocol's no-arg method signatures don't need to change.

    def has_pending_deps(self) -> tuple[bool, dict[str, list[str]]]:
        return has_pending_deps()

    def collect_plugin_dependencies(self) -> dict[str, list[str]]:
        return collect_plugin_dependencies()

    def collect_plugin_post_install(self) -> dict[str, list[dict[str, Any]]]:
        return collect_plugin_post_install()

    def deps_status(self) -> dict[str, Any]:
        plugin_deps = collect_plugin_dependencies()
        plugin_post = collect_plugin_post_install()

        cfg = _load_config_for_install()
        engine_deps: dict[str, list[str]] = {}
        engine_post: dict[str, list[dict[str, Any]]] = {}
        if cfg is not None:
            engine_deps = collect_engine_dependencies(cfg)
            engine_post = collect_engine_post_install(cfg)

        state = read_install_state() or {}
        installed_set = set(state.get("installed") or [])
        engine_installed_set = set(state.get("engine_installed") or [])
        live_hash = deps_hash(plugin_deps, plugin_post, engine_deps, engine_post)
        recorded_hash = state.get("deps_hash")

        plugins_out: dict[str, dict[str, Any]] = {}
        any_pending = False
        for name in sorted(set(plugin_deps) | set(plugin_post)):
            deps = plugin_deps.get(name) or []
            post = plugin_post.get(name) or []
            if not deps and not post:
                satisfied = True
            else:
                satisfied = (
                    name in installed_set
                    and recorded_hash == live_hash
                )
            if not satisfied:
                any_pending = True
            plugins_out[name] = {
                "dependencies": list(deps),
                "post_install": list(post),
                "installed":    name in installed_set,
                "satisfied":    satisfied,
            }

        engines_out: dict[str, dict[str, Any]] = {}
        for key in sorted(set(engine_deps) | set(engine_post)):
            deps = engine_deps.get(key) or []
            post = engine_post.get(key) or []
            if not deps and not post:
                satisfied = True
            else:
                satisfied = (
                    key in engine_installed_set
                    and recorded_hash == live_hash
                )
            if not satisfied:
                any_pending = True
            engines_out[key] = {
                "dependencies": list(deps),
                "post_install": list(post),
                "installed":    key in engine_installed_set,
                "satisfied":    satisfied,
            }

        return {
            "pending":  any_pending or recorded_hash != live_hash,
            "plugins":  plugins_out,
            "engines":  engines_out,
            "state": {
                "installed_at": state.get("installed_at"),
                "deps_hash":    recorded_hash,
                "live_hash":    live_hash,
            },
        }

    def install(
        self,
        *,
        upgrade: bool = False,
        dry_run: bool = False,
    ) -> InstallResult:
        """Run install with stdout/stderr captured into the
        result object. Used by the dashboard endpoint + the
        InstallTool — both want the captured output as data,
        not on the controlling terminal."""
        import contextlib
        import io

        out_buf = io.StringIO()
        err_buf = io.StringIO()
        try:
            with (
                contextlib.redirect_stdout(out_buf),
                contextlib.redirect_stderr(err_buf),
            ):
                rc = install(argparse.Namespace(
                    dry_run=dry_run, upgrade=upgrade,
                ))
        except Exception as e:  # noqa: BLE001
            return InstallResult(
                rc=-1,
                stdout=out_buf.getvalue(),
                stderr=err_buf.getvalue()
                       + f"\n[crash] {type(e).__name__}: {e}",
            )
        return InstallResult(
            rc=int(rc),
            stdout=out_buf.getvalue(),
            stderr=err_buf.getvalue(),
        )
