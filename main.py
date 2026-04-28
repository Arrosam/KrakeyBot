"""Root launcher: `python main.py` starts the KrakeyBot heartbeat loop.

Self-bootstrapping: if runtime deps are missing it creates `.venv`,
pip-installs `requirements.txt`, and re-launches in that interpreter.
Subsequent runs skip the install when the requirements hash is unchanged.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent
_VENV = _ROOT / ".venv"
_REQ = _ROOT / "requirements.txt"
_VENV_PY = _VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
_HASH_FILE = _VENV / ".installed-hash"


def _deps_present() -> bool:
    try:
        import aiohttp  # noqa: F401
        import aiosqlite  # noqa: F401
        import yaml  # noqa: F401
        return True
    except ImportError:
        return False


def _in_any_venv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def _req_hash() -> str:
    return hashlib.sha256(_REQ.read_bytes()).hexdigest()


def _install_into(py: Path) -> None:
    print(f"[bootstrap] installing deps with {py}", flush=True)
    subprocess.check_call([str(py), "-m", "pip", "install",
                           "-r", str(_REQ), "--disable-pip-version-check", "-q"])
    _HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
    _HASH_FILE.write_text(_req_hash(), encoding="utf-8")


def _ensure_venv() -> Path:
    if not _VENV.exists():
        print(f"[bootstrap] creating venv at {_VENV}", flush=True)
        import venv
        venv.create(_VENV, with_pip=True, clear=False)
    return _VENV_PY


def _bootstrap() -> None:
    if _deps_present():
        return

    if _in_any_venv():
        # Active venv is missing deps — install into it, then relaunch.
        print("[bootstrap] deps missing in current venv; installing...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "-r", str(_REQ),
                               "--disable-pip-version-check", "-q"])
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
        return

    py = _ensure_venv()
    existing = _HASH_FILE.read_text(encoding="utf-8").strip() if _HASH_FILE.exists() else ""
    if existing != _req_hash():
        _install_into(py)

    print(f"[bootstrap] relaunching in {_VENV}", flush=True)
    result = subprocess.run([str(py), str(Path(__file__).resolve()), *sys.argv[1:]])
    sys.exit(result.returncode)


_bootstrap()


# --- actual entry (runs once deps are guaranteed) ---
import asyncio  # noqa: E402

from krakey.main import build_runtime_from_config  # noqa: E402


if __name__ == "__main__":
    try:
        asyncio.run(build_runtime_from_config().run())
    except KeyboardInterrupt:
        pass
