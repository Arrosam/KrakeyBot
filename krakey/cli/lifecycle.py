"""Process lifecycle for the heartbeat: foreground run + daemon start/stop/status.

Pidfile lives at <repo>/workspace/.krakey.pid (atomic write via rename).
Daemon log appends to <repo>/workspace/logs/daemon.log.

`run` (foreground) writes the pidfile too, so `stop` works regardless of mode
and `start` won't double-launch on top of an active foreground run.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import _meta

_PIDFILE_REL = "workspace/.krakey.pid"
_LOG_REL = "workspace/logs/daemon.log"


def _paths() -> tuple[Path, Path, Path]:
    repo = _meta.repo_root()
    pidfile = repo / _PIDFILE_REL
    logfile = repo / _LOG_REL
    return repo, pidfile, logfile


def _write_pidfile(pidfile: Path, pid: int) -> None:
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    tmp = pidfile.with_suffix(pidfile.suffix + ".tmp")
    tmp.write_text(str(pid), encoding="utf-8")
    os.replace(tmp, pidfile)


def _read_pid(pidfile: Path) -> int | None:
    if not pidfile.exists():
        return None
    try:
        return int(pidfile.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def _is_alive(pid: int) -> bool:
    try:
        import psutil
    except ImportError:
        # Fallback: Unix only
        if os.name == "nt":
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    return psutil.pid_exists(pid) and psutil.Process(pid).is_running()


def _clear_pidfile(pidfile: Path) -> None:
    try:
        pidfile.unlink()
    except FileNotFoundError:
        pass


def _install_pidfile_cleanup(pidfile: Path) -> None:
    import atexit

    def _cleanup(*_args):
        _clear_pidfile(pidfile)
        if _args:  # called from signal handler → re-raise as clean exit
            sys.exit(0)

    atexit.register(_cleanup)
    if os.name != "nt":
        signal.signal(signal.SIGTERM, _cleanup)
        signal.signal(signal.SIGHUP, _cleanup)
    # SIGINT is handled by asyncio.run as KeyboardInterrupt anyway


def _has_chat_llm_configured(repo: Path) -> bool:
    """Cheap config peek — true iff `core_purposes.self_thinking` maps
    to a tag that has a binding. Used by the CLI to print a friendly
    "configure LLM" prompt before starting the runtime in idle mode."""
    cfg_path = repo / "config.yaml"
    if not cfg_path.exists():
        return False
    try:
        from krakey.models.config import load_config
        cfg = load_config(cfg_path)
    except Exception:  # noqa: BLE001
        return False
    tag = cfg.llm.core_purposes.get("self_thinking")
    return bool(tag) and tag in cfg.llm.tags


def _exec_runtime(repo: Path) -> int:
    """Run the heartbeat loop in this process. Blocks until exit."""
    import asyncio
    os.chdir(str(repo))
    if not _has_chat_llm_configured(repo):
        print(
            "\nkrakey: no chat LLM configured — runtime will start in "
            "IDLE mode (no heartbeat).\n"
            "        Open the dashboard's LLM tab (default "
            "http://127.0.0.1:8765) and add a provider + tag, or run\n"
            "        `krakey onboard` to redo setup. Restart krakey "
            "afterwards.\n",
            file=sys.stderr,
        )
    from krakey.main import build_runtime_from_config

    rt = build_runtime_from_config()

    # Cooperative shutdown — flip `rt._stop` on Ctrl+C / SIGTERM and let
    # the runtime exit its loop cleanly. Path differs by platform:
    #
    #   Unix: install via `loop.add_signal_handler` inside the coroutine
    #         (asyncio integrates with the signal subsystem there).
    #   Windows: `add_signal_handler` is unsupported; use `signal.signal`
    #         outside the loop. Handler sets the flag, asyncio.sleep
    #         (0.25s tick) wakes up, runtime sees `_stop=True`, exits.
    if os.name == "nt":
        def _win_request_stop(_sig, _frame):  # noqa: ARG001
            rt._stop = True
        signal.signal(signal.SIGINT, _win_request_stop)

    async def _supervised() -> None:
        if os.name != "nt":
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, lambda: setattr(rt, "_stop", True))
                except (NotImplementedError, RuntimeError):
                    pass
        try:
            await rt.run()
        finally:
            await rt.close()

    try:
        asyncio.run(_supervised())
    except KeyboardInterrupt:
        # Belt-and-suspenders: if the cooperative path didn't catch
        # Ctrl+C in time, asyncio.run raises KeyboardInterrupt out
        # of the loop. Runtime's own try/finally already cleaned up.
        pass
    return 0


# -------- public ops --------

def _ensure_config(repo: Path) -> tuple[int | None, bool]:
    """If config.yaml is missing, auto-launch onboarding and let the
    user generate one.

    Returns ``(exit_code_or_None, wizard_ran)``:
      * ``(rc, _)`` with rc != None  → caller should return rc immediately
        (wizard aborted, stdin closed, or no config written).
      * ``(None, False)`` → config existed already; caller proceeds.
      * ``(None, True)``  → wizard ran successfully; caller proceeds.
        The caller uses this flag to suppress its own banner print
        (the wizard already showed one — single-banner-per-invocation).
    """
    cfg = repo / "config.yaml"
    if cfg.exists():
        return None, False
    print("krakey: no config.yaml found — launching onboarding wizard.\n",
          file=sys.stderr)
    from krakey.onboarding import run_wizard
    try:
        run_wizard(config_path=cfg)
    except KeyboardInterrupt:
        print("\nkrakey: onboarding cancelled.", file=sys.stderr)
        return 130, False
    except EOFError:
        print("\nkrakey: onboarding ended (stdin closed).", file=sys.stderr)
        return 1, False
    if not cfg.exists():
        print("krakey: onboarding finished but no config was written; "
              "re-run `krakey onboard` when ready.", file=sys.stderr)
        return 1, False
    print()
    return None, True


def _print_runtime_banner_if_needed(wizard_ran: bool) -> None:
    """Print the KRAKEY banner before runtime startup, but only if the
    wizard didn't already print one this invocation."""
    if wizard_ran:
        return
    from . import _banner
    _banner.print_banner()


def run_foreground() -> int:
    repo, pidfile, _log = _paths()
    rc, wizard_ran = _ensure_config(repo)
    if rc is not None:
        return rc

    existing = _read_pid(pidfile)
    if existing and _is_alive(existing):
        print(f"krakey already running (pid {existing}); use `krakey stop` first",
              file=sys.stderr)
        return 1
    if existing:
        _clear_pidfile(pidfile)

    _print_runtime_banner_if_needed(wizard_ran)
    _write_pidfile(pidfile, os.getpid())
    _install_pidfile_cleanup(pidfile)
    return _exec_runtime(repo)


def start_daemon() -> int:
    repo, pidfile, logfile = _paths()
    rc, wizard_ran = _ensure_config(repo)
    if rc is not None:
        return rc

    existing = _read_pid(pidfile)
    if existing and _is_alive(existing):
        print(f"krakey already running (pid {existing})")
        return 1
    if existing:
        _clear_pidfile(pidfile)

    logfile.parent.mkdir(parents=True, exist_ok=True)

    _print_runtime_banner_if_needed(wizard_ran)
    if os.name == "nt":
        return _spawn_daemon_windows(repo, pidfile, logfile)
    return _spawn_daemon_unix(repo, pidfile, logfile)


def _spawn_daemon_unix(repo: Path, pidfile: Path, logfile: Path) -> int:
    # Double-fork to detach from controlling terminal.
    pid = os.fork()
    if pid > 0:
        # Parent: wait briefly for grandchild to write pidfile, then report.
        for _ in range(50):
            time.sleep(0.05)
            child_pid = _read_pid(pidfile)
            if child_pid and _is_alive(child_pid):
                print(f"krakey started (pid {child_pid}); log: {logfile}")
                return 0
        print("krakey: daemon failed to start within 2.5s", file=sys.stderr)
        return 1

    # First child
    os.setsid()
    pid2 = os.fork()
    if pid2 > 0:
        os._exit(0)

    # Grandchild: redirect stdio, write pidfile, run runtime
    sys.stdout.flush()
    sys.stderr.flush()
    with open(os.devnull, "rb") as devnull_r:
        os.dup2(devnull_r.fileno(), sys.stdin.fileno())
    log_fd = open(logfile, "ab", buffering=0)
    os.dup2(log_fd.fileno(), sys.stdout.fileno())
    os.dup2(log_fd.fileno(), sys.stderr.fileno())

    _write_pidfile(pidfile, os.getpid())
    _install_pidfile_cleanup(pidfile)
    _exec_runtime(repo)
    os._exit(0)


def _spawn_daemon_windows(repo: Path, pidfile: Path, logfile: Path) -> int:
    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200

    log_fh = open(logfile, "ab", buffering=0)
    proc = subprocess.Popen(
        [sys.executable, "-m", "krakey.cli.lifecycle", "--daemon-child"],
        cwd=str(repo),
        stdin=subprocess.DEVNULL,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
        close_fds=True,
    )
    # Wait for child to write its pidfile (it will, before calling _exec_runtime).
    for _ in range(50):
        time.sleep(0.05)
        cpid = _read_pid(pidfile)
        if cpid == proc.pid and _is_alive(cpid):
            print(f"krakey started (pid {cpid}); log: {logfile}")
            return 0
    print("krakey: daemon failed to start within 2.5s", file=sys.stderr)
    return 1


def stop_daemon() -> int:
    _repo, pidfile, _log = _paths()
    pid = _read_pid(pidfile)
    if pid is None:
        print("krakey: not running")
        return 1
    if not _is_alive(pid):
        _clear_pidfile(pidfile)
        print(f"krakey: not running (cleared stale pid {pid})")
        return 1

    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T"], check=False,
                       capture_output=True)
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            _clear_pidfile(pidfile)
            print("krakey: already stopped")
            return 0

    # Wait up to 10s for graceful shutdown.
    for _ in range(100):
        time.sleep(0.1)
        if not _is_alive(pid):
            _clear_pidfile(pidfile)
            print(f"krakey stopped (pid {pid})")
            return 0

    # Hard kill.
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/PID", str(pid), "/T"], check=False,
                       capture_output=True)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    _clear_pidfile(pidfile)
    print(f"krakey force-killed (pid {pid})")
    return 0


def status() -> int:
    _repo, pidfile, logfile = _paths()
    ver = _meta.version()
    pid = _read_pid(pidfile)
    if pid is None:
        print(f"krakey: stopped  (version {ver})")
        return 0
    if not _is_alive(pid):
        _clear_pidfile(pidfile)
        print(f"krakey: stopped  (version {ver}; cleared stale pid {pid})")
        return 0

    extra = ""
    try:
        import psutil
        proc = psutil.Process(pid)
        uptime = int(time.time() - proc.create_time())
        extra = f"  uptime={uptime}s"
    except Exception:
        pass

    print(f"krakey: running  pid={pid}  version={ver}{extra}")
    print(f"        log: {logfile}")
    return 0


# -------- Windows daemon-child entrypoint --------

def _daemon_child_main() -> None:
    """Invoked as `python -m src.cli.lifecycle --daemon-child` on Windows.

    Already detached by Popen flags; just write pidfile + run runtime.
    """
    repo, pidfile, _log = _paths()
    _write_pidfile(pidfile, os.getpid())
    _install_pidfile_cleanup(pidfile)
    _exec_runtime(repo)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--daemon-child":
        _daemon_child_main()
