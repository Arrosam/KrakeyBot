"""Optional auto-start for a local SearXNG Docker container.

Best-effort + idempotent:

  * If the configured URL already accepts a TCP connection, do
    nothing — an instance is already running (this plugin started it
    on a previous run, or the operator runs SearXNG outside Krakey).
  * Otherwise verify Docker is on PATH AND the daemon is responding
    (Docker Desktop must be running on Windows). Either check failing
    is logged at warning level + we return False; the tool then
    surfaces a clear connection-refused per call.
  * Finally, run the named container (pinned name → re-runs reuse
    the existing container instead of pile-launching new ones).

This module is invoked from the plugin factory (``build_tool``) only
when ``auto_start: true``. The factory's import is local so tests
don't drag this in (and so a missing ``docker`` binary doesn't crash
plugin import).

No teardown hook: a SearXNG container the plugin starts persists
across Krakey restarts, which is usually what the operator wants
(SearXNG keeps engine quotas + cache state across runs). Stop it
manually with ``docker stop <container_name>`` when you're done.

Testability
-----------
``ensure_instance_running`` accepts injected probe / runner helpers
so tests can drive every branch without real Docker. Production
callers leave the helpers at their default (real subprocess /
sockets); tests override.
"""
from __future__ import annotations

import logging
import secrets
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse


_log = logging.getLogger(__name__)


# Where the plugin keeps its SearXNG settings + a generated secret
# key (relative to Krakey's CWD). Mounted into the container at
# ``/etc/searxng:rw`` so the JSON-format toggle (required for the
# tool's HTTP API) actually takes effect.
SETTINGS_DIR = Path("workspace") / "data" / "searxng"

# Minimal SearXNG settings.yml. ``use_default_settings: true`` keeps
# the upstream defaults for everything we don't override; we only
# flip the two knobs the plugin needs:
#   * ``search.formats`` includes ``json`` so the tool can fetch
#     parseable output (default image only enables ``html``).
#   * ``server.limiter`` off — single-tenant local instance, no
#     reason to throttle ourselves.
# The secret_key placeholder is replaced with random hex on first
# setup so each installation has its own.
_SETTINGS_TEMPLATE = """\
use_default_settings: true
server:
  secret_key: "{secret}"
  limiter: false
search:
  formats:
    - html
    - json
"""


# Type aliases for the injected helpers. Kept readable so the
# orchestrator's signature still parses at a glance.
ProbeFn = Callable[[str, int], bool]
DockerOnPathFn = Callable[[], bool]
DockerDaemonFn = Callable[[], bool]
ContainerRunningFn = Callable[[str], bool]
RunContainerFn = Callable[[str, str, int], bool]
WaitForPortFn = Callable[[str, int, float], bool]


def ensure_instance_running(
    *,
    instance_url: str,
    docker_image: str,
    container_name: str,
    host_port: int,
    wait_seconds: float = 30.0,
    # Injectable probes (defaults call real subprocess / socket).
    probe: ProbeFn | None = None,
    docker_on_path: DockerOnPathFn | None = None,
    docker_daemon_alive: DockerDaemonFn | None = None,
    container_running: ContainerRunningFn | None = None,
    run_container: RunContainerFn | None = None,
    wait_for_port: WaitForPortFn | None = None,
) -> bool:
    """Probe URL; if down, start the container; wait for the port.

    Returns True if the instance is reachable when this returns,
    False if any step failed (every failure logs once at warning
    level so the operator can see what went wrong without the
    runtime tearing down).
    """
    probe = probe or _probe
    docker_on_path = docker_on_path or _has_docker
    docker_daemon_alive = docker_daemon_alive or _docker_daemon_alive
    container_running = container_running or _container_running
    run_container = run_container or _docker_run_detached
    wait_for_port = wait_for_port or _wait_for_port

    parsed = urlparse(instance_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or host_port

    if probe(host, port):
        _log.info(
            "searxng_search auto_start: instance already up at %s",
            instance_url,
        )
        return True

    if not docker_on_path():
        _log.warning(
            "searxng_search auto_start: docker not on PATH; cannot "
            "start a local instance. Either install Docker, run "
            "SearXNG yourself, or set auto_start: false and point "
            "instance_url at an existing instance.",
        )
        return False

    if not docker_daemon_alive():
        _log.warning(
            "searxng_search auto_start: ``docker`` is on PATH but "
            "the Docker daemon is not responding. On Windows, start "
            "Docker Desktop and wait for it to finish booting; on "
            "Linux check ``systemctl status docker``. Then restart "
            "Krakey (or wait — the next tool call will surface a "
            "clean connection error you can act on).",
        )
        return False

    if container_running(container_name):
        # Container exists but the port isn't accepting yet — likely
        # mid-boot. Just wait, don't try to spawn a duplicate.
        _log.info(
            "searxng_search auto_start: container %r already up, "
            "waiting for port %d", container_name, port,
        )
        return wait_for_port(host, port, wait_seconds)

    if not run_container(docker_image, container_name, host_port):
        return False

    _log.info(
        "searxng_search auto_start: launched container %r from %r "
        "on host port %d", container_name, docker_image, host_port,
    )
    return wait_for_port(host, port, wait_seconds)


# ---- default real implementations (subprocess / socket) ------------


def _probe(host: str, port: int, timeout: float = 0.5) -> bool:
    """TCP connect probe. We don't bother with an HTTP request — a
    successful connect means SearXNG's listener is up, and that's
    enough to let the tool issue real queries (which carry their
    own per-call timeout / error handling)."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _has_docker() -> bool:
    return shutil.which("docker") is not None


def _docker_daemon_alive() -> bool:
    """``docker info`` succeeds → daemon is up. ``--format`` keeps
    the output tiny (just the server version string) so the probe
    is fast even on a busy daemon."""
    try:
        out = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            check=False, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0 and bool((out.stdout or "").strip())


def _container_running(name: str) -> bool:
    """True if a running container with the exact ``name`` is found.
    Filter uses ``^name$`` regex so ``krakey-searxng`` doesn't match
    ``krakey-searxng-test``."""
    try:
        out = subprocess.run(
            [
                "docker", "ps", "--filter", f"name=^{name}$",
                "--format", "{{.Names}}",
            ],
            check=False, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return name in (out.stdout or "").splitlines()


def _docker_run_detached(
    docker_image: str, container_name: str, host_port: int,
) -> bool:
    """Run the container detached. Mounts the plugin-managed
    settings dir (creating it on first run) so JSON output is
    enabled — the tool's HTTP backend would otherwise get 403 on
    every call. Pull failures, port collisions, daemon-down all
    land in the except block; we surface stderr so the operator
    can act."""
    settings_dir_abs = ensure_settings_dir(SETTINGS_DIR).resolve()
    args = [
        "docker", "run", "-d", "--rm",
        "--name", container_name,
        "-p", f"{host_port}:8080",
        "-v", f"{settings_dir_abs}:/etc/searxng:rw",
        docker_image,
    ]
    try:
        subprocess.run(
            args, check=True, capture_output=True, text=True,
            timeout=60,
        )
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as e:
        stderr = getattr(e, "stderr", "") or ""
        _log.warning(
            "searxng_search auto_start: docker run failed: %s\n%s",
            e, stderr.strip(),
        )
        return False
    return True


def ensure_settings_dir(settings_dir: Path | str) -> Path:
    """Create ``<settings_dir>`` + write a minimal ``settings.yml``
    (with a freshly-generated secret_key) if one isn't already
    there. Returns the directory path so the caller can mount it.

    Idempotent: if the user has hand-edited settings.yml we leave
    it alone — the file is the source of truth once it exists. The
    plugin's only contract is "JSON must be enabled"; the operator
    is free to add categories / engines / theming on top.
    """
    d = Path(settings_dir)
    d.mkdir(parents=True, exist_ok=True)
    settings_path = d / "settings.yml"
    if settings_path.exists():
        return d
    settings_path.write_text(
        _SETTINGS_TEMPLATE.format(secret=secrets.token_hex(32)),
        encoding="utf-8",
    )
    _log.info(
        "searxng_search auto_start: wrote default %s (JSON output "
        "enabled, fresh secret_key)", settings_path,
    )
    return d


def _wait_for_port(
    host: str, port: int, deadline_s: float,
) -> bool:
    end = time.time() + deadline_s
    while time.time() < end:
        if _probe(host, port):
            return True
        time.sleep(0.5)
    _log.warning(
        "searxng_search auto_start: port %s:%d not reachable after "
        "%.1fs; tool will return per-call connection errors until "
        "the instance is ready.",
        host, port, deadline_s,
    )
    return False
