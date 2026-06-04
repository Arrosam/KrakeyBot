"""``DockerSandboxEnvironment`` — host-side HTTP/RPC client against the
Krakey guest agent running inside a **Docker container** (Phase D).

The wire protocol is IDENTICAL to ``SandboxEnvironment`` (the QEMU
sibling): ``POST /exec`` + ``GET /health`` with an ``X-Krakey-Token``
header. ``agent.py`` is provider-agnostic — it does not know whether
its host is a QEMU VM or a Docker container — so ``preflight.py`` is
reused unchanged.

Optional ``auto_start``: when enabled, ``__init__`` probes the agent
port and, if it's down, starts the container via ``docker run`` (best
-effort, idempotent on the pinned container name). Failure is NON-fatal:
a warning is logged and startup continues; the sandbox fails preflight
shortly after and the Router de-registers it (graceful-disable, Phase
A). The lifecycle helpers mirror ``searxng_search/lifecycle.py`` and
accept injected probes so tests can drive every branch without real
Docker.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import aiohttp

from krakey.environment.sandbox.sandbox_environment import (
    SandboxConfig,
    SandboxUnavailableError,
)

_log = logging.getLogger(__name__)


@dataclass
class DockerSandboxConfig:
    """Host-side connectivity + container-launch parameters for the
    Docker-backed sandbox. Built by the env factory from
    ``environments.sandbox`` (agent block + the ``docker`` sub-block)."""
    agent_url: str
    agent_token: str
    guest_os: str
    image: str
    container_name: str
    host_port: int
    host_bind_dirs: list[str] = field(default_factory=list)
    auto_start: bool = False
    wait_seconds: float = 30.0


class DockerSandboxEnvironment:
    """Docker-backed sandbox Environment. Implements the same
    Environment Protocol as ``SandboxEnvironment`` — a true drop-in:
    same ``run`` signature, same ``preflight`` delegation."""

    name = "sandbox"

    def __init__(self, cfg: DockerSandboxConfig) -> None:
        self._cfg = cfg
        if cfg.auto_start:
            ok = ensure_container_running(cfg)
            if not ok:
                _log.warning(
                    "docker sandbox: auto_start failed for container %r; "
                    "preflight will decide whether to de-register the "
                    "sandbox env.",
                    cfg.container_name,
                )

    async def run(
        self,
        cmd: list[str],
        *,
        cwd: Path,
        timeout: float,
        stdin: str | None = None,
    ) -> tuple[int, str, str]:
        body = {
            "cmd": list(cmd),
            "cwd": str(cwd) if cwd else None,
            "timeout": timeout,
            "stdin": stdin,
        }
        headers = {"X-Krakey-Token": self._cfg.agent_token}
        url = self._cfg.agent_url.rstrip("/") + "/exec"
        timeout_obj = aiohttp.ClientTimeout(total=timeout + 10)
        try:
            async with aiohttp.ClientSession(timeout=timeout_obj) as s:
                async with s.post(url, json=body, headers=headers) as r:
                    if r.status != 200:
                        text = await r.text()
                        raise SandboxUnavailableError(
                            f"agent returned {r.status}: {text[:200]}"
                        )
                    data = await r.json()
        except aiohttp.ClientError as e:
            raise SandboxUnavailableError(
                f"agent unreachable at {self._cfg.agent_url}: {e}"
            ) from e
        except asyncio.TimeoutError as e:
            raise SandboxUnavailableError(
                f"agent timeout at {self._cfg.agent_url} (exec/timeout)"
            ) from e
        return (
            int(data["exit"]),
            str(data.get("stdout", "")),
            str(data.get("stderr", "")),
        )

    async def preflight(self) -> dict[str, Any] | None:
        # Reuse the QEMU sibling's preflight verbatim — the guest agent
        # is host-agnostic. Construct a SandboxConfig (url + token +
        # guest_os) inline; preflight.py needs nothing else.
        from krakey.environment.sandbox.preflight import preflight
        return await preflight(SandboxConfig(
            agent_url=self._cfg.agent_url,
            agent_token=self._cfg.agent_token,
            guest_os=self._cfg.guest_os,
        ))


# ---- auto-start lifecycle (mirrors searxng_search/lifecycle.py) ------
#
# Injectable helpers (defaults call real subprocess / socket) so tests
# drive every branch without real Docker. Best-effort + idempotent;
# every failure logs once at warning level and returns False — the env
# never raises out of __init__.

ProbeFn = Callable[[str, int], bool]
DockerOnPathFn = Callable[[], bool]
DockerDaemonFn = Callable[[], bool]
ContainerRunningFn = Callable[[str], bool]
RunContainerFn = Callable[["DockerSandboxConfig"], bool]
WaitForPortFn = Callable[[str, int, float], bool]


def ensure_container_running(
    cfg: DockerSandboxConfig,
    *,
    probe: ProbeFn | None = None,
    docker_on_path: DockerOnPathFn | None = None,
    docker_daemon_alive: DockerDaemonFn | None = None,
    container_running: ContainerRunningFn | None = None,
    run_container: RunContainerFn | None = None,
    wait_for_port: WaitForPortFn | None = None,
) -> bool:
    """Probe the agent port; if down, start the container via
    ``docker run`` and wait for the port. Returns True iff the port
    accepts a TCP connection when this returns.

    The Docker agent is always mapped to ``127.0.0.1:<host_port>`` (the
    host side of ``-p <host_port>:8765``), so we probe localhost
    regardless of ``agent_url`` (which may name an in-container host)."""
    probe = probe or _probe
    docker_on_path = docker_on_path or _has_docker
    docker_daemon_alive = docker_daemon_alive or _docker_daemon_alive
    container_running = container_running or _container_running
    run_container = run_container or _docker_run_detached
    wait_for_port = wait_for_port or _wait_for_port

    host = "127.0.0.1"
    port = cfg.host_port

    if probe(host, port):
        _log.info(
            "docker sandbox: agent already reachable at %s:%d",
            host, port,
        )
        return True

    if not docker_on_path():
        _log.warning(
            "docker sandbox: auto_start=true but docker is not on PATH; "
            "cannot start container %r. Install Docker or set "
            "environments.sandbox.docker.auto_start: false.",
            cfg.container_name,
        )
        return False

    if not docker_daemon_alive():
        _log.warning(
            "docker sandbox: ``docker`` is on PATH but the Docker daemon "
            "is not responding. Start Docker Desktop (Windows/macOS) or "
            "check ``systemctl status docker`` (Linux), then restart "
            "Krakey.",
        )
        return False

    if container_running(cfg.container_name):
        _log.info(
            "docker sandbox: container %r already up, waiting for port %d",
            cfg.container_name, port,
        )
        return wait_for_port(host, port, cfg.wait_seconds)

    if not run_container(cfg):
        return False

    _log.info(
        "docker sandbox: launched container %r from image %r on host "
        "port %d", cfg.container_name, cfg.image, port,
    )
    return wait_for_port(host, port, cfg.wait_seconds)


# ---- default real implementations (subprocess / socket) ------------


def _probe(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _has_docker() -> bool:
    return shutil.which("docker") is not None


def _docker_daemon_alive() -> bool:
    try:
        out = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            check=False, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0 and bool((out.stdout or "").strip())


def _container_running(name: str) -> bool:
    """True if a running container with the exact ``name`` exists.
    ``^name$`` regex filter so ``krakey-sandbox`` doesn't match
    ``krakey-sandbox-test``."""
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


def _docker_run_detached(cfg: DockerSandboxConfig) -> bool:
    """Run the agent container detached. Maps host_port → the agent's
    in-container port 8765 and bind-mounts each ``host_bind_dirs``
    entry verbatim. Pull failures, port collisions, daemon-down all
    land in the except block; stderr is surfaced so the operator can
    act."""
    args = [
        "docker", "run", "-d", "--rm",
        "--name", cfg.container_name,
        "-p", f"{cfg.host_port}:8765",
    ]
    for vol in cfg.host_bind_dirs:
        args += ["-v", vol]
    args.append(cfg.image)
    try:
        subprocess.run(
            args, check=True, capture_output=True, text=True, timeout=120,
        )
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as e:
        stderr = getattr(e, "stderr", "") or ""
        _log.warning(
            "docker sandbox: docker run failed: %s\n%s", e, stderr.strip(),
        )
        return False
    return True


def _wait_for_port(host: str, port: int, deadline_s: float) -> bool:
    end = time.time() + deadline_s
    while time.time() < end:
        if _probe(host, port):
            return True
        time.sleep(0.5)
    _log.warning(
        "docker sandbox: port %s:%d not reachable after %.1fs; preflight "
        "will de-register the sandbox env.",
        host, port, deadline_s,
    )
    return False
