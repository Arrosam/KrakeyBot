"""Environment composition factory — the single entry point the
runtime composition root uses to build an ``EnvironmentRouter`` from
config.

This keeps the environment subsystem self-contained: runtime imports
``build_environment_router`` and nothing else from this package — never
the concrete env classes (``LocalEnvironment`` / ``SandboxEnvironment``
/ ``SandboxConfig``). All construction, the sandbox config-completeness
check, the agent-token auto-generation + persistence, and the
diagnostic status seeding live here.

Startup is NEVER blocked. Partial sandbox config, a token-write
failure, or an unknown provider all degrade to "sandbox disabled +
warning" — local-only operation always survives.
"""
from __future__ import annotations

import logging
import secrets
from pathlib import Path
from typing import Callable

import yaml

from krakey.environment.local import LocalEnvironment
from krakey.environment.router import EnvironmentRouter
from krakey.environment.sandbox import SandboxConfig, SandboxEnvironment
from krakey.interfaces.environment import Environment
from krakey.models.config import Config

_log = logging.getLogger(__name__)


def build_environment_router(
    config: Config,
    *,
    config_path: str | None = None,
    log_warn: Callable[[str], None] | None = None,
) -> EnvironmentRouter:
    """Compose Local + Sandbox-if-configured into a Router whose
    allow-list comes straight from ``config.environments``.

    Local is always registered — it's zero-config and never fails to
    start. Its allow-list is whatever the user put in
    ``environments.local.allowed_plugins`` (default empty).

    Sandbox is registered only when ``environments.sandbox`` is set AND
    fully configured. Partial config (missing guest_os / agent.url /
    agent.token) is NOT fatal: the sandbox env is left unregistered —
    treated as "feature not enabled" — and a warning names the missing
    keys. Startup must never be blocked by incomplete optional-feature
    config; plugins allow-listed for the (now absent) sandbox simply
    get ``EnvironmentDenied`` at call time, same as if the section were
    omitted entirely.

    ``config_path`` is used only by the token-persist path; ``None`` is
    valid and skips auto-generation. ``log_warn`` receives every
    warning string; defaults to this module's stdlib logger when None
    (so the environment node never depends on the runtime logger).
    """
    _warn = log_warn if log_warn is not None else _log.warning

    envs: dict[str, Environment] = {"local": LocalEnvironment()}
    envs_cfg = config.environments
    allow_list: dict[str, list[str]] = {
        "local": list(envs_cfg.local.allowed_plugins),
    }
    sb = envs_cfg.sandbox
    if sb is not None:
        missing: list[str] = []
        if not sb.guest_os:
            missing.append("environments.sandbox.guest_os")
        if not sb.agent.url:
            missing.append("environments.sandbox.agent.url")
        if not sb.agent.token:
            if not missing and config_path:
                # guest_os + agent.url are present (missing still empty) and we
                # have a writable config file -> auto-generate a shared-secret
                # token, persist it, and enable the sandbox. Opt-in is preserved
                # because we only reach here when an environments.sandbox block
                # exists with the other required fields set.
                try:
                    token = secrets.token_hex(32)
                    _persist_sandbox_token(config_path, token)
                    sb.agent.token = token
                    _warn(
                        "sandbox: generated agent.token and saved it to config.yaml. "
                        "Provision the guest VM with the SAME token, then restart krakey to enable the sandbox."
                    )
                    # Don't register sandbox THIS run: the guest cannot yet have
                    # the freshly-generated token, so preflight would waste time
                    # timing out. Next startup, the token is on disk → normal
                    # register-and-preflight path runs.
                    missing.append("environments.sandbox.agent.token")
                except Exception as e:  # noqa: BLE001 - write must never crash startup
                    _warn(
                        f"sandbox: failed to persist generated agent.token "
                        f"({e}); sandbox disabled. Set environments.sandbox.agent.token manually."
                    )
                    missing.append("environments.sandbox.agent.token")
            else:
                if config_path is None and not missing:
                    _warn(
                        "sandbox: agent.token is empty and no config_path is available; "
                        "cannot auto-generate. Set environments.sandbox.agent.token manually."
                    )
                missing.append("environments.sandbox.agent.token")
        if missing:
            _warn(
                "sandbox env config is incomplete; missing "
                + ", ".join(missing)
                + ". Sandbox environment disabled. Complete the "
                "`environments.sandbox:` block in config.yaml to "
                "enable it, or remove the section to silence this."
            )
        else:
            envs["sandbox"] = SandboxEnvironment(SandboxConfig(
                agent_url=sb.agent.url,
                agent_token=sb.agent.token,
                guest_os=sb.guest_os,
            ))
            allow_list["sandbox"] = list(sb.allowed_plugins)
    router = EnvironmentRouter(envs=envs, allow_list=allow_list)
    # Seed the diagnostic side-table BEFORE preflight runs so the
    # dashboard / Self's tool feedback can distinguish "unconfigured"
    # (we never built the env) from "unreachable"/"token_mismatch"
    # (preflight failed) when the dropped env later gets queried.
    # Local has no preflight semantics — mark it ok up front. If
    # preflight_all later runs against it, it overwrites with the
    # post-preflight status.
    router.record_status("local", "ok", "no preflight needed")
    if sb is not None and missing:
        router.record_status(
            "sandbox", "unconfigured",
            "missing fields: " + ", ".join(missing),
        )
    return router


def _persist_sandbox_token(config_path: str, token: str) -> None:
    """Persist a generated sandbox agent token into config.yaml under
    environments.sandbox.agent.token via a PyYAML round-trip. Raises on
    a missing/odd structure or any I/O error (the caller treats failure
    as non-fatal and leaves the sandbox disabled). NOTE: PyYAML safe_dump
    does not preserve comments — consistent with the dashboard's existing
    config-save path."""
    path = Path(config_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data["environments"]["sandbox"]["agent"]["token"] = token
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
