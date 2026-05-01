"""EnvironmentRouter — allow-list dispatch + preflight aggregation.

The Router is the only authority on which plugin can use which env.
Both denial paths (unknown env name, plugin not allow-listed) must
raise ``EnvironmentDenied`` distinguishably; preflight failures
must not stop the loop early; empty config must be a no-op.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from krakey.environment.router import EnvironmentRouter
from krakey.interfaces.environment import (
    EnvironmentDenied, EnvironmentUnavailableError,
)


# ---- minimal fakes ---------------------------------------------------


class _FakeEnv:
    """Skeleton Environment that records calls + lets us script
    preflight outcomes."""

    def __init__(self, name: str, *, preflight_info: dict | None = None,
                 preflight_raises: BaseException | None = None):
        self.name = name
        self._preflight_info = preflight_info
        self._preflight_raises = preflight_raises
        self.run_calls: list[tuple[list[str], Path, float]] = []
        self.preflight_calls = 0

    async def run(self, cmd: list[str], *, cwd: Path,
                    timeout: float, stdin: str | None = None
                    ) -> tuple[int, str, str]:
        self.run_calls.append((cmd, cwd, timeout))
        return 0, "", ""

    async def preflight(self) -> dict[str, Any] | None:
        self.preflight_calls += 1
        if self._preflight_raises is not None:
            raise self._preflight_raises
        return self._preflight_info


# ---- empty Router (zero-plugin invariant) ----------------------------


def test_empty_router_is_a_noop():
    """Zero envs registered → Router reports empty + asking it for
    anything raises EnvironmentDenied (lazy-call-time)."""
    r = EnvironmentRouter()
    assert r.is_empty() is True
    assert r.env_names() == []
    with pytest.raises(EnvironmentDenied):
        r.for_plugin("any_plugin", "local")


async def test_empty_router_preflight_returns_empty_list():
    r = EnvironmentRouter()
    infos = await r.preflight_all()
    assert infos == []


# ---- for_plugin: allow + deny ---------------------------------------


def test_for_plugin_returns_env_when_allow_listed():
    env = _FakeEnv("local")
    r = EnvironmentRouter(envs={"local": env},
                          allow_list={"local": ["my_plugin"]})
    got = r.for_plugin("my_plugin", "local")
    assert got is env


def test_for_plugin_denies_unknown_env():
    """Plugin asks for an env that isn't in config — distinct error
    from the not-allow-listed case so config typos are debuggable."""
    r = EnvironmentRouter(envs={"local": _FakeEnv("local")},
                          allow_list={"local": ["x"]})
    with pytest.raises(EnvironmentDenied) as ei:
        r.for_plugin("x", "sandbox")
    assert "no such environment" in str(ei.value).lower()
    assert "sandbox" in str(ei.value)


def test_for_plugin_denies_plugin_not_allow_listed():
    """Plugin asks for a configured env it isn't on the list for —
    different error message pointing at the right config field."""
    env = _FakeEnv("local")
    r = EnvironmentRouter(envs={"local": env},
                          allow_list={"local": ["other_plugin"]})
    with pytest.raises(EnvironmentDenied) as ei:
        r.for_plugin("denied_plugin", "local")
    msg = str(ei.value)
    assert "denied_plugin" in msg
    assert "local" in msg
    assert "allowed_plugins" in msg


def test_for_plugin_with_no_allow_list_at_all_denies():
    """Env registered but allow_list missing the entry entirely =
    nobody allowed (different from explicit empty list — same
    effect, but exercises the .get() fallback)."""
    r = EnvironmentRouter(envs={"local": _FakeEnv("local")})
    with pytest.raises(EnvironmentDenied):
        r.for_plugin("any", "local")


# ---- preflight_all aggregation --------------------------------------


async def test_preflight_only_runs_for_envs_with_assigned_plugins():
    """An env with empty allowed_plugins is dormant — never preflighted.
    Saves a network call when the user configured `sandbox` but
    hasn't actually opted any plugin into it yet."""
    sandbox = _FakeEnv("sandbox", preflight_info={"agent_version": "1"})
    local = _FakeEnv("local")
    r = EnvironmentRouter(
        envs={"local": local, "sandbox": sandbox},
        allow_list={"local": ["a"], "sandbox": []},  # sandbox dormant
    )
    infos = await r.preflight_all()
    assert sandbox.preflight_calls == 0  # dormant
    assert local.preflight_calls == 1
    # Local returns None → not in infos (only non-None aggregated)
    assert infos == []


async def test_preflight_aggregates_non_none_payloads():
    sandbox = _FakeEnv("sandbox", preflight_info={"agent_version": "1",
                                                   "guest_os": "linux"})
    r = EnvironmentRouter(
        envs={"sandbox": sandbox},
        allow_list={"sandbox": ["coding"]},
    )
    infos = await r.preflight_all()
    assert len(infos) == 1
    assert infos[0]["env"] == "sandbox"
    assert infos[0]["guest_os"] == "linux"


async def test_preflight_collects_all_failures_then_raises_once():
    """One env's preflight failure must not abort the loop; the
    Router collects all failures and raises a single summary so
    the user fixes everything at once."""
    bad_a = _FakeEnv("sandbox_a",
                     preflight_raises=EnvironmentUnavailableError("agent A down"))
    bad_b = _FakeEnv("sandbox_b",
                     preflight_raises=EnvironmentUnavailableError("agent B down"))
    r = EnvironmentRouter(
        envs={"sandbox_a": bad_a, "sandbox_b": bad_b},
        allow_list={"sandbox_a": ["p1"], "sandbox_b": ["p2"]},
    )
    with pytest.raises(EnvironmentUnavailableError) as ei:
        await r.preflight_all()
    msg = str(ei.value)
    # Both envs surfaced, not just the first
    assert "sandbox_a" in msg and "sandbox_b" in msg
    assert "agent A down" in msg and "agent B down" in msg
    # Both were attempted — no early abort
    assert bad_a.preflight_calls == 1
    assert bad_b.preflight_calls == 1
