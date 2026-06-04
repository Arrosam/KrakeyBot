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


async def test_preflight_deregisters_failed_envs_without_raising():
    """When every env's preflight raises EnvironmentUnavailableError the
    router must NOT re-raise; instead each failed env is de-registered.
    Both envs are still attempted — no early abort — confirming the loop
    runs to completion before mutating state."""
    bad_a = _FakeEnv("sandbox_a",
                     preflight_raises=EnvironmentUnavailableError("agent A down"))
    bad_b = _FakeEnv("sandbox_b",
                     preflight_raises=EnvironmentUnavailableError("agent B down"))
    r = EnvironmentRouter(
        envs={"sandbox_a": bad_a, "sandbox_b": bad_b},
        allow_list={"sandbox_a": ["p1"], "sandbox_b": ["p2"]},
    )
    # Must not raise — de-register silently (warning logged, not raised).
    infos = await r.preflight_all()
    assert infos == []

    # Both were attempted — no early abort.
    assert bad_a.preflight_calls == 1
    assert bad_b.preflight_calls == 1

    # Both envs removed from the registry.
    assert "sandbox_a" not in r.env_names()
    assert "sandbox_b" not in r.env_names()

    # for_plugin on either raises EnvironmentDenied ("no such environment"),
    # not any form of EnvironmentUnavailableError.
    for env_name, plugin in (("sandbox_a", "p1"), ("sandbox_b", "p2")):
        with pytest.raises(EnvironmentDenied) as ei:
            r.for_plugin(plugin, env_name)
        assert "no such environment" in str(ei.value).lower()


# ---- new edge tests: startup-resilience de-register behavior --------


async def test_preflight_mixed_batch_healthy_survives_failed_deregistered():
    """Positive / equivalence: one healthy env (returns info dict) and
    one failing env (raises EnvironmentUnavailableError) in the same
    preflight_all call.

    Assertions:
    - preflight_all does NOT raise
    - returned infos contains ONLY the healthy env's info
    - healthy env still in env_names() and resolvable via for_plugin
    - failed env is removed from env_names()
    - for_plugin on the failed env raises EnvironmentDenied with
      the "no such environment" message variant (not "not allow-listed")
    """
    good = _FakeEnv("local", preflight_info={"status": "ok", "latency_ms": 2})
    bad = _FakeEnv("sandbox",
                   preflight_raises=EnvironmentUnavailableError("guest down"))
    r = EnvironmentRouter(
        envs={"local": good, "sandbox": bad},
        allow_list={"local": ["tool_a"], "sandbox": ["tool_a"]},
    )

    infos = await r.preflight_all()

    # No raise — healthy co-existence.
    # Healthy info present, keyed by env name.
    assert len(infos) == 1
    assert infos[0]["env"] == "local"
    assert infos[0]["status"] == "ok"

    # Healthy env survives in the registry.
    assert "local" in r.env_names()
    assert r.for_plugin("tool_a", "local") is good

    # Failed env removed.
    assert "sandbox" not in r.env_names()

    with pytest.raises(EnvironmentDenied) as ei:
        r.for_plugin("tool_a", "sandbox")
    assert "no such environment" in str(ei.value).lower()
    assert "sandbox" in str(ei.value)


async def test_preflight_deregister_persists_across_second_call():
    """State transition: after preflight_all() de-registers a failed env,
    a second call to preflight_all() must NOT attempt to re-preflight it
    (it is gone from the registry) and must still not raise. The env
    stays absent after both calls — the mutation is permanent."""
    bad = _FakeEnv("sandbox",
                   preflight_raises=EnvironmentUnavailableError("agent down"))
    r = EnvironmentRouter(
        envs={"sandbox": bad},
        allow_list={"sandbox": ["coder"]},
    )

    # First call — de-registers.
    infos_first = await r.preflight_all()
    assert infos_first == []
    assert bad.preflight_calls == 1

    # Second call — env is gone; preflight must not be called again.
    infos_second = await r.preflight_all()
    assert infos_second == []
    assert bad.preflight_calls == 1  # still 1 — not retried

    # Env is still absent.
    assert "sandbox" not in r.env_names()


async def test_preflight_single_failing_env_makes_router_empty():
    """Boundary: single env, it fails preflight. After the call the router
    has no envs at all (is_empty() True), infos == [], no raise."""
    only_env = _FakeEnv("sandbox",
                        preflight_raises=EnvironmentUnavailableError("no agent"))
    r = EnvironmentRouter(
        envs={"sandbox": only_env},
        allow_list={"sandbox": ["p"]},
    )
    assert r.is_empty() is False  # one env before the call

    infos = await r.preflight_all()

    assert infos == []
    assert r.is_empty() is True
    assert r.env_names() == []

    with pytest.raises(EnvironmentDenied):
        r.for_plugin("p", "sandbox")


async def test_preflight_dormant_env_not_deregistered_on_failure():
    """Negative / guard: an env that has NO allow-listed plugins is dormant
    and is never preflighted. A dormant env that *would* fail preflight must
    NOT be de-registered — only envs that were actually preflighted-and-failed
    are removed. This guards against over-eager removal.

    To make the test meaningful we configure the dormant env to raise if
    preflight is ever called, then verify it was never called AND is still
    present in env_names() after preflight_all()."""
    dormant = _FakeEnv("sandbox",
                       preflight_raises=EnvironmentUnavailableError("should never fire"))
    r = EnvironmentRouter(
        envs={"sandbox": dormant},
        allow_list={"sandbox": []},  # no plugins → dormant
    )

    infos = await r.preflight_all()

    # Never preflighted.
    assert dormant.preflight_calls == 0
    # Still in the registry — not de-registered.
    assert "sandbox" in r.env_names()
    # infos empty because nothing was preflighted.
    assert infos == []


async def test_deregistered_env_gives_no_such_environment_not_not_allow_listed():
    """Confirms the EnvironmentDenied message variant after de-registration.

    A de-registered env must produce the "no such environment" message
    (unknown env code-path), NOT the "allowed_plugins" / "not allow-listed"
    message (which would mean the env record is still present but the plugin
    is blocked). The distinction matters for debuggability."""
    bad = _FakeEnv("sandbox",
                   preflight_raises=EnvironmentUnavailableError("dead"))
    r = EnvironmentRouter(
        envs={"sandbox": bad},
        allow_list={"sandbox": ["my_plugin"]},
    )
    await r.preflight_all()

    with pytest.raises(EnvironmentDenied) as ei:
        r.for_plugin("my_plugin", "sandbox")

    msg = str(ei.value).lower()
    assert "no such environment" in msg
    # Must NOT be the "allowed_plugins" variant.
    assert "allowed_plugins" not in msg
