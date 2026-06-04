"""Edge test: Runtime._preflight_environments() defensive guard.

Behavior under test:
  _preflight_environments() wraps `environment_router.preflight_all()`
  in try/except EnvironmentUnavailableError. On that exception it must
  NOT propagate — instead it calls self.log.hb_warn with a message that
  contains BOTH "unexpectedly" and the original exception text, then
  returns normally.

Technique applied: error-guessing / negative (primary) +
                   equivalence-partitioning positive (secondary).
"""
from __future__ import annotations

from krakey.interfaces.environment import EnvironmentUnavailableError
from tests._runtime_helpers import ScriptedLLM, build_runtime_with_fakes


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

class _RaisingRouter:
    """Fake environment_router whose preflight_all always raises."""

    def __init__(self, exc: Exception):
        self._exc = exc

    async def preflight_all(self):
        raise self._exc

    def env_status(self):
        # Real Router exposes this; runtime now reads it post-preflight to
        # publish an EnvironmentStatusEvent. Empty dict keeps the stub
        # non-crashing without inventing status data.
        return {}


class _SucceedingRouter:
    """Fake environment_router whose preflight_all returns one info dict."""

    async def preflight_all(self):
        return [{"env": "local"}]

    def env_status(self):
        return {"local": ("ok", "preflight passed")}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_preflight_environments_swallows_unavailable_error():
    """NEGATIVE / error-guessing.

    When environment_router.preflight_all() raises
    EnvironmentUnavailableError, _preflight_environments() must:
      1. NOT propagate the exception (call simply returns).
      2. Call hb_warn exactly once.
      3. The warning message contains the literal substring "unexpectedly".
      4. The warning message contains the exception's own text ("agent down").
    """
    runtime = build_runtime_with_fakes(self_llm=ScriptedLLM([]))

    # Inject a router that always raises EnvironmentUnavailableError.
    exc = EnvironmentUnavailableError("agent down")
    runtime.environment_router = _RaisingRouter(exc)

    # Capture hb_warn calls without affecting other logger behaviour.
    captured: list[str] = []
    runtime.log.hb_warn = lambda msg: captured.append(msg)

    # Primary assertion: the call must NOT raise — completing is the proof.
    await runtime._preflight_environments()

    # Exactly one warning must have been issued.
    assert len(captured) == 1, (
        f"expected exactly 1 hb_warn call, got {len(captured)}: {captured!r}"
    )

    warning = captured[0]

    assert "unexpectedly" in warning, (
        f"warning must contain 'unexpectedly', got: {warning!r}"
    )
    assert "agent down" in warning, (
        f"warning must include the original exception text 'agent down', "
        f"got: {warning!r}"
    )


async def test_preflight_environments_no_warn_on_success():
    """POSITIVE / equivalence partitioning.

    When environment_router.preflight_all() returns normally (no raise),
    _preflight_environments() must complete without calling hb_warn.
    """
    runtime = build_runtime_with_fakes(self_llm=ScriptedLLM([]))

    runtime.environment_router = _SucceedingRouter()

    warn_calls: list[str] = []
    runtime.log.hb_warn = lambda msg: warn_calls.append(msg)

    await runtime._preflight_environments()

    assert warn_calls == [], (
        f"hb_warn must NOT be called on a successful preflight, "
        f"but was called with: {warn_calls!r}"
    )
