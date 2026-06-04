"""Runtime pause/resume routes — edge tests.

Covers:
  GET  /api/runtime/state
  POST /api/runtime/pause
  POST /api/runtime/resume

Tests are written BEFORE any implementation and serve as acceptance
criteria.  All HTTP plumbing goes through ``create_app`` so the tests
stay decoupled from the internal wiring in routes/runtime.py.

Run from repo root:

    pytest krakey/plugins/dashboard
"""
from __future__ import annotations

import httpx
import pytest

from krakey.plugins.dashboard.app_factory import create_app


# ---------------------------------------------------------------------------
# Fake runtime stand-in
# ---------------------------------------------------------------------------


class _FakeRuntime:
    """Minimal stand-in for a Runtime object.

    Records every call so tests can assert the route layer actually
    forwarded the request — not just that the HTTP response was correct.
    """

    def __init__(
        self,
        *,
        paused: bool = False,
        pause_applied: bool = True,
        resume_applied: bool = True,
    ) -> None:
        self._paused = paused
        self._pause_applied = pause_applied
        self._resume_applied = resume_applied
        # Records the ``seconds`` arg from every request_pause() call.
        self.pause_calls: list = []
        self.resume_calls: int = 0

    @property
    def paused(self) -> bool:
        return self._paused

    def request_pause(self, seconds=None) -> bool:
        self.pause_calls.append(seconds)
        if self._pause_applied:
            self._paused = True
        return self._pause_applied

    def request_resume(self) -> bool:
        self.resume_calls += 1
        if self._resume_applied:
            self._paused = False
        return self._resume_applied


# ---------------------------------------------------------------------------
# No-op PluginsService — required kwarg but irrelevant to runtime routes
# ---------------------------------------------------------------------------


class _NoopPluginsService:
    """Satisfies the PluginsService Protocol without doing anything."""

    def report(self):
        return {"tools": [], "channels": []}

    def update_config(self, project, body):
        return {"project": project, "path": "", "config": {}}

    def deps_status(self):
        return {"pending": False, "plugins": {}, "engines": {}, "state": {}}

    def install(self, body):
        return {"rc": 0, "stdout": "", "stderr": ""}

    async def hot_reload(self):
        return {"reloaded": [], "added": [], "removed": [],
                "skipped": [], "errors": []}

    def find_stale_configs(self):
        return []

    def delete_stale_config(self, name):
        raise LookupError(name)


# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------


def _client_with_runtime(rt):
    """Build an httpx async client wired to the given runtime instance."""
    app = create_app(runtime=rt, plugins_service=_NoopPluginsService())
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _client_no_runtime():
    """Build a client where runtime is explicitly absent."""
    app = create_app(runtime=None, plugins_service=_NoopPluginsService())
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# ===========================================================================
# GET /api/runtime/state
# ===========================================================================


class TestGetRuntimeState:

    # --- positive / equivalence ---

    async def test_returns_200_when_not_paused(self):
        rt = _FakeRuntime(paused=False)
        async with _client_with_runtime(rt) as c:
            r = await c.get("/api/runtime/state")
        assert r.status_code == 200

    async def test_body_paused_false_when_running(self):
        rt = _FakeRuntime(paused=False)
        async with _client_with_runtime(rt) as c:
            r = await c.get("/api/runtime/state")
        assert r.json() == {"paused": False}

    async def test_returns_200_when_paused(self):
        rt = _FakeRuntime(paused=True)
        async with _client_with_runtime(rt) as c:
            r = await c.get("/api/runtime/state")
        assert r.status_code == 200

    async def test_body_paused_true_when_paused(self):
        rt = _FakeRuntime(paused=True)
        async with _client_with_runtime(rt) as c:
            r = await c.get("/api/runtime/state")
        assert r.json() == {"paused": True}

    async def test_body_contains_only_paused_key(self):
        """Response shape must be exactly {"paused": <bool>} — no extras."""
        rt = _FakeRuntime(paused=False)
        async with _client_with_runtime(rt) as c:
            r = await c.get("/api/runtime/state")
        assert set(r.json().keys()) == {"paused"}

    # --- negative: missing runtime ---

    async def test_503_when_runtime_is_none(self):
        async with _client_no_runtime() as c:
            r = await c.get("/api/runtime/state")
        assert r.status_code == 503

    # --- negative: wrong method ---

    async def test_405_or_404_post_to_state(self):
        rt = _FakeRuntime()
        async with _client_with_runtime(rt) as c:
            r = await c.post("/api/runtime/state")
        assert r.status_code in {404, 405}


# ===========================================================================
# POST /api/runtime/pause
# ===========================================================================


class TestPostRuntimePause:

    # --- positive / equivalence ---

    async def test_returns_200(self):
        rt = _FakeRuntime(paused=False)
        async with _client_with_runtime(rt) as c:
            r = await c.post("/api/runtime/pause")
        assert r.status_code == 200

    async def test_body_paused_true(self):
        rt = _FakeRuntime(paused=False)
        async with _client_with_runtime(rt) as c:
            r = await c.post("/api/runtime/pause")
        assert r.json()["paused"] is True

    async def test_body_applied_true_when_runtime_confirms(self):
        rt = _FakeRuntime(paused=False, pause_applied=True)
        async with _client_with_runtime(rt) as c:
            r = await c.post("/api/runtime/pause")
        assert r.json()["applied"] is True

    async def test_delegates_to_runtime_request_pause(self):
        """The route must call request_pause(), not just return a static body."""
        rt = _FakeRuntime(paused=False)
        async with _client_with_runtime(rt) as c:
            await c.post("/api/runtime/pause")
        assert len(rt.pause_calls) == 1

    async def test_request_pause_called_with_no_seconds_arg(self):
        """Spec says request_pause() is called with no args — seconds=None."""
        rt = _FakeRuntime(paused=False)
        async with _client_with_runtime(rt) as c:
            await c.post("/api/runtime/pause")
        assert rt.pause_calls == [None]

    async def test_subsequent_get_state_reflects_paused(self):
        """After a successful POST /pause the runtime is paused; GET must show it."""
        rt = _FakeRuntime(paused=False)
        async with _client_with_runtime(rt) as c:
            await c.post("/api/runtime/pause")
            r = await c.get("/api/runtime/state")
        assert r.json()["paused"] is True

    # --- boundary: applied=False from runtime ---

    async def test_body_applied_false_when_runtime_denies(self):
        rt = _FakeRuntime(paused=False, pause_applied=False)
        async with _client_with_runtime(rt) as c:
            r = await c.post("/api/runtime/pause")
        assert r.status_code == 200
        assert r.json()["applied"] is False

    async def test_runtime_still_called_when_pause_not_applied(self):
        """Route must forward the call even when the runtime says not applied."""
        rt = _FakeRuntime(paused=False, pause_applied=False)
        async with _client_with_runtime(rt) as c:
            await c.post("/api/runtime/pause")
        assert len(rt.pause_calls) == 1

    async def test_paused_reflects_runtime_state_when_not_applied(self):
        """When pause_applied=False (no pause-file configured), runtime stays
        unpaused. The response must report paused=False, not a hardcoded True."""
        rt = _FakeRuntime(paused=False, pause_applied=False)
        async with _client_with_runtime(rt) as c:
            r = await c.post("/api/runtime/pause")
        assert r.json()["paused"] is False

    # --- boundary: idempotent pause when already paused ---

    async def test_idempotent_pause_still_returns_200(self):
        rt = _FakeRuntime(paused=True)
        async with _client_with_runtime(rt) as c:
            r = await c.post("/api/runtime/pause")
        assert r.status_code == 200

    async def test_idempotent_pause_body_paused_true(self):
        rt = _FakeRuntime(paused=True)
        async with _client_with_runtime(rt) as c:
            r = await c.post("/api/runtime/pause")
        assert r.json()["paused"] is True

    async def test_idempotent_pause_forwards_second_call_to_runtime(self):
        """The endpoint must not short-circuit on pre-existing state.
        Both calls must reach request_pause()."""
        rt = _FakeRuntime(paused=False)
        async with _client_with_runtime(rt) as c:
            await c.post("/api/runtime/pause")
            await c.post("/api/runtime/pause")
        assert len(rt.pause_calls) == 2

    # --- negative: missing runtime ---

    async def test_503_when_runtime_is_none(self):
        async with _client_no_runtime() as c:
            r = await c.post("/api/runtime/pause")
        assert r.status_code == 503

    # --- negative: wrong method ---

    async def test_405_or_404_get_to_pause(self):
        rt = _FakeRuntime()
        async with _client_with_runtime(rt) as c:
            r = await c.get("/api/runtime/pause")
        assert r.status_code in {404, 405}


# ===========================================================================
# POST /api/runtime/resume
# ===========================================================================


class TestPostRuntimeResume:

    # --- positive / equivalence ---

    async def test_returns_200(self):
        rt = _FakeRuntime(paused=True)
        async with _client_with_runtime(rt) as c:
            r = await c.post("/api/runtime/resume")
        assert r.status_code == 200

    async def test_body_paused_false(self):
        rt = _FakeRuntime(paused=True)
        async with _client_with_runtime(rt) as c:
            r = await c.post("/api/runtime/resume")
        assert r.json()["paused"] is False

    async def test_body_applied_true_when_runtime_confirms(self):
        rt = _FakeRuntime(paused=True, resume_applied=True)
        async with _client_with_runtime(rt) as c:
            r = await c.post("/api/runtime/resume")
        assert r.json()["applied"] is True

    async def test_delegates_to_runtime_request_resume(self):
        """The route must call request_resume() exactly once."""
        rt = _FakeRuntime(paused=True)
        async with _client_with_runtime(rt) as c:
            await c.post("/api/runtime/resume")
        assert rt.resume_calls == 1

    async def test_subsequent_get_state_reflects_running(self):
        """After POST /resume, GET /state must return paused=False."""
        rt = _FakeRuntime(paused=True)
        async with _client_with_runtime(rt) as c:
            await c.post("/api/runtime/resume")
            r = await c.get("/api/runtime/state")
        assert r.json()["paused"] is False

    # --- boundary: applied=False from runtime ---

    async def test_body_applied_false_when_runtime_denies(self):
        rt = _FakeRuntime(paused=True, resume_applied=False)
        async with _client_with_runtime(rt) as c:
            r = await c.post("/api/runtime/resume")
        assert r.status_code == 200
        assert r.json()["applied"] is False

    async def test_runtime_still_called_when_resume_not_applied(self):
        rt = _FakeRuntime(paused=True, resume_applied=False)
        async with _client_with_runtime(rt) as c:
            await c.post("/api/runtime/resume")
        assert rt.resume_calls == 1

    async def test_paused_reflects_runtime_state_when_not_applied(self):
        """When resume_applied=False (no pause-file configured), runtime stays
        paused. The response must report paused=True, not a hardcoded False."""
        rt = _FakeRuntime(paused=True, resume_applied=False)
        async with _client_with_runtime(rt) as c:
            r = await c.post("/api/runtime/resume")
        assert r.json()["paused"] is True

    # --- boundary: idempotent resume when already running ---

    async def test_idempotent_resume_still_returns_200(self):
        rt = _FakeRuntime(paused=False)
        async with _client_with_runtime(rt) as c:
            r = await c.post("/api/runtime/resume")
        assert r.status_code == 200

    async def test_idempotent_resume_body_paused_false(self):
        rt = _FakeRuntime(paused=False)
        async with _client_with_runtime(rt) as c:
            r = await c.post("/api/runtime/resume")
        assert r.json()["paused"] is False

    async def test_idempotent_resume_forwards_second_call_to_runtime(self):
        """No short-circuit on pre-existing state — both calls reach runtime."""
        rt = _FakeRuntime(paused=True)
        async with _client_with_runtime(rt) as c:
            await c.post("/api/runtime/resume")
            await c.post("/api/runtime/resume")
        assert rt.resume_calls == 2

    # --- negative: missing runtime ---

    async def test_503_when_runtime_is_none(self):
        async with _client_no_runtime() as c:
            r = await c.post("/api/runtime/resume")
        assert r.status_code == 503

    # --- negative: wrong method ---

    async def test_405_or_404_get_to_resume(self):
        rt = _FakeRuntime()
        async with _client_with_runtime(rt) as c:
            r = await c.get("/api/runtime/resume")
        assert r.status_code in {404, 405}


# ===========================================================================
# State transition sequence
# ===========================================================================


class TestRuntimeStateTransitions:
    """Full pause → state → resume → state → pause → state cycle across
    a single client session.  Verifies the fake reflects mutations
    and that the GET /state endpoint reads current state each time."""

    async def test_full_cycle(self):
        rt = _FakeRuntime(paused=False)
        async with _client_with_runtime(rt) as c:

            # initial state
            r = await c.get("/api/runtime/state")
            assert r.json() == {"paused": False}

            # pause
            r = await c.post("/api/runtime/pause")
            assert r.json()["paused"] is True
            assert r.json()["applied"] is True

            # state after pause
            r = await c.get("/api/runtime/state")
            assert r.json() == {"paused": True}

            # resume
            r = await c.post("/api/runtime/resume")
            assert r.json()["paused"] is False
            assert r.json()["applied"] is True

            # state after resume
            r = await c.get("/api/runtime/state")
            assert r.json() == {"paused": False}

            # pause again
            r = await c.post("/api/runtime/pause")
            assert r.json()["paused"] is True

            # state after second pause
            r = await c.get("/api/runtime/state")
            assert r.json() == {"paused": True}

        # runtime received both pause calls and one resume
        assert len(rt.pause_calls) == 2
        assert rt.resume_calls == 1

    async def test_pause_resume_interleaved_applied_flags(self):
        """Verify applied flag tracks the runtime return accurately across
        mixed applied/not-applied scenarios in sequence."""
        rt_pause_denied = _FakeRuntime(paused=False, pause_applied=False)
        async with _client_with_runtime(rt_pause_denied) as c:
            r = await c.post("/api/runtime/pause")
        assert r.json()["applied"] is False
        # Runtime did NOT flip paused because pause_applied=False.
        assert rt_pause_denied.paused is False

        rt_resume_denied = _FakeRuntime(paused=True, resume_applied=False)
        async with _client_with_runtime(rt_resume_denied) as c:
            r = await c.post("/api/runtime/resume")
        assert r.json()["applied"] is False
        # Runtime did NOT flip paused because resume_applied=False.
        assert rt_resume_denied.paused is True

    async def test_state_reads_do_not_mutate_runtime(self):
        """GET /state must be a pure read — no side effects on the runtime."""
        rt = _FakeRuntime(paused=False)
        async with _client_with_runtime(rt) as c:
            for _ in range(5):
                await c.get("/api/runtime/state")
        # No calls to the mutating methods.
        assert rt.pause_calls == []
        assert rt.resume_calls == 0


# ===========================================================================
# 503 body sanity (missing runtime, all three endpoints)
# ===========================================================================


class TestMissingRuntimeAll:
    """Assert all three routes return 503 when create_app(runtime=None)."""

    async def test_state_503(self):
        async with _client_no_runtime() as c:
            r = await c.get("/api/runtime/state")
        assert r.status_code == 503

    async def test_pause_503(self):
        async with _client_no_runtime() as c:
            r = await c.post("/api/runtime/pause")
        assert r.status_code == 503

    async def test_resume_503(self):
        async with _client_no_runtime() as c:
            r = await c.post("/api/runtime/resume")
        assert r.status_code == 503

    async def test_503_body_is_parseable_json(self):
        """The 503 response must be parseable JSON so the UI can show a message."""
        async with _client_no_runtime() as c:
            r = await c.get("/api/runtime/state")
        # Must not raise — body must be valid JSON.
        body = r.json()
        assert body is not None

    async def test_health_still_200_without_runtime(self):
        """Sanity: /api/health is always-on; confirming it does not regress."""
        async with _client_no_runtime() as c:
            r = await c.get("/api/health")
        assert r.status_code == 200
