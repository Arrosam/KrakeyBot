"""Edge tests for the NEW engine per-impl config endpoints.

GET  /api/engines/{slot}/{impl}/config
POST /api/engines/{slot}/{impl}/config

These tests are written BEFORE the implementation and define acceptance
criteria.  They will remain RED until the routes are implemented.

Workspace isolation strategy
-----------------------------
The routes resolve config_path from the slot meta and then read/write
under a workspace-relative path (``data/engines/<slot>/<impl>.yaml``).
``monkeypatch.chdir(tmp_path)`` re-roots the working directory so all
file I/O lands under the pytest temp directory instead of the repo's
own ``workspace/``.  Absolute filesystem paths are NOT asserted; only
the returned relative ``path`` field and the API round-trip are
verified.

Real slot/impl pairs used (sourced from krakey/engines/*/meta.yaml):
  - memory  / graph_memory  → data/engines/memory/graph_memory.yaml
  - decision / hypothalamus → data/engines/decision/hypothalamus.yaml
  - reranker / passthrough  → data/engines/reranker/passthrough.yaml

Unknown examples for 404 testing:
  - slot ``nosuchslot`` (any impl)
  - slot ``memory``, impl ``nosuchimpl``
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from krakey.plugins.dashboard.app_factory import create_app


# ---------------------------------------------------------------------------
# Client helper — mirrors _client() in test_dashboard_settings.py
# ---------------------------------------------------------------------------

def _client(tmp_path: Path):
    """Build an httpx async client for the dashboard app.

    config_path is given a throwaway YAML so the settings route
    doesn't 503; it is irrelevant to the engine-config endpoints
    under test.  monkeypatch.chdir(tmp_path) must be called by the
    test BEFORE creating the client so relative path resolution lands
    under tmp_path.
    """
    cfg = tmp_path / "config.yaml"
    cfg.write_text("a: 1\n", encoding="utf-8")
    app = create_app(config_path=cfg)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# ===========================================================================
# GET /api/engines/{slot}/{impl}/config
# ===========================================================================


class TestGetEngineConfig:
    """Positive, BVA, and negative tests for the GET endpoint."""

    # -----------------------------------------------------------------------
    # Positive — valid inputs / initial state
    # -----------------------------------------------------------------------

    async def test_get_missing_file_returns_200(self, tmp_path, monkeypatch):
        """File does not exist yet — valid initial state → 200."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.get("/api/engines/memory/graph_memory/config")
        assert r.status_code == 200

    async def test_get_missing_file_returns_empty_config(self, tmp_path, monkeypatch):
        """File absent → config must be empty dict, not null/absent."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.get("/api/engines/memory/graph_memory/config")
        assert r.json()["config"] == {}

    async def test_get_missing_file_returns_correct_path(self, tmp_path, monkeypatch):
        """Returned path is the workspace-relative config_path from meta."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.get("/api/engines/memory/graph_memory/config")
        assert r.json()["path"] == "data/engines/memory/graph_memory.yaml"

    async def test_get_response_has_path_and_config_keys(self, tmp_path, monkeypatch):
        """Response shape: exactly {path, config} on initial state."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.get("/api/engines/memory/graph_memory/config")
        body = r.json()
        assert "path" in body
        assert "config" in body

    async def test_get_decision_hypothalamus_initial_state(self, tmp_path, monkeypatch):
        """decision/hypothalamus slot resolves to its own config_path."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.get("/api/engines/decision/hypothalamus/config")
        assert r.status_code == 200
        body = r.json()
        assert body["path"] == "data/engines/decision/hypothalamus.yaml"
        assert body["config"] == {}

    async def test_get_reranker_passthrough_initial_state(self, tmp_path, monkeypatch):
        """reranker/passthrough slot resolves to its own config_path."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.get("/api/engines/reranker/passthrough/config")
        assert r.status_code == 200
        body = r.json()
        assert body["path"] == "data/engines/reranker/passthrough.yaml"
        assert body["config"] == {}

    async def test_get_after_post_returns_written_values(self, tmp_path, monkeypatch):
        """After a successful POST the GET reflects the written data."""
        monkeypatch.chdir(tmp_path)
        payload = {"config": {"key": "val", "num": 42}}
        async with _client(tmp_path) as c:
            await c.post("/api/engines/memory/graph_memory/config", json=payload)
            r = await c.get("/api/engines/memory/graph_memory/config")
        assert r.status_code == 200
        assert r.json()["config"] == {"key": "val", "num": 42}

    # -----------------------------------------------------------------------
    # BVA — boundary values for path/config
    # -----------------------------------------------------------------------

    async def test_get_path_field_is_forward_slash_separated(self, tmp_path, monkeypatch):
        """Path field uses forward slashes (POSIX) regardless of OS."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.get("/api/engines/memory/graph_memory/config")
        assert "/" in r.json()["path"]
        assert "\\" not in r.json()["path"]

    async def test_get_config_is_dict_not_list_or_null(self, tmp_path, monkeypatch):
        """config value is always a mapping, never null or a list."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.get("/api/engines/memory/graph_memory/config")
        assert isinstance(r.json()["config"], dict)

    async def test_get_path_is_string(self, tmp_path, monkeypatch):
        """path field must be a string."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.get("/api/engines/memory/graph_memory/config")
        assert isinstance(r.json()["path"], str)

    async def test_get_after_empty_post_returns_empty_dict(self, tmp_path, monkeypatch):
        """POSTing {} is valid; GET after must return {} not absent/null."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            await c.post(
                "/api/engines/memory/graph_memory/config",
                json={"config": {}},
            )
            r = await c.get("/api/engines/memory/graph_memory/config")
        assert r.status_code == 200
        assert r.json()["config"] == {}

    # -----------------------------------------------------------------------
    # Negative — 404 for unknown slot/impl
    # -----------------------------------------------------------------------

    async def test_get_unknown_slot_returns_404(self, tmp_path, monkeypatch):
        """Slot that doesn't exist in any meta.yaml → 404."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.get("/api/engines/nosuchslot/x/config")
        assert r.status_code == 404

    async def test_get_known_slot_unknown_impl_returns_404(self, tmp_path, monkeypatch):
        """Slot is valid but impl is not listed in that slot's meta → 404."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.get("/api/engines/memory/nosuchimpl/config")
        assert r.status_code == 404

    async def test_get_unknown_slot_body_is_json(self, tmp_path, monkeypatch):
        """404 response must be valid JSON (so UI can show a message)."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.get("/api/engines/nosuchslot/x/config")
        body = r.json()
        assert body is not None

    async def test_get_known_slot_unknown_impl_body_is_json(self, tmp_path, monkeypatch):
        """404 response for unknown impl must be parseable JSON."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.get("/api/engines/memory/nosuchimpl/config")
        body = r.json()
        assert body is not None


# ===========================================================================
# POST /api/engines/{slot}/{impl}/config
# ===========================================================================


class TestPostEngineConfig:
    """Positive, BVA, state-transition, and negative tests for the POST endpoint."""

    # -----------------------------------------------------------------------
    # Positive — valid inputs
    # -----------------------------------------------------------------------

    async def test_post_valid_config_returns_200(self, tmp_path, monkeypatch):
        """POST a dict config → 200."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.post(
                "/api/engines/memory/graph_memory/config",
                json={"config": {"threshold": 0.5}},
            )
        assert r.status_code == 200

    async def test_post_response_status_is_saved(self, tmp_path, monkeypatch):
        """Response body must include status=saved."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.post(
                "/api/engines/memory/graph_memory/config",
                json={"config": {"threshold": 0.5}},
            )
        assert r.json()["status"] == "saved"

    async def test_post_response_restart_required_true(self, tmp_path, monkeypatch):
        """restart_required must be True — changing engine config needs a restart."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.post(
                "/api/engines/memory/graph_memory/config",
                json={"config": {"threshold": 0.5}},
            )
        assert r.json()["restart_required"] is True

    async def test_post_response_path_matches_meta(self, tmp_path, monkeypatch):
        """POST response path is the workspace-relative config_path from meta."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.post(
                "/api/engines/memory/graph_memory/config",
                json={"config": {"threshold": 0.5}},
            )
        assert r.json()["path"] == "data/engines/memory/graph_memory.yaml"

    async def test_post_response_has_required_keys(self, tmp_path, monkeypatch):
        """Response shape: {status, path, restart_required} all present."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.post(
                "/api/engines/memory/graph_memory/config",
                json={"config": {"x": 1}},
            )
        body = r.json()
        assert "status" in body
        assert "path" in body
        assert "restart_required" in body

    async def test_post_decision_hypothalamus(self, tmp_path, monkeypatch):
        """decision/hypothalamus slot can be written via POST."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.post(
                "/api/engines/decision/hypothalamus/config",
                json={"config": {"timeout": 10}},
            )
        assert r.status_code == 200
        assert r.json()["path"] == "data/engines/decision/hypothalamus.yaml"

    async def test_post_reranker_passthrough(self, tmp_path, monkeypatch):
        """reranker/passthrough slot can be written via POST."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.post(
                "/api/engines/reranker/passthrough/config",
                json={"config": {"enabled": True}},
            )
        assert r.status_code == 200
        assert r.json()["path"] == "data/engines/reranker/passthrough.yaml"

    # -----------------------------------------------------------------------
    # BVA — boundary values for config dict
    # -----------------------------------------------------------------------

    async def test_post_empty_config_dict_is_valid(self, tmp_path, monkeypatch):
        """POST {"config": {}} is valid — empty mapping is a legal config."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.post(
                "/api/engines/memory/graph_memory/config",
                json={"config": {}},
            )
        assert r.status_code == 200
        assert r.json()["status"] == "saved"

    async def test_post_empty_config_get_returns_empty_dict(self, tmp_path, monkeypatch):
        """After POSTing {}, GET must return config={}."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            await c.post(
                "/api/engines/memory/graph_memory/config",
                json={"config": {}},
            )
            r = await c.get("/api/engines/memory/graph_memory/config")
        assert r.json()["config"] == {}

    async def test_post_nested_dict_round_trips(self, tmp_path, monkeypatch):
        """Deeply nested dicts survive the write/read cycle."""
        monkeypatch.chdir(tmp_path)
        nested = {"outer": {"inner": {"deep": 99}}}
        async with _client(tmp_path) as c:
            await c.post(
                "/api/engines/memory/graph_memory/config",
                json={"config": nested},
            )
            r = await c.get("/api/engines/memory/graph_memory/config")
        assert r.json()["config"] == nested

    async def test_post_unicode_values_round_trip(self, tmp_path, monkeypatch):
        """Unicode strings (CJK, emoji, accented) survive the round-trip."""
        monkeypatch.chdir(tmp_path)
        payload = {"config": {"label": "你好", "note": "résumé", "emoji": "🧠"}}
        async with _client(tmp_path) as c:
            await c.post(
                "/api/engines/memory/graph_memory/config",
                json=payload,
            )
            r = await c.get("/api/engines/memory/graph_memory/config")
        cfg = r.json()["config"]
        assert cfg["label"] == "你好"
        assert cfg["note"] == "résumé"
        assert cfg["emoji"] == "🧠"

    async def test_post_mixed_value_types_round_trip(self, tmp_path, monkeypatch):
        """Config with int/float/bool/list values round-trips correctly."""
        monkeypatch.chdir(tmp_path)
        payload = {
            "config": {
                "count": 5,
                "ratio": 0.75,
                "enabled": False,
                "tags": ["a", "b", "c"],
            }
        }
        async with _client(tmp_path) as c:
            await c.post(
                "/api/engines/memory/graph_memory/config",
                json=payload,
            )
            r = await c.get("/api/engines/memory/graph_memory/config")
        assert r.json()["config"] == payload["config"]

    # -----------------------------------------------------------------------
    # State transitions
    # -----------------------------------------------------------------------

    async def test_post_then_get_round_trip(self, tmp_path, monkeypatch):
        """POST a config dict; GET must return exactly that dict."""
        monkeypatch.chdir(tmp_path)
        written = {"sleep_threshold": 200, "recall_limit": 50}
        async with _client(tmp_path) as c:
            post_r = await c.post(
                "/api/engines/memory/graph_memory/config",
                json={"config": written},
            )
            assert post_r.status_code == 200
            get_r = await c.get("/api/engines/memory/graph_memory/config")
        assert get_r.json()["config"] == written

    async def test_second_post_overwrites_first(self, tmp_path, monkeypatch):
        """Two successive POSTs — GET reflects only the LATEST dict."""
        monkeypatch.chdir(tmp_path)
        first = {"version": 1, "alpha": True}
        second = {"version": 2, "beta": "yes"}
        async with _client(tmp_path) as c:
            await c.post(
                "/api/engines/memory/graph_memory/config",
                json={"config": first},
            )
            await c.post(
                "/api/engines/memory/graph_memory/config",
                json={"config": second},
            )
            r = await c.get("/api/engines/memory/graph_memory/config")
        assert r.json()["config"] == second

    async def test_second_post_does_not_merge_with_first(self, tmp_path, monkeypatch):
        """Overwrite is a replace, not a merge — keys from first POST
        must NOT appear in GET after second POST if they're absent from it."""
        monkeypatch.chdir(tmp_path)
        first = {"old_key": "gone", "shared": "first"}
        second = {"shared": "second", "new_key": "kept"}
        async with _client(tmp_path) as c:
            await c.post(
                "/api/engines/memory/graph_memory/config",
                json={"config": first},
            )
            await c.post(
                "/api/engines/memory/graph_memory/config",
                json={"config": second},
            )
            r = await c.get("/api/engines/memory/graph_memory/config")
        cfg = r.json()["config"]
        assert "old_key" not in cfg
        assert cfg["shared"] == "second"
        assert cfg["new_key"] == "kept"

    async def test_get_before_any_post_then_post_then_get(self, tmp_path, monkeypatch):
        """Full initial-state → write → read cycle through the API only."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            # initial GET — file absent
            r1 = await c.get("/api/engines/decision/hypothalamus/config")
            assert r1.status_code == 200
            assert r1.json()["config"] == {}

            # write
            r2 = await c.post(
                "/api/engines/decision/hypothalamus/config",
                json={"config": {"llm_purposes": {"translator": "fast"}}},
            )
            assert r2.status_code == 200
            assert r2.json()["restart_required"] is True

            # read-back
            r3 = await c.get("/api/engines/decision/hypothalamus/config")
            assert r3.status_code == 200
            assert r3.json()["config"] == {
                "llm_purposes": {"translator": "fast"}
            }

    async def test_path_consistent_across_get_and_post(self, tmp_path, monkeypatch):
        """path returned by POST matches path returned by GET for same impl."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            post_r = await c.post(
                "/api/engines/reranker/passthrough/config",
                json={"config": {"foo": "bar"}},
            )
            get_r = await c.get("/api/engines/reranker/passthrough/config")
        assert post_r.json()["path"] == get_r.json()["path"]
        assert post_r.json()["path"] == "data/engines/reranker/passthrough.yaml"

    async def test_slots_are_independent(self, tmp_path, monkeypatch):
        """Writing to one slot/impl does not affect a different slot/impl."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            # write memory slot
            await c.post(
                "/api/engines/memory/graph_memory/config",
                json={"config": {"mem_key": "mem_val"}},
            )
            # decision slot should still be unset
            r = await c.get("/api/engines/decision/hypothalamus/config")
        assert r.json()["config"] == {}

    # -----------------------------------------------------------------------
    # Negative — bad requests / unknown resources
    # -----------------------------------------------------------------------

    async def test_post_missing_config_key_returns_400(self, tmp_path, monkeypatch):
        """Body with no 'config' key → 400."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.post(
                "/api/engines/memory/graph_memory/config",
                json={"not_config": {"a": 1}},
            )
        assert r.status_code == 400

    async def test_post_config_as_list_returns_400(self, tmp_path, monkeypatch):
        """config value is a list (not a mapping) → 400."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.post(
                "/api/engines/memory/graph_memory/config",
                json={"config": ["item1", "item2"]},
            )
        assert r.status_code == 400

    async def test_post_config_as_string_returns_400(self, tmp_path, monkeypatch):
        """config value is a string (not a mapping) → 400."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.post(
                "/api/engines/memory/graph_memory/config",
                json={"config": "just a string"},
            )
        assert r.status_code == 400

    async def test_post_config_as_int_returns_400(self, tmp_path, monkeypatch):
        """config value is an int (not a mapping) → 400."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.post(
                "/api/engines/memory/graph_memory/config",
                json={"config": 42},
            )
        assert r.status_code == 400

    async def test_post_config_as_null_returns_400(self, tmp_path, monkeypatch):
        """config=null (None) is not a mapping → 400."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.post(
                "/api/engines/memory/graph_memory/config",
                json={"config": None},
            )
        assert r.status_code == 400

    async def test_post_empty_body_returns_400(self, tmp_path, monkeypatch):
        """Body with no keys at all → 400 (no 'config' key present)."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.post(
                "/api/engines/memory/graph_memory/config",
                json={},
            )
        assert r.status_code == 400

    async def test_post_impl_without_config_path_returns_400_not_500(
        self, tmp_path, monkeypatch,
    ):
        """An impl whose meta declares NO config_path has nowhere to persist
        to. POST must return a clean 400 — not let FileEngineConfigStore.
        write('') raise ValueError → 500. (No shipped engine omits
        config_path, so we synthesise one via the slot loader.)"""
        from krakey.engine_system.catalog import EngineImpl

        def _fake_load_slot_meta(slot, **kwargs):
            return (
                {"noconfig": EngineImpl(
                    cls=object, description="no settings file",
                    config_path="",
                )},
                "noconfig",
            )

        monkeypatch.setattr(
            "krakey.engine_system.meta_loader.load_slot_meta",
            _fake_load_slot_meta,
        )
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.post(
                "/api/engines/memory/noconfig/config",
                json={"config": {"a": 1}},
            )
        assert r.status_code == 400, r.text

    async def test_get_impl_without_config_path_returns_200_empty(
        self, tmp_path, monkeypatch,
    ):
        """The GET sibling tolerates an empty config_path (read('') → {}),
        so opening the form is fine even when there's nowhere to save —
        the asymmetry with POST's 400 is intentional."""
        from krakey.engine_system.catalog import EngineImpl

        def _fake_load_slot_meta(slot, **kwargs):
            return (
                {"noconfig": EngineImpl(
                    cls=object, description="no settings file",
                    config_path="",
                )},
                "noconfig",
            )

        monkeypatch.setattr(
            "krakey.engine_system.meta_loader.load_slot_meta",
            _fake_load_slot_meta,
        )
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.get("/api/engines/memory/noconfig/config")
        assert r.status_code == 200
        assert r.json()["config"] == {}

    async def test_post_unknown_slot_returns_404(self, tmp_path, monkeypatch):
        """Slot that doesn't exist in any meta.yaml → 404."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.post(
                "/api/engines/nosuchslot/x/config",
                json={"config": {"a": 1}},
            )
        assert r.status_code == 404

    async def test_post_unknown_impl_returns_404(self, tmp_path, monkeypatch):
        """Known slot but impl not in that slot's meta → 404."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            r = await c.post(
                "/api/engines/memory/nosuchimpl/config",
                json={"config": {"a": 1}},
            )
        assert r.status_code == 404

    async def test_post_bad_request_does_not_affect_existing_config(
        self, tmp_path, monkeypatch
    ):
        """A 400 response must not clobber a previously saved config."""
        monkeypatch.chdir(tmp_path)
        good = {"stable": "value"}
        async with _client(tmp_path) as c:
            await c.post(
                "/api/engines/memory/graph_memory/config",
                json={"config": good},
            )
            # bad POST
            await c.post(
                "/api/engines/memory/graph_memory/config",
                json={"config": "bad"},
            )
            r = await c.get("/api/engines/memory/graph_memory/config")
        assert r.json()["config"] == good

    async def test_post_404_does_not_create_file(self, tmp_path, monkeypatch):
        """A 404 response must not create any file under tmp_path."""
        monkeypatch.chdir(tmp_path)
        async with _client(tmp_path) as c:
            await c.post(
                "/api/engines/nosuchslot/x/config",
                json={"config": {"a": 1}},
            )
        # No engine data directory should have been created.
        assert not (tmp_path / "data" / "engines" / "nosuchslot").exists()
