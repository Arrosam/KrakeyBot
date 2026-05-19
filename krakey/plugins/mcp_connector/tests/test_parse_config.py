"""Edge tests for ``_parse_config`` in ``krakey.plugins.mcp_connector.plugin``.

Bug under test
--------------
The ``meta.yaml`` schema declares ``servers`` as ``type: text`` with
``default: ""``.  Operators (particularly through the dashboard text input)
supply ``servers`` as a JSON-encoded string.  The current implementation does::

    servers_raw = raw.get("servers")
    if not isinstance(servers_raw, list):
        servers_raw = []

This silently discards a valid JSON string and registers zero servers.

Required behaviour after the fix
---------------------------------
1. JSON string → list of valid dicts  →  parsed and filtered as if the list
   had been supplied directly.
2. Empty string ``""`` or whitespace-only string  →  ``{"servers": [], ...}``,
   no raise.
3. Non-JSON string (malformed)  →  ``{"servers": [], ...}``, no raise.
4. JSON string that decodes to a non-list (dict, int, …)  →  ``{"servers": [], ...}``,
   no raise.
5. Real Python ``list`` (YAML path)  →  unchanged behaviour.
6. Non-string non-list types (``None``, ``int``, ``dict``)  →  ``{"servers": []}``
   fallback, no raise.
7. Per-entry filtering still applies after JSON decoding:
   non-dict entries skipped, entries without truthy ``id`` skipped,
   entries with ``enabled: false`` skipped, entries defaulting to enabled kept.
8. ``reconnect`` handling is unchanged: ``bool(raw.get("reconnect", True))``.

Run from repo root::

    pytest krakey/plugins/mcp_connector/tests/test_parse_config.py

Import note
-----------
``_parse_config`` is a module-level function in
``krakey.plugins.mcp_connector.plugin`` but is NOT re-exported from the
package ``__init__``.  We import it directly from the module, which is the
only correct path.  The sibling ``test_mcp_connector.py`` uses the same
package for ``build_tool`` imports.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

# Direct module import — _parse_config is intentionally internal.
from krakey.plugins.mcp_connector.plugin import _parse_config


# ---------------------------------------------------------------------------
# Helpers: build representative server-entry dicts
# ---------------------------------------------------------------------------

def _valid_entry(
    server_id: str = "srv1",
    enabled: bool | None = True,
    *,
    include_enabled: bool = True,
) -> dict[str, Any]:
    """Return a minimal valid server entry dict."""
    entry: dict[str, Any] = {
        "id": server_id,
        "transport": "stdio",
        "command": ["python", "server.py"],
    }
    if include_enabled:
        entry["enabled"] = enabled
    return entry


def _disabled_entry(server_id: str = "off") -> dict[str, Any]:
    return _valid_entry(server_id=server_id, enabled=False)


def _no_id_entry() -> dict[str, Any]:
    return {"transport": "stdio", "command": ["python", "server.py"], "enabled": True}


def _non_dict_entry() -> str:
    return "i_am_not_a_dict"


# ---------------------------------------------------------------------------
# Section 1 — JSON string → parsed as list (behaviour 1)
# ---------------------------------------------------------------------------


class TestServersAsJsonString:
    """Behaviour 1 — servers is a JSON-encoded string encoding a list."""

    def test_valid_json_string_single_valid_entry_is_parsed(self):
        """JSON string with one valid, enabled entry produces one server."""
        entry = _valid_entry("alpha")
        raw = {"servers": json.dumps([entry])}
        result = _parse_config(raw)
        assert result["servers"] == [entry], (
            "A JSON string encoding a single valid server dict must be parsed "
            "and returned in the servers list."
        )

    def test_valid_json_string_two_valid_entries_both_kept(self):
        """JSON string with two valid, enabled entries produces both servers."""
        e1 = _valid_entry("srv_a")
        e2 = _valid_entry("srv_b")
        raw = {"servers": json.dumps([e1, e2])}
        result = _parse_config(raw)
        assert len(result["servers"]) == 2
        ids = {s["id"] for s in result["servers"]}
        assert ids == {"srv_a", "srv_b"}

    def test_valid_json_string_result_structure_matches_direct_list(self):
        """JSON-string path and direct-list path must produce identical output."""
        entry = _valid_entry("match_me")
        via_list = _parse_config({"servers": [entry]})
        via_string = _parse_config({"servers": json.dumps([entry])})
        assert via_string == via_list, (
            "Passing servers as a JSON string must produce exactly the same "
            "result as passing the equivalent Python list."
        )

    def test_json_string_with_mixed_entries_filters_correctly(self):
        """JSON string: disabled entry skipped, no-id skipped, non-dict skipped, valid kept."""
        valid = _valid_entry("keep_me")
        disabled = _disabled_entry("skip_disabled")
        no_id = _no_id_entry()
        non_dict_val = "not_a_dict"

        raw = {"servers": json.dumps([disabled, no_id, non_dict_val, valid])}
        result = _parse_config(raw)

        assert len(result["servers"]) == 1, (
            "Only the one valid enabled entry with an id should survive filtering "
            f"after JSON decode; got {result['servers']!r}"
        )
        assert result["servers"][0]["id"] == "keep_me"

    def test_json_string_disabled_entry_is_excluded(self):
        """JSON string: a server with enabled=false must be excluded."""
        entry = _disabled_entry("off_server")
        raw = {"servers": json.dumps([entry])}
        result = _parse_config(raw)
        assert result["servers"] == [], (
            "A disabled server supplied via JSON string must be skipped."
        )

    def test_json_string_no_id_entry_is_excluded(self):
        """JSON string: entry without a truthy 'id' must be excluded."""
        entry = _no_id_entry()
        raw = {"servers": json.dumps([entry])}
        result = _parse_config(raw)
        assert result["servers"] == [], (
            "A server entry with no 'id' supplied via JSON string must be skipped."
        )

    def test_json_string_non_dict_list_element_is_excluded(self):
        """JSON string: a list element that is not a dict must be excluded."""
        raw = {"servers": json.dumps(["just_a_string", 42, None])}
        result = _parse_config(raw)
        assert result["servers"] == [], (
            "Non-dict elements inside a JSON-decoded list must all be skipped."
        )

    def test_json_string_entry_missing_enabled_defaults_to_true(self):
        """JSON string: entry with no 'enabled' key is treated as enabled=True."""
        entry = _valid_entry("implicit_on", include_enabled=False)
        raw = {"servers": json.dumps([entry])}
        result = _parse_config(raw)
        assert len(result["servers"]) == 1, (
            "An entry without an 'enabled' key should default to enabled and be kept."
        )
        assert result["servers"][0]["id"] == "implicit_on"

    def test_json_string_entry_explicit_enabled_true_is_kept(self):
        """JSON string: entry with enabled=true is kept."""
        entry = _valid_entry("explicit_on", enabled=True)
        raw = {"servers": json.dumps([entry])}
        result = _parse_config(raw)
        assert len(result["servers"]) == 1
        assert result["servers"][0]["id"] == "explicit_on"

    def test_json_string_empty_list_json_produces_no_servers(self):
        """JSON string encoding an empty list '[]' must produce servers=[]."""
        raw = {"servers": "[]"}
        result = _parse_config(raw)
        assert result["servers"] == [], (
            "The JSON string '[]' must produce an empty servers list."
        )


# ---------------------------------------------------------------------------
# Section 2 — Empty / whitespace-only string (behaviour 2)
# ---------------------------------------------------------------------------


class TestServersEmptyOrWhitespaceString:
    """Behaviour 2 — empty string or whitespace-only string → no servers, no raise."""

    def test_empty_string_produces_empty_servers_list(self):
        """Empty string '' (the schema default) must produce servers=[]."""
        raw = {"servers": ""}
        result = _parse_config(raw)
        assert result["servers"] == [], (
            "An empty string (schema default) must be treated as no servers."
        )

    def test_empty_string_does_not_raise(self):
        """Empty string must not raise any exception."""
        try:
            _parse_config({"servers": ""})
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"_parse_config raised {type(exc).__name__} for empty string: {exc}"
            )

    def test_whitespace_only_string_produces_empty_servers_list(self):
        """A whitespace-only string must be treated as no servers."""
        for ws in ("   ", "\t", "\n", "  \n  \t  "):
            result = _parse_config({"servers": ws})
            assert result["servers"] == [], (
                f"Whitespace string {ws!r} must produce servers=[], "
                f"got {result['servers']!r}"
            )

    def test_whitespace_only_string_does_not_raise(self):
        """Whitespace-only string must not raise any exception."""
        try:
            _parse_config({"servers": "   "})
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"_parse_config raised {type(exc).__name__} for whitespace string: {exc}"
            )


# ---------------------------------------------------------------------------
# Section 3 — Malformed JSON string (behaviour 3)
# ---------------------------------------------------------------------------


class TestServesMalformedJsonString:
    """Behaviour 3 — a string that is not valid JSON → no servers, no raise."""

    def test_malformed_json_string_produces_empty_servers_list(self):
        """A non-JSON string must produce servers=[]."""
        raw = {"servers": "this is not json at all"}
        result = _parse_config(raw)
        assert result["servers"] == [], (
            "A malformed JSON string must be treated as no servers."
        )

    def test_malformed_json_string_does_not_raise(self):
        """A non-JSON string must not raise any exception."""
        try:
            _parse_config({"servers": "not valid JSON }{{"})
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"_parse_config raised {type(exc).__name__} for malformed JSON: {exc}"
            )

    def test_truncated_json_does_not_raise(self):
        """A truncated JSON string (starts but does not close) must not raise."""
        try:
            result = _parse_config({"servers": '[{"id": "x"'})
            assert result["servers"] == []
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"_parse_config raised {type(exc).__name__} for truncated JSON: {exc}"
            )

    def test_json_object_at_top_level_string_does_not_raise(self):
        """The string '{\"id\": \"x\"}' (a JSON object, not a list) must not raise."""
        # This is technically valid JSON but decodes to a dict, not a list.
        # Covered in depth by TestServersJsonStringNonList, but must also not raise.
        try:
            _parse_config({"servers": '{"id": "x"}'})
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"_parse_config raised {type(exc).__name__} for JSON-object string: {exc}"
            )


# ---------------------------------------------------------------------------
# Section 4 — Valid JSON string but decoded value is not a list (behaviour 4)
# ---------------------------------------------------------------------------


class TestServersJsonStringNonList:
    """Behaviour 4 — JSON string decodes to a non-list value → servers=[], no raise."""

    def test_json_string_decodes_to_dict_gives_empty_servers(self):
        """JSON string encoding a dict '{ \"id\": \"x\" }' → servers=[]."""
        raw = {"servers": json.dumps({"id": "x", "transport": "stdio"})}
        result = _parse_config(raw)
        assert result["servers"] == [], (
            "A JSON string that decodes to a dict (not a list) must produce servers=[]."
        )

    def test_json_string_decodes_to_int_gives_empty_servers(self):
        """JSON string encoding an integer '42' → servers=[]."""
        raw = {"servers": "42"}
        result = _parse_config(raw)
        assert result["servers"] == [], (
            "A JSON string that decodes to an int must produce servers=[]."
        )

    def test_json_string_decodes_to_null_gives_empty_servers(self):
        """JSON string 'null' → servers=[]."""
        raw = {"servers": "null"}
        result = _parse_config(raw)
        assert result["servers"] == [], (
            "A JSON string that decodes to null must produce servers=[]."
        )

    def test_json_string_decodes_to_string_gives_empty_servers(self):
        """JSON string encoding a quoted string '\"hello\"' → servers=[]."""
        raw = {"servers": '"hello"'}
        result = _parse_config(raw)
        assert result["servers"] == [], (
            "A JSON string that decodes to a string (not a list) must produce servers=[]."
        )

    def test_json_string_decodes_to_bool_gives_empty_servers(self):
        """JSON string 'true' (decodes to bool, not list) → servers=[]."""
        raw = {"servers": "true"}
        result = _parse_config(raw)
        assert result["servers"] == [], (
            "A JSON string that decodes to a bool must produce servers=[]."
        )

    def test_json_string_non_list_does_not_raise(self):
        """None of the non-list decode cases must raise."""
        bad_strings = [
            json.dumps({"id": "x"}),  # dict
            "42",                      # int
            "null",                    # null
            '"hello"',                 # string
            "true",                    # bool
        ]
        for s in bad_strings:
            try:
                _parse_config({"servers": s})
            except Exception as exc:  # noqa: BLE001
                pytest.fail(
                    f"_parse_config raised {type(exc).__name__} for JSON string "
                    f"{s!r} that decodes to a non-list: {exc}"
                )


# ---------------------------------------------------------------------------
# Section 5 — Real Python list still works (behaviour 5)
# ---------------------------------------------------------------------------


class TestServersAsRealList:
    """Behaviour 5 — a real Python list still works (YAML config-file path)."""

    def test_real_list_single_valid_entry_is_kept(self):
        """Real list with one valid entry is unchanged."""
        entry = _valid_entry("yaml_srv")
        result = _parse_config({"servers": [entry]})
        assert result["servers"] == [entry]

    def test_real_list_multiple_entries_all_kept(self):
        """Real list with multiple valid entries — all kept."""
        entries = [_valid_entry(f"s{i}") for i in range(3)]
        result = _parse_config({"servers": entries})
        assert len(result["servers"]) == 3

    def test_real_list_disabled_entry_skipped(self):
        """Real list: disabled entry is skipped (unchanged original behaviour)."""
        entries = [_valid_entry("on"), _disabled_entry("off")]
        result = _parse_config({"servers": entries})
        assert len(result["servers"]) == 1
        assert result["servers"][0]["id"] == "on"

    def test_real_list_no_id_entry_skipped(self):
        """Real list: entry without 'id' is skipped (unchanged original behaviour)."""
        entries = [_no_id_entry(), _valid_entry("has_id")]
        result = _parse_config({"servers": entries})
        assert len(result["servers"]) == 1
        assert result["servers"][0]["id"] == "has_id"

    def test_real_list_non_dict_element_skipped(self):
        """Real list: a non-dict element is skipped (unchanged original behaviour)."""
        entries = ["not_a_dict", _valid_entry("dict_entry")]
        result = _parse_config({"servers": entries})  # type: ignore[arg-type]
        assert len(result["servers"]) == 1
        assert result["servers"][0]["id"] == "dict_entry"

    def test_real_empty_list_produces_empty_servers(self):
        """Real empty list [] → servers=[]."""
        result = _parse_config({"servers": []})
        assert result["servers"] == []


# ---------------------------------------------------------------------------
# Section 6 — Non-string, non-list fallback (behaviour 6)
# ---------------------------------------------------------------------------


class TestServersNonStringNonList:
    """Behaviour 6 — None, int, dict, etc. fall back to servers=[], no raise."""

    def test_servers_none_produces_empty_list(self):
        """servers=None → servers=[]."""
        result = _parse_config({"servers": None})
        assert result["servers"] == []

    def test_servers_int_produces_empty_list(self):
        """servers=42 (int) → servers=[]."""
        result = _parse_config({"servers": 42})
        assert result["servers"] == []

    def test_servers_dict_produces_empty_list(self):
        """servers={} (a dict, not a list) → servers=[]."""
        result = _parse_config({"servers": {"id": "bad"}})
        assert result["servers"] == []

    def test_servers_missing_key_produces_empty_list(self):
        """No 'servers' key at all → servers=[]."""
        result = _parse_config({})
        assert result["servers"] == []

    def test_none_does_not_raise(self):
        """servers=None must not raise."""
        try:
            _parse_config({"servers": None})
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"_parse_config raised {type(exc).__name__} for servers=None: {exc}"
            )

    def test_int_does_not_raise(self):
        """servers=0 (int) must not raise."""
        try:
            _parse_config({"servers": 0})
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"_parse_config raised {type(exc).__name__} for servers=0: {exc}"
            )

    def test_dict_does_not_raise(self):
        """servers={} (a dict) must not raise."""
        try:
            _parse_config({"servers": {}})
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"_parse_config raised {type(exc).__name__} for servers={{}}: {exc}"
            )


# ---------------------------------------------------------------------------
# Section 7 — reconnect handling (behaviour 8)
# ---------------------------------------------------------------------------


class TestReconnectHandling:
    """Behaviour 8 — reconnect field is unchanged by the JSON-string fix."""

    def test_reconnect_defaults_to_true_when_absent(self):
        """No 'reconnect' key → reconnect=True."""
        result = _parse_config({"servers": []})
        assert result["reconnect"] is True

    def test_reconnect_true_when_explicitly_true(self):
        """reconnect=True → reconnect=True."""
        result = _parse_config({"servers": [], "reconnect": True})
        assert result["reconnect"] is True

    def test_reconnect_false_when_explicitly_false(self):
        """reconnect=False → reconnect=False."""
        result = _parse_config({"servers": [], "reconnect": False})
        assert result["reconnect"] is False

    def test_reconnect_is_bool_type(self):
        """reconnect in the result is always a Python bool."""
        result = _parse_config({"servers": [], "reconnect": 1})
        assert isinstance(result["reconnect"], bool), (
            "reconnect must be coerced to a bool (bool(raw.get('reconnect', True)))."
        )

    def test_reconnect_truthy_int_becomes_true(self):
        """reconnect=1 (truthy int) → reconnect=True."""
        result = _parse_config({"servers": [], "reconnect": 1})
        assert result["reconnect"] is True

    def test_reconnect_zero_becomes_false(self):
        """reconnect=0 (falsy int) → reconnect=False."""
        result = _parse_config({"servers": [], "reconnect": 0})
        assert result["reconnect"] is False

    def test_reconnect_preserved_when_servers_is_json_string(self):
        """reconnect is correctly read even when servers is supplied as JSON string."""
        entry = _valid_entry("r_srv")
        result = _parse_config({
            "servers": json.dumps([entry]),
            "reconnect": False,
        })
        assert result["reconnect"] is False

    def test_reconnect_preserved_when_servers_is_empty_string(self):
        """reconnect is correctly read even when servers is the empty-string default."""
        result = _parse_config({"servers": "", "reconnect": False})
        assert result["reconnect"] is False


# ---------------------------------------------------------------------------
# Section 8 — Return structure contract
# ---------------------------------------------------------------------------


class TestReturnStructure:
    """_parse_config always returns a dict with exactly the two expected keys."""

    def test_result_has_servers_key(self):
        """Return dict always has 'servers' key."""
        result = _parse_config({})
        assert "servers" in result

    def test_result_has_reconnect_key(self):
        """Return dict always has 'reconnect' key."""
        result = _parse_config({})
        assert "reconnect" in result

    def test_servers_value_is_always_a_list(self):
        """'servers' in result is always a Python list, never None or other type."""
        inputs = [
            {},
            {"servers": ""},
            {"servers": "[]"},
            {"servers": json.dumps([_valid_entry("x")])},
            {"servers": []},
            {"servers": None},
            {"servers": "bad json"},
        ]
        for raw in inputs:
            result = _parse_config(raw)
            assert isinstance(result["servers"], list), (
                f"result['servers'] is not a list for input {raw!r}: "
                f"got {type(result['servers'])}"
            )

    def test_reconnect_value_is_always_bool(self):
        """'reconnect' in result is always a bool."""
        inputs = [
            {},
            {"reconnect": True},
            {"reconnect": False},
            {"reconnect": 1},
            {"reconnect": 0},
        ]
        for raw in inputs:
            result = _parse_config(raw)
            assert isinstance(result["reconnect"], bool), (
                f"result['reconnect'] is not bool for input {raw!r}: "
                f"got {type(result['reconnect'])}"
            )
