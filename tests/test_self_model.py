import pytest

from src.models.self_model import SelfModelStore, default_self_model


def test_load_missing_returns_default(tmp_path):
    p = tmp_path / "sm.yaml"
    store = SelfModelStore(p)
    data = store.load()
    assert data == default_self_model()
    assert data["identity"]["name"] == ""
    assert data["state"]["bootstrap_complete"] is False


def test_save_then_load_roundtrip(tmp_path):
    p = tmp_path / "sm.yaml"
    store = SelfModelStore(p)
    data = default_self_model()
    data["identity"]["name"] = "Krakey"
    data["state"]["bootstrap_complete"] = True
    store.save(data)

    again = SelfModelStore(p).load()
    assert again["identity"]["name"] == "Krakey"
    assert again["state"]["bootstrap_complete"] is True


def test_update_deep_merges(tmp_path):
    p = tmp_path / "sm.yaml"
    store = SelfModelStore(p)
    store.save(default_self_model())
    # Identity persona survives an identity.name update; bootstrap
    # flag is unaffected.
    store.update({"identity": {"name": "Krakey"}})

    data = store.load()
    assert data["identity"]["name"] == "Krakey"
    assert data["identity"]["persona"] == ""  # default, untouched
    assert data["state"]["bootstrap_complete"] is False  # untouched


def test_load_existing_preserves_data(tmp_path):
    p = tmp_path / "sm.yaml"
    p.write_text("identity:\n  name: Existing\n", encoding="utf-8")
    store = SelfModelStore(p)
    data = store.load()
    assert data["identity"]["name"] == "Existing"


def test_slim_schema_has_only_two_top_level_keys():
    """Regression for the 2026-04-25 slim. Adding fields back to
    self-model should be a deliberate decision and break this test."""
    d = default_self_model()
    assert set(d.keys()) == {"identity", "state"}
    assert set(d["identity"].keys()) == {"name", "persona"}
    assert set(d["state"].keys()) == {"bootstrap_complete"}
