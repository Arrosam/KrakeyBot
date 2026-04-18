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
    store.update({"state": {"focus_topic": "astronomy"},
                  "statistics": {"total_heartbeats": 5}})

    data = store.load()
    assert data["state"]["focus_topic"] == "astronomy"
    assert data["state"]["bootstrap_complete"] is False  # untouched
    assert data["statistics"]["total_heartbeats"] == 5


def test_update_appends_goal(tmp_path):
    p = tmp_path / "sm.yaml"
    store = SelfModelStore(p)
    store.save(default_self_model())
    store.update({"goals": {"active": ["greet user"]}})
    data = store.load()
    assert data["goals"]["active"] == ["greet user"]


def test_load_existing_preserves_data(tmp_path):
    p = tmp_path / "sm.yaml"
    p.write_text("identity:\n  name: Existing\n", encoding="utf-8")
    store = SelfModelStore(p)
    data = store.load()
    assert data["identity"]["name"] == "Existing"
