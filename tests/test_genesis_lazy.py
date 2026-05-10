"""GENESIS.md is only read when the bootstrap modifier is ACTIVE
and Self prompts are being built — plus the Option D auto-removal
of "bootstrap" from central config.yaml on completion.

Steady-state runtimes (bootstrap complete; agent has lived
experience) must never touch the file. After the Engine refactor
(2026-05) GENESIS handling moved out of Runtime into the bootstrap
plugin's BootstrapModifier — these tests now exercise the plugin's
lazy load behavior.

Pinned guarantees:
  1. Construction touches no GENESIS file.
  2. ``modify_prompt`` reads GENESIS exactly once (cached) — repeat
     calls during a long bootstrap don't re-read the file.
  3. A modifier whose active flag is False (post-completion or
     pinned via ``force_active``) does NOT call the loader.
  4. On auto-completion, "bootstrap" is removed from the central
     config.yaml's ``modifiers:`` and ``plugins:`` lists so the
     plugin won't run on the next start. Re-enable from the
     dashboard = re-bootstrap.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from krakey.plugins.bootstrap.modifier import BootstrapModifier


class _FakeStore:
    """Minimal SelfModelStore stand-in — load() returns a self_model
    dict the test controls."""

    def __init__(self, sm: dict):
        self._sm = sm
        self.update_calls: list = []

    def load(self):
        return self._sm

    def update(self, delta):
        self.update_calls.append(delta)


class _FakeEvents:
    """EventBus stand-in — records subscribers."""
    def __init__(self):
        self.subscribers: list = []

    def subscribe(self, cb): self.subscribers.append(cb)


def _make_modifier(*, sm, genesis_path, central_config_path=None):
    """Construct BootstrapModifier with the post-Option-D constructor
    signature. ``central_config_path`` defaults to None — only the
    Option-D self-removal tests pass a real path."""
    return BootstrapModifier(
        self_model_store=_FakeStore(sm),
        events=_FakeEvents(),
        central_config_path=central_config_path,
        genesis_path=str(genesis_path),
    )


def test_modifier_construction_does_not_read_genesis(tmp_path, monkeypatch):
    """BootstrapModifier.__init__ must NOT touch GENESIS."""
    reads = {"n": 0}
    real_read = Path.read_text

    def _spy_read(self, *args, **kwargs):
        if self.name.lower() == "genesis.md":
            reads["n"] += 1
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _spy_read)

    (tmp_path / "GENESIS.md").write_text("hi", encoding="utf-8")
    _make_modifier(
        sm={"state": {"bootstrap_complete": False}},
        genesis_path=tmp_path / "GENESIS.md",
    )
    assert reads["n"] == 0


def test_modify_prompt_reads_genesis_once_and_caches(tmp_path, monkeypatch):
    """First modify_prompt call reads GENESIS; subsequent calls hit
    the cache (no new disk reads)."""
    reads = {"n": 0}
    real_read = Path.read_text

    def _spy_read(self, *args, **kwargs):
        if self.name.lower() == "genesis.md":
            reads["n"] += 1
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _spy_read)

    genesis_file = tmp_path / "GENESIS.md"
    genesis_file.write_text("hello young agent", encoding="utf-8")
    mod = _make_modifier(
        sm={"state": {"bootstrap_complete": False}},
        genesis_path=genesis_file,
    )

    elements = {}
    mod.modify_prompt(elements)
    assert reads["n"] == 1
    assert "hello young agent" in elements.get("bootstrap_intro", "")

    # Second call — cache hit, no new read
    mod.modify_prompt({})
    assert reads["n"] == 1


def test_inactive_modifier_does_not_read_genesis(tmp_path, monkeypatch):
    """A modifier pinned inactive must NOT load GENESIS even if
    modify_prompt fires. Post Option D the active flag is no longer
    inferred from ``state.bootstrap_complete`` (the plugin removes
    itself from the central config on completion instead), so this
    test pins the flag explicitly via ``force_active(False)``."""
    reads = {"n": 0}
    real_read = Path.read_text

    def _spy_read(self, *args, **kwargs):
        if self.name.lower() == "genesis.md":
            reads["n"] += 1
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _spy_read)

    (tmp_path / "GENESIS.md").write_text("placeholder", encoding="utf-8")
    mod = _make_modifier(
        sm={"state": {"bootstrap_complete": True}},
        genesis_path=tmp_path / "GENESIS.md",
    )
    mod.force_active(False)

    elements = {}
    mod.modify_prompt(elements)
    assert reads["n"] == 0
    # No bootstrap_intro injected
    assert "bootstrap_intro" not in elements


def test_missing_genesis_returns_placeholder_lazily(tmp_path):
    """No file at genesis_path → modifier injects the placeholder
    rather than crashing the heartbeat."""
    mod = _make_modifier(
        sm={"state": {"bootstrap_complete": False}},
        genesis_path=tmp_path / "does_not_exist.md",
    )
    elements = {}
    mod.modify_prompt(elements)
    intro = elements.get("bootstrap_intro", "")
    assert intro
    assert "blank-slate" in intro.lower() or "GENESIS" in intro


# ---------------- Option D: central-config self-removal ----------------


def _trigger_completion(mod):
    """Push a NoteEvent with identity fields so _handle_note runs and
    auto-completes. Tests fake the event with bare attribute access —
    the modifier only reads ``event.kind`` + ``event.text``."""
    class _Ev:
        kind = "note"
        text = ""
    mod._on_event(_Ev())


def test_completion_removes_bootstrap_from_modifiers_list(tmp_path):
    """Pre: identity already set; completion triggered. Post: central
    config.yaml's ``modifiers:`` list no longer contains 'bootstrap'."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "modifiers:\n  - bootstrap\n  - hypothalamus\n"
        "plugins: []\n",
        encoding="utf-8",
    )
    sm = {"identity": {"name": "Krakey", "persona": "curious"}}
    mod = _make_modifier(
        sm=sm,
        genesis_path=tmp_path / "GENESIS.md",
        central_config_path=cfg,
    )
    _trigger_completion(mod)

    assert mod.is_active is False
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert "bootstrap" not in data["modifiers"]
    assert "hypothalamus" in data["modifiers"]


def test_completion_removes_from_plugins_list_too(tmp_path):
    """Defensive: if bootstrap somehow ended up in ``plugins:`` (manual
    edit), removal still wipes it from there."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "plugins:\n  - bootstrap\n  - dashboard\n",
        encoding="utf-8",
    )
    sm = {"identity": {"name": "Krakey", "persona": "curious"}}
    mod = _make_modifier(
        sm=sm,
        genesis_path=tmp_path / "GENESIS.md",
        central_config_path=cfg,
    )
    _trigger_completion(mod)

    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert "bootstrap" not in data["plugins"]
    assert "dashboard" in data["plugins"]


def test_completion_with_no_central_config_path_still_completes(tmp_path):
    """Missing config_path is non-fatal — self_model still flips,
    plugin still deactivates. Worst case: bootstrap re-runs next
    start, which the user can disable from the dashboard."""
    sm = {"identity": {"name": "Krakey", "persona": "curious"}}
    store = _FakeStore(sm)
    mod = BootstrapModifier(
        self_model_store=store,
        events=_FakeEvents(),
        central_config_path=None,
        genesis_path=str(tmp_path / "GENESIS.md"),
    )
    _trigger_completion(mod)

    assert mod.is_active is False
    assert any(
        "bootstrap_complete" in str(call) for call in store.update_calls
    ), "self_model should still receive bootstrap_complete=True"


def test_completion_partial_identity_does_not_remove(tmp_path):
    """name set but persona empty → criteria not met → no removal,
    plugin stays active."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "modifiers:\n  - bootstrap\n",
        encoding="utf-8",
    )
    sm = {"identity": {"name": "Krakey", "persona": ""}}
    mod = _make_modifier(
        sm=sm,
        genesis_path=tmp_path / "GENESIS.md",
        central_config_path=cfg,
    )
    _trigger_completion(mod)

    assert mod.is_active is True
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert "bootstrap" in data["modifiers"]


def test_completion_idempotent_on_missing_entry(tmp_path):
    """Already removed (user manually edited) → no write, no error."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "modifiers: []\nplugins: []\n",
        encoding="utf-8",
    )
    mtime_before = cfg.stat().st_mtime_ns
    sm = {"identity": {"name": "Krakey", "persona": "curious"}}
    mod = _make_modifier(
        sm=sm,
        genesis_path=tmp_path / "GENESIS.md",
        central_config_path=cfg,
    )
    _trigger_completion(mod)

    # Modifier still flips inactive even though the file didn't need
    # writing — completion is driven by self_model state.
    assert mod.is_active is False
    assert cfg.stat().st_mtime_ns == mtime_before, (
        "config.yaml was rewritten despite no removal needed"
    )
