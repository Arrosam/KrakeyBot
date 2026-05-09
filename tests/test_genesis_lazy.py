"""GENESIS.md is only read when the bootstrap modifier is ACTIVE
and Self prompts are being built.

Steady-state runtimes (bootstrap complete; agent has lived
experience) must never touch the file. After the Engine refactor
(2026-05) GENESIS handling moved out of Runtime into the bootstrap
plugin's BootstrapModifier — these tests now exercise the plugin's
lazy load behavior.

Pinned guarantees:
  1. Construction touches no GENESIS file.
  2. ``modify_prompt`` reads GENESIS exactly once (cached) — repeat
     calls during a long bootstrap don't re-read the file.
  3. A modifier whose active flag is False does NOT call the
     loader, regardless of how many times modify_prompt fires.
"""
from __future__ import annotations

from pathlib import Path

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


class _FakeMemory:
    async def count_nodes(self): return 0
    async def list_kbs(self, *, include_archived=False): return []


class _FakeEvents:
    """EventBus stand-in — records subscribers."""
    def __init__(self):
        self.subscribers: list = []

    def subscribe(self, cb): self.subscribers.append(cb)


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
    BootstrapModifier(
        self_model_store=_FakeStore({"state": {"bootstrap_complete": False}}),
        memory=_FakeMemory(),
        events=_FakeEvents(),
        genesis_path=str(tmp_path / "GENESIS.md"),
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
    mod = BootstrapModifier(
        self_model_store=_FakeStore({"state": {"bootstrap_complete": False}}),
        memory=_FakeMemory(),
        events=_FakeEvents(),
        genesis_path=str(genesis_file),
    )

    elements = {}
    mod.modify_prompt(elements)
    assert reads["n"] == 1
    assert "hello young agent" in elements.get("bootstrap_intro", "")

    # Second call — cache hit, no new read
    mod.modify_prompt({})
    assert reads["n"] == 1


def test_inactive_modifier_does_not_read_genesis(tmp_path, monkeypatch):
    """When the modifier is inactive (bootstrap complete) it must
    NOT load GENESIS even if modify_prompt fires."""
    reads = {"n": 0}
    real_read = Path.read_text

    def _spy_read(self, *args, **kwargs):
        if self.name.lower() == "genesis.md":
            reads["n"] += 1
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _spy_read)

    (tmp_path / "GENESIS.md").write_text("placeholder", encoding="utf-8")
    mod = BootstrapModifier(
        self_model_store=_FakeStore({
            "state": {"bootstrap_complete": True},
        }),
        memory=_FakeMemory(),
        events=_FakeEvents(),
        genesis_path=str(tmp_path / "GENESIS.md"),
    )
    # Active flag was inferred from the persisted complete=True; modifier
    # is inactive.
    assert mod.is_active is False

    elements = {}
    mod.modify_prompt(elements)
    assert reads["n"] == 0
    # No bootstrap_intro injected
    assert "bootstrap_intro" not in elements


def test_missing_genesis_returns_placeholder_lazily(tmp_path):
    """No file at genesis_path → modifier injects the placeholder
    rather than crashing the heartbeat."""
    mod = BootstrapModifier(
        self_model_store=_FakeStore({"state": {"bootstrap_complete": False}}),
        memory=_FakeMemory(),
        events=_FakeEvents(),
        genesis_path=str(tmp_path / "does_not_exist.md"),
    )
    elements = {}
    mod.modify_prompt(elements)
    intro = elements.get("bootstrap_intro", "")
    assert intro
    assert "blank-slate" in intro.lower() or "GENESIS" in intro
