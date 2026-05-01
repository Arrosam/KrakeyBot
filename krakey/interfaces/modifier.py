"""Modifier plugin interface — protocols + registry.

Sibling to ``channel.py`` and ``tool.py``: defines the contract
the runtime depends on and the registry it stores instances in.
Concrete Modifiers live under ``krakey/plugins/<plugin>/``.

Each Modifier declares a ``role`` string. The registry rejects a
second registration claiming an already-taken role: roles are
unique. The runtime does not interpret role names — it just looks
up a role and calls its protocol-specific methods. Plugins free to
mint new role names; they only collide with each other if they
chose the same string.

Optional advisory protocols below (HypothalamusModifier, ...) document
the method shapes the runtime expects when a particular role is
used. Modifiers don't have to inherit from them — structural typing
keeps plugin code free of interface imports it doesn't need.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from krakey.memory.recall import RecallLike


# --------------------------------------------------------------------
# Contract dataclasses — cross the Modifier ↔ runtime boundary
# --------------------------------------------------------------------


@dataclass
class ToolCall:
    """Structured tool invocation produced by a decision-translator
    Modifier's ``translate()``. Consumed by the dispatcher and by the
    script-only action executor (when no translator is registered)."""
    tool: str
    intent: str
    params: dict[str, Any] = field(default_factory=dict)
    adrenalin: bool = False


@dataclass
class DecisionResult:
    """Aggregate result of one decision-translation pass: the tool
    calls to dispatch, plus any memory side-effects and the sleep
    flag. Produced by either the hypothalamus role's translate() or
    the bare tool-call parser fallback; the dispatcher consumes it
    without caring which path produced it."""
    tool_calls: list[ToolCall] = field(default_factory=list)
    memory_writes: list[dict[str, Any]] = field(default_factory=list)
    memory_updates: list[dict[str, Any]] = field(default_factory=list)
    sleep: bool = False


@dataclass
class HeartbeatContext:
    """Bundle passed to ``on_heartbeat_start`` / ``on_heartbeat_end``."""
    heartbeat_id: int
    phase: str  # "start" | "end"


# --------------------------------------------------------------------
# Protocols — Modifier shapes (advisory; runtime uses by_role lookup)
# --------------------------------------------------------------------


@runtime_checkable
class Modifier(Protocol):
    """Base shape — every Modifier has a name + role."""
    name: str
    role: str


@runtime_checkable
class HypothalamusModifier(Protocol):
    """Optional shape advised for Modifiers that translate Self's
    [DECISION] text into structured tool calls."""
    name: str
    role: str

    async def translate(
        self, decision: str, tools: list[dict[str, Any]],
    ) -> DecisionResult: ...


@runtime_checkable
class RecallAnchorModifier(Protocol):
    """Optional shape advised for Modifiers that build the per-beat
    recall instance."""
    name: str
    role: str

    def make_recall(self, runtime: Any) -> "RecallLike": ...


@runtime_checkable
class InMindModifier(Protocol):
    """Optional shape advised for Modifiers that own Self's persistent
    in-mind state (thoughts / mood / focus)."""
    name: str
    role: str

    def read(self) -> dict[str, str]: ...

    def update(
        self,
        thoughts: str | None = None,
        mood: str | None = None,
        focus: str | None = None,
    ) -> dict[str, str]: ...


# --------------------------------------------------------------------
# Registry — role-keyed, role-unique
# --------------------------------------------------------------------


class ModifierRegistry:
    """Role-keyed registry for Modifiers.

    Each role is held by at most one Modifier. Registering a second
    Modifier with the same role raises — the runtime can't reasonably
    decide which one to use, so the user has to fix the conflict in
    config.

    The registry is intentionally narrow: the runtime queries by role
    string (``by_role("hypothalamus")``), checks existence
    (``has_role(...)``), or iterates everything (``all()``). It does
    NOT interpret role names or know what methods exist for any
    particular role — that's the caller's responsibility.
    """

    def __init__(self):
        self._by_role: dict[str, Modifier] = {}
        self._order: list[str] = []  # registration order, for `all()`

    # ---- registration ------------------------------------------------

    def register(self, modifier: Modifier) -> None:
        """Register a Modifier under its declared role. Raises if the
        role is already taken or the modifier is missing required
        attributes."""
        role = getattr(modifier, "role", None)
        name = getattr(modifier, "name", None)
        if not role:
            raise ValueError(
                f"Modifier {modifier!r} missing required `role` attribute"
            )
        if not name:
            raise ValueError(
                f"Modifier {modifier!r} missing required `name` attribute"
            )
        if role in self._by_role:
            existing = self._by_role[role]
            raise ValueError(
                f"role {role!r} already claimed by Modifier "
                f"{existing.name!r}; cannot register {name!r}"
            )
        self._by_role[role] = modifier
        self._order.append(role)

    # ---- lookup ------------------------------------------------------

    def by_role(self, role: str) -> Modifier | None:
        """The Modifier for ``role``, or ``None`` if no Modifier has
        claimed it."""
        return self._by_role.get(role)

    def has_role(self, role: str) -> bool:
        return role in self._by_role

    def roles(self) -> list[str]:
        """All claimed role names, in registration order."""
        return list(self._order)

    def names(self) -> list[str]:
        """All registered Modifier names, in registration order."""
        return [self._by_role[r].name for r in self._order]

    def all(self) -> list[Modifier]:
        """All registered Modifiers, in registration order."""
        return [self._by_role[r] for r in self._order]

    # ---- lifecycle hook ---------------------------------------------

    def attach_all(self, runtime: Any) -> None:
        """One-time post-registration lifecycle hook.

        Each registered Modifier that defines an ``attach`` method
        gets called with the runtime so it can wire up its own
        runtime-coupled assets that don't fit the meta.yaml
        ``components:`` shape (in-tree plugins ship sibling tools
        as additional components instead, but the hook stays available
        for plugins that need direct runtime references at startup).

        Errors in one Modifier's attach must not block the others —
        plugins are strictly additive (CLAUDE.md invariant).
        """
        import logging
        log = logging.getLogger(__name__)
        for modifier in self.all():
            attach = getattr(modifier, "attach", None)
            if attach is None:
                continue
            try:
                attach(runtime)
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "Modifier %r attach() raised %s; continuing "
                    "without its runtime hooks",
                    getattr(modifier, "name", "?"), e,
                )
