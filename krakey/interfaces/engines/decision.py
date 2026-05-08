"""``DecisionEngine`` — translate Self's [DECISION] text into structure.

Two impls ship in-tree, mutually exclusive (the slot picks one):

  * ``ToolCallParserDecisionEngine`` (default) — scripted scan for
    ``<tool_call>{...}</tool_call>`` blocks. No LLM call. Fast, no
    extra cost, but only as smart as the parser regex.
  * ``HypothalamusDecisionEngine`` — LLM-based translator that takes
    Self's free-form [DECISION] text and produces structured
    ``ToolCall``s + memory writes + sleep flag. Costs an LLM call but
    handles ambiguous decisions ("remember that X", "stop doing Y")
    that the script parser would miss.

The Engine slot replaces the previous Modifier-role gimmick (where
"hypothalamus" was a Modifier the heartbeat probed via
``modifiers.by_role``). Decision translation is a core flow that
must always work; making it an Engine guarantees one impl is wired.

``DecisionResult`` and ``ToolCall`` move here from
``krakey/interfaces/modifier.py`` because they cross the
heartbeat ↔ Engine boundary, not the Modifier boundary. ``ParseFailure``
is the diagnostic for malformed ``<tool_call>`` blocks — the heartbeat
surfaces these as corrective system_event Stimuli back to Self.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolCall:
    """Structured tool invocation produced by a ``DecisionEngine``."""
    tool: str
    intent: str
    params: dict[str, Any] = field(default_factory=dict)
    adrenalin: bool = False


@dataclass
class ParseFailure:
    """One ``<tool_call>`` block whose parsing surfaced a problem.

    ``salvaged=False`` (default): call was lost; orchestrator pushes a
    corrective stimulus reporting the failed dispatch.
    ``salvaged=True``: JSON had trailing junk but the parser recovered
    the object; the call DID dispatch and a ``ToolCall`` was emitted
    alongside this ``ParseFailure``. Surfaced so Self gets format-
    correction feedback even on the salvage path.
    """
    payload: str
    error: str
    block_index: int
    salvaged: bool = False


@dataclass
class DecisionResult:
    """Aggregate result of one decision-translation pass.

    Consumed by ``DispatchEngine`` (which fires the tool calls + memory
    writes/updates) and the heartbeat (which checks ``sleep`` to gate
    the sleep transition).

    ``parse_failures`` is non-empty only when the impl is the scripted
    parser AND it encountered malformed blocks. LLM-based impls
    typically return [] here — they either parse cleanly or raise.
    """
    tool_calls: list[ToolCall] = field(default_factory=list)
    memory_writes: list[dict[str, Any]] = field(default_factory=list)
    memory_updates: list[dict[str, Any]] = field(default_factory=list)
    sleep: bool = False
    parse_failures: list[ParseFailure] = field(default_factory=list)


@runtime_checkable
class DecisionEngine(Protocol):
    """Translate Self's response into a ``DecisionResult``.

    Impls receive the parsed ``decision`` text, the full ``raw``
    response (some impls scan it as a fallback if [DECISION] is empty),
    and the live tool descriptions list (in case the impl wants to
    validate names or pass them to an LLM). Returns the structured
    result; never raises for parse-level problems (those go in
    ``parse_failures``). Reserved exceptions: ``ValueError`` for
    impl-internal config errors, network errors for LLM-based impls
    — the heartbeat catches and surfaces those as system_event stimuli.
    """

    async def translate(
        self,
        decision: str,
        raw: str,
        tools: list[dict[str, Any]],
    ) -> DecisionResult: ...
