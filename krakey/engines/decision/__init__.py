"""``decision`` Engine — translate Self's [DECISION] text into structure.

Two impls ship in-tree, mutually exclusive (the slot picks one):

  * ``ToolCallParserDecisionEngine`` (default) — scripted scan for
    ``<tool_call>{...}</tool_call>`` blocks. No LLM call. Fast, cheap,
    but only as smart as the parser regex.
  * ``HypothalamusDecisionEngine`` (alt, lifted from the retired
    plugin in step 7b) — LLM-based translator that takes Self's
    free-form [DECISION] text and produces structured ToolCalls +
    memory writes + sleep flag. Costs an LLM call but handles
    ambiguous decisions ("remember that X", "stop doing Y") that the
    script parser would miss.

Engine override mechanism replaces the previous Modifier-role gimmick
where "hypothalamus" was a Modifier the heartbeat probed via
``modifiers.by_role``. Decision translation is a core flow that must
always work; making it an Engine guarantees one impl is wired.
"""
from krakey.engines.decision.tool_call_parser import (
    ToolCallParserDecisionEngine,
)

__all__ = ["ToolCallParserDecisionEngine"]
