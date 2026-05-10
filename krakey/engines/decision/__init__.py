"""``decision`` Engine — translate Self's [DECISION] text into structure.

Two impls ship in-tree, mutually exclusive (the slot picks one):

  * ``tool_call_parser`` (default) — scripted scan for
    ``<tool_call>{...}</tool_call>`` blocks. No LLM call. Fast, cheap,
    but only as smart as the parser regex.
  * ``hypothalamus`` — LLM-based translator that takes Self's free-form
    [DECISION] text and produces structured ToolCalls + memory writes
    + sleep flag. Costs an LLM call but handles ambiguous decisions
    ("remember that X", "stop doing Y") that the script parser misses.
    Bind ``llm.core_purposes.hypothalamus`` to a tag before switching.
"""
from krakey.engines.catalog import EngineImpl
from krakey.engines.decision.hypothalamus import HypothalamusDecisionEngine
from krakey.engines.decision.tool_call_parser import (
    ToolCallParserDecisionEngine,
)

BUILTIN_ENGINES = {
    "tool_call_parser": EngineImpl(
        cls=ToolCallParserDecisionEngine,
        description=(
            "Scripted <tool_call>{...}</tool_call> parser. No LLM "
            "call. Default."
        ),
    ),
    "hypothalamus": EngineImpl(
        cls=HypothalamusDecisionEngine,
        description=(
            "LLM-based natural-language translator. Bind "
            "core_purposes.hypothalamus to a tag before use."
        ),
    ),
}

DEFAULT_ENGINE = "tool_call_parser"

__all__ = [
    "BUILTIN_ENGINES",
    "DEFAULT_ENGINE",
    "ToolCallParserDecisionEngine",
    "HypothalamusDecisionEngine",
]
