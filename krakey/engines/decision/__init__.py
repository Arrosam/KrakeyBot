"""``decision`` Engine — translate Self's [DECISION] text into structure.

The slot's catalog (default = ``tool_call_parser``, alternative =
``hypothalamus``) lives in ``meta.yaml`` next to this file. The
DecisionEngine Protocol lives at
``krakey.interfaces.engines.decision``.
"""
from krakey.engines.decision.hypothalamus import HypothalamusDecisionEngine
from krakey.engines.decision.tool_call_parser import (
    ToolCallParserDecisionEngine,
)

__all__ = [
    "ToolCallParserDecisionEngine",
    "HypothalamusDecisionEngine",
]
