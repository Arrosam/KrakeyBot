"""Static prose layers injected by ``PromptBuilder`` (non-DNA).

Only ``HEARTBEAT_QUESTION`` lives here now — it's the trailing closer
that wraps every beat regardless of which decision/memory/recall engine
is wired in, so it belongs to the prompt builder.

The ``[ACTION FORMAT]`` prose used to live here too, but that prose is
**engine-specific** — different decision engines teach Self different
output shapes (``<tool_call>`` JSON for the parser engine; natural
language for the hypothalamus engine). Each engine now owns its own
prose and injects it via the ``modify_prompt`` hook on its
DecisionEngine impl. PromptBuilder pre-allocates an empty
``action_format`` slot; whichever engine is wired in fills it. This
keeps prompt builder code engine-agnostic.
"""
from __future__ import annotations


HEARTBEAT_QUESTION = (
    "# [HEARTBEAT]\n"
    "What do you notice? What matters? What do you do?\n"
    "Respond using [THINKING] / [DECISION] / [NOTE] / [IDLE]."
)
