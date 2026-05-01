"""``update_in_mind`` tool — Self's interface to mutate the
in_mind state.

Built by ``build_tool(ctx)`` as the second component of the
``in_mind_note`` plugin. Pulls the already-built modifier instance
from ``ctx.plugin_cache`` (the modifier factory ran first because
``components:`` lists it first). The execute() result is a feedback
receipt for Self only — there's no separate human-facing channel
to broadcast to (in_mind is pure inner state).

Argument shape (matches what Self emits in the <tool_call> block):

    {"name": "update_in_mind", "arguments": {
        "thoughts": "...",   # all three optional
        "mood": "...",
        "focus": "..."
    }}

Semantics:
  * field omitted (key not present in arguments) → leave alone
  * field present, empty string → explicit clear
  * field present, non-empty string → set
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from datetime import datetime

from krakey.interfaces.tool import Tool
from krakey.models.stimulus import Stimulus
from krakey.plugins.in_mind_note import _CACHE_KEY

if TYPE_CHECKING:
    from krakey.interfaces.plugin_context import PluginContext
    from krakey.plugins.in_mind_note.modifier import (
        InMindModifierImpl,
    )


def build_tool(ctx: "PluginContext") -> "UpdateInMindTool | None":
    """Factory for the second component. Grabs the modifier instance
    that the modifier factory stashed in ``ctx.plugin_cache`` and wires
    the tool to it. Returns ``None`` (opt-out) if the modifier
    factory didn't run — the additive-plugin invariant: a missing
    half degrades, doesn't crash."""
    modifier = ctx.plugin_cache.get(_CACHE_KEY)
    if modifier is None:
        import logging
        logging.getLogger(__name__).warning(
            "in_mind_note tool skipped: modifier not in "
            "plugin_cache. Components likely loaded out of order.",
        )
        return None
    return UpdateInMindTool(modifier)


class UpdateInMindTool(Tool):
    """Self-facing tool that calls back into the in_mind Modifier.

    Held by reference, not registry lookup, so the link is direct
    and immune to modifier re-registration weirdness.
    """

    def __init__(self, modifier: "InMindModifierImpl"):
        self._modifier = modifier

    @property
    def name(self) -> str:
        return "update_in_mind"

    @property
    def description(self) -> str:
        return (
            "Update your in_mind mental state (thoughts / mood / focus). "
            "All three parameters are optional: omit = leave unchanged, "
            "empty string = explicit clear, non-empty = set. Call this "
            "whenever the topic on your mind, your mood, or your focus "
            "changes."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "thoughts": {
                    "type": "string",
                    "description": (
                        "The most important thing on your mind right now "
                        "(one sentence is enough). Empty string = clear."
                    ),
                },
                "mood": {
                    "type": "string",
                    "description": (
                        "Current mood + a brief reason. Empty string = clear."
                    ),
                },
                "focus": {
                    "type": "string",
                    "description": (
                        "What you are concretely focused on. "
                        "Empty string = clear."
                    ),
                },
            },
        }

    async def execute(
        self, intent: str, params: dict[str, Any],
    ) -> Stimulus:
        # Pull only the three known fields; ignore extras silently
        # (forward-compat if Self over-specifies).
        new_state = self._modifier.update(
            thoughts=params.get("thoughts"),
            mood=params.get("mood"),
            focus=params.get("focus"),
        )
        # Build a compact human-readable receipt that names what changed
        # so Self can verify the update landed.
        changed = [
            k for k in ("thoughts", "mood", "focus")
            if k in params
        ]
        if not changed:
            content = (
                "in_mind: no field updated (no `thoughts` / `mood` / "
                "`focus` argument passed). Current state preserved."
            )
        else:
            shown = ", ".join(
                f"{k}={new_state.get(k, '')!r}" for k in changed
            )
            content = f"in_mind updated: {shown}"
        return Stimulus(
            type="tool_feedback", source=f"tool:{self.name}",
            content=content, timestamp=datetime.now(), adrenalin=False,
        )
