"""Prompt-side rendering for the in_mind Modifier.

Two pieces:

  * ``IN_MIND_INSTRUCTIONS_LAYER`` — constant string injected as a
    standalone prompt layer (between [CAPABILITIES] and [STIMULUS]).
    Cache-friendly: only present / absent depending on whether the
    in_mind Modifier is registered, never changing content within a
    run. Its job is to teach Self **when** to call ``update_in_mind``.
  * ``render_virtual_round(state)`` — formats the per-beat virtual
    round prepended to [HISTORY] when at least one in_mind field is
    populated.
"""
from __future__ import annotations

from krakey.plugins.in_mind_note.state import InMindState


IN_MIND_INSTRUCTIONS_LAYER = """# [IN MIND — operating constraint]
Your "mental state" (Thoughts / Mood / Focus) is the single source
other systems read for "what is on your mind right now"; it shows up
continuously at the top of [HISTORY] inside the
"Heartbeat #now (in mind)" block.

Whenever ANY of the following happens, you must immediately call the
update_in_mind tool to update the corresponding field:

- Your thinking focus shifts (new topic / new question / new lead)
- Your mood changes meaningfully
- What you are concretely focused on changes

Only update the fields that changed. All three parameters are optional:

  thoughts: the most important thing on your mind (one sentence is fine)
  mood:     current mood + brief reason
  focus:    the concrete thing you are focused on

Omit a field = leave it unchanged; pass an empty string = explicit clear.

Not updating = other systems see stale info and make wrong decisions.
This is a standing instruction; it applies every beat — no need to wait
for a reminder."""


def render_virtual_round(state: InMindState) -> str | None:
    """Return the multi-line block to prepend at the head of
    [HISTORY], or None if every field is empty (no point inserting
    a round of nothing — wastes prompt tokens + adds visual noise).
    """
    if state.is_empty():
        return None
    lines = ["--- Heartbeat #now (in mind) ---"]
    # Render only non-empty fields so Self isn't told "Mood: " on a
    # line. Empty = "I never set this", not "I am literally feeling
    # nothing".
    if state.thoughts:
        lines.append(f"Thoughts: {state.thoughts}")
    if state.mood:
        lines.append(f"Mood: {state.mood}")
    if state.focus:
        lines.append(f"Focus: {state.focus}")
    return "\n".join(lines)
