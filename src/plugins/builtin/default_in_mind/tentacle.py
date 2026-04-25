"""``update_in_mind`` tentacle — Self's interface to mutate the
in_mind state.

Registered automatically by ``InMindReflect.attach(runtime)`` after
the Reflect is loaded. Internal (``is_internal=True``) — the result
is just a feedback receipt for Self, not something to broadcast to
the human.

Argument shape (matches what Self emits in the [ACTION] JSONL block):

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

from src.interfaces.tentacle import Tentacle
from src.models.stimulus import Stimulus

if TYPE_CHECKING:
    from src.plugins.builtin.default_in_mind.reflect import (
        InMindReflectImpl,
    )


class UpdateInMindTentacle(Tentacle):
    """Self-facing tentacle that calls back into the in_mind Reflect.

    Held by reference, not registry lookup, so the link is direct
    and immune to reflect re-registration weirdness.
    """

    def __init__(self, reflect: "InMindReflectImpl"):
        self._reflect = reflect

    @property
    def name(self) -> str:
        return "update_in_mind"

    @property
    def description(self) -> str:
        return (
            "更新你的 in_mind 心智状态 (thoughts / mood / focus). "
            "三个参数都可选: 不传 = 不动该字段, 空字符串 = 显式清空, "
            "非空字符串 = 设置。每当心头主题、情绪或专注改变, 立刻调用。"
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "thoughts": {
                    "type": "string",
                    "description": (
                        "当下心头最重要的事 (一句话即可)。空字符串 = 清空。"
                    ),
                },
                "mood": {
                    "type": "string",
                    "description": (
                        "当前情绪 + 简短原因。空字符串 = 清空。"
                    ),
                },
                "focus": {
                    "type": "string",
                    "description": "正在专注的具体事。空字符串 = 清空。",
                },
            },
        }

    @property
    def sandboxed(self) -> bool:
        # Pure in-process state mutation, no external surface.
        return False

    @property
    def is_internal(self) -> bool:
        return True

    async def execute(
        self, intent: str, params: dict[str, Any],
    ) -> Stimulus:
        # Pull only the three known fields; ignore extras silently
        # (forward-compat if Self over-specifies).
        new_state = self._reflect.update(
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
            type="tentacle_feedback", source=f"tentacle:{self.name}",
            content=content, timestamp=datetime.now(), adrenalin=False,
        )
