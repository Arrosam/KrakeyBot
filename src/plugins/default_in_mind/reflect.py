"""``default_in_mind`` Reflect — owner of Self's in-mind state.

Imported lazily by ``src.plugin_system.discovery.load_component``.
The plugin contributes TWO components: this reflect (owns the state)
and ``tentacle.UpdateInMindTentacle`` (lets Self mutate the state via
the normal action-dispatch pipeline). The factories share the
reflect instance via ``ctx.plugin_cache`` — same pattern telegram +
web_chat use to share a client / history across their components.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from src.plugins.default_in_mind import state as state_mod
from src.plugins.default_in_mind.state import InMindState  # noqa: F401
from src.plugins.default_in_mind.tentacle import _CACHE_KEY

if TYPE_CHECKING:
    from src.interfaces.plugin_context import PluginContext

_log = logging.getLogger(__name__)


# Lock-in (Samuel 2026-04-25): state files live under
# workspace/data/, not workspace/reflects/. State ≠ config.
DEFAULT_STATE_PATH = Path("workspace") / "data" / "in_mind.json"


class InMindReflectImpl:
    """Runtime owner of Self's in_mind state.

    Reads from / writes to the JSON state file. Exposes ``read`` /
    ``update`` for the prompt builder + the ``update_in_mind``
    tentacle. No LLM in this Reflect — pure state plumbing.
    """

    name = "default_in_mind"
    kind = "in_mind"

    def __init__(self, state_path: str | Path = DEFAULT_STATE_PATH):
        self._state_path = Path(state_path)
        self._state = state_mod.load(self._state_path)

    # ---- public surface (InMindReflect Protocol) -----------------------

    def read(self) -> dict[str, str]:
        """Snapshot dict — safe to mutate; we don't share the inner
        dataclass across callers."""
        return dict(self._state.to_dict())

    def update(
        self,
        thoughts: str | None = None,
        mood: str | None = None,
        focus: str | None = None,
    ) -> dict[str, str]:
        """Patch state.

        Field semantics:
          * ``None``         — leave alone
          * empty string     — clear that field
          * non-empty string — set that field

        Persists immediately (atomic write). Returns the post-update
        snapshot for the tentacle's feedback receipt.
        """
        if thoughts is not None:
            self._state.thoughts = thoughts
        if mood is not None:
            self._state.mood = mood
        if focus is not None:
            self._state.focus = focus
        # Bump timestamp only if SOMETHING was passed in. Keeping
        # updated_at sticky on no-op calls would be a confusing log
        # signal.
        if (thoughts is not None or mood is not None
                or focus is not None):
            self._state.updated_at = state_mod.now_iso()
        try:
            state_mod.save(self._state, self._state_path)
        except OSError as e:
            # Best-effort: in-memory state still updated, but disk
            # write failed. Log + continue. The next successful
            # update will re-persist; meanwhile the prompt still
            # reads the in-memory copy correctly.
            _log.warning(
                "in_mind: state save to %s failed: %s; in-memory "
                "state remains current",
                self._state_path, e,
            )
        return self.read()


def build_reflect(ctx: "PluginContext") -> InMindReflectImpl:
    """Factory invoked by ``load_component``. Stashes the instance in
    ``ctx.plugin_cache`` so the sibling tentacle factory (loaded next,
    same plugin) can wire to the same reflect.

    ``ctx.deps.in_mind_state_path`` is honored when provided so tests
    can isolate the state file. Production leaves it ``None`` and
    the Reflect uses the locked-in
    ``workspace/data/in_mind.json``. No LLM purposes declared.
    """
    state_path = (
        getattr(ctx.deps, "in_mind_state_path", None)
        or DEFAULT_STATE_PATH
    )
    reflect = InMindReflectImpl(state_path=state_path)
    ctx.plugin_cache[_CACHE_KEY] = reflect
    return reflect
