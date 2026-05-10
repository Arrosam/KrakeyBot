"""``ContextEngine`` — the prompt-assembly slot's Protocol.

Default impl ``DefaultContextEngine`` (was ``PromptBuilder``) walks
DNA + self-model + capabilities + recall + history + status + stimulus
into ``PromptElements``, lets each Modifier mutate, then renders to a
single string for the Self LLM.

Users replace via ``cfg.core_implementations.context``. A custom Engine
controls layer order, layer rendering, and the final string format —
no constraints except the two methods below + the Modifier mutation
contract (the runtime gives every Modifier a ``modify_prompt`` chance
between ``build_default_elements`` and ``render``).

**Layer-order guidance** (not Protocol-enforced): the default ordering
is most-stable cacheable prefix first → most-volatile last, to maximize
LLM prefix-cache hit rate. Custom engines that reorder will work but
pay a cache penalty. Document this on the Engine, don't try to encode
in the Protocol.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from krakey.interfaces.engines.recall import RecallResult
    from krakey.models.stimulus import Stimulus
    # PromptElements + view dataclasses live in the default impl folder
    # (engines/context/) but the Protocol references them as types.
    # Forward-stringed to avoid the engines→interfaces cycle at runtime.


@runtime_checkable
class ContextEngine(Protocol):
    """Minimal surface the heartbeat invokes on the prompt-builder slot."""

    def build_default_elements(
        self,
        *,
        self_model: dict[str, Any],
        capabilities: list[Any],         # CapabilityView from default impl
        status: Any,                     # StatusSnapshot from default impl
        recall: "RecallResult",
        window: list[Any],               # ExplicitHistoryRound list
        stimuli: list["Stimulus"],
        current_time: datetime | None = None,
    ) -> Any:
        """Assemble the default ``PromptElements`` for this beat. The
        runtime hands the result to each Modifier's ``modify_prompt``
        before serialization.

        The return type is the Engine's own PromptElements-shaped
        object — typed as ``Any`` here because the concrete class
        ships with the impl. Modifiers receive a per-plugin binding
        and write/read string keys; the protocol doesn't pin the
        binding shape, only that it's the same object passed through.
        """
        ...

    def render(self, elements: Any) -> str:
        """Serialize a (possibly Modifier-mutated) ``PromptElements``
        into the final prompt string."""
        ...
