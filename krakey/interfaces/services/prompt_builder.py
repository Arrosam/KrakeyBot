"""``PromptBuilderLike`` — the prompt-assembly slot's Protocol.

Default implementation: ``krakey.prompt.builder.PromptBuilder``. Users
swap in a custom one via ``core_implementations.prompt_builder`` in
config.yaml, e.g. ``my_pkg.prompts:CustomPromptBuilder``.

Only the methods the runtime actually invokes are part of the Protocol.
Test conveniences (``PromptBuilder.build``) and per-layer renderers
(``render_status`` etc.) stay private to the default implementation —
a custom builder is free to compose the prompt however it wants as long
as ``build_default_elements`` produces a fully-populated
``PromptElements`` and ``render`` serializes it to a string.

**Layer-order contract**: the runtime depends on the canonical layer
order documented in ``krakey/prompt/builder.py`` (most-stable cacheable
prefix first → most-volatile last) for Anthropic prefix-cache hit
rate. Custom builders that reorder layers will work but pay a cache
penalty — Protocol can't enforce this; document it instead.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from krakey.memory.recall import RecallResult
    from krakey.models.stimulus import Stimulus
    from krakey.prompt.elements import PromptElements
    from krakey.prompt.views import (
        CapabilityView,
        SlidingWindowRound,
        StatusSnapshot,
    )


@runtime_checkable
class PromptBuilderLike(Protocol):
    """Minimal surface the runtime invokes on the prompt builder slot."""

    def build_default_elements(
        self,
        *,
        self_model: dict[str, Any],
        capabilities: list["CapabilityView"],
        status: "StatusSnapshot",
        recall: "RecallResult",
        window: list["SlidingWindowRound"],
        stimuli: list["Stimulus"],
        current_time: datetime | None = None,
    ) -> "PromptElements":
        """Assemble the canonical default ``PromptElements`` for this beat.

        After this returns, the runtime gives each Modifier a chance to
        mutate the elements before serialization (see
        ``heartbeat_orchestrator.build_self_prompt``).
        """
        ...

    def render(self, elements: "PromptElements") -> str:
        """Serialize a (possibly modifier-mutated) ``PromptElements``
        into the final prompt string sent to the Self LLM."""
        ...
