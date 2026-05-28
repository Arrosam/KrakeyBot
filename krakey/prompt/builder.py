"""Prompt assembler — Layers ordered for LLM prefix cache (DevSpec §3.6).

Layer order (most-stable cacheable prefix first → most-volatile last):

    1. dna                  — never changes at runtime
    2. self_model           — only changes when Self writes <self-model>
    3. capabilities         — only changes on plugin reload
    4. action_format        — taught only when no decision-translator
                              role is registered (plugins delete this
                              key when they own the dispatch path)
    5. in_mind_instructions — standing instruction (added by in_mind plugin)
    6. recall               — derived from stimulus
    7. in_mind_round        — virtual "Heartbeat #now (in mind)" round
                              (filled by in_mind plugin; empty otherwise)
    8. history              — appends every beat but stable prefix
    9. stimulus             — volatile per-beat content placed AFTER the
                              stable history prefix; accepted prompt
                              prefix-cache trade-off (history cacheable,
                              stimulus invalidates only trailing tokens)
    10. status              — every beat changes; near the end so it
                              doesn't invalidate the cacheable prefix
    11. heartbeat_question  — end anchor

Plugins receive a ``PromptElements`` per heartbeat (via their Modifier's
``modify_prompt`` hook) and can read/write/delete any element. The
runtime tracks per-element modifications and warns on conflicts.

This module owns the LAYER RENDERING logic (``render_*`` helpers).
The runtime side composes a PromptElements dict from these renders +
hands it to plugins + serializes to a string.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import yaml

from krakey.models.stimulus import Stimulus
from krakey.prompt.dna import get_dna
from krakey.prompt.elements import PromptElements
from krakey.prompt.layers import HEARTBEAT_QUESTION
from krakey.prompt.views import (
    CapabilityView,
    ExplicitHistoryRound,
    StatusSnapshot,
)

if TYPE_CHECKING:
    from krakey.interfaces.engines.recall import RecallResult


# Default ordered element keys. Runtime constructs a PromptElements
# with these keys (in order) before letting plugins modify.
DEFAULT_ELEMENT_KEYS: tuple[str, ...] = (
    # ``bootstrap_intro`` is empty by default; the bootstrap plugin
    # populates it via modify_prompt during the bootstrap phase.
    # Pre-allocating the slot at the head of the list ensures the
    # injected intro renders BEFORE DNA — without this the plugin's
    # __setitem__ would append a brand-new key past
    # heartbeat_question, putting BOOTSTRAP_PROMPT after Self's
    # trailing prompt.
    "bootstrap_intro",
    "dna",
    "self_model",
    "capabilities",
    "action_format",
    "in_mind_instructions",
    "recall",
    "in_mind_round",
    "history",
    "stimulus",
    "status",
    "heartbeat_question",
)


def _format_stim(s: Stimulus) -> list[str]:
    lines = [
        "---",
        f"source: {s.source} | adrenalin: {s.adrenalin}",
        f"content: {s.content}",
    ]
    # When recall couldn't find any GraphMemory context for this
    # stimulus on the previous beat, the orchestrator re-pushes it
    # with an incremented `recall_retries` counter. Surface
    # that to Self so it knows the [GRAPH MEMORY] layer has nothing to
    # offer for this signal — Self can ask follow-up questions or fall
    # back to its own knowledge instead of assuming silent context.
    retries = s.metadata.get("recall_retries", 0)
    if retries:
        lines.append(
            f"⚠ no related graph memory recalled on the previous "
            f"heartbeat (retry #{retries})"
        )
    return lines


class PromptBuilder:
    """Renders the canonical default elements, then assembles a
    PromptElements collection plugins can mutate.

    Two entry points:
      * ``build_default_elements(...)`` — produces a fully-populated
        PromptElements (with all DEFAULT_ELEMENT_KEYS set, some to
        empty strings if the layer is contextually absent).
      * ``render(elements)`` — concatenates the elements into the
        final prompt string.
    """

    def build_default_elements(
        self,
        *,
        self_model: dict[str, Any],
        capabilities: list[CapabilityView],
        status: StatusSnapshot,
        recall: "RecallResult",
        window: list[ExplicitHistoryRound],
        stimuli: list[Stimulus],
        current_time: datetime | None = None,
    ) -> PromptElements:
        """Build the default-state PromptElements before plugin
        modification. Each known key gets a value; empty-string slots
        (in_mind_instructions, in_mind_round) are reserved for plugins
        to fill in."""
        return PromptElements(initial=[
            # Empty by default; bootstrap plugin populates this via
            # modify_prompt when bootstrap is active. Pre-allocated
            # at the head of the order so the intro lands before
            # DNA + ahead of every other element.
            ("bootstrap_intro", ""),
            # ``get_dna()`` re-reads ``krakey/prompt/dna.txt`` only
            # when its mtime changes — live prompt-tuning during
            # development without a runtime restart.
            ("dna", get_dna()),
            ("self_model", self.render_self_model(self_model)),
            ("capabilities", self.render_capabilities(capabilities)),
            # The decision engine owns this slot; its modify_prompt
            # hook fills it (see e.g. ToolCallParserDecisionEngine
            # and HypothalamusDecisionEngine). Builder stays
            # engine-agnostic — pre-allocated empty string here just
            # pins the position in the rendered prompt order.
            ("action_format", ""),
            ("in_mind_instructions", ""),
            ("recall", self.render_recall(recall)),
            ("in_mind_round", ""),
            ("history", self.render_history(window)),
            ("stimulus", self.render_stimulus(stimuli, current_time)),
            ("status", self.render_status(status)),
            ("heartbeat_question", HEARTBEAT_QUESTION),
        ])

    def render(self, elements: PromptElements) -> str:
        return elements.render()

    def build(
        self,
        *,
        self_model: dict[str, Any],
        capabilities: list[CapabilityView],
        status: StatusSnapshot,
        recall: "RecallResult",
        window: list[ExplicitHistoryRound],
        stimuli: list[Stimulus],
        current_time: datetime | None = None,
    ) -> str:
        """Convenience: build the default elements (no plugin
        modification) and serialize. Used by tests that exercise the
        basic prompt shape without the runtime's full Modifier pipeline.
        Production callers should use ``build_default_elements`` +
        plugin ``modify_prompt`` hooks + ``render`` instead."""
        return self.render(self.build_default_elements(
            self_model=self_model, capabilities=capabilities,
            status=status, recall=recall, window=window, stimuli=stimuli,
            current_time=current_time,
        ))

    # ---- layer renderers (used by build_default_elements) -----------

    def render_self_model(self, sm: dict[str, Any]) -> str:
        body = (
            yaml.safe_dump(sm, allow_unicode=True, sort_keys=False).strip()
            if sm
            else "(empty)"
        )
        return f"# [SELF-MODEL]\n{body}"

    def render_capabilities(self, tools: list[CapabilityView]) -> str:
        if tools:
            lines = "\n".join(
                f"- {t.name}: {t.description}" for t in tools
            )
        else:
            lines = "(none)"
        return (
            "# [CAPABILITIES]\n"
            "Available tools (registered this beat):\n"
            f"{lines}"
        )

    def render_status(self, s: StatusSnapshot) -> str:
        return (
            "# [STATUS]\n"
            f"Graph Memory: {s.gm_node_count} nodes, "
            f"{s.gm_edge_count} edges\n"
            f"Fatigue: {s.fatigue_pct}% {s.fatigue_hint}\n"
            f"Last Sleep: {s.last_sleep_time}\n"
            f"Heartbeats since last Sleep: {s.heartbeats_since_sleep}"
        )

    def render_recall(self, recall: "RecallResult") -> str:
        if not recall.nodes and not recall.edges:
            return "# [GRAPH MEMORY]\n(no recall)"
        lines = ["# [GRAPH MEMORY]"]
        recalled_names: set[str] = set()
        for n in recall.nodes:
            recalled_names.add(n.get("name", ""))
        unexplored_count = 0
        kb_refs: list[str] = []
        for n in recall.nodes:
            kw = ", ".join(n.get("neighbor_keywords", []))
            lines.append(
                f"- [{n.get('name', '?')}] ({n.get('category', '?')}) — "
                f"{n.get('description', '')}"
            )
            if kw:
                lines.append(f"  neighbors: {kw}")
                for k in n.get("neighbor_keywords", []):
                    if k not in recalled_names:
                        unexplored_count += 1
            meta = n.get("metadata") or {}
            if meta.get("is_kb_index"):
                kb_id = meta.get("kb_id", n.get("name", "?"))
                kb_refs.append(kb_id)
        for e in recall.edges:
            lines.append(
                f"- [{e.get('source', '?')}] "
                f"--{e.get('predicate', '?')}--> "
                f"[{e.get('target', '?')}]"
            )
        if unexplored_count > 0 or kb_refs:
            lines.append("")
            lines.append("Exploration hints:")
            if unexplored_count > 0:
                lines.append(
                    f"- {unexplored_count} neighbor(s) not shown "
                    f"(use memory_recall to explore)"
                )
            for kb_id in kb_refs:
                lines.append(
                    f"- KB \"{kb_id}\" available — "
                    f"use memory_recall with kb_id to browse full content"
                )
        return "\n".join(lines)

    def render_history(self, window: list[ExplicitHistoryRound]) -> str:
        """Render the [HISTORY] layer from real heartbeat rounds only.

        The in-mind virtual round (``--- Heartbeat #now (in mind) ---``)
        is rendered by the in_mind plugin into the separate
        ``in_mind_round`` element, which renders just before this one.
        """
        lines = ["# [HISTORY]"]
        if not window:
            return "# [HISTORY]\n(empty)"
        # Locate the LAST heartbeat_id reset within the window.
        # heartbeat_id resets to 1 each process start (never persisted),
        # so a non-monotonic step signals a session boundary.
        boundary_idx = None
        for i in range(1, len(window)):
            if window[i].heartbeat_id <= window[i - 1].heartbeat_id:
                boundary_idx = i   # keep updating → ends on the LAST reset
        for idx, r in enumerate(window):
            if boundary_idx is not None and idx == boundary_idx:
                lines.append(
                    "--- SESSION BOUNDARY"
                    " (above: previous session | below: current session) ---"
                )
            lines.append(f"--- Heartbeat #{r.heartbeat_id} ---")
            lines.append(f"Stimulus: {r.stimulus_summary}")
            if r.recall_summary:
                lines.append(f"Recalled: {r.recall_summary}")
            if r.thinking_text:
                lines.append(f"Thinking: {r.thinking_text}")
            lines.append(f"Decision: {r.decision_text}")
            if r.note_text:
                lines.append(f"Note: {r.note_text}")
        return "\n".join(lines)

    def render_stimulus(
        self,
        stimuli: list[Stimulus],
        current_time: datetime | None,
    ) -> str:
        if not stimuli:
            lines = ["# [STIMULUS]\n(no new signals)"]
        else:
            incoming: list[Stimulus] = []
            own_actions: list[Stimulus] = []
            system: list[Stimulus] = []
            for s in stimuli:
                if s.type == "user_message":
                    incoming.append(s)
                elif s.type == "tool_feedback":
                    own_actions.append(s)
                else:
                    system.append(s)

            lines = [
                f"# [STIMULUS]\nReceived {len(stimuli)} signals, grouped by source:"
            ]

            if incoming:
                lines.append(
                    "\n## INCOMING (user / external input — what others said "
                    "to you; needs a response)"
                )
                for s in incoming:
                    lines.extend(_format_stim(s))

            if own_actions:
                lines.append(
                    "\n## YOUR RECENT ACTIONS (feedback from what YOU just "
                    "said/did — not the user talking to you!)"
                )
                for s in own_actions:
                    lines.extend(_format_stim(s))

            if system:
                lines.append("\n## SYSTEM events")
                for s in system:
                    lines.extend(_format_stim(s))

        if current_time is not None:
            ts = current_time.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"\ncurrent time: {ts}")

        return "\n".join(lines)
