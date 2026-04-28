"""Prompt assembler — Layers ordered for LLM prefix cache (DevSpec §3.6).

Layer order (most-stable cacheable prefix first → most-volatile last):

    1. dna                  — never changes at runtime
    2. self_model           — only changes when Self writes <self-model>
    3. capabilities         — only changes on plugin reload
    4. action_format        — taught only when no decision-translator
                              role is registered (plugins delete this
                              key when they own the dispatch path)
    5. in_mind_instructions — standing instruction (added by in_mind plugin)
    6. stimulus             — often empty / repeated tool feedback
    7. recall               — derived from stimulus
    8. in_mind_round        — virtual "Heartbeat #now (in mind)" round
                              (filled by in_mind plugin; empty otherwise)
    9. history              — appends every beat but stable prefix
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
from krakey.prompt.dna import DNA
from krakey.prompt.elements import PromptElements
from krakey.prompt.layers import ACTION_FORMAT_LAYER, HEARTBEAT_QUESTION
from krakey.prompt.views import (
    CapabilityView,
    SlidingWindowRound,
    StatusSnapshot,
)

if TYPE_CHECKING:
    from krakey.memory.recall import RecallResult


# Default ordered element keys. Runtime constructs a PromptElements
# with these keys (in order) before letting plugins modify.
DEFAULT_ELEMENT_KEYS: tuple[str, ...] = (
    "dna",
    "self_model",
    "capabilities",
    "action_format",
    "in_mind_instructions",
    "stimulus",
    "recall",
    "in_mind_round",
    "history",
    "status",
    "heartbeat_question",
)


def _format_stim(s: Stimulus) -> list[str]:
    lines = [
        "---",
        f"来源: {s.source} | adrenalin: {s.adrenalin}",
        f"内容: {s.content}",
    ]
    # When the recall_anchor plugin couldn't find any GraphMemory
    # context for this stimulus on the previous beat, the orchestrator
    # re-pushes it with an incremented `recall_retries` counter. Surface
    # that to Self so it knows the [GRAPH MEMORY] layer has nothing to
    # offer for this signal — Self can ask follow-up questions or fall
    # back to its own knowledge instead of assuming silent context.
    retries = s.metadata.get("recall_retries", 0)
    if retries:
        lines.append(
            f"⚠ 上一次心跳未召回到与本条相关的图记忆 (重试第 {retries} 次)"
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
        window: list[SlidingWindowRound],
        stimuli: list[Stimulus],
        current_time: datetime | None = None,
    ) -> PromptElements:
        """Build the default-state PromptElements before plugin
        modification. Each known key gets a value; empty-string slots
        (in_mind_instructions, in_mind_round) are reserved for plugins
        to fill in."""
        return PromptElements(initial=[
            ("dna", DNA),
            ("self_model", self.render_self_model(self_model)),
            ("capabilities", self.render_capabilities(capabilities)),
            ("action_format", ACTION_FORMAT_LAYER),
            ("in_mind_instructions", ""),
            ("stimulus", self.render_stimulus(stimuli, current_time)),
            ("recall", self.render_recall(recall)),
            ("in_mind_round", ""),
            ("history", self.render_history(window)),
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
        window: list[SlidingWindowRound],
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
            "可用 Tools (本跳已注册):\n"
            f"{lines}"
        )

    def render_status(self, s: StatusSnapshot) -> str:
        return (
            "# [STATUS]\n"
            f"Graph Memory: {s.gm_node_count} nodes, "
            f"{s.gm_edge_count} edges\n"
            f"疲惫度: {s.fatigue_pct}% {s.fatigue_hint}\n"
            f"上次 Sleep: {s.last_sleep_time}\n"
            f"心跳数 (自上次 Sleep): {s.heartbeats_since_sleep}"
        )

    def render_recall(self, recall: "RecallResult") -> str:
        if not recall.nodes and not recall.edges:
            return "# [GRAPH MEMORY]\n(no recall)"
        lines = ["# [GRAPH MEMORY]"]
        for n in recall.nodes:
            kw = ", ".join(n.get("neighbor_keywords", []))
            lines.append(
                f"- [{n.get('name', '?')}] ({n.get('category', '?')}) — "
                f"{n.get('description', '')}"
            )
            if kw:
                lines.append(f"  相邻: {kw}")
        for e in recall.edges:
            lines.append(
                f"- [{e.get('source', '?')}] "
                f"--{e.get('predicate', '?')}--> "
                f"[{e.get('target', '?')}]"
            )
        return "\n".join(lines)

    def render_history(self, window: list[SlidingWindowRound]) -> str:
        """Render the [HISTORY] layer from real heartbeat rounds only.

        The in-mind virtual round (``--- Heartbeat #now (in mind) ---``)
        is rendered by the in_mind plugin into the separate
        ``in_mind_round`` element, which renders just before this one.
        """
        lines = ["# [HISTORY]"]
        if not window:
            return "# [HISTORY]\n(empty)"
        for r in window:
            lines.append(f"--- Heartbeat #{r.heartbeat_id} ---")
            lines.append(f"Stimulus: {r.stimulus_summary}")
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
                f"# [STIMULUS]\n本次收到 {len(stimuli)} 条信号，按来源分组："
            ]

            if incoming:
                lines.append(
                    "\n## 用户/外部输入 (INCOMING — 别人对你说的话, 需要回应)"
                )
                for s in incoming:
                    lines.extend(_format_stim(s))

            if own_actions:
                lines.append(
                    "\n## 你自己刚才行动的结果 (YOUR RECENT ACTIONS — "
                    "这是你刚才说/做的话和动作的回执, 不是用户在跟你互动!)"
                )
                for s in own_actions:
                    lines.extend(_format_stim(s))

            if system:
                lines.append("\n## 系统事件 (SYSTEM)")
                for s in system:
                    lines.extend(_format_stim(s))

        if current_time is not None:
            ts = current_time.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"\n当前时间: {ts}")

        return "\n".join(lines)
