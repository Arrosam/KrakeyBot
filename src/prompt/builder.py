"""Prompt assembler — Layers ordered for LLM prefix cache (DevSpec §3.6).

Layer order goes from most stable (cacheable prefix) to most volatile
(cache-breaking tail):

    1. DNA                  — never changes at runtime
    2. [SELF-MODEL]         — only changes when Self writes <self-model>
    3. [CAPABILITIES]       — only changes on plugin reload
    4. [STIMULUS]            — often empty / repeated tentacle feedback,
                               so bimodal: stable on quiet beats
    5. [GRAPH MEMORY]       — derived from [STIMULUS]; synchronized
                               cache state with it
    6. [HISTORY]            — appends every beat but has a stable prefix
    7. [STATUS]             — every beat changes (heartbeat counter,
                               fatigue); kept near the end so it does
                               NOT invalidate the stable prefix above
    8. [HEARTBEAT] question — end anchor

Per-stimulus timestamps are intentionally omitted — the trailer
``当前时间: YYYY-MM-DD HH:MM:SS`` at the bottom of [STIMULUS] is the
single authoritative "now" read by Self. Beat-level temporal ordering
lives in [HISTORY] via ``heartbeat_id``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import yaml

from typing import TYPE_CHECKING

from src.models.stimulus import Stimulus
from src.prompt.dna import DNA

if TYPE_CHECKING:
    from src.memory.recall import RecallResult


def _format_stim(s: Stimulus) -> list[str]:
    return [
        "---",
        f"来源: {s.source} | adrenalin: {s.adrenalin}",
        f"内容: {s.content}",
    ]


HEARTBEAT_QUESTION = (
    "# [HEARTBEAT]\n"
    "What do you notice? What matters? What do you do?\n"
    "Respond using [THINKING] / [DECISION] / [NOTE] / [HIBERNATE]."
)


@dataclass
class SlidingWindowRound:
    heartbeat_id: int
    stimulus_summary: str
    decision_text: str
    note_text: str


@dataclass
class StatusSnapshot:
    """Per-beat runtime numbers rendered in the [STATUS] layer.

    Replaces the previous ``status: dict[str, Any]`` contract. Used to
    be a free dict where producer / consumer / test fixture all kept
    their own copy of the schema; typoing a key (e.g. ``fatigue_pct``
    → ``fatigue_percent``) silently rendered the default value with no
    error. Now: producer constructs a ``StatusSnapshot``; field typo
    is a TypeError at construction.
    """
    gm_node_count: int
    gm_edge_count: int
    fatigue_pct: int
    fatigue_hint: str
    last_sleep_time: str
    heartbeats_since_sleep: int


@dataclass
class CapabilityView:
    """One row in the [CAPABILITIES] layer — name + one-line blurb.

    Replaces ``list[dict[str, Any]]`` with name/description keys.
    """
    name: str
    description: str


ACTION_FORMAT_LAYER = """# [ACTION FORMAT]
当你想调用 tentacles 时, 在你的回复里写一段 [ACTION]...[/ACTION] 块,
块内每行写一个 JSON 对象 (OpenAI tool_calls 风格):

[ACTION]
{"name": "<tentacle_name>", "arguments": {...}}
{"name": "<another>", "arguments": {...}, "adrenalin": true}
[/ACTION]

字段:
- name (str, 必需): 从 [CAPABILITIES] 里挑一个 tentacle 名字
- arguments (object, 可选): 该 tentacle 的参数, 不传 = 空对象 {}
- adrenalin (bool, 可选): 紧急标志, 不传 = false; 只在你想要这次动作的
  反馈打断后续 hibernate 时设 true

不需要调用 tentacle 的心跳 (例: 只是思考 / 写 [NOTE]) 直接省略 [ACTION] 块。
[ACTION] 块可以出现在 [DECISION] / [THINKING] 里, 也可以出现在 [DECISION]
之后, 都会被解析。一行解析失败不影响其它行。"""


class PromptBuilder:
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
        suppress_action_format: bool = False,
        in_mind: dict[str, str] | None = None,
        in_mind_instructions: str | None = None,
    ) -> str:
        """Assemble the Self prompt.

        ``suppress_action_format``: True → omit the ``[ACTION FORMAT]``
        layer. Set when a Reflect of kind="hypothalamus" is registered.

        ``in_mind`` / ``in_mind_instructions``: when an in_mind Reflect
        is registered, Runtime passes its state snapshot (``in_mind``)
        + the standing instruction text (``in_mind_instructions``) to
        the builder. The instructions block is added as a stable layer
        between [ACTION FORMAT] and [STIMULUS]; the state is
        prepended as a virtual "Heartbeat #now (in mind)" round at
        the head of [HISTORY] when at least one field is non-empty.
        Both are absent (no virtual round, no instruction layer) when
        no in_mind Reflect is registered — zero-plugin invariant.
        """
        layers: list[str] = [
            DNA,
            self._layer_self_model(self_model),
            self._layer_capabilities(capabilities),
        ]
        if not suppress_action_format:
            layers.append(ACTION_FORMAT_LAYER)
        if in_mind_instructions:
            layers.append(in_mind_instructions)
        layers.extend([
            self._layer_stimulus(stimuli, current_time),
            self._layer_recall(recall),
            self._layer_history(window, in_mind=in_mind),
            self._layer_status(status),
            HEARTBEAT_QUESTION,
        ])
        return "\n\n".join(layers)

    def _layer_self_model(self, sm: dict[str, Any]) -> str:
        body = (
            yaml.safe_dump(sm, allow_unicode=True, sort_keys=False).strip()
            if sm
            else "(empty)"
        )
        return f"# [SELF-MODEL]\n{body}"

    def _layer_capabilities(self, tentacles: list[CapabilityView]) -> str:
        if tentacles:
            lines = "\n".join(
                f"- {t.name}: {t.description}" for t in tentacles
            )
        else:
            lines = "(none)"
        return (
            "# [CAPABILITIES]\n"
            "可用 Tentacles (本跳已注册):\n"
            f"{lines}"
        )

    def _layer_status(self, s: StatusSnapshot) -> str:
        return (
            "# [STATUS]\n"
            f"Graph Memory: {s.gm_node_count} nodes, "
            f"{s.gm_edge_count} edges\n"
            f"疲惫度: {s.fatigue_pct}% {s.fatigue_hint}\n"
            f"上次 Sleep: {s.last_sleep_time}\n"
            f"心跳数 (自上次 Sleep): {s.heartbeats_since_sleep}"
        )

    def _layer_recall(self, recall: "RecallResult") -> str:
        if not recall.nodes and not recall.edges:
            return "# [GRAPH MEMORY]\n(no recall)"
        lines = ["# [GRAPH MEMORY]"]
        # Nodes/edges are dicts (originate from GraphMemory rows). All
        # field reads use .get() with sensible defaults so a slimmer GM
        # row schema doesn't crash the prompt builder; the previous mix
        # of [] + .get() on the same dict was inconsistent and could
        # KeyError for missing 'name' or 'category'.
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

    def _layer_history(
        self, window: list[SlidingWindowRound],
        *, in_mind: dict[str, str] | None = None,
    ) -> str:
        """Render the [HISTORY] layer.

        When ``in_mind`` is provided and at least one of its content
        fields (``thoughts`` / ``mood`` / ``focus``) is non-empty, a
        virtual "Heartbeat #now (in mind)" round is prepended before
        the real heartbeat history. This is the single channel by
        which Self's current mental state reaches every prompt
        consumer (Self LLM, recall LLM, future Reflects) — see
        docs/design/reflects-and-self-model.md Reflect #3 (Part 3).
        """
        lines = ["# [HISTORY]"]
        virtual = self._format_in_mind_round(in_mind)
        if virtual is not None:
            lines.append(virtual)
        if not window:
            if virtual is None:
                # Nothing at all in history — keep the historical
                # "(empty)" sentinel so existing tests / Self's
                # mental model stay consistent.
                return "# [HISTORY]\n(empty)"
            return "\n".join(lines)
        for r in window:
            lines.append(f"--- Heartbeat #{r.heartbeat_id} ---")
            lines.append(f"Stimulus: {r.stimulus_summary}")
            lines.append(f"Decision: {r.decision_text}")
            if r.note_text:
                lines.append(f"Note: {r.note_text}")
        return "\n".join(lines)

    def _format_in_mind_round(
        self, in_mind: dict[str, str] | None,
    ) -> str | None:
        """Format the virtual round at the head of [HISTORY], or
        return None if no in_mind state was passed / every field is
        empty. Lives on the builder rather than in the in_mind
        Reflect's prompt.py because the builder owns the [HISTORY]
        layer's exact line shape — keeping the formatting consistent
        with real heartbeat rounds matters for Self's pattern-match.
        """
        if not in_mind:
            return None
        thoughts = (in_mind.get("thoughts") or "").strip()
        mood = (in_mind.get("mood") or "").strip()
        focus = (in_mind.get("focus") or "").strip()
        if not (thoughts or mood or focus):
            return None
        out = ["--- Heartbeat #now (in mind) ---"]
        if thoughts:
            out.append(f"Thoughts: {thoughts}")
        if mood:
            out.append(f"Mood: {mood}")
        if focus:
            out.append(f"Focus: {focus}")
        return "\n".join(out)

    def _layer_stimulus(
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
                elif s.type == "tentacle_feedback":
                    own_actions.append(s)
                else:  # batch_complete | system_event | unknown
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
