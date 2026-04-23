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

from src.models.stimulus import Stimulus
from src.prompt.dna import DNA


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


class PromptBuilder:
    def build(
        self,
        *,
        self_model: dict[str, Any],
        capabilities: list[dict[str, Any]],
        status: dict[str, Any],
        recall: dict[str, Any],
        window: list[SlidingWindowRound],
        stimuli: list[Stimulus],
        current_time: datetime | None = None,
    ) -> str:
        layers = [
            DNA,
            self._layer_self_model(self_model),
            self._layer_capabilities(capabilities),
            self._layer_stimulus(stimuli, current_time),
            self._layer_recall(recall),
            self._layer_history(window),
            self._layer_status(status),
            HEARTBEAT_QUESTION,
        ]
        return "\n\n".join(layers)

    def _layer_self_model(self, sm: dict[str, Any]) -> str:
        body = (
            yaml.safe_dump(sm, allow_unicode=True, sort_keys=False).strip()
            if sm
            else "(empty)"
        )
        return f"# [SELF-MODEL]\n{body}"

    def _layer_capabilities(self, tentacles: list[dict[str, Any]]) -> str:
        if tentacles:
            lines = "\n".join(
                f"- {t['name']}: {t['description']}" for t in tentacles
            )
        else:
            lines = "(none)"
        return (
            "# [CAPABILITIES]\n"
            "可用 Tentacles (本跳已注册):\n"
            f"{lines}"
        )

    def _layer_status(self, s: dict[str, Any]) -> str:
        return (
            "# [STATUS]\n"
            f"Graph Memory: {s.get('gm_node_count', 0)} nodes, "
            f"{s.get('gm_edge_count', 0)} edges\n"
            f"疲惫度: {s.get('fatigue_pct', 0)}% {s.get('fatigue_hint', '')}\n"
            f"上次 Sleep: {s.get('last_sleep_time', 'never')}\n"
            f"心跳数 (自上次 Sleep): {s.get('heartbeats_since_sleep', 0)}"
        )

    def _layer_recall(self, recall: dict[str, Any]) -> str:
        nodes = recall.get("nodes", [])
        edges = recall.get("edges", [])
        if not nodes and not edges:
            return "# [GRAPH MEMORY]\n(no recall)"
        lines = ["# [GRAPH MEMORY]"]
        for n in nodes:
            kw = ", ".join(n.get("neighbor_keywords", []))
            lines.append(
                f"- [{n['name']}] ({n['category']}) — "
                f"{n.get('description', '')}"
            )
            if kw:
                lines.append(f"  相邻: {kw}")
        for e in edges:
            lines.append(
                f"- [{e['source']}] --{e['predicate']}--> [{e['target']}]"
            )
        return "\n".join(lines)

    def _layer_history(self, window: list[SlidingWindowRound]) -> str:
        if not window:
            return "# [HISTORY]\n(empty)"
        lines = ["# [HISTORY]"]
        for r in window:
            lines.append(f"--- Heartbeat #{r.heartbeat_id} ---")
            lines.append(f"Stimulus: {r.stimulus_summary}")
            lines.append(f"Decision: {r.decision_text}")
            if r.note_text:
                lines.append(f"Note: {r.note_text}")
        return "\n".join(lines)

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
