"""Prompt assembler — Layers 0–5 (DevSpec §3.6)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml

from src.models.stimulus import Stimulus
from src.prompt.dna import DNA


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
    def build(self, *, self_model: dict[str, Any], status: dict[str, Any],
              recall: dict[str, Any], window: list[SlidingWindowRound],
              stimuli: list[Stimulus]) -> str:
        layers = [
            DNA,
            self._layer_self_model(self_model),
            self._layer_status(status),
            self._layer_recall(recall),
            self._layer_history(window),
            self._layer_stimulus(stimuli),
            HEARTBEAT_QUESTION,
        ]
        return "\n\n".join(layers)

    def _layer_self_model(self, sm: dict[str, Any]) -> str:
        body = yaml.safe_dump(sm, allow_unicode=True, sort_keys=False).strip() if sm else "(empty)"
        return f"# [SELF-MODEL]\n{body}"

    def _layer_status(self, s: dict[str, Any]) -> str:
        tentacle_lines = "\n".join(
            f"- {t['name']}: {t['description']}" for t in s.get("tentacles", [])
        ) or "(none)"
        return (
            "# [STATUS]\n"
            f"Graph Memory: {s.get('gm_node_count', 0)} nodes, "
            f"{s.get('gm_edge_count', 0)} edges\n"
            f"疲惫度: {s.get('fatigue_pct', 0)}% {s.get('fatigue_hint', '')}\n"
            f"上次 Sleep: {s.get('last_sleep_time', 'never')}\n"
            f"心跳数 (自上次 Sleep): {s.get('heartbeats_since_sleep', 0)}\n\n"
            "可用 Tentacles:\n"
            f"{tentacle_lines}"
        )

    def _layer_recall(self, recall: dict[str, Any]) -> str:
        nodes = recall.get("nodes", [])
        edges = recall.get("edges", [])
        if not nodes and not edges:
            return "# [GRAPH MEMORY]\n(no recall)"
        lines = ["# [GRAPH MEMORY]"]
        for n in nodes:
            kw = ", ".join(n.get("neighbor_keywords", []))
            lines.append(f"- [{n['name']}] ({n['category']}) — {n.get('description', '')}")
            if kw:
                lines.append(f"  相邻: {kw}")
        for e in edges:
            lines.append(f"- [{e['source']}] --{e['predicate']}--> [{e['target']}]")
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

    def _layer_stimulus(self, stimuli: list[Stimulus]) -> str:
        if not stimuli:
            return "# [STIMULUS]\n(no new signals)"
        lines = [f"# [STIMULUS]\n本次收到 {len(stimuli)} 条信号："]
        for s in stimuli:
            lines.append("---")
            lines.append(f"来源: {s.source} | 时间: {s.timestamp.isoformat()} "
                         f"| adrenalin: {s.adrenalin}")
            lines.append(f"内容: {s.content}")
        lines.append("---")
        return "\n".join(lines)
