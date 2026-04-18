"""CogniBot entrypoint + main heartbeat loop (DevSpec §6.4, Phase-0 subset).

Phase 0 skips: GM recall, compact, fatigue, classify, auto_ingest, sleep.
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from src.hypothalamus import Hypothalamus, HypothalamusResult, TentacleCall
from src.interfaces.sensory import SensoryRegistry
from src.interfaces.tentacle import TentacleRegistry
from src.llm.client import LLMClient
from src.models.config import Config, load_config
from src.models.stimulus import Stimulus
from src.prompt.builder import PromptBuilder, SlidingWindowRound
from src.runtime.fatigue import fatigue_hint
from src.runtime.hibernate import hibernate
from src.runtime.stimulus_buffer import StimulusBuffer
from src.self_agent import parse_self_output
from src.sensories.cli_input import CliInputSensory
from src.tentacles.action import ActionTentacle


class ChatLike(Protocol):
    async def chat(self, messages, **kwargs) -> str: ...


@dataclass
class RuntimeDeps:
    config: Config
    self_llm: ChatLike
    hypo_llm: ChatLike
    action_llm: ChatLike
    reader = None  # line reader for CLI (optional, default stdin)


HARDCODED_SELF_MODEL: dict[str, Any] = {
    "identity": {"name": "Klarky", "persona": "nascent cognitive entity"},
    "state": {"mood_baseline": "neutral", "energy_level": 1.0,
              "focus_topic": "", "is_sleeping": False,
              "bootstrap_complete": True},
    "goals": {"active": [], "completed": []},
    "relationships": {"users": []},
    "statistics": {"total_heartbeats": 0, "total_sleep_cycles": 0,
                   "uptime_hours": 0.0, "first_boot": "",
                   "last_heartbeat": "", "last_sleep": ""},
}


class Runtime:
    def __init__(self, deps: RuntimeDeps, *, hibernate_min: float | None = None,
                 hibernate_max: float | None = None):
        self.config = deps.config
        self.self_llm = deps.self_llm
        self.hypothalamus = Hypothalamus(deps.hypo_llm)
        self.buffer = StimulusBuffer()
        self.window: list[SlidingWindowRound] = []
        self.builder = PromptBuilder()

        self.tentacles = TentacleRegistry()
        self.tentacles.register(ActionTentacle(
            llm=deps.action_llm,
            max_context_tokens=self.config.tentacle.get("action", {})
                .get("max_context_tokens", 4096),
        ))

        self.sensories = SensoryRegistry()
        if self.config.sensory.get("cli_input", {}).get("enabled", False):
            self.sensories.register(CliInputSensory(
                default_adrenalin=self.config.sensory["cli_input"]
                    .get("default_adrenalin", True),
                reader=deps.reader,
            ))

        self.self_model = dict(HARDCODED_SELF_MODEL)
        self.heartbeat_count = 0
        self._stop = False
        self._min = hibernate_min if hibernate_min is not None else self.config.hibernate.min_interval
        self._max = hibernate_max if hibernate_max is not None else self.config.hibernate.max_interval

    async def run(self, iterations: int | None = None) -> None:
        await self.sensories.start_all(self.buffer)
        try:
            count = 0
            while not self._stop:
                await self._heartbeat()
                count += 1
                if iterations is not None and count >= iterations:
                    return
        finally:
            await self.sensories.stop_all()

    async def _heartbeat(self) -> None:
        self.heartbeat_count += 1
        stimuli = self.buffer.drain()

        print(f"[HB #{self.heartbeat_count}] stimuli={len(stimuli)} "
              f"(thinking...)", flush=True)

        prompt = self.builder.build(
            self_model=self.self_model,
            status=self._status(),
            recall={"nodes": [], "edges": []},
            window=list(self.window),
            stimuli=stimuli,
        )
        try:
            raw = await self.self_llm.chat([{"role": "user", "content": prompt}])
        except Exception as e:  # noqa: BLE001
            print(f"[HB #{self.heartbeat_count}] Self LLM error: {e}", flush=True)
            await asyncio.sleep(self._min)
            return
        parsed = parse_self_output(raw)

        self.window.append(SlidingWindowRound(
            heartbeat_id=self.heartbeat_count,
            stimulus_summary=_summarize_stimuli(stimuli),
            decision_text=parsed.decision,
            note_text=parsed.note,
        ))

        snippet = parsed.decision.strip().replace("\n", " ")[:120]
        print(f"[HB #{self.heartbeat_count}] decision: {snippet or '(none)'}",
              flush=True)

        decision = parsed.decision.strip().lower()
        if decision and decision not in ("no action", "无行动"):
            tentacle_descs = self.tentacles.list_descriptions()
            try:
                result = await self.hypothalamus.translate(parsed.decision, tentacle_descs)
            except Exception as e:  # noqa: BLE001
                print(f"[HB #{self.heartbeat_count}] Hypothalamus error: {e}",
                      flush=True)
                result = None

            if result is not None:
                if result.sleep:
                    print("[runtime] sleep requested (not implemented in Phase 0)",
                          file=sys.stderr, flush=True)
                for call in result.tentacle_calls:
                    asyncio.create_task(self._dispatch(call))

        interval = parsed.hibernate_seconds or self.config.hibernate.default_interval
        print(f"[HB #{self.heartbeat_count}] hibernate {interval}s", flush=True)
        await hibernate(interval, self.buffer,
                        min_interval=self._min, max_interval=self._max)

    async def _dispatch(self, call: TentacleCall) -> None:
        try:
            tentacle = self.tentacles.get(call.tentacle)
        except KeyError:
            await self.buffer.push(Stimulus(
                type="system_event", source="runtime",
                content=f"Unknown tentacle: {call.tentacle}",
                timestamp=datetime.now(), adrenalin=False,
            ))
            print(f"[dispatch] unknown tentacle: {call.tentacle}", flush=True)
            return

        print(f"[dispatch] {call.tentacle} ← {call.intent!r}"
              f"{' (adrenalin)' if call.adrenalin else ''}", flush=True)
        try:
            stim = await tentacle.execute(call.intent, call.params)
        except Exception as e:  # noqa: BLE001
            print(f"[dispatch] {call.tentacle} error: {e}", flush=True)
            await self.buffer.push(Stimulus(
                type="system_event", source=f"tentacle:{call.tentacle}",
                content=f"error: {e}", timestamp=datetime.now(),
                adrenalin=call.adrenalin,
            ))
            return

        # Adrenalin inheritance (DevSpec §4.4)
        if call.adrenalin and not stim.adrenalin:
            stim.adrenalin = True
        print(f"[{call.tentacle}] {stim.content}", flush=True)
        await self.buffer.push(stim)

    def _status(self) -> dict[str, Any]:
        pct = 0  # Phase-0: GM not yet wired; stays at 0 until Phase 1.
        return {
            "gm_node_count": 0,
            "gm_edge_count": 0,
            "fatigue_pct": pct,
            "fatigue_hint": fatigue_hint(pct, self.config.fatigue.thresholds),
            "last_sleep_time": "never",
            "heartbeats_since_sleep": self.heartbeat_count,
            "tentacles": self.tentacles.list_descriptions(),
        }


def _summarize_stimuli(stimuli: list[Stimulus]) -> str:
    if not stimuli:
        return "(none)"
    return " | ".join(f"{s.source}: {s.content[:60]}" for s in stimuli)


def build_runtime_with_fakes(*, self_llm: ChatLike, hypo_llm: ChatLike,
                              action_llm: ChatLike,
                              hibernate_min: float = 0.01,
                              hibernate_max: float = 5.0) -> Runtime:
    """Test helper: build a Runtime with an in-memory config skeleton.

    CLI sensory is disabled so we do not spawn stdin readers in tests.
    """
    from src.models.config import (
        Config, FatigueSection, GraphMemorySection, HibernateSection,
        KnowledgeBaseSection, LLMSection, SafetySection, SleepSection,
        SlidingWindowSection,
    )

    cfg = Config(
        llm=LLMSection(providers={}, roles={}),
        hibernate=HibernateSection(min_interval=1, max_interval=60, default_interval=1),
        fatigue=FatigueSection(gm_node_soft_limit=200, force_sleep_threshold=120,
                               thresholds={}),
        sliding_window=SlidingWindowSection(max_tokens=4096),
        graph_memory=GraphMemorySection(db_path="", auto_ingest_similarity_threshold=0.9,
                                         recall_per_stimulus_k=5, max_recall_nodes=20,
                                         neighbor_expand_depth=1),
        knowledge_base=KnowledgeBaseSection(dir=""),
        sensory={"cli_input": {"enabled": False, "default_adrenalin": True}},
        tentacle={"action": {"enabled": True, "max_context_tokens": 4096,
                              "sandboxed": True}},
        sleep=SleepSection(max_duration_seconds=7200),
        safety=SafetySection(gm_node_hard_limit=500, max_consecutive_no_action=50),
    )
    deps = RuntimeDeps(config=cfg, self_llm=self_llm, hypo_llm=hypo_llm,
                       action_llm=action_llm)
    return Runtime(deps, hibernate_min=hibernate_min, hibernate_max=hibernate_max)


def build_runtime_from_config(config_path: str = "config.yaml") -> Runtime:
    cfg = load_config(config_path)
    self_role = cfg.llm.roles["self"]
    hypo_role = cfg.llm.roles["hypothalamus"]
    tentacle_role = cfg.llm.roles["tentacle_default"]

    self_llm = LLMClient(cfg.llm.providers[self_role.provider], self_role.model)
    hypo_llm = LLMClient(cfg.llm.providers[hypo_role.provider], hypo_role.model)
    action_llm = LLMClient(cfg.llm.providers[tentacle_role.provider], tentacle_role.model)

    deps = RuntimeDeps(config=cfg, self_llm=self_llm, hypo_llm=hypo_llm,
                       action_llm=action_llm)
    return Runtime(deps)


async def _amain() -> None:
    runtime = build_runtime_from_config()
    await runtime.run()


if __name__ == "__main__":
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        pass
