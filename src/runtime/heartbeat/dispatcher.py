"""Hypothalamus side-effects executor — extracted from Runtime.

After the hypothalamus Reflect (or the [ACTION] executor fallback)
turns Self's natural-language [DECISION] into a structured
``HypothalamusResult``, four side-effects need to fire:

  1. **Log + publish** the summary (counts + sleep flag).
  2. **Dispatch** each TentacleCall as an async task and register
     the batch with the BatchTracker so completion can wake Self.
  3. **Apply memory writes** (LLM-extracted nodes/edges via
     ``GraphMemory.explicit_write``).
  4. **Apply memory updates** (category flips like TARGET → FACT
     via ``GraphMemory.update_node_category``).

These were five Runtime methods that touched the same five
collaborators (tentacles, batch_tracker, buffer, gm, log+events)
and nothing from the heartbeat loop's state. Clean seam.

Each entry method takes ``heartbeat_id`` as a parameter — the
dispatcher doesn't track the beat counter; Runtime owns it.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.models.stimulus import Stimulus
from src.runtime.events.event_types import DispatchEvent, HypothalamusEvent, TentacleResultEvent

if TYPE_CHECKING:
    from src.interfaces.reflect import HypothalamusResult, TentacleCall
    from src.interfaces.tentacle import TentacleRegistry
    from src.memory.graph_memory import GraphMemory
    from src.runtime.stimuli.batch_tracker import BatchTrackerSensory
    from src.runtime.events.event_bus import EventBus
    from src.runtime.console.heartbeat_logger import HeartbeatLogger
    from src.runtime.stimuli.stimulus_buffer import StimulusBuffer


class HypothalamusDispatcher:
    """Executes the four side-effects of a HypothalamusResult."""

    def __init__(
        self,
        *,
        tentacles: "TentacleRegistry",
        batch_tracker: "BatchTrackerSensory",
        buffer: "StimulusBuffer",
        gm: "GraphMemory",
        log: "HeartbeatLogger",
        events: "EventBus",
    ):
        self._tentacles = tentacles
        self._batch_tracker = batch_tracker
        self._buffer = buffer
        self._gm = gm
        self._log = log
        self._events = events

    # ---- summary --------------------------------------------------------

    def log_summary(self, heartbeat_id: int,
                    result: "HypothalamusResult") -> None:
        """Log + publish HypothalamusEvent (counts + sleep flag)."""
        self._log.hypo(
            f"tentacle_calls={len(result.tentacle_calls)} "
            f"memory_writes={len(result.memory_writes)} "
            f"memory_updates={len(result.memory_updates)} "
            f"sleep={result.sleep}"
        )
        self._events.publish(HypothalamusEvent(
            heartbeat_id=heartbeat_id,
            tentacle_calls_count=len(result.tentacle_calls),
            memory_writes_count=len(result.memory_writes),
            memory_updates_count=len(result.memory_updates),
            sleep_requested=result.sleep,
        ))

    # ---- tentacle dispatch ---------------------------------------------

    async def dispatch_tentacle_calls(
        self, heartbeat_id: int, calls: list["TentacleCall"],
    ) -> None:
        """Schedule each call as an async task and register the batch
        so the BatchTracker can wake Self when all complete."""
        call_ids: list[str] = []
        for idx, call in enumerate(calls):
            cid = f"hb{heartbeat_id}_c{idx}"
            call_ids.append(cid)
            asyncio.create_task(self._dispatch_one(heartbeat_id, call, cid))
        if call_ids:
            self._batch_tracker.register_batch(call_ids)

    async def _dispatch_one(
        self, heartbeat_id: int, call: "TentacleCall", call_id: str,
    ) -> None:
        try:
            tentacle = self._tentacles.get(call.tentacle)
        except KeyError:
            await self._buffer.push(Stimulus(
                type="system_event", source="runtime",
                content=f"Unknown tentacle: {call.tentacle}",
                timestamp=datetime.now(), adrenalin=False,
            ))
            self._log.dispatch(f"unknown tentacle: {call.tentacle}")
            await self._batch_tracker.mark_completed(call_id)
            return

        self._log.dispatch(
            f"{call.tentacle} ← {call.intent!r}"
            f"{' (adrenalin)' if call.adrenalin else ''}"
        )
        self._events.publish(DispatchEvent(
            heartbeat_id=heartbeat_id, tentacle=call.tentacle,
            intent=call.intent, adrenalin=call.adrenalin,
        ))
        try:
            stim = await tentacle.execute(call.intent, call.params)
        except Exception as e:  # noqa: BLE001
            # Catastrophic tentacle crash — worth waking Self regardless
            # of whether the original call was urgent.
            self._log.dispatch(f"{call.tentacle} error: {e}")
            await self._buffer.push(Stimulus(
                type="system_event", source=f"tentacle:{call.tentacle}",
                content=f"error: {e}", timestamp=datetime.now(),
                adrenalin=True,
            ))
            await self._batch_tracker.mark_completed(call_id)
            return

        # Tentacle-feedback stimuli are low-priority receipts by design.
        # The tentacle itself decides whether its outcome is worth
        # interrupting Self's hibernate (failures typically set
        # adrenalin=True in their own return). Do NOT inherit adrenalin
        # from the dispatch: by the time feedback arrives Self has
        # already acted on the urgent upstream signal, and re-waking
        # for the echo just produces avoidable heartbeats.
        #
        # All tentacle output goes through ONE log channel — the
        # previous internal/chat split was driven by a self-declared
        # ``Tentacle.is_internal`` flag, removed because a malicious
        # plugin could set it True to hide its actions from operator
        # view. Operator transparency wins over log-color aesthetics.
        self._log.chat(call.tentacle, stim.content)
        self._events.publish(TentacleResultEvent(
            tentacle=call.tentacle, content=stim.content,
        ))
        await self._buffer.push(stim)
        await self._batch_tracker.mark_completed(call_id)

    # ---- memory side-effects -------------------------------------------

    async def apply_memory_writes(
        self, writes: list[dict[str, Any]],
        recall_nodes: list[dict[str, Any]],
        heartbeat_id: int,
    ) -> None:
        """Run gm.explicit_write per write entry; per-write failures log
        but don't abort the rest."""
        for w in writes:
            self._log.hypo(f"memory_write: {w.get('content', '')[:80]}")
            try:
                await self._gm.explicit_write(
                    w["content"],
                    importance=w.get("importance", "normal"),
                    recall_context=recall_nodes,
                    source_heartbeat=heartbeat_id,
                )
            except Exception as e:  # noqa: BLE001
                self._log.runtime_error(f"explicit_write error: {e}")

    async def apply_memory_updates(
        self, updates: list[dict[str, Any]],
    ) -> None:
        """Run gm.update_node_category per update; per-update failures
        log but don't abort the rest."""
        for u in updates:
            self._log.hypo(f"memory_update: {u.get('node_name')} → "
                              f"{u.get('new_category')}")
            try:
                await self._gm.update_node_category(
                    u["node_name"], u["new_category"],
                )
            except Exception as e:  # noqa: BLE001
                self._log.runtime_error(f"update_category error: {e}")
