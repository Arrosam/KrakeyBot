"""Per-beat heartbeat algorithm — extracted from Runtime.

Owns the orchestration of one heartbeat: phase ordering, sleep
short-circuits, command consumption, the prompt-build / Self-call /
decision-dispatch / idle flow.

Composition over inheritance: takes a ``Runtime`` reference and reads
state through it (``rt.memory``, ``rt.explicit_history``,
``rt.heartbeat_count``, …). Almost every Runtime field gets touched here, so wiring the
orchestrator with 13+ narrow protocols would just reproduce Runtime's
shape — pass the runtime ref and accept the verbosity. The split
buys two things anyway:

  1. **Read clarity** — Runtime is now ~600 lines of resources +
     lifecycle; the heartbeat algorithm is in one file.
  2. **Stable seam for future replacement** — a different orchestrator
     (e.g. test fake, or Phase 3 multi-stage scheduler) can be
     swapped in without touching Runtime construction.

The orchestrator owns NO state of its own. All beat-loop counters
(``heartbeat_count``, ``_recall``, ``_classify_tasks``, …) live on
Runtime so the existing test surface keeps reading them directly.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from krakey.models.config import LLMParams
from krakey.models.stimulus import Stimulus
from krakey.prompt.views import ExplicitHistoryRound
from krakey.engines.heartbeat.compact import compact_if_needed
from krakey.runtime.events.event_types import (
    DecisionEvent, GMStatsEvent, HeartbeatStartEvent, IdleEvent,
    SelfOutputEvent, NoteEvent, PromptBuiltEvent, SleepDoneEvent, 
    SleepFailedEvent, SleepStartEvent, StimuliQueuedEvent, ThinkingEvent
)
from krakey.engines.heartbeat.fatigue import calculate_fatigue
from krakey.engines.heartbeat.idle import idle_with_recall, wait_or_adrenalin
from krakey.runtime.commands.commands import (
    CommandAction, handle_command, parse_command,
)
from krakey.self_agent import parse_self_output

if TYPE_CHECKING:
    from krakey.interfaces.engines.recall import RecallResult, RecallSession
    from krakey.prompt.views import CapabilityView, StatusSnapshot
    from krakey.runtime.runtime import Runtime


MAX_RECALL_RETRIES = 1
"""Cap on uncovered stimulus re-tries to prevent infinite pushback loops
when GM has no related nodes yet (e.g. first-ever user message)."""

_REQUIRED_SELF_TAGS = frozenset({"THINKING", "DECISION"})
"""Tags that must appear in Self's response for it to count as
structurally valid.  Missing either triggers the structured-output
retry loop rather than forwarding a half-parsed result."""


def _raw_requests_builtin_sleep(raw: str) -> bool:
    """True iff ``raw`` contains a <tool_call> block whose JSON
    ``name`` is the built-in sleep tool. Used to waive the [IDLE]
    requirement on sleep beats. Deliberately narrow: only a cleanly
    parsed <tool_call>{"name":"sleep"} qualifies — malformed blocks
    and natural-language phrasing do NOT (the hypothalamus decides
    NL sleep post-translation; no pre-translation predicate exists)."""
    import json
    import re
    from krakey.runtime.builtin_tools import SLEEP_TOOL_NAME
    for block in re.findall(r"<tool_call>(.*?)</tool_call>", raw, re.DOTALL):
        try:
            data = json.loads(block.strip())
        except Exception:  # noqa: BLE001
            continue
        if isinstance(data, dict) and data.get("name") == SLEEP_TOOL_NAME:
            return True
    return False


@dataclass
class _GMCounts:
    """Snapshot from one heartbeat's fatigue phase, threaded into later
    phases so they don't re-query GM redundantly."""
    node_count: int
    edge_count: int
    fatigue_pct: int
    fatigue_hint: str


def _delta_str(delta: int) -> str:
    if delta > 0:
        return f" (+{delta})"
    if delta < 0:
        return f" ({delta})"
    return ""


def _summarize_stimuli(stimuli: list[Stimulus]) -> str:
    """Render the stimulus list for persistence in an ``ExplicitHistoryRound``.

    This text is what Self sees in the ``[HISTORY]`` layer every
    subsequent beat — truncation here is destructive: downstream
    mechanisms (recall, compact summarization, user-message echo
    checks) rely on the full content. The window's token budget
    handles overflow via compact_if_needed, so we don't need a blunt
    character cap here.
    """
    if not stimuli:
        return "(none)"
    return " | ".join(f"{s.source}: {s.content}" for s in stimuli)


def _summarize_recall(recall_result: "RecallResult") -> str:
    """Compact one-liner for history persistence: node names + categories + edge count."""
    if not recall_result.nodes:
        return ""
    parts = []
    for n in recall_result.nodes[:8]:
        name = n.get("name", "?")
        cat = n.get("category", "?")
        parts.append(f"[{name}]({cat})")
    overflow = len(recall_result.nodes) - 8
    summary = ", ".join(parts)
    if overflow > 0:
        summary += f" +{overflow} more"
    if recall_result.edges:
        summary += f" | {len(recall_result.edges)} edge(s)"
    return summary


# Per-block payload truncation in the format-failure stimulus.
# Tool-call arguments can carry large content (file writes, search
# queries with quoted text, …) and dumping the whole thing back to
# Self bloats the next prompt without helping diagnosis. 300 chars
# is enough to spot the structural problem.
_FAILURE_PAYLOAD_PREVIEW_CHARS = 300


def _format_parse_failure_stimulus(
    failures: list, *, total_blocks: int,
) -> str:
    """Build the system_event content for tool_call parse failures.

    Two failure modes appear in the same list:
      * **Lost** — JSON decode failure with no safe truncation
        point, OR a structural problem (missing name, non-object).
        The call did NOT dispatch.
      * **Salvaged** — "Extra data" (trailing junk after the JSON
        object). The parser truncated and recovered; the call DID
        dispatch this beat. The diagnostic surfaces so Self can
        clean up the format on the next beat.

    The header tells Self how many of each happened. Salvaged
    failures use ``repr()`` for the payload so invisible chars
    (zero-width space, BOM, control chars) appear as escape
    sequences rather than rendering as nothing — the "I can't
    see what's wrong" trap that previously stuck Self in a 120+
    beat loop.

    Inlined the format reminder rather than importing
    ``ACTION_FORMAT_LAYER_TOOL_CALL`` from prompt/layers — the layer
    is for prompt assembly, not for stimulus content; coupling them
    would drag layer concerns into the runtime path. The phrasing is
    copy-aligned with the layer; if either drifts, dual-update.
    """
    n_salv = sum(1 for f in failures if getattr(f, "salvaged", False))
    n_lost = len(failures) - n_salv
    header_parts: list[str] = []
    if n_lost > 0:
        header_parts.append(
            f"{n_lost} of {total_blocks} <tool_call> block(s) "
            "FAILED to parse — those actions did NOT dispatch."
        )
    if n_salv > 0:
        header_parts.append(
            f"{n_salv} of {total_blocks} <tool_call> block(s) had "
            "trailing junk after the JSON; salvaged this time but "
            "fix the format next beat."
        )
    lines = ["\n".join(header_parts), ""]
    for f in failures:
        preview = f.payload
        if len(preview) > _FAILURE_PAYLOAD_PREVIEW_CHARS:
            preview = preview[:_FAILURE_PAYLOAD_PREVIEW_CHARS] + "…"
        lines.append(f"Block {f.block_index}: {preview}")
        lines.append(f"  → {f.error}")
    lines.extend([
        "",
        "Each <tool_call> block must contain ONLY a JSON object — "
        "no XML tags after the JSON:",
        "",
        '  <tool_call>{"name": "<tool_name>", "arguments": {...}}</tool_call>',
        "",
        "Do NOT append </arg_value>, </function>, </function_call>, "
        "zero-width spaces, or any other characters after the closing "
        "}. The trailing-bytes line above shows EXACTLY what was "
        "appended — including invisible characters as escape "
        "sequences.",
    ])
    return "\n".join(lines)


class HeartbeatOrchestrator:
    """Runs one heartbeat per ``beat()`` call. Pure logic over Runtime
    state — owns no fields of its own."""

    def __init__(self, runtime: "Runtime"):
        self._rt = runtime

    # ---- one full beat -------------------------------------------------

    async def beat(self) -> None:
        """Orchestrates one heartbeat. Sleep can short-circuit at two
        points (force-sleep at fatigue threshold, voluntary sleep from
        Self's [DECISION]). Slash-commands (/kill, /sleep) can also
        short-circuit."""
        rt = self._rt
        rt.heartbeat_count += 1
        rt.log.set_heartbeat(rt.heartbeat_count)
        # Reload self_model from disk so plugin writes (e.g. the
        # bootstrap modifier's self-model patches in response to
        # Self's NOTE on the previous beat) flow into the prompt
        # this beat. Cheap (one YAML parse per beat); the file is
        # rarely re-written so the read is mostly a stat + cache.
        try:
            rt.self_model = rt._self_model_store.load()
        except Exception as e:  # noqa: BLE001
            rt.log.runtime_error(f"self_model reload failed: {e}")
        stimuli = await self._phase_drain_and_seed_recall()

        # Slash-commands run out-of-band (Self never sees /<cmd>).
        stimuli, command_action = await self._phase_handle_commands(stimuli)
        if command_action is CommandAction.KILL:
            return
        if command_action is CommandAction.SLEEP:
            await self._perform_sleep(
                "manual /sleep command",
                wake_msg="Completed a manually-triggered sleep.",
            )
            return

        counts = await self._phase_compute_fatigue()

        if counts.fatigue_pct >= rt.config.fatigue.force_sleep_threshold:
            await self._perform_sleep(
                f"force-sleep at fatigue {counts.fatigue_pct}%",
                wake_msg="Fell asleep earlier due to extreme fatigue.",
            )
            return

        await self._phase_compact()
        recall_result = await self._phase_finalize_recall_and_pushback()
        parsed = await self._phase_run_self(stimuli, recall_result, counts)
        if parsed is None:
            return  # Self LLM error already logged + slept
        self._phase_save_round(parsed, stimuli, recall_result)
        # ``_phase_log_self_output`` publishes a ``NoteEvent`` when
        # parsed.note is non-empty; the bootstrap plugin (and any
        # future Note observer) subscribes to that event for its
        # signal parsing — no orchestrator-side branch needed.
        self._phase_log_self_output(parsed)
        await self._phase_auto_ingest_feedback(stimuli)
        sleep_requested = await self._phase_apply_decision(
            parsed, recall_result, counts,
        )
        if sleep_requested:
            await self._perform_sleep(
                "voluntary sleep requested by Self",
                wake_msg=("Completed a full sleep cycle (clustering + KB "
                          "migration + index rebuild). Feeling refreshed."),
            )
            return
        self._phase_schedule_classify()
        await self._phase_idle(parsed, recall_result)

    # ---- phases --------------------------------------------------------

    async def _phase_handle_commands(
        self, stimuli: list[Stimulus],
    ) -> tuple[list[Stimulus], "CommandAction | None"]:
        """Scan drained stimuli for /<cmd>. Each match is consumed (Self
        never sees it) and executed out-of-band. Returns the filtered
        stimulus list + the highest-priority action (KILL > SLEEP)."""
        rt = self._rt
        filtered: list[Stimulus] = []
        triggered: CommandAction | None = None
        for s in stimuli:
            if s.type != "user_message":
                filtered.append(s)
                continue
            cmd = parse_command(s.content)
            if cmd is None:
                filtered.append(s)
                continue
            result = await handle_command(cmd, rt)
            rt.log.hb(f"command /{cmd}: {result.output}")
            if result.action is CommandAction.KILL:
                triggered = CommandAction.KILL
                rt.request_stop()
                break
            if (result.action is CommandAction.SLEEP
                    and triggered is not CommandAction.KILL):
                triggered = CommandAction.SLEEP
            # Self can still see informational commands as system events
            if result.action is CommandAction.NONE:
                await rt.buffer.push(Stimulus(
                    type="system_event", source="system:command",
                    content=f"/{cmd}: {result.output}",
                    timestamp=datetime.now(), adrenalin=False,
                ))
        return filtered, triggered

    async def _phase_drain_and_seed_recall(self) -> list[Stimulus]:
        rt = self._rt
        stimuli = rt.buffer.drain()
        rt.log.hb(f"stimuli={len(stimuli)} (thinking...)")
        rt.events.publish(HeartbeatStartEvent(
            heartbeat_id=rt.heartbeat_count, stimulus_count=len(stimuli),
        ))
        rt.events.publish(StimuliQueuedEvent(stimuli=[
            {"type": s.type, "source": s.source, "content": s.content,
             "adrenalin": s.adrenalin, "ts": s.timestamp.isoformat()}
            for s in stimuli
        ]))
        assert rt._recall is not None
        already = {id(s) for s in rt._recall.processed_stimuli}
        new_for_recall = [s for s in stimuli if id(s) not in already]
        if new_for_recall:
            await rt._recall.add_stimuli(new_for_recall)
        return stimuli

    async def _phase_compute_fatigue(self) -> _GMCounts:
        rt = self._rt
        node_count = await rt.memory.count_nodes()
        edge_count = await rt.memory.count_edges()
        pct, hint = calculate_fatigue(
            node_count=node_count,
            soft_limit=rt.config.fatigue.gm_node_soft_limit,
            thresholds=rt.config.fatigue.thresholds,
        )
        node_delta = node_count - rt._last_node_count
        edge_delta = edge_count - rt._last_edge_count
        rt._last_node_count = node_count
        rt._last_edge_count = edge_count
        rt.log.hb(
            f"gm: nodes={node_count}{_delta_str(node_delta)}, "
            f"edges={edge_count}{_delta_str(edge_delta)}, fatigue={pct}%"
        )
        if pct >= rt.config.fatigue.force_sleep_threshold:
            rt.log.hb_warn(
                f"force-sleep threshold reached (fatigue={pct}%); "
                "entering sleep mode."
            )
        rt.events.publish(GMStatsEvent(
            heartbeat_id=rt.heartbeat_count,
            node_count=node_count, edge_count=edge_count, fatigue_pct=pct,
        ))
        return _GMCounts(node_count=node_count, edge_count=edge_count,
                          fatigue_pct=pct, fatigue_hint=hint)

    async def _phase_compact(self) -> None:
        rt = self._rt
        async def _recall_fn(text: str):
            return await rt.memory.fts_search(text, top_k=10)
        await compact_if_needed(
            rt.explicit_history, rt.memory, rt.compact_llm,
            recall_fn=_recall_fn,
            include_recall_context=rt.config.sliding_window.compact_include_recall,
        )

    async def _phase_finalize_recall_and_pushback(self):
        """Finalize recall + cap-1 retry of uncovered stimuli."""
        rt = self._rt
        assert rt._recall is not None
        recall_result = await rt._recall.finalize()
        for s in recall_result.uncovered_stimuli:
            retries = s.metadata.get("recall_retries", 0)
            if retries >= MAX_RECALL_RETRIES:
                continue
            s.metadata["recall_retries"] = retries + 1
            await rt.buffer.push(s)
        return recall_result

    async def _phase_run_self(self, stimuli, recall_result,
                                counts: "_GMCounts"):
        """Build prompt + call Self LLM + parse + validate.

        Two retry layers share one ``while`` loop:

        1. **HTTP / timeout failures** — the ``except`` branch.
           Retries indefinitely at ``llm_failure_retry_interval``.
        2. **Structured-output failures** — the LLM returned text but
           required tags (THINKING, DECISION) are missing.  Fast
           retries (``struct_output_fast_retries`` attempts at
           ``llm_failure_retry_interval``), then infinite slow
           retries at ``struct_output_slow_retry_interval``.

        ``heartbeat_count`` stays frozen during both retry tiers —
        failed or malformed calls don't pollute heartbeat history,
        fatigue accounting, or sleep distance. Loop exits when a
        structurally valid response arrives OR
        ``runtime.stop_requested`` flips True (Ctrl+C / /kill).

        Each per-attempt call is wrapped in ``asyncio.wait_for`` with
        ``idle.self_max_wall_seconds`` ceiling — protects against a
        server stuck in an infinite generation loop. A wait_for
        timeout counts as one failed HTTP attempt."""
        rt = self._rt
        prompt, recall_result = await self.enforce_input_budget(
            stimuli, recall_result, counts,
        )
        self.record_prompt(rt.heartbeat_count, prompt)
        rt.events.publish(PromptBuiltEvent(
            heartbeat_id=rt.heartbeat_count,
            layers={"full_prompt": prompt},
        ))
        idle_cfg = rt.config.idle
        retry_idle = idle_cfg.llm_failure_retry_interval
        wall_cap = idle_cfg.self_max_wall_seconds
        fast_budget = idle_cfg.struct_output_fast_retries
        slow_interval = idle_cfg.struct_output_slow_retry_interval

        http_attempt = 0
        struct_attempt = 0
        parsed = None
        while not rt.stop_requested:
            # ---- LLM call (HTTP-failure retry) ----
            try:
                raw = await asyncio.wait_for(
                    rt.self_llm.chat(
                        [{"role": "user", "content": prompt}],
                    ),
                    timeout=wall_cap,
                )
            except Exception as e:  # noqa: BLE001
                http_attempt += 1
                rt.log.hb(
                    f"Self LLM error (attempt {http_attempt}): "
                    f"{type(e).__name__}: {e}; sleeping "
                    f"{retry_idle}s before retrying same beat "
                    f"(heartbeat_count frozen at {rt.heartbeat_count})"
                )
                interrupted = await wait_or_adrenalin(rt.buffer, retry_idle)
                if interrupted:
                    stimuli, recall_result, prompt = await self._reassemble_on_adrenalin(
                        stimuli, recall_result, counts,
                    )
                    if stimuli is None:
                        # KILL or SLEEP handled inside _reassemble; stop_requested
                        # is set (KILL) or None return signals SLEEP to caller.
                        return None
                    http_attempt = 0
                    struct_attempt = 0
                continue

            # ---- Structured-output validation ----
            parsed = parse_self_output(raw)
            required = _REQUIRED_SELF_TAGS | (
                frozenset() if _raw_requests_builtin_sleep(raw)
                else frozenset({"IDLE"})
            )
            missing = required - parsed.found_tags
            if not missing:
                break

            struct_attempt += 1
            if struct_attempt <= fast_budget:
                delay = retry_idle
                tier = f"fast {struct_attempt}/{fast_budget}"
            else:
                if struct_attempt == fast_budget + 1:
                    rt.log.hb_warn(
                        f"struct output retries exhausted fast tier "
                        f"({fast_budget} attempts); switching to slow "
                        f"interval ({slow_interval}s)"
                    )
                delay = slow_interval
                tier = f"slow, {slow_interval}s interval"
            rt.log.hb(
                f"Self output missing required tags "
                f"(found: {{{', '.join(sorted(parsed.found_tags))}}}, "
                f"required: {{{', '.join(sorted(required))}}}); "
                f"struct retry {struct_attempt} ({tier}), sleeping {delay}s"
            )
            interrupted = await wait_or_adrenalin(rt.buffer, delay)
            if interrupted:
                stimuli, recall_result, prompt = await self._reassemble_on_adrenalin(
                    stimuli, recall_result, counts,
                )
                if stimuli is None:
                    # KILL or SLEEP handled inside _reassemble; stop_requested
                    # is set (KILL) or None return signals SLEEP to caller.
                    return None
                http_attempt = 0
                struct_attempt = 0

        if rt.stop_requested:
            return None
        # Pair the raw response with the prompt-log entry created
        # above so the dashboard's Prompts tab can render the
        # unparsed output side-by-side with the prompt.
        self.record_raw_output(rt.heartbeat_count, raw)
        rt.events.publish(SelfOutputEvent(
            heartbeat_id=rt.heartbeat_count, raw=raw,
        ))
        return parsed

    async def _reassemble_on_adrenalin(
        self, stimuli, recall_result, counts: "_GMCounts",
    ):
        """Drain the buffer, handle commands, fold new stimuli in, rebuild
        the recall session and prompt in response to an adrenalin interrupt
        during a retry wait.

        Returns a (stimuli, recall_result, prompt) triple with the refreshed
        context if the beat should continue, or (None, None, None) if it
        should stop (KILL or SLEEP handled here).

        Called only from _phase_run_self when wait_or_adrenalin returns True.
        heartbeat_count is NOT incremented — the beat is still in progress.
        The LLM call is non-interruptible; only the between-attempt waits
        use this path.
        """
        rt = self._rt

        # a. Drain new stimuli.
        new = rt.buffer.drain()

        # b. Handle any slash-commands in the new batch.
        filtered_new, cmd_action = await self._phase_handle_commands(new)
        if cmd_action is CommandAction.KILL:
            # rt.request_stop() already called by _phase_handle_commands.
            return None, None, None
        if cmd_action is CommandAction.SLEEP:
            await self._perform_sleep(
                'voluntary sleep requested by Self',
                wake_msg=(
                    'Completed a full sleep cycle (clustering + KB '
                    'migration + index rebuild). Feeling refreshed.'
                ),
            )
            return None, None, None

        # c. Fold surviving non-command stimuli into this beat.
        stimuli = stimuli + filtered_new

        # d. Fresh recall session (mirror enforce_input_budget pattern).
        fresh = rt.recall.new_session()
        await fresh.add_stimuli(stimuli)
        recall_result = await fresh.finalize()
        rt._recall = fresh

        # e. Rebuild prompt with updated recall + stimuli.
        prompt, recall_result = await self.enforce_input_budget(
            stimuli, recall_result, counts,
        )

        # f. Update the existing prompt-log entry IN PLACE (do NOT append).
        #    record_prompt would push a second dict — that creates a phantom
        #    'no output' entry visible in the Prompts tab dashboard.
        for entry in reversed(rt._prompt_log):
            if entry.get('heartbeat_id') == rt.heartbeat_count:
                entry['full_prompt'] = prompt
                break

        # g. Re-publish the rebuilt prompt so subscribers see the refreshed version.
        rt.events.publish(PromptBuiltEvent(
            heartbeat_id=rt.heartbeat_count,
            layers={'full_prompt': prompt},
        ))

        # i. Log the reassembly.
        rt.log.hb(
            'beat reassembled on adrenalin during retry; '
            'rebuilt prompt, retry counters reset'
        )

        return stimuli, recall_result, prompt

    def _phase_save_round(self, parsed, stimuli, recall_result) -> None:
        rt = self._rt
        rt.explicit_history.append(ExplicitHistoryRound(
            heartbeat_id=rt.heartbeat_count,
            stimulus_summary=_summarize_stimuli(stimuli),
            decision_text=parsed.decision,
            note_text=parsed.note,
            thinking_text=parsed.thinking,
            recall_summary=_summarize_recall(recall_result),
        ))

    def _phase_log_self_output(self, parsed) -> None:
        rt = self._rt
        decision_text = parsed.decision.strip() or "(none)"
        rt.log.hb_thought("decision", decision_text)
        rt.events.publish(DecisionEvent(
            heartbeat_id=rt.heartbeat_count, text=decision_text,
        ))
        if parsed.thinking:
            rt.log.hb_thought("thinking", parsed.thinking)
            rt.events.publish(ThinkingEvent(
                heartbeat_id=rt.heartbeat_count,
                text=parsed.thinking.strip(),
            ))
        if parsed.note:
            rt.log.hb_thought("note", parsed.note)
            rt.events.publish(NoteEvent(
                heartbeat_id=rt.heartbeat_count, text=parsed.note.strip(),
            ))

    async def _phase_auto_ingest_feedback(self, stimuli) -> None:
        rt = self._rt
        for s in stimuli:
            if s.type != "tool_feedback":
                continue
            try:
                await rt.memory.auto_ingest(
                    s.content, source_heartbeat=rt.heartbeat_count,
                )
            except Exception as e:  # noqa: BLE001
                rt.log.runtime_error(f"auto_ingest error: {e}")

    async def _phase_apply_decision(self, parsed, recall_result,
                                     counts: "_GMCounts") -> bool:
        """Convert Self's response into tool calls + dispatch.

        ``rt.decision.translate(...)`` produces the structured
        ``DecisionResult``. The default engine is the scripted
        ``<tool_call>`` parser; users swap in any DecisionEngine impl
        (e.g. ``HypothalamusDecisionEngine`` for LLM-driven
        translation) via ``cfg.core_implementations.decision``.

        Malformed ``<tool_call>`` blocks surface to Self via a
        corrective ``system_event`` stimulus on the next beat — the
        per-block payload + parser error + a tight format reminder.
        Successful blocks in the same response still dispatch; partial
        parse must not block what DID parse.

        Returns True iff Self requested sleep.
        """
        rt = self._rt

        decision = parsed.decision.strip().lower()
        if not decision or decision == "no action":
            return False

        parse_failures: list = []
        try:
            # The engine receives the full ``raw`` response so an
            # alternative impl can scan beyond [DECISION] if it wants
            # (default impl scans decision-only — see the Engine's
            # docstring for the [NOTE]-echo rationale).
            result = await rt.decision.translate(
                parsed.decision,
                parsed.raw,
                rt.tools.list_descriptions(),
            )
            parse_failures = list(result.parse_failures)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e!r}"
            rt.log.hb(f"Decision dispatch error: {err}")
            await rt.buffer.push(Stimulus(
                type="system_event", source="system:decision",
                content=(
                    "Your last [DECISION] could not be translated "
                    f"({err}). Nothing was dispatched. Try re-stating "
                    "the intent more explicitly next beat."
                ),
                timestamp=datetime.now(),
                adrenalin=True,
            ))
            return False
        if parse_failures:
            # Salvaged failures appear in BOTH ``result.tool_calls``
            # (the recovered call) and ``parse_failures`` (the
            # diagnostic). To get the true block count we add and
            # subtract the salvaged duplicate.
            n_salvaged = sum(
                1 for f in parse_failures
                if getattr(f, "salvaged", False)
            )
            total_blocks = (
                len(result.tool_calls) + len(parse_failures) - n_salvaged
            )
            await rt.buffer.push(Stimulus(
                type="system_event", source="system:tool_call_parse",
                content=_format_parse_failure_stimulus(
                    parse_failures, total_blocks=total_blocks,
                ),
                timestamp=datetime.now(),
                adrenalin=True,
            ))
        # Intercept built-in `sleep` tool calls before dispatch.
        # Sleep is a heartbeat-lifecycle transition that must run
        # in the beat loop AFTER this method returns, not as a
        # fire-and-forget async task in the dispatcher. Filter
        # any sleep call out of tool_calls and set result.sleep
        # so the caller's existing `if sleep_requested:` branch
        # picks it up. This works for BOTH paths uniformly: the
        # hypothalamus translator can also emit
        # ``tool_calls=[{"tool":"sleep"}]`` and this intercept
        # collapses it into the sleep flag, redundant with the
        # legacy `sleep: true` JSON field but uniform.
        from krakey.runtime.builtin_tools import SLEEP_TOOL_NAME
        if any(c.tool == SLEEP_TOOL_NAME for c in result.tool_calls):
            result.sleep = True
            result.tool_calls = [
                c for c in result.tool_calls
                if c.tool != SLEEP_TOOL_NAME
            ]
        # Guard: refuse voluntary sleep when energy is high. Applies to
        # BOTH the sleep-tool-call path above AND a DecisionEngine that
        # sets result.sleep directly (e.g. the Hypothalamus boolean) —
        # otherwise that engine bypasses the policy entirely. Mirrors
        # the condition under which fatigue_hint() returns
        # LOW_FATIGUE_HINT — pct below every configured threshold.
        if result.sleep:
            thresholds = rt.config.fatigue.thresholds
            if thresholds and counts.fatigue_pct < min(thresholds):
                result.sleep = False
                await rt.buffer.push(Stimulus(
                    type="system_event", source="system:sleep",
                    content=(
                        f"Sleep refused: energy is high (fatigue "
                        f"{counts.fatigue_pct}% is below the minimum sleep "
                        f"threshold {min(thresholds)}%). Stay active."
                    ),
                    timestamp=datetime.now(), adrenalin=False,
                ))
        # Engine-mediated dispatch — runs the 4 side-effects (log +
        # publish, dispatch tool calls, apply memory writes, apply
        # memory updates) in one call. The DispatchEngine slot lets
        # users replace local in-process dispatch with a remote worker
        # / VM-backed executor / approval-gate wrapper without
        # touching this phase.
        await rt.dispatch.dispatch(
            rt.heartbeat_count, result, rt,
            recall_context=recall_result.nodes,
        )
        return bool(result.sleep)

    def _phase_schedule_classify(self) -> None:
        """Background classify+link doesn't block the heartbeat."""
        rt = self._rt
        rt._classify_tasks.append(
            asyncio.create_task(rt.memory.classify_and_link_pending()),
        )

    async def _phase_idle(self, parsed, recall_result) -> None:
        rt = self._rt
        if recall_result.uncovered_stimuli:
            interval = rt.config.idle.min_interval
        else:
            interval = (parsed.idle_seconds
                    or rt.config.idle.default_interval)
        rt.log.hb(f"idle {interval}s")
        rt.events.publish(IdleEvent(
            heartbeat_id=rt.heartbeat_count, interval_seconds=interval,
        ))
        rt._recall = self._rt.recall.new_session()
        await idle_with_recall(
            interval, rt.buffer, rt._recall,
            min_interval=rt._min, max_interval=rt._max,
        )

    # ---- prompt assembly (used by _phase_run_self) ----------------------

    def build_self_prompt(self, stimuli, recall_result,
                              counts: "_GMCounts") -> str:
        rt = self._rt
        # 1. Runtime constructs the canonical default elements.
        elements = rt.context.build_default_elements(
            self_model=rt.self_model,
            capabilities=self._capabilities(),
            status=self._status(counts.node_count, counts.edge_count,
                                  counts.fatigue_pct, counts.fatigue_hint),
            recall=recall_result,
            window=rt.explicit_history.get_rounds(),
            stimuli=stimuli,
            current_time=datetime.now(),
        )
        # 2. The DecisionEngine gets first crack at the elements via
        #    its optional ``modify_prompt`` hook — used by impls that
        #    teach Self a different action surface (e.g.
        #    HypothalamusDecisionEngine drops the [ACTION FORMAT]
        #    element because its translator handles natural-language
        #    decisions).
        decision_modify = getattr(rt.decision, "modify_prompt", None)
        if decision_modify is not None:
            try:
                decision_modify(
                    elements.for_plugin(f"engine:{type(rt.decision).__name__}"),
                )
            except Exception as e:  # noqa: BLE001
                rt.log.runtime_error(
                    f"DecisionEngine modify_prompt raised "
                    f"{type(e).__name__}: {e}; ignoring its modifications"
                )
        # 3. Each Modifier that defines a modify_prompt hook gets its
        #    chance to mutate. Modifications are tracked per element;
        #    PromptElements logs a warning on second-write conflicts.
        for modifier in rt.modifiers.all():
            modify = getattr(modifier, "modify_prompt", None)
            if modify is None:
                continue
            try:
                modify(elements.for_plugin(modifier.name))
            except Exception as e:  # noqa: BLE001
                rt.log.runtime_error(
                    f"Modifier {modifier.name!r} modify_prompt raised "
                    f"{type(e).__name__}: {e}; ignoring its modifications"
                )
        return elements.render()

    async def enforce_input_budget(self, stimuli, recall_result,
                                       counts: "_GMCounts"):
        """Overall prompt-budget enforcement (DevSpec §10.2).

        After recall is finalized and we have a candidate prompt, if
        the full prompt exceeds the Self role's ``max_input_tokens``,
        prune the oldest history round into GM (normal compact path)
        and re-run recall (GM changed → new nodes may be more
        relevant). Repeat until the prompt fits or the window is empty.

        This is the second line of defense: ``_phase_compact`` already
        caps history at ``max_input_tokens * history_token_fraction``,
        but the rest of the prompt (DNA + self-model + capabilities +
        stimulus + recall + status) can push the total over budget
        even when history is within its own share. When that happens
        we borrow from history (oldest rounds are least valuable) and
        promote them to GM so nothing is lost.

        Returns the final (prompt, recall_result) pair. Hard cap on
        iterations so a pathological configuration can't spin forever.
        """
        from krakey.engines.heartbeat.compact import compact_round
        from krakey.utils.tokens import estimate_tokens

        rt = self._rt
        self_params = rt.config.llm.core_params("self_thinking") or LLMParams()
        budget = int(self_params.max_input_tokens or 128_000)

        async def _recall_fn(text: str):
            return await rt.memory.fts_search(text, top_k=10)

        prompt = self.build_self_prompt(stimuli, recall_result, counts)
        max_iters = 10  # safety bound — should never need more than 2-3
        for _ in range(max_iters):
            total = estimate_tokens(prompt)
            if total <= budget:
                return prompt, recall_result
            if not rt.explicit_history.rounds:
                rt.log.hb_warn(
                    f"prompt {total} > max_input_tokens {budget} and "
                    "window is empty; sending anyway"
                )
                return prompt, recall_result
            oldest = rt.explicit_history.pop_oldest()
            assert oldest is not None
            rt.log.hb(
                f"input budget: prompt {total} > {budget}; pruning oldest "
                f"round (heartbeat #{oldest.heartbeat_id}) into GM"
            )
            try:
                await compact_round(
                oldest, rt.memory, rt.compact_llm, _recall_fn,
                include_recall_context=rt.config.sliding_window.compact_include_recall,
            )
            except Exception as e:  # noqa: BLE001 — never crash the beat
                rt.log.hb_warn(
                    f"budget-driven compact failed: {e} — round "
                    f"#{oldest.heartbeat_id} dropped without GM write"
                )
            fresh = self._rt.recall.new_session()
            await fresh.add_stimuli(stimuli)
            recall_result = await fresh.finalize()
            rt._recall = fresh
            prompt = self.build_self_prompt(stimuli, recall_result, counts)
        rt.log.hb_warn(
            f"input budget not satisfied after {max_iters} prune "
            f"iterations; sending oversized prompt "
            f"({estimate_tokens(prompt)} > {budget})"
        )
        return prompt, recall_result

    def record_prompt(self, heartbeat_id: int, prompt: str) -> None:
        self._rt._prompt_log.append({
            "heartbeat_id": heartbeat_id,
            "ts": datetime.now().isoformat(),
            "full_prompt": prompt,
            # Filled in by record_raw_output once the LLM returns;
            # left None when the LLM call errored so the dashboard
            # can show a "(no output)" placeholder rather than guess.
            "raw_output": None,
        })

    def record_raw_output(self, heartbeat_id: int, raw: str) -> None:
        """Attach the raw LLM response to the most recent prompt-log
        entry matching ``heartbeat_id``. Called after a successful
        ``self_llm.chat`` so the dashboard's Prompts tab can show the
        unparsed text alongside the prompt.
        """
        # Walk newest-to-oldest — an in-flight beat is always near
        # the tail. Failed lookup is non-fatal (e.g. log was rotated
        # past the entry while the LLM call was outstanding).
        for entry in reversed(self._rt._prompt_log):
            if entry.get("heartbeat_id") == heartbeat_id:
                entry["raw_output"] = raw
                return

    def _capabilities(self) -> list["CapabilityView"]:
        """Tool list for the [CAPABILITIES] layer. Only changes on
        plugin reload, so this gets rendered high in the prompt above
        the cache-breaking volatile layers."""
        from krakey.prompt.views import CapabilityView
        return [
            CapabilityView(name=t["name"], description=t["description"])
            for t in self._rt.tools.list_descriptions()
        ]

    def _status(self, node_count: int, edge_count: int,
                  pct: int, hint: str) -> "StatusSnapshot":
        """Runtime status numbers — changes every beat (heartbeat
        counter, fatigue), so this section is deliberately placed near
        the end of the prompt to preserve the cacheable prefix above it."""
        from krakey.prompt.views import StatusSnapshot
        return StatusSnapshot(
            gm_node_count=node_count,
            gm_edge_count=edge_count,
            fatigue_pct=pct,
            fatigue_hint=hint,
            last_sleep_time="never",
            heartbeats_since_sleep=self._rt.heartbeat_count,
        )

    # ---- sleep ---------------------------------------------------------

    async def _perform_sleep(self, reason: str, *, wake_msg: str) -> None:
        """Run 7-phase Sleep, persist self-model bookkeeping, push wake-up
        stimulus, reset incremental recall (GM state changed)."""
        rt = self._rt
        rt.log.hb(f"sleep started — {reason}")
        rt.events.publish(SleepStartEvent(reason=reason))
        try:
            sl = rt.config.sleep
            # Sleep flows through the MemoryEngine — a custom
            # MemoryEngine impl can override sleep_cycle to ship
            # consolidation to a remote worker, skip migration
            # entirely, etc. The default GraphMemoryEngine wraps
            # the in-tree enter_sleep_mode pipeline.
            stats = await rt.memory.sleep_cycle(
                channels=rt.buffer,
                log_dir=rt.sleep_log_dir,
                config={
                    "llm": rt.compact_llm,
                    "reranker": rt.reranker,
                    "min_community_size": sl.min_community_size,
                    "kb_consolidation_threshold":
                        sl.kb_consolidation_threshold,
                    "kb_index_max":          sl.kb_index_max,
                    "kb_archive_pct":        sl.kb_archive_pct,
                    "kb_revive_threshold":   sl.kb_revive_threshold,
                },
            )
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            # Three-way surfacing so a sleep failure isn't silent:
            # (1) stderr via hb_warn, since `hb` previously went to
            #     stdout where it could blend with normal beat lines
            # (2) event bus → dashboard sees a SleepFailed card
            # (3) system_event stimulus → Self sees on next beat
            #     that its sleep request didn't take effect AND
            #     gets the underlying error so it can react
            rt.log.hb_warn(f"sleep failed: {err}")
            rt.events.publish(SleepFailedEvent(reason=reason, error=err))
            await rt.buffer.push(Stimulus(
                type="system_event", source="system:sleep",
                content=(
                    f"Sleep transition failed: {err}. "
                    "Runtime is continuing without entering sleep "
                    "state. Likely causes: compact_llm not bound or "
                    "unreachable, GM/KB I/O error during clustering "
                    "or migration. Check the runtime stderr for the "
                    "stack trace."
                ),
                timestamp=datetime.now(), adrenalin=True,
            ))
            return
        rt.events.publish(SleepDoneEvent(stats=stats))
        rt.log.hb(
            f"sleep done: facts_migrated={stats['facts_migrated']}, "
            f"focus_cleared={stats['focus_cleared']}, "
            f"kbs={stats['kbs_created']}, index_nodes={stats['index_nodes']}"
        )
        # Sleep bookkeeping is a per-process runtime concern, not
        # something Self needs to remember across restarts.
        rt._sleep_cycles += 1
        # Wake-up stimulus
        await rt.buffer.push(Stimulus(
            type="system_event", source="system:sleep",
            content=wake_msg, timestamp=datetime.now(),
            adrenalin=False,
        ))
        # GM changed underneath us — start a fresh recall
        rt._recall = self._rt.recall.new_session()
