"""Sliding window — dynamic token-bounded recent-heartbeat buffer
(DevSpec §10.1). Each round stores (stimulus_summary, decision, note).

Compaction is triggered by TWO independent conditions (both checked
by ``needs_compact()``):

1. Token count: window total exceeds ``compact_threshold`` (default
   2048 tokens, configurable via engine config).
2. Round count: number of buffered rounds exceeds
   ``max_history_rounds`` (default 20, configurable via engine config).

Both thresholds are read from the engine's config dict at construction
time and sanitised via ``_sanitize_positive_int``. Invalid, zero, or
negative values fall back to the respective defaults.

The ``history_token_budget`` constructor parameter is retained for
Protocol compliance (declared by the ExplicitHistoryEngine Protocol)
and for diagnostics/logging; it is no longer used by
``needs_compact()``.

Token estimation goes through ``src.utils.tokens.estimate_tokens``
(tiktoken cl100k_base) — replaces the previous char/4 heuristic which
undercounted Chinese text ~4-8×.

Persistence (Samuel 2026-05-07): the live rounds list is mirrored
to ``state_path`` (atomic write) on every mutation so a process
restart restores the working memory exactly. Without this, every
restart wiped Self's most-recent context — only the rounds already
compacted to GM survived, the last few in-flight rounds vanished.
GM/KB/self_model already persist; the sliding window was the
last in-memory hole. Pass ``state_path=None`` to opt out (unit specs).
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from krakey.interfaces.engines.explicit_history import ExplicitHistoryRound
from krakey.utils.tokens import estimate_tokens

_log = logging.getLogger(__name__)


# Bump if the on-disk shape changes. Older files with a different
# version are ignored (window starts empty + a stderr nudge).
_STATE_SCHEMA_VERSION = 2


def _sanitize_positive_int(value, default):
    # bool is an int subclass — reject it explicitly. Only a genuine
    # positive int is valid; everything else (None, float, str, bool,
    # negative, 0) falls back to the default.
    if isinstance(value, bool):
        return default
    if isinstance(value, int) and value > 0:
        return value
    return default


class SlidingWindow:
    """Bounded buffer of recent heartbeat rounds.

    ``history_token_budget`` is retained for Protocol compliance and
    diagnostics; compaction is now governed by ``compact_threshold``
    (token cap) and ``max_history_rounds`` (round cap), both read from
    the engine config dict supplied at construction.

    When ``needs_compact()`` is True the compactor loop pops the
    oldest round and asks the compact LLM to extract GM nodes.

    ``state_path`` is the on-disk mirror. ``None`` opts out of
    persistence (used by ephemeral-window integration specs).
    """

    def __init__(
        self,
        history_token_budget: int,
        *,
        state_path: str | Path | None = None,
        config: dict | None = None,
    ):
        self.history_token_budget: int = int(history_token_budget)
        self._state_path: Path | None = (
            Path(state_path) if state_path is not None else None
        )
        _cfg = config or {}
        self._compact_threshold = _sanitize_positive_int(
            _cfg.get("compact_threshold"), 2048
        )
        self._max_history_rounds = _sanitize_positive_int(
            _cfg.get("max_history_rounds"), 20
        )
        self.rounds: list[ExplicitHistoryRound] = []
        if self._state_path is not None:
            self._load_from_disk()

    def append(self, r: ExplicitHistoryRound) -> None:
        self.rounds.append(r)
        self._persist()

    def get_rounds(self) -> list[ExplicitHistoryRound]:
        return list(self.rounds)

    def pop_oldest(self) -> ExplicitHistoryRound | None:
        if not self.rounds:
            return None
        round_ = self.rounds.pop(0)
        # Persist BEFORE returning so a crash between pop + the
        # caller's GM write doesn't resurrect the round on restart
        # (otherwise the round would re-appear in the window AND
        # its extracted nodes would already be in GM → duplicate).
        self._persist()
        return round_

    def total_tokens(self) -> int:
        return sum(
            estimate_tokens(r.stimulus_summary)
            + estimate_tokens(r.decision_text)
            + estimate_tokens(r.note_text)
            + estimate_tokens(r.thinking_text)
            + estimate_tokens(r.recall_summary)
            for r in self.rounds
        )

    def needs_compact(self) -> bool:
        total = self.total_tokens()
        over_tokens = total > self._compact_threshold
        over_rounds = len(self.rounds) > self._max_history_rounds
        _log.debug(
            "needs_compact: total_tokens=%d compact_threshold=%d rounds=%d "
            "max_history_rounds=%d → compact=%s",
            total, self._compact_threshold, len(self.rounds),
            self._max_history_rounds, over_tokens or over_rounds,
        )
        return over_tokens or over_rounds

    # ---- persistence ---------------------------------------------------

    def _load_from_disk(self) -> None:
        """Read ``state_path`` if it exists. Missing file → empty
        window (silent -- first run). Corrupt file or wrong schema
        version → empty window + stderr warning (don't crash).
        """
        assert self._state_path is not None
        if not self._state_path.exists():
            return
        try:
            raw = self._state_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as e:
            _log.warning(
                "sliding_window: failed to read %s (%s); "
                "starting with empty window",
                self._state_path, e,
            )
            return
        if not isinstance(data, dict):
            _log.warning(
                "sliding_window: %s top-level is %s, expected mapping; "
                "starting with empty window",
                self._state_path, type(data).__name__,
            )
            return
        version = data.get("schema_version")
        if version != _STATE_SCHEMA_VERSION:
            _log.warning(
                "sliding_window: %s has schema_version=%r "
                "(expected %d); starting with empty window",
                self._state_path, version, _STATE_SCHEMA_VERSION,
            )
            return
        rounds_raw = data.get("rounds") or []
        if not isinstance(rounds_raw, list):
            _log.warning(
                "sliding_window: %s `rounds` is %s, expected list; "
                "starting with empty window",
                self._state_path, type(rounds_raw).__name__,
            )
            return
        loaded: list[ExplicitHistoryRound] = []
        for entry in rounds_raw:
            if not isinstance(entry, dict):
                continue
            try:
                loaded.append(ExplicitHistoryRound(
                    heartbeat_id=int(entry["heartbeat_id"]),
                    stimulus_summary=str(entry.get("stimulus_summary", "")),
                    decision_text=str(entry.get("decision_text", "")),
                    note_text=str(entry.get("note_text", "")),
                    thinking_text=str(entry.get("thinking_text", "")),
                    recall_summary=str(entry.get("recall_summary", "")),
                ))
            except (KeyError, TypeError, ValueError) as e:
                _log.warning(
                    "sliding_window: skipping malformed round in %s "
                    "(%s)", self._state_path, e,
                )
        self.rounds = loaded

    def _persist(self) -> None:
        """Atomic write: serialize the rounds list to a sibling
        ``.tmp`` file then ``os.replace`` onto the target path.
        Failure is logged but never raised -- a transient I/O hiccup
        shouldn't crash the heartbeat loop. The next mutation will
        re-attempt.
        """
        if self._state_path is None:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "schema_version": _STATE_SCHEMA_VERSION,
                "rounds": [asdict(r) for r in self.rounds],
            }
            # Write to a tempfile in the SAME directory so os.replace
            # is a same-filesystem rename (atomic on POSIX + Windows).
            tmp_fd, tmp_name = tempfile.mkstemp(
                prefix=self._state_path.name + ".",
                suffix=".tmp",
                dir=str(self._state_path.parent),
            )
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp_name, self._state_path)
            except Exception:
                # Cleanup partial tmpfile on any failure inside the
                # write/replace cycle.
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        except OSError as e:
            _log.warning(
                "sliding_window: persist to %s failed (%s); "
                "in-memory state retained, retry on next mutation",
                self._state_path, e,
            )