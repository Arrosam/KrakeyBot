"""Phase 1.4: SlidingWindow (dynamic token-based window)."""
import pytest

from krakey.runtime.heartbeat.sliding_window import SlidingWindow, SlidingWindowRound


def _round(i, stim="stim", decision="dec", note=""):
    return SlidingWindowRound(heartbeat_id=i, stimulus_summary=stim,
                                decision_text=decision, note_text=note)


def test_append_and_get_rounds():
    w = SlidingWindow(max_tokens=4096)
    w.append(_round(1))
    w.append(_round(2))
    assert [r.heartbeat_id for r in w.get_rounds()] == [1, 2]


def test_needs_compact_false_when_under_limit():
    w = SlidingWindow(max_tokens=10000)
    w.append(_round(1, stim="short"))
    assert w.needs_compact() is False


def test_needs_compact_true_when_over_limit():
    # tight max_tokens + long content
    w = SlidingWindow(max_tokens=10)
    w.append(_round(1, stim="a" * 200, decision="b" * 200))
    assert w.needs_compact() is True


def test_pop_oldest_returns_and_removes():
    w = SlidingWindow(max_tokens=4096)
    w.append(_round(1))
    w.append(_round(2))
    popped = w.pop_oldest()
    assert popped.heartbeat_id == 1
    assert [r.heartbeat_id for r in w.get_rounds()] == [2]


def test_pop_oldest_on_empty_returns_none():
    w = SlidingWindow(max_tokens=4096)
    assert w.pop_oldest() is None


def test_needs_compact_clears_after_popping():
    # Use content long enough to blow past the 50-token cap under the
    # real (tiktoken cl100k_base) estimator. Previously-relied-upon
    # char/4 heuristic undercounted dramatically, so the small round
    # needed bigger payloads than expected.
    w = SlidingWindow(max_tokens=50)
    w.append(_round(1, stim="hello world " * 60,
                      decision="goodbye world " * 60))
    w.append(_round(2, stim="c"))
    assert w.needs_compact() is True
    w.pop_oldest()
    assert w.needs_compact() is False


def test_total_tokens_approximation_scales_with_content():
    w = SlidingWindow(max_tokens=4096)
    small = SlidingWindow(max_tokens=4096)
    w.append(_round(1, stim="x" * 400))
    small.append(_round(1, stim="x" * 40))
    assert w.total_tokens() > small.total_tokens()


# ---------------- persistence ----------------


def test_persistence_round_trip(tmp_path):
    """Append rounds in instance A, construct instance B pointing at
    the same file → B sees the same rounds. This is the core
    "working memory survives restart" guarantee."""
    state = tmp_path / "sw.json"
    a = SlidingWindow(history_token_budget=4096, state_path=state)
    a.append(_round(1, stim="hello", decision="reply"))
    a.append(_round(2, stim="follow-up", note="user pleased"))

    b = SlidingWindow(history_token_budget=4096, state_path=state)
    rounds = b.get_rounds()
    assert [r.heartbeat_id for r in rounds] == [1, 2]
    assert rounds[0].stimulus_summary == "hello"
    assert rounds[0].decision_text == "reply"
    assert rounds[1].note_text == "user pleased"


def test_persistence_pop_oldest_persists_before_returning(tmp_path):
    """pop_oldest must persist BEFORE returning the round. Otherwise
    a crash between pop + the caller's GM write would resurrect the
    round on restart AND its extracted nodes would already be in
    GM → double-write. Verify the file reflects the post-pop state
    by the time pop returns."""
    state = tmp_path / "sw.json"
    a = SlidingWindow(history_token_budget=4096, state_path=state)
    a.append(_round(1))
    a.append(_round(2))

    popped = a.pop_oldest()
    assert popped.heartbeat_id == 1

    b = SlidingWindow(history_token_budget=4096, state_path=state)
    assert [r.heartbeat_id for r in b.get_rounds()] == [2]


def test_persistence_missing_file_starts_empty(tmp_path):
    """First-ever start: no state file. Window must come up empty,
    no error, no warning."""
    state = tmp_path / "nonexistent.json"
    w = SlidingWindow(history_token_budget=4096, state_path=state)
    assert w.get_rounds() == []
    assert not state.exists()  # no spurious write before first append


def test_persistence_corrupt_file_recovers_to_empty(tmp_path, caplog):
    """Power failure or disk corruption left half-written JSON.
    Window must recover to empty + warn — never crash startup."""
    state = tmp_path / "sw.json"
    state.write_text('{"schema_version": 1, "rounds": [{"heartbe',
                     encoding="utf-8")
    w = SlidingWindow(history_token_budget=4096, state_path=state)
    assert w.get_rounds() == []
    # Subsequent appends should overwrite the corrupt file.
    w.append(_round(1))
    b = SlidingWindow(history_token_budget=4096, state_path=state)
    assert [r.heartbeat_id for r in b.get_rounds()] == [1]


def test_persistence_wrong_schema_version_starts_empty(tmp_path):
    """A future-version state file shouldn't be silently parsed
    against the current schema (could mis-load fields). Detect
    version mismatch → empty window + warn."""
    state = tmp_path / "sw.json"
    state.write_text(
        '{"schema_version": 99, "rounds": ['
        '{"heartbeat_id": 1, "stimulus_summary": "old"}]}',
        encoding="utf-8",
    )
    w = SlidingWindow(history_token_budget=4096, state_path=state)
    assert w.get_rounds() == []


def test_persistence_state_path_none_skips_disk(tmp_path):
    """state_path=None opts out — for tests/benchmarks that want
    pure in-memory windows. Mutations don't write anywhere."""
    w = SlidingWindow(history_token_budget=4096, state_path=None)
    w.append(_round(1))
    w.append(_round(2))
    # No file at any path; the parent directory shouldn't have
    # anything created.
    assert not any(tmp_path.iterdir())


def test_persistence_atomic_no_partial_writes(tmp_path, monkeypatch):
    """Simulate a failure mid-replace. The on-disk file must remain
    a valid JSON of the PRIOR state, not a half-written tmpfile."""
    import os as _os
    state = tmp_path / "sw.json"
    w = SlidingWindow(history_token_budget=4096, state_path=state)
    w.append(_round(1, stim="committed"))
    # File now has heartbeat_id=1 written. Hijack os.replace to
    # blow up on the next call → simulate crash between rename steps.
    real_replace = _os.replace

    def boom(*a, **kw):
        raise OSError("disk full simulation")

    monkeypatch.setattr(_os, "replace", boom)
    # Append should swallow the OSError (logged warning) and leave
    # the on-disk file unchanged from the PRIOR successful write.
    w.append(_round(2, stim="should not land"))
    monkeypatch.setattr(_os, "replace", real_replace)

    b = SlidingWindow(history_token_budget=4096, state_path=state)
    rounds = b.get_rounds()
    # Only the committed round is on disk; the failed append is
    # in memory but never persisted.
    assert [r.heartbeat_id for r in rounds] == [1]
    assert rounds[0].stimulus_summary == "committed"


def test_persistence_unicode_round_trips(tmp_path):
    """Self's actual content is heavy on Chinese / emoji / symbols.
    JSON must persist + reload them losslessly."""
    state = tmp_path / "sw.json"
    a = SlidingWindow(history_token_budget=4096, state_path=state)
    a.append(_round(
        1,
        stim="user: 你好，今天天气如何？",
        decision="检查天气 🌤️",
        note="user prefers 中文",
    ))
    b = SlidingWindow(history_token_budget=4096, state_path=state)
    r = b.get_rounds()[0]
    assert r.stimulus_summary == "user: 你好，今天天气如何？"
    assert r.decision_text == "检查天气 🌤️"
    assert r.note_text == "user prefers 中文"


# ---------------- end-to-end across Runtime restart ----------------


async def test_runtime_restart_preserves_sliding_window(tmp_path):
    """The actual user-facing guarantee: spin up a Runtime, run a
    heartbeat that writes to the window, dispose the runtime,
    spin up a NEW Runtime pointing at the same workspace file, and
    confirm the new window has the same rounds.

    Pre-fix: the new Runtime's window was empty — Self lost its
    most-recent context on every restart. GM survived but the last
    few uncompacted beats vanished.
    """
    from datetime import datetime
    from krakey.models.stimulus import Stimulus
    from tests._runtime_helpers import (
        ScriptedLLM, build_runtime_with_fakes,
    )

    sw_path = tmp_path / "shared_sw.json"

    rt_a = build_runtime_with_fakes(
        self_llm=ScriptedLLM([
            "[DECISION]\nNo action.\n[IDLE]\n1",
        ]),
        hypo_llm=ScriptedLLM([]),
    )
    # Repoint to the shared path. (The helper assigned its own
    # per-test tmpfile; we want both runtimes on the same file.)
    from krakey.runtime.heartbeat.sliding_window import SlidingWindow as SW
    rt_a.window = SW(
        history_token_budget=rt_a.window.history_token_budget,
        state_path=sw_path,
    )

    await rt_a.buffer.push(Stimulus(
        type="user_message", source="channel:cli_input",
        content="hello there", timestamp=datetime.now(),
        adrenalin=True,
    ))
    await rt_a.run(iterations=1)
    rounds_after_first = rt_a.window.get_rounds()
    assert rounds_after_first, "first runtime didn't append any rounds"
    first_hb_ids = [r.heartbeat_id for r in rounds_after_first]
    first_summaries = [r.stimulus_summary for r in rounds_after_first]
    await rt_a.close()

    # Second runtime — fresh process modeled by a fresh build.
    rt_b = build_runtime_with_fakes(
        self_llm=ScriptedLLM([]),
        hypo_llm=ScriptedLLM([]),
    )
    rt_b.window = SW(
        history_token_budget=rt_b.window.history_token_budget,
        state_path=sw_path,
    )

    rounds_b = rt_b.window.get_rounds()
    assert [r.heartbeat_id for r in rounds_b] == first_hb_ids
    assert [r.stimulus_summary for r in rounds_b] == first_summaries
    # Self's most-recent user context survives the restart.
    assert any("hello there" in r.stimulus_summary for r in rounds_b)

    await rt_b.close()
