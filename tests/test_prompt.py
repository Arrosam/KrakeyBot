from datetime import datetime

import pytest

from krakey.interfaces.engines.recall import RecallResult
from krakey.models.stimulus import Stimulus
from krakey.prompt.builder import PromptBuilder
from krakey.prompt.dna import DNA
from krakey.prompt.views import CapabilityView, ExplicitHistoryRound, StatusSnapshot


def test_dna_has_all_sections():
    for tag in ["[SELF-MODEL]", "[CAPABILITIES]", "[STATUS]",
                "[GRAPH MEMORY]", "[HISTORY]", "[STIMULUS]",
                "[THINKING]", "[DECISION]", "[NOTE]", "[IDLE]"]:
        assert tag in DNA, f"DNA missing section: {tag}"
    assert "Caveman" in DNA


def test_dna_disambiguates_sleep_and_idle():
    """DNA must teach that Sleep and Idle are different mechanisms +
    warn against ambiguous wording ("rest", "pause") that should NOT
    enter Sleep. The exact Sleep trigger shape (sleep tool call vs.
    NL phrase) lives in the per-engine [ACTION FORMAT] layer, not
    DNA — so this test no longer asserts a specific phrasing."""
    assert "Idle" in DNA and "Sleep" in DNA
    assert "rest" in DNA.lower()
    # DNA must point Self at [ACTION FORMAT] for the actual trigger
    # (since it varies by decision engine), instead of pinning a
    # specific phrase or call shape itself.
    assert "[ACTION FORMAT]" in DNA


def test_dna_mentions_active_memory_recall():
    """Self must know it can dispatch a recall tool to actively
    explore GM, not just receive passive auto-recall. DNA does not
    name a specific tool (the live name lives in [CAPABILITIES]
    — naming it in DNA would couple the always-on prompt prefix to a
    swappable plugin), but it must teach the *concept* of proactive
    recall and point Self at [CAPABILITIES] to find the actual name.
    """
    d = DNA.lower()
    assert "proactive" in d or "explicit" in d
    assert "recall" in d
    assert "modifier" in d
    assert "[capabilities]" in d


def test_dna_warns_about_self_vs_external_signals():
    """Regression for the 'Self echoes its own tool output as if user
    replied' bug. DNA must explain that tool_feedback in [STIMULUS]
    is the bot's own action, not user input.
    """
    assert "INCOMING" in DNA
    assert "tool_feedback" in DNA.lower() or "YOUR" in DNA
    # Must explicitly warn against the self-echo loop
    assert "self-echo" in DNA.lower()


def test_dna_points_tool_lookup_at_capabilities_not_status():
    """After the [STATUS] split, the sentence telling Self where to look
    up tool names must point to [CAPABILITIES], not [STATUS]."""
    # "look it up in [CAPABILITIES]" should be present
    assert "[CAPABILITIES]" in DNA
    # the old phrasing "look it up in [STATUS]" must be gone
    assert "Look it up in `[STATUS]`" not in DNA
    assert "name not in `[STATUS]`" not in DNA


def _basic_status() -> StatusSnapshot:
    return StatusSnapshot(
        gm_node_count=0, gm_edge_count=0,
        fatigue_pct=0, fatigue_hint="",
        last_sleep_time="", heartbeats_since_sleep=0,
    )


def test_builder_assembles_all_layers():
    self_model = {"identity": {"name": "Krakey", "persona": "curious bot"}}
    status = StatusSnapshot(
        gm_node_count=12, gm_edge_count=5,
        fatigue_pct=25, fatigue_hint="",
        last_sleep_time="never", heartbeats_since_sleep=7,
    )
    capabilities = [CapabilityView(name="action", description="general")]
    recall = RecallResult()
    window = [ExplicitHistoryRound(heartbeat_id=1,
                                  stimulus_summary="hi",
                                  decision_text="reply",
                                  note_text="")]
    stimuli = [Stimulus(type="user_message", source="channel:cli_input",
                         content="hello", timestamp=datetime(2026, 4, 18))]

    prompt = PromptBuilder().build(
        self_model=self_model, capabilities=capabilities, status=status,
        recall=recall, window=window, stimuli=stimuli,
        current_time=datetime(2026, 4, 24, 15, 32, 47),
    )

    assert "Krakey" in prompt
    assert "action" in prompt
    assert "hello" in prompt
    assert "hi" in prompt
    assert "reply" in prompt
    assert "# [SELF-MODEL]" in prompt
    assert "# [CAPABILITIES]" in prompt
    assert "# [STIMULUS]" in prompt
    assert "# [STATUS]" in prompt
    assert "current time: 2026-04-24 15:32:47" in prompt


def test_builder_layer_order_is_cache_optimal():
    """Stable-first ordering is load-bearing for prompt cache hits.
    DNA → SELF-MODEL → CAPABILITIES → GRAPH MEMORY → HISTORY →
    STIMULUS → STATUS → HEARTBEAT.
    Note: STIMULUS is placed AFTER HISTORY (accepted prefix-cache
    trade-off) rather than in the original stable-first position."""
    p = PromptBuilder().build(
        self_model={"identity": {"name": "K"}},
        capabilities=[CapabilityView(name="t", description="d")],
        status=_basic_status(),
        recall=RecallResult(),
        window=[],
        stimuli=[],
        current_time=datetime(2026, 4, 24),
    )
    order = [
        "# [SELF-MODEL]",
        "# [CAPABILITIES]",
        "# [GRAPH MEMORY]",
        "# [HISTORY]",
        "# [STIMULUS]",
        "# [STATUS]",
        "# [HEARTBEAT]",
    ]
    positions = [p.index(tag) for tag in order]
    assert positions == sorted(positions), \
        f"layers out of order: {list(zip(order, positions))}"


def test_builder_stimulus_has_no_per_stim_timestamps():
    """Per-stim ISO timestamps were removed for cache stability. Only
    the single trailer 'current time' survives in [STIMULUS]."""
    stim_ts = datetime(2026, 4, 24, 12, 34, 56)
    stimuli = [
        Stimulus(type="user_message", source="channel:cli", content="hi",
                  timestamp=stim_ts),
        Stimulus(type="tool_feedback", source="tool:action",
                  content="done", timestamp=stim_ts),
    ]
    p = PromptBuilder().build(
        self_model={}, capabilities=[], status=_basic_status(),
        recall=RecallResult(), window=[], stimuli=stimuli,
        current_time=datetime(2026, 4, 24, 13, 0, 0),
    )
    stim_block = p[p.rindex("# [STIMULUS]"):p.index("# [STATUS]")]
    # the stim's own ISO timestamp must not leak into the block
    assert "2026-04-24T12:34:56" not in stim_block
    assert "12:34:56" not in stim_block
    # the single "current time" trailer is present, seconds precision
    assert "current time: 2026-04-24 13:00:00" in stim_block


def test_builder_current_time_present_even_on_empty_stimulus():
    """Empty stimulus is the common quiet-beat case; the current-time
    trailer must still appear so Self always has a 'now' anchor."""
    p = PromptBuilder().build(
        self_model={}, capabilities=[], status=_basic_status(),
        recall=RecallResult(), window=[], stimuli=[],
        current_time=datetime(2026, 4, 24, 9, 15, 3),
    )
    assert "(no new signals)" in p
    assert "current time: 2026-04-24 09:15:03" in p


def test_builder_capabilities_layer_renders_tools():
    p = PromptBuilder().build(
        self_model={},
        capabilities=[
            CapabilityView(name="search", description="web search"),
            CapabilityView(name="memory_recall", description="GM recall"),
        ],
        status=_basic_status(),
        recall=RecallResult(), window=[], stimuli=[],
        current_time=datetime(2026, 4, 24),
    )
    cap_block = p[p.index("# [CAPABILITIES]"):p.index("# [STIMULUS]")]
    assert "search: web search" in cap_block
    assert "memory_recall: GM recall" in cap_block


def test_builder_status_has_no_tool_info():
    """After split, tool list must NOT appear under [STATUS] (cache
    hygiene — status changes every beat, capabilities shouldn't)."""
    p = PromptBuilder().build(
        self_model={},
        capabilities=[CapabilityView(name="search", description="web search")],
        status=_basic_status(),
        recall=RecallResult(), window=[], stimuli=[],
        current_time=datetime(2026, 4, 24),
    )
    status_block = p[p.rindex("# [STATUS]"):p.index("# [HEARTBEAT]")]
    assert "search" not in status_block
    assert "web search" not in status_block


def test_builder_splits_stimulus_by_source_type():
    """Phase 1.7 fix: STIMULUS must group by type so Self does not confuse
    its own tool outputs with user input."""
    stimuli = [
        Stimulus(type="user_message", source="channel:cli_input",
                  content="hello bot", timestamp=datetime(2026, 4, 19)),
        Stimulus(type="tool_feedback", source="tool:action",
                  content="hi user!", timestamp=datetime(2026, 4, 19)),
        Stimulus(type="batch_complete", source="channel:batch_tracker",
                  content="batch done", timestamp=datetime(2026, 4, 19)),
    ]
    p = PromptBuilder().build(
        self_model={}, capabilities=[], status=_basic_status(),
        recall=RecallResult(), window=[], stimuli=stimuli,
        current_time=datetime(2026, 4, 24),
    )

    # Scope search to the [STIMULUS] block (DNA also mentions these names).
    stim_block = p[p.rindex("# [STIMULUS]"):p.index("# [STATUS]")]

    assert "INCOMING" in stim_block
    assert "YOUR RECENT ACTIONS" in stim_block
    assert "SYSTEM" in stim_block

    incoming_idx = stim_block.find("INCOMING")
    own_idx = stim_block.find("YOUR RECENT ACTIONS")
    user_pos = stim_block.find("hello bot")
    bot_pos = stim_block.find("hi user!")
    assert incoming_idx >= 0 and own_idx >= 0
    assert incoming_idx < user_pos < own_idx, \
        "user_message must land under INCOMING, before YOUR RECENT ACTIONS"
    assert own_idx < bot_pos, \
        "tool_feedback must land under YOUR RECENT ACTIONS"


def test_builder_marks_recall_retry_stimuli():
    """When recall couldn't find any GraphMemory context for a
    stimulus on the prior beat, the orchestrator re-pushes it with
    ``recall_retries`` incremented. The prompt must surface that flag
    so Self knows the [GRAPH MEMORY] layer has no related context for
    that signal."""
    fresh = Stimulus(
        type="user_message", source="channel:cli",
        content="brand new question",
        timestamp=datetime(2026, 4, 19),
    )
    retried = Stimulus(
        type="user_message", source="channel:cli",
        content="lonely message",
        timestamp=datetime(2026, 4, 19),
        metadata={"recall_retries": 1},
    )
    p = PromptBuilder().build(
        self_model={}, capabilities=[], status=_basic_status(),
        recall=RecallResult(), window=[], stimuli=[fresh, retried],
        current_time=datetime(2026, 4, 24),
    )
    stim_block = p[p.rindex("# [STIMULUS]"):p.index("# [STATUS]")]

    # Fresh stimulus has NO retry marker
    fresh_idx = stim_block.find("brand new question")
    assert fresh_idx >= 0
    next_sep = stim_block.find("---", fresh_idx)
    fresh_chunk = (
        stim_block[fresh_idx:next_sep] if next_sep != -1
        else stim_block[fresh_idx:]
    )
    assert "no related graph memory recalled" not in fresh_chunk

    # Retried stimulus has the marker (with retry count)
    retried_idx = stim_block.find("lonely message")
    assert retried_idx >= 0
    assert "no related graph memory recalled" in stim_block[retried_idx:]
    assert "retry #1" in stim_block[retried_idx:]


def test_builder_omits_empty_subsections():
    """If no incoming, that subsection header should not appear."""
    stimuli = [
        Stimulus(type="tool_feedback", source="tool:action",
                  content="my reply", timestamp=datetime(2026, 4, 19)),
    ]
    p = PromptBuilder().build(
        self_model={}, capabilities=[], status=_basic_status(),
        recall=RecallResult(), window=[], stimuli=stimuli,
        current_time=datetime(2026, 4, 24),
    )
    stim_block = p[p.rindex("# [STIMULUS]"):p.index("# [STATUS]")]
    assert "YOUR RECENT ACTIONS" in stim_block
    assert "INCOMING" not in stim_block
    assert "SYSTEM" not in stim_block


def test_builder_handles_empty_recall_and_window():
    prompt = PromptBuilder().build(
        self_model={}, capabilities=[], status=_basic_status(),
        recall=RecallResult(), window=[], stimuli=[],
        current_time=datetime(2026, 4, 24),
    )
    assert "[STIMULUS]" in prompt
    # heartbeat question always appended
    assert "?" in prompt or "question" in prompt.lower() or "heartbeat" in prompt.lower()


def test_builder_injects_recall_nodes_and_edges():
    recall = RecallResult(
        nodes=[{"name": "Apple", "category": "FACT",
                 "description": "a fruit",
                 "neighbor_keywords": ["fruit", "tree"]}],
        edges=[{"source": "Apple", "predicate": "RELATED_TO",
                 "target": "Banana"}],
    )
    p = PromptBuilder().build(
        self_model={}, capabilities=[], status=_basic_status(),
        recall=recall, window=[], stimuli=[],
        current_time=datetime(2026, 4, 24),
    )
    assert "Apple" in p
    assert "FACT" in p
    assert "RELATED_TO" in p
    assert "Banana" in p
    assert "fruit" in p


def test_builder_status_field_values_render_through():
    """Producer-side typo guard: every StatusSnapshot field must surface
    in the rendered prompt with its actual value. If a producer typoes
    a field name (or the renderer reads the wrong attribute), one of
    these substring assertions fails — much earlier than waiting for
    Self to start saying weird things about its own fatigue."""
    status = StatusSnapshot(
        gm_node_count=42, gm_edge_count=17,
        fatigue_pct=66, fatigue_hint="(getting tired)",
        last_sleep_time="2026-04-26 03:14",
        heartbeats_since_sleep=199,
    )
    p = PromptBuilder().build(
        self_model={}, capabilities=[], status=status,
        recall=RecallResult(), window=[], stimuli=[],
        current_time=datetime(2026, 4, 24),
    )
    assert "42 nodes" in p
    assert "17 edges" in p
    assert "66%" in p
    assert "(getting tired)" in p
    assert "2026-04-26 03:14" in p
    assert "199" in p


# ---------------------------------------------------------------------------
# Part D — STIMULUS layer moves to directly below HISTORY
# New canonical order: SELF-MODEL → CAPABILITIES → (ACTION FORMAT) →
#   GRAPH MEMORY → HISTORY → STIMULUS → STATUS → HEARTBEAT
# ---------------------------------------------------------------------------

_DIVIDER = "--- SESSION BOUNDARY (above: previous session | below: current session) ---"


def _build_prompt(stimuli=None, window=None):
    """Helper: build a full prompt with default-empty parts."""
    return PromptBuilder().build(
        self_model={"identity": {"name": "K"}},
        capabilities=[CapabilityView(name="t", description="d")],
        status=_basic_status(),
        recall=RecallResult(),
        window=window if window is not None else [],
        stimuli=stimuli if stimuli is not None else [],
        current_time=datetime(2026, 5, 1),
    )


def _make_round(heartbeat_id: int) -> ExplicitHistoryRound:
    return ExplicitHistoryRound(
        heartbeat_id=heartbeat_id,
        stimulus_summary=f"s{heartbeat_id}",
        decision_text=f"d{heartbeat_id}",
        note_text="",
    )


def _make_stimulus(content: str = "hi") -> Stimulus:
    return Stimulus(
        type="user_message", source="channel:cli",
        content=content, timestamp=datetime(2026, 5, 1),
    )


class TestPartD_LayerOrder:
    """Part D: STIMULUS layer is now positioned AFTER HISTORY and before STATUS."""

    # --- positive: full prompt, non-empty stimulus and history ---

    def test_stimulus_after_history_in_full_prompt(self):
        """Positive: HISTORY < STIMULUS in position when both are populated."""
        window = [_make_round(1)]
        stimuli = [_make_stimulus("hello")]
        p = _build_prompt(stimuli=stimuli, window=window)
        assert p.index("# [HISTORY]") < p.index("# [STIMULUS]"), \
            "# [HISTORY] must appear before # [STIMULUS]"

    def test_stimulus_after_graph_memory_in_full_prompt(self):
        """Positive: GRAPH MEMORY < STIMULUS in position."""
        window = [_make_round(1)]
        stimuli = [_make_stimulus("hello")]
        p = _build_prompt(stimuli=stimuli, window=window)
        assert p.index("# [GRAPH MEMORY]") < p.index("# [STIMULUS]"), \
            "# [GRAPH MEMORY] must appear before # [STIMULUS]"

    def test_stimulus_before_status_in_full_prompt(self):
        """Positive: STIMULUS < STATUS in position."""
        window = [_make_round(1)]
        stimuli = [_make_stimulus("hello")]
        p = _build_prompt(stimuli=stimuli, window=window)
        assert p.index("# [STIMULUS]") < p.index("# [STATUS]"), \
            "# [STIMULUS] must appear before # [STATUS]"

    def test_full_order_verified_at_once_non_empty(self):
        """Positive: the combined GRAPH MEMORY < HISTORY < STIMULUS < STATUS
        chain holds in one assertion when prompt is fully populated."""
        window = [_make_round(1)]
        stimuli = [_make_stimulus("hello")]
        p = _build_prompt(stimuli=stimuli, window=window)
        gm = p.index("# [GRAPH MEMORY]")
        hist = p.index("# [HISTORY]")
        stim = p.index("# [STIMULUS]")
        stat = p.index("# [STATUS]")
        assert gm < hist < stim < stat, (
            f"Expected GRAPH MEMORY({gm}) < HISTORY({hist}) < "
            f"STIMULUS({stim}) < STATUS({stat})"
        )

    # --- equivalence: empty stimuli still obey the order ---

    def test_order_preserved_with_empty_stimuli(self):
        """Equivalence: empty stimuli list — STIMULUS header still exists
        and still sits between HISTORY and STATUS."""
        p = _build_prompt(stimuli=[], window=[_make_round(5)])
        assert p.index("# [HISTORY]") < p.index("# [STIMULUS]") < p.index("# [STATUS]")

    def test_order_preserved_with_non_empty_stimuli(self):
        """Equivalence: populated stimuli list — order constraint identical."""
        stimuli = [_make_stimulus("msg1"), _make_stimulus("msg2")]
        p = _build_prompt(stimuli=stimuli, window=[_make_round(5)])
        assert p.index("# [HISTORY]") < p.index("# [STIMULUS]") < p.index("# [STATUS]")

    # --- boundary: empty window + empty stimuli ---

    def test_order_preserved_with_empty_window_and_empty_stimuli(self):
        """BVA: minimal prompt — no history rounds, no stimuli. All
        section headers must still appear in canonical order."""
        p = _build_prompt(stimuli=[], window=[])
        order = [
            "# [SELF-MODEL]",
            "# [CAPABILITIES]",
            "# [GRAPH MEMORY]",
            "# [HISTORY]",
            "# [STIMULUS]",
            "# [STATUS]",
            "# [HEARTBEAT]",
        ]
        positions = [p.index(tag) for tag in order]
        assert positions == sorted(positions), \
            f"layers out of order on empty inputs: {list(zip(order, positions))}"

    def test_stimulus_is_not_before_graph_memory(self):
        """BVA / regression: the OLD position (STIMULUS before GRAPH MEMORY)
        must no longer exist. STIMULUS must not precede GRAPH MEMORY."""
        p = _build_prompt(stimuli=[_make_stimulus()], window=[])
        assert p.index("# [STIMULUS]") > p.index("# [GRAPH MEMORY]"), \
            "STIMULUS must not appear before GRAPH MEMORY (old order regression)"


# ---------------------------------------------------------------------------
# Part E — single cross-session divider in render_history
# ---------------------------------------------------------------------------

class TestPartE_RenderHistoryDivider:
    """Part E: render_history inserts exactly one cross-session divider
    at the LAST index where heartbeat_id resets (window[i] <= window[i-1])."""

    # --- positive: canonical multi-session window ---

    def test_divider_present_in_multi_session_window(self):
        """Positive: ids [48,49,50,1,2] — exactly one divider."""
        window = [_make_round(hid) for hid in [48, 49, 50, 1, 2]]
        rendered = PromptBuilder().render_history(window)
        assert rendered.count(_DIVIDER) == 1

    def test_divider_exact_substring(self):
        """Positive: the divider line matches the contractual literal string."""
        window = [_make_round(hid) for hid in [48, 49, 50, 1, 2]]
        rendered = PromptBuilder().render_history(window)
        assert _DIVIDER in rendered

    def test_divider_after_id50_block_before_second_id1_block(self):
        """Positive: divider is AFTER the Heartbeat #50 block and BEFORE
        the second id=1 block (index 3 in the window)."""
        window = [_make_round(hid) for hid in [48, 49, 50, 1, 2]]
        rendered = PromptBuilder().render_history(window)
        divider_pos = rendered.index(_DIVIDER)
        # 'Heartbeat #50' text rendered for window[2] (id=50)
        # The SECOND occurrence of '1' as a heartbeat id is for window[3].
        # We use the stimulus_summary/decision text 's1'/'d1' but those
        # also appear for the first window[3]=id1 entry; instead we rely
        # on positional ordering: divider must come after some text from
        # id=50 and before some text from the boundary round.
        # The round at index 2 (id=50) has stimulus_summary='s50'.
        pos_s50 = rendered.index("s50")
        # The round at index 3 (id=1, the reset round) has
        # stimulus_summary='s1' — find the SECOND occurrence since
        # window[0] also has id=1... wait, window[0] has id=48.
        # Stimulus summary for the boundary round (index 3) is 's1'.
        # Round at index 0 has id=48 → 's48'.  Round at index 3 → 's1'.
        # So 's1' only appears once here.
        pos_boundary_round = rendered.index("s1")
        assert pos_s50 < divider_pos, \
            "Divider must come AFTER the id=50 round's content"
        assert divider_pos < pos_boundary_round, \
            "Divider must come BEFORE the boundary (id=1 reset) round's content"

    # --- boundary / equivalence: no divider cases ---

    def test_monotonic_ids_no_divider(self):
        """BVA: ids [1,2,3] strictly increasing — no session boundary."""
        window = [_make_round(hid) for hid in [1, 2, 3]]
        rendered = PromptBuilder().render_history(window)
        assert _DIVIDER not in rendered

    def test_single_round_no_divider(self):
        """BVA: single-element window — no comparison possible, no divider."""
        window = [_make_round(7)]
        rendered = PromptBuilder().render_history(window)
        assert _DIVIDER not in rendered

    def test_empty_window_no_divider_returns_empty_marker(self):
        """BVA: empty window — no divider, and the existing (empty) marker
        is returned (matching the pre-existing empty-path contract)."""
        rendered = PromptBuilder().render_history([])
        assert _DIVIDER not in rendered
        assert "(empty)" in rendered

    def test_two_rounds_with_reset_one_divider(self):
        """BVA: minimal session boundary — [5, 1] has one reset at i=1."""
        window = [_make_round(5), _make_round(1)]
        rendered = PromptBuilder().render_history(window)
        assert rendered.count(_DIVIDER) == 1
        assert _DIVIDER in rendered

    def test_two_rounds_no_reset_no_divider(self):
        """BVA: [5, 6] — strictly increasing, no divider."""
        window = [_make_round(5), _make_round(6)]
        rendered = PromptBuilder().render_history(window)
        assert _DIVIDER not in rendered

    # --- state / multi-session: LAST reset wins ---

    def test_multiple_resets_only_last_divider_drawn(self):
        """State: ids [1,2,1,2,1] has resets at i=2 and i=4. Divider must
        appear ONLY at the last reset (before index 4), exactly once."""
        window = [_make_round(hid) for hid in [1, 2, 1, 2, 1]]
        rendered = PromptBuilder().render_history(window)
        assert rendered.count(_DIVIDER) == 1
        # The last round (index 4, id=1) has stimulus_summary='s1'.
        # There are multiple rounds with id=1; we need to verify the
        # divider falls before the LAST one. The last round's decision
        # text is 'd1'. We find the last occurrence of 'd1' in rendered.
        # The round at index 2 also has id=1 — both emit the same text.
        # We can only assert count==1 + that the divider exists.
        # For positional proof: divider must come after content from
        # index 3 (id=2, summary='s2' — last occurrence of 's2').
        last_s2_pos = rendered.rindex("s2")
        divider_pos = rendered.index(_DIVIDER)
        assert divider_pos > last_s2_pos, \
            "Divider must come AFTER the last id=2 round (index 3), not at the earlier reset"

    # --- negative / edge: equal adjacent ids count as reset ---

    def test_equal_adjacent_ids_count_as_reset(self):
        """Negative/edge: [3, 3] — equal ids satisfy window[i] <= window[i-1],
        so this is treated as a reset; one divider before index 1."""
        window = [_make_round(3), _make_round(3)]
        rendered = PromptBuilder().render_history(window)
        assert rendered.count(_DIVIDER) == 1

    def test_equal_adjacent_divider_before_second_round(self):
        """Negative/edge: [3, 3] — divider must precede the second round's content."""
        window = [_make_round(3), _make_round(3)]
        rendered = PromptBuilder().render_history(window)
        divider_pos = rendered.index(_DIVIDER)
        # Both rounds have same id, same summary 's3'. We need the
        # first round's content to appear before the divider.
        # The first round is rendered first; its decision text 'd3'
        # appears before the divider, and the second 'd3' after.
        first_d3 = rendered.index("d3")
        assert first_d3 < divider_pos, \
            "First round content must precede the divider for equal-id [3,3] window"

    def test_three_equal_ids_only_one_divider_at_last_reset(self):
        """Negative/edge: [10, 10, 10] — resets at i=1 and i=2.
        Only ONE divider, at the LAST reset (before index 2)."""
        window = [_make_round(10), _make_round(10), _make_round(10)]
        rendered = PromptBuilder().render_history(window)
        assert rendered.count(_DIVIDER) == 1
        # The divider must fall after the content from the middle round
        # (index 1) and before the content from the last round (index 2).
        # All three rounds emit the same text since id=10 and same helper.
        # Positional ordering: the second occurrence of the round content
        # block appears before the divider; the third is after it.
        # We can at minimum assert exactly one divider and that its
        # position is past the midpoint of the content.
        mid = len(rendered) // 2
        divider_pos = rendered.index(_DIVIDER)
        # The last reset is at index 2 (the 3rd round). If the divider
        # is drawn ONLY at the last reset, it must appear in the latter
        # portion of the rendered string (after 2 of 3 rounds are shown).
        assert divider_pos > mid, \
            "Divider at last reset should appear after the majority of content"

    def test_at_most_one_divider_invariant(self):
        """Negative: no matter how many resets exist, count is always <= 1."""
        # Alternating session ids: many resets
        window = [_make_round(hid) for hid in [5, 1, 5, 1, 5, 1, 5]]
        rendered = PromptBuilder().render_history(window)
        count = rendered.count(_DIVIDER)
        assert count <= 1, f"Expected at most 1 divider, got {count}"

    def test_divider_exact_string_no_variant(self):
        """Negative: any variant of the divider string (different dashes, extra
        spaces, different wording) must NOT be treated as the real divider."""
        wrong_variants = [
            "--- Session Boundary ---",
            "--- SESSION BOUNDARY ---",
            "-- SESSION BOUNDARY (above: previous session | below: current session) --",
            "--- session boundary (above: previous session | below: current session) ---",
        ]
        window = [_make_round(5), _make_round(1)]
        rendered = PromptBuilder().render_history(window)
        # The real divider is present
        assert _DIVIDER in rendered
        # None of the wrong variants should appear (the impl uses the exact string)
        for variant in wrong_variants:
            assert variant not in rendered, \
                f"Unexpected variant divider found: {variant!r}"
