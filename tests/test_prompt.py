from datetime import datetime

import pytest

from krakey.memory.recall import RecallResult
from krakey.models.stimulus import Stimulus
from krakey.prompt.builder import PromptBuilder
from krakey.prompt.dna import DNA
from krakey.prompt.views import CapabilityView, SlidingWindowRound, StatusSnapshot


def test_dna_has_all_sections():
    for tag in ["[SELF-MODEL]", "[CAPABILITIES]", "[STATUS]",
                "[GRAPH MEMORY]", "[HISTORY]", "[STIMULUS]",
                "[THINKING]", "[DECISION]", "[NOTE]", "[IDLE]"]:
        assert tag in DNA, f"DNA missing section: {tag}"
    assert "Caveman" in DNA


def test_dna_disambiguates_sleep_and_idle():
    """Regression: DNA must teach that Sleep and Idle are different
    mechanisms + warn against ambiguous 'rest' wording that would
    otherwise slip past the Hypothalamus as a sleep trigger."""
    assert "Idle" in DNA and "Sleep" in DNA
    assert "rest" in DNA.lower()
    # The explicit trigger phrase the Hypothalamus recognises must be stated
    assert "enter sleep mode" in DNA.lower()


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
    window = [SlidingWindowRound(heartbeat_id=1,
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
    DNA → SELF-MODEL → CAPABILITIES → STIMULUS → GRAPH MEMORY →
    HISTORY → STATUS → HEARTBEAT."""
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
        "# [STIMULUS]",
        "# [GRAPH MEMORY]",
        "# [HISTORY]",
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
    stim_block = p[p.rindex("# [STIMULUS]"):p.index("# [GRAPH MEMORY]")]
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
    stim_block = p[p.rindex("# [STIMULUS]"):p.index("# [GRAPH MEMORY]")]

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
    """When the recall_anchor plugin couldn't find any GraphMemory
    context for a stimulus on the prior beat, the orchestrator
    re-pushes it with `recall_retries` incremented. The prompt must
    surface that flag so Self knows the [GRAPH MEMORY] layer has no
    related context for that signal."""
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
    stim_block = p[p.rindex("# [STIMULUS]"):p.index("# [GRAPH MEMORY]")]

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
    stim_block = p[p.rindex("# [STIMULUS]"):p.index("# [GRAPH MEMORY]")]
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
