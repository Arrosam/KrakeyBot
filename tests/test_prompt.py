from datetime import datetime

import pytest

from src.models.stimulus import Stimulus
from src.prompt.builder import PromptBuilder, SlidingWindowRound
from src.prompt.dna import DNA


def test_dna_has_all_sections():
    for tag in ["[SELF-MODEL]", "[CAPABILITIES]", "[STATUS]",
                "[GRAPH MEMORY]", "[HISTORY]", "[STIMULUS]",
                "[THINKING]", "[DECISION]", "[NOTE]", "[HIBERNATE]"]:
        assert tag in DNA, f"DNA missing section: {tag}"
    assert "Caveman" in DNA or "精简" in DNA


def test_dna_disambiguates_sleep_and_hibernate():
    """Regression: DNA must teach that Sleep and Hibernate are different
    mechanisms + warn against ambiguous 'rest/休息' wording that would
    otherwise slip past the Hypothalamus as a sleep trigger."""
    assert "Hibernate" in DNA and "Sleep" in DNA
    assert "休息" in DNA or "rest" in DNA.lower()
    # The explicit trigger phrase the Hypothalamus recognises must be stated
    assert "进入睡眠" in DNA or "enter sleep mode" in DNA.lower()


def test_dna_mentions_active_memory_recall():
    """Self must know it can dispatch memory_recall to actively explore GM,
    not just receive passive auto-recall."""
    assert "memory_recall" in DNA
    d = DNA.lower()
    assert "proactive" in d or "主动" in DNA or "explicit" in d
    assert "reflect" in d or "反思" in DNA


def test_dna_warns_about_self_vs_external_signals():
    """Regression for the 'Self echoes its own tentacle output as if user
    replied' bug. DNA must explain that tentacle_feedback in [STIMULUS]
    is the bot's own action, not user input.
    """
    assert "INCOMING" in DNA or "外部" in DNA
    assert "tentacle_feedback" in DNA.lower() or "你自己" in DNA or "YOUR" in DNA
    # Must explicitly warn against the self-echo loop
    assert "自言自语" in DNA or "self-echo" in DNA.lower() or "不是用户" in DNA


def test_dna_points_tentacle_lookup_at_capabilities_not_status():
    """After the [STATUS] split, the sentence telling Self where to look
    up tentacle names must point to [CAPABILITIES], not [STATUS]."""
    # "look it up in [CAPABILITIES]" should be present
    assert "[CAPABILITIES]" in DNA
    # the old phrasing "look it up in [STATUS]" must be gone
    assert "Look it up in `[STATUS]`" not in DNA
    assert "name not in `[STATUS]`" not in DNA


def _basic_status():
    return {
        "gm_node_count": 0,
        "gm_edge_count": 0,
        "fatigue_pct": 0,
        "fatigue_hint": "",
        "last_sleep_time": "",
        "heartbeats_since_sleep": 0,
    }


def test_builder_assembles_all_layers():
    self_model = {"identity": {"name": "Krakey", "persona": "curious bot"}}
    status = {
        "gm_node_count": 12,
        "gm_edge_count": 5,
        "fatigue_pct": 25,
        "fatigue_hint": "",
        "last_sleep_time": "never",
        "heartbeats_since_sleep": 7,
    }
    capabilities = [{"name": "action", "description": "general"}]
    recall = {"nodes": [], "edges": []}
    window = [SlidingWindowRound(heartbeat_id=1,
                                  stimulus_summary="hi",
                                  decision_text="reply",
                                  note_text="")]
    stimuli = [Stimulus(type="user_message", source="sensory:cli_input",
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
    assert "当前时间: 2026-04-24 15:32:47" in prompt


def test_builder_layer_order_is_cache_optimal():
    """Stable-first ordering is load-bearing for prompt cache hits.
    DNA → SELF-MODEL → CAPABILITIES → STIMULUS → GRAPH MEMORY →
    HISTORY → STATUS → HEARTBEAT."""
    p = PromptBuilder().build(
        self_model={"identity": {"name": "K"}},
        capabilities=[{"name": "t", "description": "d"}],
        status=_basic_status(),
        recall={"nodes": [], "edges": []},
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
    the single trailer '当前时间' survives in [STIMULUS]."""
    stim_ts = datetime(2026, 4, 24, 12, 34, 56)
    stimuli = [
        Stimulus(type="user_message", source="sensory:cli", content="hi",
                  timestamp=stim_ts),
        Stimulus(type="tentacle_feedback", source="tentacle:action",
                  content="done", timestamp=stim_ts),
    ]
    p = PromptBuilder().build(
        self_model={}, capabilities=[], status=_basic_status(),
        recall={"nodes": [], "edges": []}, window=[], stimuli=stimuli,
        current_time=datetime(2026, 4, 24, 13, 0, 0),
    )
    stim_block = p[p.rindex("# [STIMULUS]"):p.index("# [GRAPH MEMORY]")]
    # the stim's own ISO timestamp must not leak into the block
    assert "2026-04-24T12:34:56" not in stim_block
    assert "12:34:56" not in stim_block
    # the single "current time" trailer is present, seconds precision
    assert "当前时间: 2026-04-24 13:00:00" in stim_block


def test_builder_current_time_present_even_on_empty_stimulus():
    """Empty stimulus is the common quiet-beat case; the current-time
    trailer must still appear so Self always has a 'now' anchor."""
    p = PromptBuilder().build(
        self_model={}, capabilities=[], status=_basic_status(),
        recall={"nodes": [], "edges": []}, window=[], stimuli=[],
        current_time=datetime(2026, 4, 24, 9, 15, 3),
    )
    assert "(no new signals)" in p
    assert "当前时间: 2026-04-24 09:15:03" in p


def test_builder_capabilities_layer_renders_tentacles():
    p = PromptBuilder().build(
        self_model={},
        capabilities=[
            {"name": "search", "description": "web search"},
            {"name": "memory_recall", "description": "GM recall"},
        ],
        status=_basic_status(),
        recall={"nodes": [], "edges": []}, window=[], stimuli=[],
        current_time=datetime(2026, 4, 24),
    )
    cap_block = p[p.index("# [CAPABILITIES]"):p.index("# [STIMULUS]")]
    assert "search: web search" in cap_block
    assert "memory_recall: GM recall" in cap_block


def test_builder_status_has_no_tentacle_info():
    """After split, tentacle list must NOT appear under [STATUS] (cache
    hygiene — status changes every beat, capabilities shouldn't)."""
    p = PromptBuilder().build(
        self_model={},
        capabilities=[{"name": "search", "description": "web search"}],
        status=_basic_status(),
        recall={"nodes": [], "edges": []}, window=[], stimuli=[],
        current_time=datetime(2026, 4, 24),
    )
    status_block = p[p.rindex("# [STATUS]"):p.index("# [HEARTBEAT]")]
    assert "search" not in status_block
    assert "web search" not in status_block


def test_builder_splits_stimulus_by_source_type():
    """Phase 1.7 fix: STIMULUS must group by type so Self does not confuse
    its own tentacle outputs with user input."""
    stimuli = [
        Stimulus(type="user_message", source="sensory:cli_input",
                  content="hello bot", timestamp=datetime(2026, 4, 19)),
        Stimulus(type="tentacle_feedback", source="tentacle:action",
                  content="hi user!", timestamp=datetime(2026, 4, 19)),
        Stimulus(type="batch_complete", source="sensory:batch_tracker",
                  content="batch done", timestamp=datetime(2026, 4, 19)),
    ]
    p = PromptBuilder().build(
        self_model={}, capabilities=[], status=_basic_status(),
        recall={"nodes": [], "edges": []}, window=[], stimuli=stimuli,
        current_time=datetime(2026, 4, 24),
    )

    # Scope search to the [STIMULUS] block (DNA also mentions these names).
    stim_block = p[p.rindex("# [STIMULUS]"):p.index("# [GRAPH MEMORY]")]

    assert "INCOMING" in stim_block or "用户/外部输入" in stim_block
    assert "YOUR RECENT ACTIONS" in stim_block or "你自己刚才行动的结果" in stim_block
    assert "SYSTEM" in stim_block or "系统事件" in stim_block

    incoming_idx = max(stim_block.find("INCOMING"),
                        stim_block.find("用户/外部输入"))
    own_idx = max(stim_block.find("YOUR RECENT ACTIONS"),
                   stim_block.find("你自己刚才行动的结果"))
    user_pos = stim_block.find("hello bot")
    bot_pos = stim_block.find("hi user!")
    assert incoming_idx >= 0 and own_idx >= 0
    assert incoming_idx < user_pos < own_idx, \
        "user_message must land under INCOMING, before YOUR RECENT ACTIONS"
    assert own_idx < bot_pos, \
        "tentacle_feedback must land under YOUR RECENT ACTIONS"


def test_builder_omits_empty_subsections():
    """If no incoming, that subsection header should not appear."""
    stimuli = [
        Stimulus(type="tentacle_feedback", source="tentacle:action",
                  content="my reply", timestamp=datetime(2026, 4, 19)),
    ]
    p = PromptBuilder().build(
        self_model={}, capabilities=[], status=_basic_status(),
        recall={"nodes": [], "edges": []}, window=[], stimuli=stimuli,
        current_time=datetime(2026, 4, 24),
    )
    stim_block = p[p.rindex("# [STIMULUS]"):p.index("# [GRAPH MEMORY]")]
    assert "你自己刚才行动的结果" in stim_block
    assert "用户/外部输入" not in stim_block
    assert "系统事件" not in stim_block


def test_builder_handles_empty_recall_and_window():
    prompt = PromptBuilder().build(
        self_model={}, capabilities=[], status=_basic_status(),
        recall={"nodes": [], "edges": []}, window=[], stimuli=[],
        current_time=datetime(2026, 4, 24),
    )
    assert "[STIMULUS]" in prompt
    # heartbeat question always appended
    assert "?" in prompt or "question" in prompt.lower() or "heartbeat" in prompt.lower()


def test_builder_injects_recall_nodes_and_edges():
    recall = {
        "nodes": [{"name": "Apple", "category": "FACT",
                   "description": "a fruit", "neighbor_keywords": ["fruit", "tree"]}],
        "edges": [{"source": "Apple", "predicate": "RELATED_TO", "target": "Banana"}],
    }
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
