from datetime import datetime

import pytest

from src.models.stimulus import Stimulus
from src.prompt.builder import PromptBuilder, SlidingWindowRound
from src.prompt.dna import DNA


def test_dna_has_all_sections():
    for tag in ["[SELF-MODEL]", "[STATUS]", "[GRAPH MEMORY]", "[HISTORY]",
                "[STIMULUS]", "[THINKING]", "[DECISION]", "[NOTE]", "[HIBERNATE]"]:
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


def test_builder_assembles_all_layers():
    self_model = {"identity": {"name": "Krakey", "persona": "curious bot"}}
    status = {
        "gm_node_count": 12,
        "gm_edge_count": 5,
        "fatigue_pct": 25,
        "fatigue_hint": "",
        "last_sleep_time": "never",
        "heartbeats_since_sleep": 7,
        "tentacles": [{"name": "action", "description": "general"}],
    }
    recall = {"nodes": [], "edges": []}
    window = [SlidingWindowRound(heartbeat_id=1,
                                  stimulus_summary="hi",
                                  decision_text="reply",
                                  note_text="")]
    stimuli = [Stimulus(type="user_message", source="sensory:cli_input",
                         content="hello", timestamp=datetime(2026, 4, 18))]

    prompt = PromptBuilder().build(self_model=self_model, status=status,
                                    recall=recall, window=window, stimuli=stimuli)

    assert "Krakey" in prompt
    assert "action" in prompt
    assert "hello" in prompt
    assert "hi" in prompt
    assert "reply" in prompt
    assert "[SELF-MODEL]" in prompt
    assert "[STIMULUS]" in prompt
    assert "# [STATUS]" in prompt


def test_builder_splits_stimulus_by_source_type():
    """Phase 1.7 fix: STIMULUS must group by type so Self does not confuse
    its own tentacle outputs with user input."""
    status = {"gm_node_count": 0, "gm_edge_count": 0, "fatigue_pct": 0,
              "fatigue_hint": "", "last_sleep_time": "", "heartbeats_since_sleep": 0,
              "tentacles": []}
    stimuli = [
        Stimulus(type="user_message", source="sensory:cli_input",
                  content="hello bot", timestamp=datetime(2026, 4, 19)),
        Stimulus(type="tentacle_feedback", source="tentacle:action",
                  content="hi user!", timestamp=datetime(2026, 4, 19)),
        Stimulus(type="batch_complete", source="sensory:batch_tracker",
                  content="batch done", timestamp=datetime(2026, 4, 19)),
    ]
    p = PromptBuilder().build(self_model={}, status=status,
                                recall={"nodes": [], "edges": []},
                                window=[], stimuli=stimuli)

    # Scope search to the [STIMULUS] block (DNA also mentions these names).
    stim_block = p[p.rindex("# [STIMULUS]"):]

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
    status = {"gm_node_count": 0, "gm_edge_count": 0, "fatigue_pct": 0,
              "fatigue_hint": "", "last_sleep_time": "", "heartbeats_since_sleep": 0,
              "tentacles": []}
    stimuli = [
        Stimulus(type="tentacle_feedback", source="tentacle:action",
                  content="my reply", timestamp=datetime(2026, 4, 19)),
    ]
    p = PromptBuilder().build(self_model={}, status=status,
                                recall={"nodes": [], "edges": []},
                                window=[], stimuli=stimuli)
    stim_block = p[p.rindex("# [STIMULUS]"):]
    assert "你自己刚才行动的结果" in stim_block
    assert "用户/外部输入" not in stim_block
    assert "系统事件" not in stim_block


def test_builder_handles_empty_recall_and_window():
    prompt = PromptBuilder().build(
        self_model={},
        status={"gm_node_count": 0, "gm_edge_count": 0, "fatigue_pct": 0,
                "fatigue_hint": "", "last_sleep_time": "",
                "heartbeats_since_sleep": 0, "tentacles": []},
        recall={"nodes": [], "edges": []},
        window=[],
        stimuli=[])
    assert "[STIMULUS]" in prompt
    # heartbeat question always appended
    assert "?" in prompt or "question" in prompt.lower() or "heartbeat" in prompt.lower()


def test_builder_injects_recall_nodes_and_edges():
    status = {"gm_node_count": 1, "gm_edge_count": 0, "fatigue_pct": 0,
              "fatigue_hint": "", "last_sleep_time": "",
              "heartbeats_since_sleep": 0, "tentacles": []}
    recall = {
        "nodes": [{"name": "Apple", "category": "FACT",
                   "description": "a fruit", "neighbor_keywords": ["fruit", "tree"]}],
        "edges": [{"source": "Apple", "predicate": "RELATED_TO", "target": "Banana"}],
    }
    p = PromptBuilder().build(self_model={}, status=status, recall=recall,
                              window=[], stimuli=[])
    assert "Apple" in p
    assert "FACT" in p
    assert "RELATED_TO" in p
    assert "Banana" in p
    assert "fruit" in p
