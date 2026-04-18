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
