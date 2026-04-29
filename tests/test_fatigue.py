import pytest

from krakey.runtime.heartbeat.fatigue import calculate_fatigue, fatigue_hint


THRESHOLDS = {
    50: "(may sleep when not busy)",
    75: "(fatigued; should proactively sleep)",
    100: "(very fatigued; find an opportunity to sleep immediately)",
}


def test_zero_fatigue_shows_low_hint():
    hint = fatigue_hint(0, THRESHOLDS)
    assert "energy" in hint or "no need" in hint


def test_below_smallest_threshold_shows_low_hint():
    hint = fatigue_hint(30, THRESHOLDS)
    assert "energy" in hint or "no need" in hint


def test_matches_50_threshold_exactly():
    assert fatigue_hint(50, THRESHOLDS) == THRESHOLDS[50]


def test_between_50_and_75_uses_50_hint():
    assert fatigue_hint(60, THRESHOLDS) == THRESHOLDS[50]


def test_75_threshold():
    assert fatigue_hint(75, THRESHOLDS) == THRESHOLDS[75]


def test_100_threshold():
    assert fatigue_hint(100, THRESHOLDS) == THRESHOLDS[100]


def test_above_max_threshold_stays_at_max():
    assert fatigue_hint(150, THRESHOLDS) == THRESHOLDS[100]


def test_empty_thresholds_returns_empty_string():
    assert fatigue_hint(0, {}) == ""
    assert fatigue_hint(99, {}) == ""


def test_unordered_dict_still_works():
    d = {100: "c", 50: "a", 75: "b"}
    assert fatigue_hint(0, d).startswith("(")  # low hint
    assert fatigue_hint(60, d) == "a"
    assert fatigue_hint(80, d) == "b"
    assert fatigue_hint(200, d) == "c"


# ----- calculate_fatigue -----

def test_calculate_fatigue_zero_nodes():
    pct, hint = calculate_fatigue(node_count=0, soft_limit=200,
                                     thresholds=THRESHOLDS)
    assert pct == 0
    assert "energy" in hint


def test_calculate_fatigue_scales_with_node_count():
    pct1, _ = calculate_fatigue(node_count=50, soft_limit=200, thresholds={})
    pct2, _ = calculate_fatigue(node_count=150, soft_limit=200, thresholds={})
    assert pct1 == 25
    assert pct2 == 75


def test_calculate_fatigue_can_exceed_100():
    pct, _ = calculate_fatigue(node_count=300, soft_limit=200, thresholds={})
    assert pct == 150


def test_calculate_fatigue_soft_limit_zero_safe():
    # Avoid division by zero — treat as 0%
    pct, hint = calculate_fatigue(node_count=10, soft_limit=0, thresholds={})
    assert pct == 0


def test_calculate_fatigue_at_threshold_shows_matching_hint():
    _, hint = calculate_fatigue(node_count=100, soft_limit=200,
                                   thresholds=THRESHOLDS)
    # 50% → 50-threshold hint
    assert hint == THRESHOLDS[50]


def test_calculate_fatigue_at_force_threshold():
    pct, hint = calculate_fatigue(node_count=240, soft_limit=200,
                                     thresholds=THRESHOLDS)
    assert pct == 120
    # highest threshold applies
    assert hint == THRESHOLDS[100]
