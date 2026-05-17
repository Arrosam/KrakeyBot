"""Part B — IncrementalRecall multi-query expansion contract tests.

Tests the `enricher` keyword-only parameter on IncrementalRecall and the
intra-stimulus weight-dedup rule.

New contracts under test (implementation does NOT exist yet):
  - IncrementalRecall(enricher=<enricher | None>)
  - enricher=None  → identical to today's behavior (regression guard)
  - enricher returns [q1,q2,q3]  → 3 vec-searches per stimulus, but ONE
    processed_stimuli entry and ONE per-stimulus tracking entry per
    original stimulus.
  - enricher returns None or raises → single-query fallback (stimulus.content)
  - Intra-stimulus dedup: multiple phrases of the SAME stimulus hitting
    the SAME node accumulate weight only ONCE, not once per phrase.
  - Cross-stimulus accumulation is PRESERVED: two different stimuli hitting
    the same node still add weight twice.
  - covered/uncovered classification remains keyed on original stimuli.
  - adrenalin ×10 still applies per original stimulus (not per phrase).

Techniques applied:
  - Positive / equivalence-partition  (7 tests)
  - Boundary value analysis           (5 tests)
  - State transitions                 (4 tests)
  - Negative / error-guessing         (5 tests)
"""
from __future__ import annotations

from datetime import datetime

import pytest

from krakey.engines.memory._internal.graph_memory import GraphMemory
from krakey.engines.recall._internal.incremental import IncrementalRecall
from krakey.engines.recall._internal.scoring import ScoringWeights
from krakey.interfaces.engines.recall import RecallResult
from krakey.models.stimulus import Stimulus


# ---------------------------------------------------------------------------
# Test doubles (mirror existing test_incremental_recall.py style)
# ---------------------------------------------------------------------------

class MappingEmbedder:
    """Returns a fixed pre-defined vector for each text key.
    Raises KeyError for unknown keys to surface test-setup bugs."""

    def __init__(self, mapping):
        self._m = mapping

    async def __call__(self, text: str) -> list[float]:
        if text not in self._m:
            raise KeyError(f"no embedding for {text!r}")
        return list(self._m[text])


class _ScriptedEnricher:
    """Duck-types the SemanticAssociationEnricher public surface.
    Returns a fixed phrase list on every call. Records call count."""

    def __init__(self, phrases: list[str] | None):
        self._phrases = phrases
        self.call_count = 0
        self.texts_seen: list[str] = []

    async def enrich(self, text: str, *, now: datetime) -> list[str] | None:
        self.call_count += 1
        self.texts_seen.append(text)
        return list(self._phrases) if self._phrases is not None else None


class _RaisingEnricher:
    """Enricher that always raises — treated as fallback-to-None."""

    async def enrich(self, text: str, *, now: datetime) -> list[str] | None:
        raise RuntimeError("enricher exploded")


def _stim(content: str, *, adrenalin: bool = False) -> Stimulus:
    return Stimulus(
        type="user_message",
        source="test",
        content=content,
        timestamp=datetime(2026, 5, 17),
        adrenalin=adrenalin,
    )


async def _populate(tmp_path, embed_map: dict[str, list[float]]) -> GraphMemory:
    gm = GraphMemory(tmp_path / "gm.sqlite",
                      embedder=MappingEmbedder(embed_map))
    await gm.initialize()
    for name, vec in embed_map.items():
        await gm.insert_node(name=name, category="FACT",
                              description=name, embedding=vec)
    return gm


def _recall(gm: GraphMemory, *, enricher=None, **kw) -> IncrementalRecall:
    defaults = dict(
        per_stimulus_k=5,
        recall_token_budget=100_000,
        weights=ScoringWeights(),
        now=lambda: datetime(2026, 5, 17),
        enricher=enricher,
    )
    defaults.update(kw)
    return IncrementalRecall(gm, embedder=gm._embedder, **defaults)


# ---------------------------------------------------------------------------
# Positive tests
# ---------------------------------------------------------------------------

class TestMultiQueryPositive:

    async def test_enricher_none_baseline_unchanged(self, tmp_path):
        """enricher=None must produce identical behavior to existing tests —
        one query per stimulus, one processed_stimuli entry per stimulus."""
        embed_map = {"apple": [1.0, 0.0], "car": [0.0, 1.0]}
        gm = await _populate(tmp_path, embed_map)

        r = _recall(gm, enricher=None, per_stimulus_k=1)
        s = _stim("apple")
        await r.add_stimuli([s])
        result = await r.finalize()

        assert [n["name"] for n in result.nodes] == ["apple"]
        assert result.covered_stimuli == [s]
        assert result.uncovered_stimuli == []
        assert r.processed_stimuli == [s]
        await gm.close()

    async def test_enricher_phrases_expand_recall(self, tmp_path):
        """3 enricher phrases → 3 vec-searches, potentially surfacing
        nodes not reachable from stimulus.content alone."""
        # apple matches "apple" query; banana matches "banana" query;
        # the stimulus content "query" only matches "query" (absent in GM).
        embed_map = {
            "query": [0.5, 0.5],
            "apple": [1.0, 0.0],
            "banana": [0.0, 1.0],
            "phrase_a": [1.0, 0.0],   # phrase that maps to apple
            "phrase_b": [0.0, 1.0],   # phrase that maps to banana
            "phrase_c": [0.5, 0.5],   # phrase that maps to query region
        }
        gm = await _populate(tmp_path, embed_map)

        enricher = _ScriptedEnricher(["phrase_a", "phrase_b", "phrase_c"])
        r = _recall(gm, enricher=enricher, per_stimulus_k=3)
        s = _stim("query")
        await r.add_stimuli([s])
        result = await r.finalize()

        node_names = {n["name"] for n in result.nodes}
        # phrase_a and phrase_b searches should surface apple and banana
        assert "apple" in node_names
        assert "banana" in node_names
        await gm.close()

    async def test_enricher_phrases_single_processed_stimuli_entry(self, tmp_path):
        """Multiple enricher phrases still produce exactly ONE
        processed_stimuli entry per original stimulus."""
        embed_map = {"q": [1.0, 0.0], "p1": [1.0, 0.0], "p2": [0.0, 1.0]}
        gm = await _populate(tmp_path, embed_map)

        enricher = _ScriptedEnricher(["p1", "p2"])
        r = _recall(gm, enricher=enricher)
        s = _stim("q")
        await r.add_stimuli([s])

        assert len(r.processed_stimuli) == 1
        assert r.processed_stimuli[0] is s
        await gm.close()

    async def test_enricher_none_return_falls_back_to_content(self, tmp_path):
        """When enricher.enrich() returns None, the engine falls back to
        querying with stimulus.content — same result as enricher=None."""
        embed_map = {"hello": [1.0, 0.0]}
        gm = await _populate(tmp_path, embed_map)

        enricher = _ScriptedEnricher(phrases=None)  # returns None
        r = _recall(gm, enricher=enricher, per_stimulus_k=1)
        s = _stim("hello")
        await r.add_stimuli([s])
        result = await r.finalize()

        assert [n["name"] for n in result.nodes] == ["hello"]
        assert result.covered_stimuli == [s]
        await gm.close()

    async def test_cross_stimulus_weight_accumulation_preserved(self, tmp_path):
        """Two different stimuli hitting the same node still produce weight 2.0.
        This is the existing cross-stimulus accumulation — must not regress."""
        gm = GraphMemory(tmp_path / "gm.sqlite",
                          embedder=MappingEmbedder({
                              "stim_one": [1.0, 0.0],
                              "stim_two": [0.95, 0.31],
                          }))
        await gm.initialize()
        await gm.insert_node(name="apple", category="FACT",
                              description="apple", embedding=[1.0, 0.0])

        r = IncrementalRecall(
            gm, embedder=gm._embedder,
            per_stimulus_k=1,
            recall_token_budget=100_000,
            weights=ScoringWeights(),
            now=lambda: datetime(2026, 5, 17),
            # enricher intentionally omitted → None
        )
        await r.add_stimuli([_stim("stim_one"), _stim("stim_two")])
        result = await r.finalize()

        apple = next(n for n in result.nodes if n["name"] == "apple")
        assert apple["score"] == 2.0
        await gm.close()

    async def test_intra_stimulus_dedup_single_weight(self, tmp_path):
        """Critical dedup rule: multiple enricher phrases of the SAME stimulus
        hitting the SAME node must accumulate weight only ONCE (weight=1.0),
        not once per phrase (weight=N)."""
        # Only one node exists: "node_a". All phrases embed near it.
        embed_map = {
            "original": [1.0, 0.0],
            "phrase_x": [1.0, 0.0],
            "phrase_y": [0.99, 0.1],
        }
        gm = GraphMemory(tmp_path / "gm.sqlite",
                          embedder=MappingEmbedder(embed_map))
        await gm.initialize()
        await gm.insert_node(name="node_a", category="FACT",
                              description="node_a", embedding=[1.0, 0.0])

        enricher = _ScriptedEnricher(["phrase_x", "phrase_y"])
        r = _recall(gm, enricher=enricher, per_stimulus_k=1)
        await r.add_stimuli([_stim("original")])
        result = await r.finalize()

        node_a = next(n for n in result.nodes if n["name"] == "node_a")
        # One stimulus, multiple phrases, but all hitting same node →
        # weight must be 1.0 (not 2.0 or 3.0).
        assert node_a["score"] == pytest.approx(1.0)
        await gm.close()

    async def test_intra_dedup_does_not_suppress_cross_stimulus(self, tmp_path):
        """Intra-stimulus dedup affects per-stimulus counting only.
        Two SEPARATE stimuli both hitting node_a still produce weight 2.0."""
        embed_map = {
            "stim_a": [1.0, 0.0],
            "stim_b": [0.98, 0.1],
            "phrase_a1": [1.0, 0.0],
            "phrase_a2": [0.99, 0.05],
        }
        gm = GraphMemory(tmp_path / "gm.sqlite",
                          embedder=MappingEmbedder(embed_map))
        await gm.initialize()
        await gm.insert_node(name="node_a", category="FACT",
                              description="node_a", embedding=[1.0, 0.0])

        # stim_a has enricher phrases that also hit node_a (deduped to 1.0)
        # stim_b has no enricher (or its phrases also hit node_a → +1.0)
        # Total weight from two stimuli: 1.0 + 1.0 = 2.0
        enricher = _ScriptedEnricher(["phrase_a1", "phrase_a2"])
        r = _recall(gm, enricher=enricher, per_stimulus_k=1)
        await r.add_stimuli([_stim("stim_a"), _stim("stim_b")])
        result = await r.finalize()

        node_a = next(n for n in result.nodes if n["name"] == "node_a")
        assert node_a["score"] == pytest.approx(2.0)
        await gm.close()


# ---------------------------------------------------------------------------
# Boundary value analysis
# ---------------------------------------------------------------------------

class TestMultiQueryBVA:

    async def test_enricher_returns_empty_list_falls_back_to_content(self, tmp_path):
        """enricher returning [] (empty, not None) must fall back to
        querying with stimulus.content."""
        embed_map = {"hello": [1.0, 0.0]}
        gm = await _populate(tmp_path, embed_map)

        enricher = _ScriptedEnricher(phrases=[])  # empty list
        r = _recall(gm, enricher=enricher, per_stimulus_k=1)
        s = _stim("hello")
        await r.add_stimuli([s])
        result = await r.finalize()

        assert [n["name"] for n in result.nodes] == ["hello"]
        await gm.close()

    async def test_enricher_returns_one_phrase(self, tmp_path):
        """Exactly one enricher phrase → exactly one expanded query."""
        embed_map = {"q": [0.5, 0.5], "target": [1.0, 0.0], "phrase1": [1.0, 0.0]}
        gm = await _populate(tmp_path, embed_map)

        enricher = _ScriptedEnricher(["phrase1"])
        r = _recall(gm, enricher=enricher, per_stimulus_k=2)
        await r.add_stimuli([_stim("q")])
        result = await r.finalize()

        assert "target" in {n["name"] for n in result.nodes}
        await gm.close()

    async def test_zero_stimuli_empty_result(self, tmp_path):
        """Passing an empty stimuli list with an enricher produces an
        empty RecallResult and zero processed_stimuli."""
        embed_map = {"x": [1.0, 0.0]}
        gm = await _populate(tmp_path, embed_map)

        enricher = _ScriptedEnricher(["x"])
        r = _recall(gm, enricher=enricher)
        await r.add_stimuli([])
        result = await r.finalize()

        assert result.nodes == []
        assert result.edges == []
        assert result.covered_stimuli == []
        assert result.uncovered_stimuli == []
        assert r.processed_stimuli == []
        assert enricher.call_count == 0
        await gm.close()

    async def test_single_stimulus_covered_when_any_phrase_hits(self, tmp_path):
        """A stimulus is 'covered' if ANY of its sub-query phrase hits
        survive selection — not just the base content query."""
        embed_map = {
            "unique_content": [0.0, 0.0, 1.0],  # no matching node
            "phrase_hit": [1.0, 0.0, 0.0],        # matches 'apple'
            "apple": [1.0, 0.0, 0.0],
        }
        gm = GraphMemory(tmp_path / "gm.sqlite",
                          embedder=MappingEmbedder(embed_map))
        await gm.initialize()
        await gm.insert_node(name="apple", category="FACT",
                              description="apple",
                              embedding=[1.0, 0.0, 0.0])

        enricher = _ScriptedEnricher(["phrase_hit"])
        r = _recall(gm, enricher=enricher, per_stimulus_k=1)
        s = _stim("unique_content")
        await r.add_stimuli([s])
        result = await r.finalize()

        assert s in result.covered_stimuli
        assert s not in result.uncovered_stimuli
        await gm.close()

    async def test_adrenalin_weight_applies_per_stimulus_not_per_phrase(self, tmp_path):
        """Adrenalin ×10 multiplier applies once per stimulus. With 3 enricher
        phrases all hitting the same node from an adrenalin stimulus, the node
        weight must be 10.0 (not 30.0)."""
        embed_map = {
            "urgent_msg": [1.0, 0.0],
            "ph1": [1.0, 0.0],
            "ph2": [0.99, 0.1],
            "ph3": [0.98, 0.15],
        }
        gm = GraphMemory(tmp_path / "gm.sqlite",
                          embedder=MappingEmbedder(embed_map))
        await gm.initialize()
        await gm.insert_node(name="node_a", category="FACT",
                              description="node_a", embedding=[1.0, 0.0])

        enricher = _ScriptedEnricher(["ph1", "ph2", "ph3"])
        r = _recall(gm, enricher=enricher, per_stimulus_k=1)
        await r.add_stimuli([_stim("urgent_msg", adrenalin=True)])
        result = await r.finalize()

        node_a = next(n for n in result.nodes if n["name"] == "node_a")
        # One adrenalin stimulus → weight = 10.0 regardless of phrase count
        assert node_a["score"] == pytest.approx(10.0)
        await gm.close()


# ---------------------------------------------------------------------------
# State transition tests
# ---------------------------------------------------------------------------

class TestMultiQueryStateTransitions:

    async def test_add_stimuli_called_twice_accumulates_both(self, tmp_path):
        """add_stimuli() is non-idempotent: calling it twice with one
        stimulus each time accumulates both into processed_stimuli."""
        embed_map = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        gm = await _populate(tmp_path, embed_map)

        r = _recall(gm)
        s1 = _stim("a")
        s2 = _stim("b")
        await r.add_stimuli([s1])
        await r.add_stimuli([s2])

        assert r.processed_stimuli == [s1, s2]
        await gm.close()

    async def test_add_stimuli_twice_with_enricher_two_tracking_entries(self, tmp_path):
        """Two separate add_stimuli calls with enricher=present still result
        in exactly one tracking entry per original stimulus (two total)."""
        embed_map = {"a": [1.0, 0.0], "b": [0.0, 1.0],
                     "pa": [1.0, 0.0], "pb": [0.0, 1.0]}
        gm = await _populate(tmp_path, embed_map)

        enricher = _ScriptedEnricher(["pa", "pb"])
        r = _recall(gm, enricher=enricher)
        s1 = _stim("a")
        s2 = _stim("b")
        await r.add_stimuli([s1])
        await r.add_stimuli([s2])

        assert len(r.processed_stimuli) == 2
        assert r.processed_stimuli[0] is s1
        assert r.processed_stimuli[1] is s2
        await gm.close()

    async def test_finalize_after_two_add_stimuli_calls_returns_both_covered(
        self, tmp_path
    ):
        """finalize() after two add_stimuli calls correctly classifies
        both stimuli if they both hit nodes."""
        embed_map = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        gm = await _populate(tmp_path, embed_map)

        r = _recall(gm, per_stimulus_k=1)
        s1 = _stim("a")
        s2 = _stim("b")
        await r.add_stimuli([s1])
        await r.add_stimuli([s2])
        result = await r.finalize()

        assert s1 in result.covered_stimuli
        assert s2 in result.covered_stimuli
        assert result.uncovered_stimuli == []
        await gm.close()

    async def test_enricher_called_once_per_add_stimuli_invocation(self, tmp_path):
        """The enricher is called once per original stimulus, regardless of
        how many add_stimuli batches are used."""
        embed_map = {"x": [1.0, 0.0], "y": [0.0, 1.0],
                     "px": [1.0, 0.0], "py": [0.0, 1.0]}
        gm = await _populate(tmp_path, embed_map)

        enricher = _ScriptedEnricher(["px"])
        r = _recall(gm, enricher=enricher)
        await r.add_stimuli([_stim("x")])   # 1st call
        await r.add_stimuli([_stim("y")])   # 2nd call
        await r.finalize()

        assert enricher.call_count == 2
        await gm.close()


# ---------------------------------------------------------------------------
# Negative tests
# ---------------------------------------------------------------------------

class TestMultiQueryNegative:

    async def test_enricher_raises_falls_back_to_content_query(self, tmp_path):
        """If enricher.enrich() raises, the engine must not propagate the
        exception — it falls back to querying with stimulus.content."""
        embed_map = {"hello": [1.0, 0.0]}
        gm = await _populate(tmp_path, embed_map)

        enricher = _RaisingEnricher()
        r = _recall(gm, enricher=enricher, per_stimulus_k=1)
        s = _stim("hello")
        await r.add_stimuli([s])
        result = await r.finalize()

        assert [n["name"] for n in result.nodes] == ["hello"]
        assert result.covered_stimuli == [s]
        await gm.close()

    async def test_enricher_none_does_not_add_novel_nodes(self, tmp_path):
        """enricher=None → only stimulus.content drives recall. Nodes only
        reachable via expansion phrases are NOT surfaced."""
        embed_map = {
            "unrelated": [0.0, 0.0, 1.0],   # only accessible via a phrase
            "apple": [1.0, 0.0, 0.0],         # accessible via 'apple' content
        }
        gm = GraphMemory(tmp_path / "gm.sqlite",
                          embedder=MappingEmbedder(embed_map))
        await gm.initialize()
        await gm.insert_node(name="apple", category="FACT",
                              description="apple",
                              embedding=[1.0, 0.0, 0.0])
        await gm.insert_node(name="unrelated", category="FACT",
                              description="unrelated",
                              embedding=[0.0, 0.0, 1.0])

        r = _recall(gm, enricher=None, per_stimulus_k=1)
        await r.add_stimuli([_stim("apple")])
        result = await r.finalize()

        node_names = {n["name"] for n in result.nodes}
        assert "apple" in node_names
        assert "unrelated" not in node_names
        await gm.close()

    async def test_stimulus_uncovered_if_no_phrase_hits(self, tmp_path):
        """Stimulus with enricher where no phrase hits anything → uncovered."""
        embed_map = {
            "far_away": [0.0, 0.0, 1.0],
            "phrase_a": [0.0, 0.0, 1.0],
        }
        gm = GraphMemory(tmp_path / "gm.sqlite",
                          embedder=MappingEmbedder(embed_map))
        await gm.initialize()
        # No matching node exists for these embeddings
        # (don't insert any node)

        enricher = _ScriptedEnricher(["phrase_a"])
        r = _recall(gm, enricher=enricher, per_stimulus_k=1)
        s = _stim("far_away")
        await r.add_stimuli([s])
        result = await r.finalize()

        assert result.nodes == []
        assert s in result.uncovered_stimuli
        await gm.close()

    async def test_enricher_phrases_do_not_create_duplicate_processed_stimuli(
        self, tmp_path
    ):
        """processed_stimuli must contain EXACTLY the original stimulus objects,
        never duplicated because of multiple phrase expansions."""
        embed_map = {"q": [1.0, 0.0], "p1": [1.0, 0.0], "p2": [1.0, 0.0]}
        gm = await _populate(tmp_path, embed_map)

        enricher = _ScriptedEnricher(["p1", "p2"])
        r = _recall(gm, enricher=enricher, per_stimulus_k=2)
        s = _stim("q")
        await r.add_stimuli([s])

        assert r.processed_stimuli.count(s) == 1
        await gm.close()

    async def test_enricher_receives_original_stimulus_content(self, tmp_path):
        """The enricher is called with the original stimulus.content,
        not with a transformed or embedded version of it."""
        embed_map = {"my exact content": [1.0, 0.0]}
        gm = await _populate(tmp_path, embed_map)

        enricher = _ScriptedEnricher(["my exact content"])
        r = _recall(gm, enricher=enricher, per_stimulus_k=1)
        await r.add_stimuli([_stim("my exact content")])
        await r.finalize()

        assert enricher.texts_seen == ["my exact content"]
        await gm.close()
