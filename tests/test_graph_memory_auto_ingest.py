"""Phase 1.2c: auto_ingest (embedding + dedup) and cosine similarity."""
import pytest

from src.memory.graph_memory import GraphMemory, cosine_similarity


class MappingEmbedder:
    """Returns a canned vector per text; raises if unseen."""

    def __init__(self, mapping):
        self._m = mapping
        self.calls = []

    async def __call__(self, text: str) -> list[float]:
        self.calls.append(text)
        if text not in self._m:
            raise KeyError(f"no embedding for: {text!r}")
        return list(self._m[text])


def test_cosine_similarity_identical_vectors():
    assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)


def test_cosine_similarity_opposite_direction():
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_safe():
    # zero vector has no direction; convention: return 0.0 (not NaN / crash)
    assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0


async def test_auto_ingest_creates_new_fact_node(tmp_path):
    embed = MappingEmbedder({"apple is red": [1.0, 0.0, 0.0]})
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=embed)
    await gm.initialize()

    result = await gm.auto_ingest("apple is red")

    assert result["created"] is True
    assert result["node_id"] is not None
    assert await gm.count_nodes() == 1

    node = await gm.get_node(result["node_id"])
    assert node["category"] == "FACT"
    assert node["source_type"] == "auto"
    assert node["description"] == "apple is red"
    assert node["embedding"] == [1.0, 0.0, 0.0]
    await gm.close()


async def test_auto_ingest_similar_content_bumps_importance(tmp_path):
    embed = MappingEmbedder({
        "apple is red": [1.0, 0.0, 0.0],
        "the apple is red":      [0.99, 0.14, 0.0],  # cosine ~0.99 > 0.92
    })
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=embed)
    await gm.initialize()

    first = await gm.auto_ingest("apple is red")
    before = await gm.get_node(first["node_id"])

    second = await gm.auto_ingest("the apple is red")
    assert second["created"] is False
    assert second["node_id"] == first["node_id"]
    assert await gm.count_nodes() == 1

    after = await gm.get_node(first["node_id"])
    assert after["importance"] == before["importance"] + 0.5
    await gm.close()


async def test_auto_ingest_dissimilar_content_creates_second_node(tmp_path):
    embed = MappingEmbedder({
        "apple is red": [1.0, 0.0, 0.0],
        "the sky is blue": [0.0, 1.0, 0.0],  # orthogonal → sim 0
    })
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=embed)
    await gm.initialize()

    r1 = await gm.auto_ingest("apple is red")
    r2 = await gm.auto_ingest("the sky is blue")

    assert r1["created"] is True
    assert r2["created"] is True
    assert r1["node_id"] != r2["node_id"]
    assert await gm.count_nodes() == 2
    await gm.close()


async def test_auto_ingest_node_name_is_short(tmp_path):
    long_text = "a" * 200
    embed = MappingEmbedder({long_text: [1.0, 0.0, 0.0]})
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=embed)
    await gm.initialize()

    r = await gm.auto_ingest(long_text)
    node = await gm.get_node(r["node_id"])
    assert len(node["name"]) <= 80
    # description preserves full content
    assert node["description"] == long_text
    await gm.close()


async def test_auto_ingest_respects_similarity_threshold(tmp_path):
    # two vectors with cosine ~0.90 < 0.92 → should NOT dedupe
    embed = MappingEmbedder({
        "fooooo": [1.0, 0.0],
        "barrrr": [0.9, 0.436],  # cos ≈ 0.9
    })
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=embed,
                      auto_ingest_threshold=0.92)
    await gm.initialize()
    await gm.auto_ingest("fooooo")
    await gm.auto_ingest("barrrr")
    assert await gm.count_nodes() == 2
    await gm.close()


async def test_auto_ingest_skips_too_short_content(tmp_path):
    embed = MappingEmbedder({})  # never called
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=embed)
    await gm.initialize()
    result = await gm.auto_ingest("hi")  # 2 chars
    assert result["skipped"] is True
    assert result["created"] is False
    assert await gm.count_nodes() == 0
    assert embed.calls == []  # didn't even embed
    await gm.close()


async def test_auto_ingest_skips_pure_symbols(tmp_path):
    """Regression: '✓' / emoji / pure punctuation must not enter GM."""
    embed = MappingEmbedder({})
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=embed)
    await gm.initialize()
    for content in ("✓", "✅", "...", "!!!", "    ", "❤️🎉"):
        result = await gm.auto_ingest(content)
        assert result["skipped"] is True, f"should skip {content!r}"
    assert await gm.count_nodes() == 0
    await gm.close()


async def test_auto_ingest_keeps_short_alphanumeric(tmp_path):
    """A short but informative string like 'NZX50' should still pass."""
    embed = MappingEmbedder({"NZX50": [1.0, 0.0]})
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=embed)
    await gm.initialize()
    result = await gm.auto_ingest("NZX50")
    assert result["created"] is True
    assert await gm.count_nodes() == 1
    await gm.close()


async def test_auto_ingest_accepts_explicit_threshold(tmp_path):
    # Same vectors, but lower threshold → should dedupe
    embed = MappingEmbedder({
        "fooooo": [1.0, 0.0],
        "barrrr": [0.9, 0.436],
    })
    gm = GraphMemory(tmp_path / "gm.sqlite", embedder=embed,
                      auto_ingest_threshold=0.85)
    await gm.initialize()
    await gm.auto_ingest("fooooo")
    await gm.auto_ingest("barrrr")
    assert await gm.count_nodes() == 1
    await gm.close()
