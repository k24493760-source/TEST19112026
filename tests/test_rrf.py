"""RRF: одинаковый чанк в dense и sparse поднимается выше уникальных."""

from types import SimpleNamespace

from app.services.search import reciprocal_rank_fusion


def _hit(chunk_id: str):
    chunk = SimpleNamespace(id=chunk_id)
    doc = SimpleNamespace(id=f"doc-{chunk_id}")
    return (chunk, doc)


def test_rrf_prefers_overlap():
    dense = [_hit("a"), _hit("b"), _hit("c")]
    sparse = [_hit("c"), _hit("d"), _hit("e")]
    fused = reciprocal_rank_fusion(dense, sparse, k=60, limit=3)
    ids = [chunk.id for chunk, _doc in fused]
    assert ids[0] == "c"
    assert len(ids) == 3


def test_rrf_empty():
    assert reciprocal_rank_fusion([], [], limit=3) == []


def test_rrf_single_list_keeps_order():
    dense = [_hit("a"), _hit("b"), _hit("c"), _hit("d")]
    fused = reciprocal_rank_fusion(dense, [], k=60, limit=3)
    assert [chunk.id for chunk, _ in fused] == ["a", "b", "c"]
