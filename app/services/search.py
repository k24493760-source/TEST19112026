"""Гибридный поиск по опубликованному CorpusRelease: dense (pgvector) + FTS + RRF."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import func, or_, text
from sqlalchemy.orm import Session

from app.config import EMBEDDING_MODEL, USES_PGVECTOR
from app.embedder import cosine_similarity, embed_text
from app.models import (
    ChunkEmbedding,
    CorpusRelease,
    CorpusReleaseItem,
    KnowledgeChunk,
    KnowledgeDocument,
)
from app.services.corpus import document_is_valid
from app.services.embedding_pipeline import load_embedding_config
from app.services.publication import resolve_search_release

logger = logging.getLogger("most.search")

TOP_K = 3
CANDIDATE_K = 10
RRF_K = 60
MIN_COSINE = 0.75
MAX_COSINE_DISTANCE = 1.0 - MIN_COSINE
REVIEWED_CAP = 12
PROD_DB = "most_db"


@dataclass
class SearchResult:
    rows: list[tuple[KnowledgeChunk, KnowledgeDocument]]
    reviewed_sources: list[dict] = field(default_factory=list)
    corpus_version: str | None = None
    release_id: str | None = None
    no_published_release: bool = False
    error_code: str | None = None
    timings: dict = field(default_factory=dict)
    citation_meta: dict = field(default_factory=dict)


def approved_chunk_query(
    db: Session,
    now: datetime,
    corpus_version: str | None = None,
):
    """Устаревший контур без членства в релизе. Не вызывается из retrieve_chunks."""
    query = (
        db.query(KnowledgeChunk, KnowledgeDocument)
        .join(KnowledgeDocument, KnowledgeChunk.document_id == KnowledgeDocument.id)
        .filter(KnowledgeChunk.approval_status == "approved")
        .filter(KnowledgeDocument.status == "approved")
        .filter(
            or_(
                KnowledgeDocument.valid_to.is_(None),
                KnowledgeDocument.valid_to > now,
            )
        )
        .filter(
            or_(
                KnowledgeDocument.valid_from.is_(None),
                KnowledgeDocument.valid_from <= now,
            )
        )
    )
    version = (corpus_version or "").strip()
    if version:
        query = query.filter(KnowledgeDocument.version == version)
    return query


def _search_terms(question: str) -> list[str]:
    import re

    words = re.findall(r"[0-9A-Za-zА-Яа-яЁё\-]+", question.lower())
    return [w for w in words if len(w) >= 4]


def _keep_valid(
    rows: list[tuple[KnowledgeChunk, KnowledgeDocument]],
) -> list[tuple[KnowledgeChunk, KnowledgeDocument]]:
    return [(chunk, doc) for chunk, doc in rows if document_is_valid(doc)]


def reciprocal_rank_fusion(
    *ranked_lists: list[tuple[KnowledgeChunk, KnowledgeDocument]],
    k: int = RRF_K,
    limit: int = TOP_K,
) -> list[tuple[KnowledgeChunk, KnowledgeDocument]]:
    """RRF: score = Σ 1 / (k + rank), rank начинается с 1. Ключ — chunk_id."""
    scores: dict[str, float] = {}
    payloads: dict[str, tuple[KnowledgeChunk, KnowledgeDocument]] = {}
    for ranked in ranked_lists:
        for rank, (chunk, doc) in enumerate(ranked, start=1):
            key = chunk.id
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            payloads[key] = (chunk, doc)
    ordered = sorted(scores, key=lambda item: scores[item], reverse=True)
    return [payloads[item] for item in ordered[:limit]]


def _keyword_search(
    db: Session,
    question: str,
    now: datetime,
    corpus_version: str | None = None,
) -> list[tuple[KnowledgeChunk, KnowledgeDocument]]:
    base = approved_chunk_query(db, now, corpus_version=corpus_version)
    text_lower = func.lower(KnowledgeChunk.text)
    rows = base.filter(text_lower.like(f"%{question.lower()}%")).all()
    valid = _keep_valid(rows)[:TOP_K]
    if valid:
        return valid
    terms = _search_terms(question)
    if not terms:
        return []
    conditions = [text_lower.like(f"%{term}%") for term in terms]
    return _keep_valid(base.filter(or_(*conditions)).all())[:TOP_K]


def _sqlite_cosine_search(
    db: Session,
    query_vector: list[float],
    now: datetime,
    corpus_version: str | None = None,
) -> list[tuple[KnowledgeChunk, KnowledgeDocument]]:
    scored = []
    for chunk, doc in approved_chunk_query(db, now, corpus_version=corpus_version).all():
        if not document_is_valid(doc):
            continue
        score = cosine_similarity(chunk.embedding, query_vector)
        if score >= MIN_COSINE:
            scored.append((score, chunk, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(chunk, doc) for _, chunk, doc in scored[:CANDIDATE_K]]


def _source_entry(doc: KnowledgeDocument) -> dict:
    return {"id": doc.id, "title": doc.title or doc.id}


def _unique_sources(
    rows: list[tuple[KnowledgeChunk, KnowledgeDocument]],
) -> list[dict]:
    seen: dict[str, dict] = {}
    for _chunk, doc in rows:
        if doc.id not in seen:
            seen[doc.id] = _source_entry(doc)
    return list(seen.values())


def _dense_search(
    db: Session,
    query_vector: list[float],
    now: datetime,
    corpus_version: str | None = None,
) -> tuple[
    list[tuple[KnowledgeChunk, KnowledgeDocument]],
    list[tuple[KnowledgeChunk, KnowledgeDocument]],
]:
    rows = (
        approved_chunk_query(db, now, corpus_version=corpus_version)
        .filter(KnowledgeChunk.embedding.isnot(None))
        .order_by(KnowledgeChunk.embedding.cosine_distance(query_vector))
        .limit(CANDIDATE_K)
        .all()
    )
    inspected = _keep_valid(rows)
    relevant = [
        (chunk, doc)
        for chunk, doc in inspected
        if cosine_similarity(chunk.embedding, query_vector) >= MIN_COSINE
    ]
    return inspected, relevant


def _sparse_search(
    db: Session,
    question: str,
    now: datetime,
    corpus_version: str | None = None,
) -> list[tuple[KnowledgeChunk, KnowledgeDocument]]:
    tsquery = func.plainto_tsquery("russian", question)
    rows = (
        approved_chunk_query(db, now, corpus_version=corpus_version)
        .filter(KnowledgeChunk.search_vector.op("@@")(tsquery))
        .order_by(func.ts_rank_cd(KnowledgeChunk.search_vector, tsquery).desc())
        .limit(CANDIDATE_K)
        .all()
    )
    return _keep_valid(rows)


def _fallback_reviewed(
    db: Session,
    now: datetime,
    corpus_version: str | None,
) -> list[dict]:
    query = db.query(KnowledgeDocument).filter(KnowledgeDocument.status == "approved")
    version = (corpus_version or "").strip()
    if version:
        query = query.filter(KnowledgeDocument.version == version)
    return [_source_entry(doc) for doc in query.limit(REVIEWED_CAP).all()]


def _empty_result(**kwargs) -> SearchResult:
    return SearchResult(rows=[], reviewed_sources=[], **kwargs)


def query_model_compatible(release: CorpusRelease) -> bool:
    cfg = load_embedding_config()
    live = str(cfg.get("embedding_model") or EMBEDDING_MODEL)
    if (release.embedding_model or "") not in {live, EMBEDDING_MODEL, f"sentence-transformers/{live}"}:
        return False
    cfg_ver = str(cfg.get("embedding_config_version") or "")
    if release.embedding_config_version and cfg_ver and release.embedding_config_version != cfg_ver:
        return False
    return True


def citation_fields(
    chunk: KnowledgeChunk,
    doc: KnowledgeDocument,
    release: CorpusRelease,
) -> dict:
    fmt = (doc.file_format or "").lower().lstrip(".")
    page = chunk.page
    if fmt == "docx" and page is None:
        page_out = None
    else:
        page_out = page
    return {
        "document_id": doc.id,
        "document_version": doc.version,
        "chunk_id": chunk.id,
        "corpus_version": release.version,
        "section": chunk.section,
        "page": page_out,
        "source_element_ids": chunk.source_element_ids,
        "evidence_text": chunk.evidence_text,
        "status": doc.status,
        "document_class": doc.document_class,
    }


def _collapse_chunk_hits(
    rows: list[tuple[KnowledgeChunk, KnowledgeDocument]],
) -> list[tuple[KnowledgeChunk, KnowledgeDocument]]:
    seen: set[str] = set()
    out: list[tuple[KnowledgeChunk, KnowledgeDocument]] = []
    for chunk, doc in rows:
        if chunk.id in seen:
            continue
        seen.add(chunk.id)
        out.append((chunk, doc))
    return out


def _published_base(db: Session, release: CorpusRelease, now: datetime):
    item_chunk_ids = db.query(CorpusReleaseItem.chunk_id).filter(
        CorpusReleaseItem.release_id == release.id
    )
    return (
        db.query(KnowledgeChunk, KnowledgeDocument)
        .join(KnowledgeDocument, KnowledgeChunk.document_id == KnowledgeDocument.id)
        .filter(KnowledgeChunk.id.in_(item_chunk_ids))
        .filter(KnowledgeChunk.approval_status == "approved")
        .filter(KnowledgeDocument.status == "approved")
        .filter(KnowledgeDocument.status.notin_(("draft", "rejected", "ready_for_review")))
        .filter(
            or_(
                KnowledgeDocument.valid_to.is_(None),
                KnowledgeDocument.valid_to > now,
            )
        )
        .filter(
            or_(
                KnowledgeDocument.valid_from.is_(None),
                KnowledgeDocument.valid_from <= now,
            )
        )
    )


def _published_dense(
    db: Session,
    release: CorpusRelease,
    query_vector: list[float],
    now: datetime,
) -> tuple[list[tuple[KnowledgeChunk, KnowledgeDocument]], list[tuple[KnowledgeChunk, KnowledgeDocument]]]:
    cfg = load_embedding_config()
    cfg_ver = str(cfg.get("embedding_config_version") or release.embedding_config_version)
    q = (
        db.query(ChunkEmbedding, KnowledgeChunk, KnowledgeDocument)
        .join(KnowledgeChunk, ChunkEmbedding.chunk_id == KnowledgeChunk.id)
        .join(KnowledgeDocument, KnowledgeChunk.document_id == KnowledgeDocument.id)
        .join(
            CorpusReleaseItem,
            (CorpusReleaseItem.embedding_id == ChunkEmbedding.embedding_id)
            & (CorpusReleaseItem.chunk_id == KnowledgeChunk.id)
            & (CorpusReleaseItem.release_id == release.id),
        )
        .filter(ChunkEmbedding.embedding_status == "approved")
        .filter(ChunkEmbedding.embedding_status.notin_(("draft", "ready", "superseded", "error", "pending")))
        .filter(ChunkEmbedding.embedding_model == release.embedding_model)
        .filter(ChunkEmbedding.model_revision == release.model_revision)
        .filter(ChunkEmbedding.embedding_config_version == (release.embedding_config_version or cfg_ver))
        .filter(KnowledgeChunk.approval_status == "approved")
        .filter(KnowledgeDocument.status == "approved")
        .filter(ChunkEmbedding.embedding.isnot(None))
        .order_by(ChunkEmbedding.embedding.cosine_distance(query_vector))
        .limit(CANDIDATE_K * 3)
    )
    triples = q.all()
    inspected_raw: list[tuple[KnowledgeChunk, KnowledgeDocument]] = []
    relevant_raw: list[tuple[KnowledgeChunk, KnowledgeDocument]] = []
    for emb, chunk, doc in triples:
        if not document_is_valid(doc):
            continue
        pair = (chunk, doc)
        inspected_raw.append(pair)
        score = cosine_similarity(emb.embedding, query_vector)
        if score >= 0.2:
            relevant_raw.append(pair)
    return _collapse_chunk_hits(inspected_raw), _collapse_chunk_hits(relevant_raw)


def _published_sparse(
    db: Session,
    release: CorpusRelease,
    question: str,
    now: datetime,
) -> list[tuple[KnowledgeChunk, KnowledgeDocument]]:
    tsquery = func.plainto_tsquery("russian", question)
    rows = (
        _published_base(db, release, now)
        .filter(KnowledgeChunk.search_vector.op("@@")(tsquery))
        .order_by(func.ts_rank_cd(KnowledgeChunk.search_vector, tsquery).desc())
        .limit(CANDIDATE_K)
        .all()
    )
    return _collapse_chunk_hits(_keep_valid(rows))


def retrieve_chunks(
    db: Session,
    question: str,
    now: datetime | None = None,
    corpus_version: str | None = None,
) -> SearchResult:
    """Поиск только по опубликованному составу. Без релиза — пустой результат, без legacy-fallback."""
    now = now or datetime.utcnow()
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        release = resolve_search_release(db, corpus_version)
        if release is None:
            return _empty_result(
                no_published_release=True,
                error_code="no_published_release",
                corpus_version=(corpus_version or None),
                timings={"total_s": time.perf_counter() - t0},
            )
        if not query_model_compatible(release):
            return _empty_result(
                error_code="model_mismatch",
                corpus_version=release.version,
                release_id=release.id,
                timings={"total_s": time.perf_counter() - t0},
            )

        t_emb = time.perf_counter()
        query_vector = embed_text(question)
        timings["embed_query_s"] = time.perf_counter() - t_emb

        t_search = time.perf_counter()
        if USES_PGVECTOR:
            inspected_dense, dense = _published_dense(db, release, query_vector, now)
            sparse = _published_sparse(db, release, question, now)
            reviewed = _unique_sources(inspected_dense + sparse)
            hits = reciprocal_rank_fusion(dense, sparse, k=RRF_K, limit=TOP_K)
        else:
            dense = []
            sparse = _published_sparse(db, release, question, now)
            hits = sparse[:TOP_K]
            reviewed = _unique_sources(hits)
        timings["search_s"] = time.perf_counter() - t_search
        timings["total_s"] = time.perf_counter() - t0
        meta = {chunk.id: citation_fields(chunk, doc, release) for chunk, doc in hits}
        return SearchResult(
            rows=hits,
            reviewed_sources=reviewed[:REVIEWED_CAP],
            corpus_version=release.version,
            release_id=release.id,
            timings=timings,
            citation_meta=meta,
        )
    except Exception as exc:
        logger.error("retrieve_chunks failed without draft fallback: %s", exc)
        db.rollback()
        return _empty_result(
            error_code="search_error",
            corpus_version=(corpus_version or None),
            timings={"total_s": time.perf_counter() - t0},
        )


def retrieve_chunks_legacy_unscoped(
    db: Session,
    question: str,
    now: datetime | None = None,
    corpus_version: str | None = None,
) -> SearchResult:
    """Сохранён для регрессии. Не используется /ask."""
    now = now or datetime.utcnow()
    query_vector = embed_text(question)
    if USES_PGVECTOR:
        inspected_dense, dense = _dense_search(
            db, query_vector, now, corpus_version=corpus_version
        )
        sparse = _sparse_search(db, question, now, corpus_version=corpus_version)
        reviewed = _unique_sources(inspected_dense + sparse)
        hits = reciprocal_rank_fusion(dense, sparse, k=RRF_K, limit=TOP_K)
        if not hits and not reviewed:
            reviewed = _fallback_reviewed(db, now, corpus_version)
        return SearchResult(rows=hits, reviewed_sources=reviewed[:REVIEWED_CAP])
    dense = _sqlite_cosine_search(db, query_vector, now, corpus_version=corpus_version)
    hits = dense[:TOP_K] if dense else _keyword_search(
        db, question, now, corpus_version=corpus_version
    )
    reviewed = _unique_sources(dense or hits)
    if not hits and not reviewed:
        reviewed = _fallback_reviewed(db, now, corpus_version)
    return SearchResult(rows=hits, reviewed_sources=reviewed[:REVIEWED_CAP])
