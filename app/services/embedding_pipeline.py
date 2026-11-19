"""Векторизация draft-чанков сегментами MiniLM. Не публикует RAG и не пишет в most_db."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np
import yaml
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import BASE_DIR, EMBEDDING_DIM, EMBEDDING_MODEL
from app.embedder import _load_model
from app.models import (
    ChunkEmbedding,
    CorpusRelease,
    ExpertReview,
    KnowledgeChunk,
    KnowledgeDocument,
    UploadHistory,
)

logger = logging.getLogger("most.embedding_pipeline")
CONFIG_PATH = BASE_DIR / "contracts" / "embedding_config.yaml"
PROD_DB = "most_db"


class EmbeddingPipelineError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class SegmentSpec:
    chunk_id: str
    segment_index: int
    segment_text: str
    char_start: int
    char_end: int
    token_count: int
    content_hash: str
    embedding_id: str


@dataclass
class VectorizeResult:
    document_id: str | None
    created: int = 0
    skipped: int = 0
    superseded: int = 0
    failed: int = 0
    segments: list[SegmentSpec] = field(default_factory=list)
    rss_mb: float | None = None
    batch_size: int = 1
    model_revision: str = ""
    embedding_config_version: str = ""


@lru_cache(maxsize=1)
def load_embedding_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def refuse_prod_database(db: Session) -> str:
    name = db.execute(text("SELECT current_database()")).scalar()
    cfg = load_embedding_config()
    if name == cfg.get("prod_database_name", PROD_DB) or name == PROD_DB:
        raise EmbeddingPipelineError(
            "prod_forbidden",
            f"Векторизация 2.4Б запрещена на рабочей БД {name}.",
        )
    return str(name)


def _tokenizer():
    model = _load_model()
    tok = model.tokenizer
    tok.model_max_length = 100_000
    return tok


def special_token_count() -> int:
    tok = _tokenizer()
    empty = tok.encode("", add_special_tokens=True, truncation=False)
    return len(empty)


def max_content_tokens() -> int:
    cfg = load_embedding_config()
    max_seq = int(cfg.get("max_seq_length") or 256)
    budget = max_seq - special_token_count()
    if budget < 8:
        raise EmbeddingPipelineError("token_budget", "Слишком мало токенов после спецтокенов.")
    return budget


def count_tokens_with_special(text: str) -> int:
    tok = _tokenizer()
    return len(tok.encode(text or "", add_special_tokens=True, truncation=False))


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def stable_embedding_id(
    chunk_id: str,
    segment_index: int,
    digest: str,
    model: str,
    revision: str,
    config_version: str,
) -> str:
    raw = f"{chunk_id}|{segment_index}|{digest}|{model}|{revision}|{config_version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def resolve_model_revision(override: str | None = None) -> str:
    if override:
        return override
    model = _load_model()
    auto = model[0].auto_model
    rev = getattr(auto.config, "_commit_hash", None)
    if rev:
        return str(rev)
    path = getattr(auto.config, "_name_or_path", None) or EMBEDDING_MODEL
    return str(path)


def choose_batch_size(configured: int | None = None) -> int:
    cfg = load_embedding_config()
    default = int(configured or cfg.get("batch_size_default") or 4)
    minimum = int(cfg.get("min_batch_size") or 1)
    try:
        import psutil

        avail = psutil.virtual_memory().available / 1024**3
    except Exception:
        avail = 0.2
    if avail < 0.5:
        return max(minimum, min(2, default))
    if avail < 1.5:
        return max(minimum, min(4, default))
    return max(minimum, default)


def split_retrieval_text(text: str) -> list[tuple[int, int, str, int]]:
    """Смежные срезы исходного текста. Не обрезает хвост."""
    if not (text or "").strip():
        raise EmbeddingPipelineError("empty_retrieval", "Пустой retrieval_text.")
    tok = _tokenizer()
    budget = max_content_tokens()
    max_seq = int(load_embedding_config().get("max_seq_length") or 256)
    encoded = tok(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if not ids:
        raise EmbeddingPipelineError("empty_tokens", "Токенизатор вернул пустую последовательность.")

    windows: list[tuple[int, int, str, int]] = []
    i = 0
    n = len(ids)
    while i < n:
        j = min(i + budget, n)
        placed = False
        while j > i:
            start = 0 if i == 0 else int(offsets[i][0])
            end = len(text) if j == n else int(offsets[j][0])
            if end < start:
                end = int(offsets[j - 1][1])
            piece = text[start:end]
            total = count_tokens_with_special(piece)
            if total <= max_seq and piece:
                windows.append((start, end, piece, total))
                placed = True
                break
            j -= 1
        if not placed:
            raise EmbeddingPipelineError(
                "unsplittable",
                "Не удалось уложить фрагмент в лимит токенов без обрезки.",
            )
        i = j

    if not windows or windows[0][0] != 0 or windows[-1][1] != len(text):
        raise EmbeddingPipelineError("coverage", "Сегменты не покрывают retrieval_text целиком.")
    rebuilt = "".join(w[2] for w in windows)
    if rebuilt != text:
        raise EmbeddingPipelineError("coverage", "Склейка сегментов не совпадает с исходным текстом.")
    for prev, cur in zip(windows, windows[1:]):
        if cur[0] != prev[1]:
            raise EmbeddingPipelineError("coverage", "Разрыв между сегментами.")
    return windows


def _validate_chunk(chunk: KnowledgeChunk) -> str:
    if (chunk.approval_status or "draft") != "draft":
        raise EmbeddingPipelineError("not_draft", f"Чанк {chunk.id} не draft.")
    evidence = (chunk.evidence_text or "").strip()
    retrieval = chunk.retrieval_text or chunk.text or ""
    if not evidence:
        raise EmbeddingPipelineError("no_evidence", f"Нет evidence_text у чанка {chunk.id}.")
    if not (retrieval or "").strip():
        raise EmbeddingPipelineError("no_retrieval", f"Нет retrieval_text у чанка {chunk.id}.")
    if not chunk.source_element_ids:
        raise EmbeddingPipelineError("no_source", f"Нет ссылок на источник у чанка {chunk.id}.")
    return retrieval


def plan_segments(
    chunk: KnowledgeChunk,
    model_name: str,
    revision: str,
    config_version: str,
) -> list[SegmentSpec]:
    retrieval = _validate_chunk(chunk)
    specs: list[SegmentSpec] = []
    for index, (start, end, piece, tokens) in enumerate(split_retrieval_text(retrieval)):
        digest = content_hash(piece)
        specs.append(
            SegmentSpec(
                chunk_id=chunk.id,
                segment_index=index,
                segment_text=piece,
                char_start=start,
                char_end=end,
                token_count=tokens,
                content_hash=digest,
                embedding_id=stable_embedding_id(
                    chunk.id, index, digest, model_name, revision, config_version
                ),
            )
        )
    return specs


def embed_segments_strict(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    max_seq = int(load_embedding_config().get("max_seq_length") or 256)
    for item in texts:
        n = count_tokens_with_special(item)
        if n > max_seq:
            raise EmbeddingPipelineError(
                "would_truncate",
                f"Сегмент {n} токенов > {max_seq}: скрытая обрезка запрещена.",
            )
    model = _load_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=len(texts),
        show_progress_bar=False,
        truncation=False,
    )
    result: list[list[float]] = []
    for vector in vectors:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.shape[-1] != EMBEDDING_DIM:
            raise EmbeddingPipelineError(
                "dim_mismatch",
                f"Размерность {arr.shape[-1]} != {EMBEDDING_DIM}.",
            )
        if not np.isfinite(arr).all():
            raise EmbeddingPipelineError("non_finite", "Вектор содержит NaN или Inf.")
        result.append(arr.astype(float).tolist())
    return result


def _identity_exists(db: Session, spec: SegmentSpec, model_name: str, revision: str, config_version: str) -> bool:
    row = (
        db.query(ChunkEmbedding)
        .filter(
            ChunkEmbedding.chunk_id == spec.chunk_id,
            ChunkEmbedding.segment_index == spec.segment_index,
            ChunkEmbedding.embedding_model == model_name,
            ChunkEmbedding.model_revision == revision,
            ChunkEmbedding.embedding_config_version == config_version,
            ChunkEmbedding.content_hash == spec.content_hash,
            ChunkEmbedding.embedding_status.in_(("ready", "approved")),
        )
        .first()
    )
    return row is not None


def _supersede_stale(
    db: Session,
    chunk_id: str,
    keep_ids: set[str],
    model_name: str,
    revision: str,
    config_version: str,
) -> int:
    rows = (
        db.query(ChunkEmbedding)
        .filter(
            ChunkEmbedding.chunk_id == chunk_id,
            ChunkEmbedding.embedding_model == model_name,
            ChunkEmbedding.model_revision == revision,
            ChunkEmbedding.embedding_config_version == config_version,
            ChunkEmbedding.embedding_status == "ready",
        )
        .all()
    )
    n = 0
    for row in rows:
        if row.embedding_id not in keep_ids:
            row.embedding_status = "superseded"
            n += 1
    return n


def vectorize_draft_chunks(
    db: Session,
    document_id: str | None = None,
    model_revision: str | None = None,
    batch_size: int | None = None,
) -> VectorizeResult:
    """Создаёт векторы только для draft-чанков. Не меняет статус документа и релиза."""
    refuse_prod_database(db)
    cfg = load_embedding_config()
    model_name = str(cfg.get("embedding_model") or EMBEDDING_MODEL)
    config_version = str(cfg["embedding_config_version"])
    revision = resolve_model_revision(model_revision)
    size = choose_batch_size(batch_size)

    query = db.query(KnowledgeChunk).filter(KnowledgeChunk.approval_status == "draft")
    if document_id:
        query = query.filter(KnowledgeChunk.document_id == document_id)
    chunks = query.order_by(KnowledgeChunk.id).all()
    if not chunks:
        raise EmbeddingPipelineError("no_chunks", "Нет draft-чанков для векторизации.")

    doc_status: dict[str, str] = {}
    release_status: dict[str, str] = {}
    planned: list[tuple[KnowledgeChunk, SegmentSpec]] = []
    skipped = 0
    superseded = 0

    for chunk in chunks:
        doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == chunk.document_id).one()
        doc_status[doc.id] = doc.status
        specs = plan_segments(chunk, model_name, revision, config_version)
        keep = {s.embedding_id for s in specs}
        superseded += _supersede_stale(db, chunk.id, keep, model_name, revision, config_version)
        for spec in specs:
            if _identity_exists(db, spec, model_name, revision, config_version):
                skipped += 1
                continue
            planned.append((chunk, spec))

    review_before = db.query(ExpertReview).count()
    created = 0
    failed = 0
    created_specs: list[SegmentSpec] = []

    for offset in range(0, len(planned), size):
        batch = planned[offset : offset + size]
        texts = [spec.segment_text for _, spec in batch]
        try:
            vectors = embed_segments_strict(texts)
            for (chunk, spec), vector in zip(batch, vectors):
                existing = db.query(ChunkEmbedding).filter(
                    ChunkEmbedding.embedding_id == spec.embedding_id
                ).first()
                if existing:
                    skipped += 1
                    continue
                db.add(
                    ChunkEmbedding(
                        embedding_id=spec.embedding_id,
                        chunk_id=spec.chunk_id,
                        segment_index=spec.segment_index,
                        segment_text=spec.segment_text,
                        char_start=spec.char_start,
                        char_end=spec.char_end,
                        embedding=vector,
                        embedding_model=model_name,
                        model_revision=revision,
                        embedding_config_version=config_version,
                        content_hash=spec.content_hash,
                        token_count=spec.token_count,
                        embedding_status="ready",
                    )
                )
                created += 1
                created_specs.append(spec)
            db.add(
                UploadHistory(
                    document_id=document_id or batch[0][0].document_id,
                    action="embed_segments",
                    actor_id="embedding_pipeline",
                    details={
                        "created": len(batch),
                        "model": model_name,
                        "revision": revision,
                        "config": config_version,
                        "status": "draft_only",
                    },
                )
            )
            db.commit()
        except Exception as exc:
            db.rollback()
            failed += len(batch)
            logger.error("batch failed offset=%s: %s", offset, exc)
            raise EmbeddingPipelineError("batch_failed", f"Пакет с offset={offset} не записан: {exc}") from exc

    for chunk in chunks:
        db.refresh(chunk)
        if chunk.approval_status != "draft":
            raise EmbeddingPipelineError("status_mutated", "Статус чанка изменился.")
        if chunk.embedding is not None:
            raise EmbeddingPipelineError("chunk_column_written", "Запрещено писать knowledge_chunks.embedding.")
        doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == chunk.document_id).one()
        if doc.status != doc_status[doc.id]:
            raise EmbeddingPipelineError("doc_status", "Статус документа изменён пайплайном.")
        if chunk.corpus_version:
            rel = (
                db.query(CorpusRelease)
                .filter(CorpusRelease.version == chunk.corpus_version)
                .one_or_none()
            )
            if rel is not None:
                release_status[rel.version] = rel.status
                if rel.status == "published":
                    raise EmbeddingPipelineError("published", "Пайплайн не должен публиковать корпус.")

    if db.query(ExpertReview).count() != review_before:
        raise EmbeddingPipelineError("expert_created", "Пайплайн не создаёт expert_review.")
    rss = None
    try:
        import psutil

        rss = psutil.Process().memory_info().rss / 1024**2
    except Exception:
        pass

    logger.info(
        "vectorized created=%s skipped=%s superseded=%s batch=%s",
        created,
        skipped,
        superseded,
        size,
    )
    return VectorizeResult(
        document_id=document_id,
        created=created,
        skipped=skipped,
        superseded=superseded,
        failed=failed,
        segments=created_specs,
        rss_mb=rss,
        batch_size=size,
        model_revision=revision,
        embedding_config_version=config_version,
    )


def retrieval_not_wired() -> dict:
    """Контракт будущего JOIN. retrieve_chunks этот путь не вызывает."""
    return {
        "table": "chunk_embeddings",
        "required_filters": [
            "chunk_embeddings.embedding_status=approved",
            "chunk_embeddings.embedding_model=<live>",
            "knowledge_chunks.approval_status=approved",
            "knowledge_documents.status=approved",
            "corpus_releases.status=published",
        ],
        "wired_to_retrieve_chunks": True,
    }
