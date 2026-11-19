"""Фоновые задания генерации Q&A. Не блокируют загрузку документа вызовом LLM."""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import AuditEvent, KnowledgeChunk, KnowledgeDocument, QaCard, QaGenerationJob, generate_uuid
from app.services.qa_generator import ChunkContext, GenerateRequest, QaGenerator, default_generator
from app.services.qa_llm_client import LlmTransportError
from app.services.qa_llm_config import resolved_llm_config
from app.services.qa_service import QaServiceError, create_candidate

logger = logging.getLogger("most.qa_jobs")


def _idempotency_key(chunk: KnowledgeChunk, cfg: dict) -> str:
    blob = "|".join(
        [
            chunk.id,
            str(chunk.document_version or ""),
            str(chunk.chunking_version or ""),
            str(cfg.get("adapter") or ""),
            str(cfg.get("model") or ""),
            str(cfg.get("revision") or ""),
            str(cfg.get("prompt_version") or ""),
        ]
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def enqueue_for_chunk(db: Session, chunk: KnowledgeChunk, *, cfg: dict | None = None) -> QaGenerationJob | None:
    cfg = cfg or resolved_llm_config()
    if not cfg.get("enqueue_after_chunking"):
        return None
    key = _idempotency_key(chunk, cfg)
    existing = db.query(QaGenerationJob).filter(QaGenerationJob.idempotency_key == key).one_or_none()
    if existing:
        return existing
    job = QaGenerationJob(
        id=generate_uuid(),
        document_id=chunk.document_id,
        chunk_id=chunk.id,
        status="queued",
        idempotency_key=key,
        model=cfg.get("model") or None,
        revision=cfg.get("revision") or None,
        prompt_version=cfg.get("prompt_version"),
        adapter=cfg.get("adapter"),
        document_version=chunk.document_version,
        chunking_version=chunk.chunking_version,
    )
    db.add(job)
    try:
        with db.begin_nested():
            db.flush()
    except IntegrityError:
        return db.query(QaGenerationJob).filter(QaGenerationJob.idempotency_key == key).one_or_none()
    db.add(
        AuditEvent(
            actor="qa-job",
            action="qa_generate_enqueue",
            object=job.id,
            payload={"chunk_id": chunk.id, "document_id": chunk.document_id, "model": job.model},
        )
    )
    return job


def enqueue_for_document(db: Session, document_id: str) -> int:
    cfg = resolved_llm_config()
    if not cfg.get("enqueue_after_chunking"):
        return 0
    chunks = (
        db.query(KnowledgeChunk)
        .filter(KnowledgeChunk.document_id == document_id)
        .all()
    )
    n = 0
    for chunk in chunks:
        if enqueue_for_chunk(db, chunk, cfg=cfg):
            n += 1
    db.commit()
    logger.info("qa_jobs_enqueued document_id=%s n=%s", document_id, n)
    return n


def _chunk_context(db: Session, chunk: KnowledgeChunk) -> ChunkContext:
    doc = db.get(KnowledgeDocument, chunk.document_id)
    return ChunkContext(
        chunk_id=chunk.id,
        evidence_text=chunk.evidence_text or chunk.text or "",
        section_path=chunk.section,
        page=chunk.page,
        document_id=chunk.document_id,
        document_version=chunk.document_version or (doc.version if doc else None),
        document_class=doc.document_class if doc else None,
        source_element_ids=list(chunk.source_element_ids or []),
    )


def persist_candidates(
    db: Session,
    *,
    candidates,
    chunk: KnowledgeChunk,
    cfg: dict,
    actor_id: str = "qa-llm",
) -> list[QaCard]:
    saved: list[QaCard] = []
    for cand in candidates:
        try:
            card, _report = create_candidate(
                db,
                question_text=cand.question_text,
                reference_answer=cand.reference_answer,
                chunk_ids=cand.source_chunk_ids or [chunk.id],
                quotes=cand.quotes,
                question_type=cand.question_type,
                difficulty=cand.difficulty,
                golden_test_kind=cand.golden_test_kind,
                expected_answer_status=cand.expected_answer_status,
                expected_chunk_ids=cand.source_chunk_ids or [chunk.id],
                corpus_version=chunk.corpus_version,
                generation_model=cfg.get("model") or cfg.get("adapter"),
                generation_prompt_version=cfg.get("prompt_version"),
                warnings=cand.warnings,
                ambiguity_flags=cand.ambiguity_flags,
                document_version=chunk.document_version,
                actor_id=actor_id,
            )
        except QaServiceError as exc:
            if exc.code == "exact_duplicate":
                continue
            raise
        if card.golden_status != "none" or card.learning_status != "none":
            raise RuntimeError("auto_approve_forbidden")
        saved.append(card)
    return saved


def run_job(db: Session, job_id: str, generator: QaGenerator | None = None) -> QaGenerationJob:
    job = db.get(QaGenerationJob, job_id)
    if job is None:
        raise KeyError(job_id)
    cfg = resolved_llm_config()
    if job.status == "done":
        return job
    job.status = "running"
    job.attempt = int(job.attempt or 0) + 1
    job.updated_at = datetime.utcnow()
    db.commit()
    chunk = db.get(KnowledgeChunk, job.chunk_id)
    if chunk is None:
        job.status = "failed"
        job.error_code = "chunk_missing"
        db.commit()
        return job
    gen = generator or default_generator()
    try:
        result = gen.generate(GenerateRequest(primary=_chunk_context(db, chunk), config=cfg))
    except LlmTransportError as exc:
        job.status = "failed"
        job.error_code = exc.code
        job.skipped_reason = None
        db.add(
            AuditEvent(
                actor="qa-job",
                action="qa_generate_failed",
                object=job.id,
                payload={"code": exc.code, "attempt": job.attempt},
            )
        )
        db.commit()
        return job
    if not result.candidates:
        job.status = "skipped"
        job.error_code = None
        job.skipped_reason = result.skipped_reason
        job.n_candidates = 0
        job.n_persisted = 0
        db.add(
            AuditEvent(
                actor="qa-job",
                action="qa_generate_skipped",
                object=job.id,
                payload={"reason": result.skipped_reason},
            )
        )
        db.commit()
        return job
    try:
        saved = persist_candidates(db, candidates=result.candidates, chunk=chunk, cfg=cfg)
    except Exception:
        job.status = "failed"
        job.error_code = "persist_failed"
        db.commit()
        return job
    job.status = "done"
    job.error_code = None
    job.skipped_reason = None
    job.n_candidates = len(result.candidates)
    job.n_persisted = len(saved)
    db.add(
        AuditEvent(
            actor="qa-job",
            action="qa_generate_done",
            object=job.id,
            payload={"n_candidates": job.n_candidates, "n_persisted": job.n_persisted, "model": job.model},
        )
    )
    db.commit()
    return job


def process_queued(db: Session, *, limit: int = 1, generator: QaGenerator | None = None) -> list[QaGenerationJob]:
    jobs = (
        db.query(QaGenerationJob)
        .filter(QaGenerationJob.status.in_(["queued", "failed"]))
        .order_by(QaGenerationJob.created_at)
        .limit(limit)
        .all()
    )
    out = []
    for job in jobs:
        out.append(run_job(db, job.id, generator=generator))
    return out
