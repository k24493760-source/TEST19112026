"""Создание и версионирование карточек Q&A. Без автопубликации и без LLM."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime

from sqlalchemy.orm import Session, selectinload

from app.models import (
    AuditEvent,
    KnowledgeChunk,
    KnowledgeDocument,
    LearningModule,
    QaCard,
    QaEvalSnapshot,
    QaEvidenceQuote,
    QaSourceChunk,
    Question,
    Rubric,
    generate_uuid,
)
from app.services.qa_duplicates import find_exact_duplicates
from app.services.qa_validation import ValidationReport, validate_card

TRACK_NONE = "none"
TRACK_PENDING = "pending"
TRACK_APPROVED = "approved"
TRACK_REJECTED = "rejected"
TRACK_REVISION = "revision"
TRACK_STALE = "stale"


class QaServiceError(Exception):
    def __init__(self, code: str, message: str, http_status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


def _hash_payload(payload: dict) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def compute_content_hash(
    question_text: str,
    reference_answer: str,
    chunk_ids: list[str],
    quotes: list[str],
) -> str:
    return _hash_payload(
        {
            "q": (question_text or "").strip(),
            "a": (reference_answer or "").strip(),
            "chunks": list(chunk_ids),
            "quotes": [(q or "").strip() for q in quotes],
        }
    )


def _load(db: Session, qa_id: str) -> QaCard:
    card = (
        db.query(QaCard)
        .options(selectinload(QaCard.sources), selectinload(QaCard.quotes))
        .filter(QaCard.qa_id == qa_id)
        .one_or_none()
    )
    if card is None:
        raise QaServiceError("not_found", "Карточка Q&A не найдена.", 404)
    return card


def _bind_sources(db: Session, card: QaCard, chunk_ids: list[str]) -> None:
    card.sources.clear()
    db.flush()
    for index, chunk_id in enumerate(chunk_ids):
        chunk = db.get(KnowledgeChunk, chunk_id)
        if chunk is None:
            raise QaServiceError("chunk_missing", f"Чанк {chunk_id} не найден.")
        doc = db.get(KnowledgeDocument, chunk.document_id)
        card.sources.append(
            QaSourceChunk(
                id=generate_uuid(),
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                document_version=(chunk.document_version or (doc.version if doc else None)),
                section=chunk.section,
                page=chunk.page,
                role="primary" if index == 0 else "related",
            )
        )
        if index == 0:
            card.document_version = chunk.document_version or (doc.version if doc else None)


def _bind_quotes(card: QaCard, quotes: list[dict]) -> None:
    card.quotes.clear()
    for item in quotes:
        card.quotes.append(
            QaEvidenceQuote(
                id=generate_uuid(),
                chunk_id=item.get("chunk_id"),
                quote=item.get("quote") or "",
                source_element_ids=item.get("source_element_ids"),
            )
        )


def create_candidate(
    db: Session,
    *,
    question_text: str,
    reference_answer: str,
    chunk_ids: list[str],
    quotes: list[dict] | None = None,
    question_type: str = "factual",
    difficulty: str | None = None,
    golden_test_kind: str | None = "single",
    expected_answer_status: str | None = "answered",
    expected_chunk_ids: list[str] | None = None,
    eval_split: str | None = "holdout",
    corpus_version: str | None = None,
    generation_model: str | None = "stub",
    generation_prompt_version: str | None = None,
    competency: str | None = None,
    module_id: str | None = None,
    rubric_id: str | None = None,
    answer_type: str | None = None,
    answer_key: dict | list | None = None,
    required_elements: list | None = None,
    allowed_formulations: list | None = None,
    critical_errors: list | None = None,
    actor_id: str = "qa-author",
    warnings: list | None = None,
    ambiguity_flags: list | None = None,
    document_version: str | None = None,
) -> tuple[QaCard, ValidationReport]:
    quotes = quotes or []
    digest = compute_content_hash(
        question_text,
        reference_answer,
        chunk_ids,
        [q.get("quote") or "" for q in quotes],
    )
    if find_exact_duplicates(db, digest):
        raise QaServiceError("exact_duplicate", "Точный дубль уже существует.")
    if module_id and db.get(LearningModule, module_id) is None:
        raise QaServiceError("module_missing", "Учебный модуль не найден.")
    rubric = db.get(Rubric, rubric_id) if rubric_id else None
    qa_id = generate_uuid()
    card = QaCard(
        qa_id=qa_id,
        lineage_id=qa_id,
        qa_version="1.0",
        question_text=question_text.strip(),
        reference_answer=(reference_answer or "").strip(),
        question_type=question_type,
        difficulty=difficulty,
        corpus_version=corpus_version,
        generation_model=generation_model,
        generation_prompt_version=generation_prompt_version,
        content_hash=digest,
        generation_status="draft",
        golden_status=TRACK_NONE,
        learning_status=TRACK_NONE,
        golden_test_kind=golden_test_kind,
        expected_answer_status=expected_answer_status,
        expected_chunk_ids=expected_chunk_ids or list(chunk_ids),
        eval_split=eval_split,
        competency=competency,
        module_id=module_id,
        rubric_id=rubric_id,
        rubric_version=rubric.version if rubric else None,
        answer_type=answer_type,
        answer_key=answer_key,
        required_elements=required_elements,
        allowed_formulations=allowed_formulations,
        critical_errors=critical_errors,
        warnings=warnings,
        ambiguity_flags=ambiguity_flags,
        document_version=document_version,
    )
    db.add(card)
    db.flush()
    _bind_sources(db, card, chunk_ids)
    _bind_quotes(card, quotes)
    report = validate_card(db, card)
    card.generation_status = "validated" if report.ok else "draft"
    db.add(
        AuditEvent(
            actor=actor_id,
            action="qa_create",
            object=card.qa_id,
            payload={"hash": digest, "ok": report.ok},
        )
    )
    db.commit()
    db.refresh(card)
    return card, report


def revise_candidate(
    db: Session,
    qa_id: str,
    *,
    question_text: str,
    reference_answer: str,
    chunk_ids: list[str],
    quotes: list[dict] | None = None,
    actor_id: str = "qa-author",
    **kwargs,
) -> tuple[QaCard, ValidationReport]:
    previous = _load(db, qa_id)
    previous.generation_status = "superseded"
    previous.updated_at = datetime.utcnow()
    quotes = quotes or []
    digest = compute_content_hash(
        question_text,
        reference_answer,
        chunk_ids,
        [q.get("quote") or "" for q in quotes],
    )
    major, _, minor = (previous.qa_version or "1.0").partition(".")
    try:
        nxt = f"{int(major)}.{int(minor or 0) + 1}"
    except ValueError:
        nxt = previous.qa_version + ".1"
    card = QaCard(
        qa_id=generate_uuid(),
        lineage_id=previous.lineage_id,
        parent_qa_id=previous.qa_id,
        qa_version=nxt,
        question_text=question_text.strip(),
        reference_answer=(reference_answer or "").strip(),
        question_type=kwargs.get("question_type") or previous.question_type,
        difficulty=kwargs.get("difficulty") if "difficulty" in kwargs else previous.difficulty,
        corpus_version=kwargs.get("corpus_version", previous.corpus_version),
        generation_model=previous.generation_model,
        generation_prompt_version=previous.generation_prompt_version,
        content_hash=digest,
        generation_status="draft",
        golden_status=TRACK_NONE,
        learning_status=TRACK_NONE,
        golden_test_kind=kwargs.get("golden_test_kind", previous.golden_test_kind),
        expected_answer_status=kwargs.get("expected_answer_status", previous.expected_answer_status),
        expected_chunk_ids=kwargs.get("expected_chunk_ids") or list(chunk_ids),
        eval_split=previous.eval_split,
        competency=kwargs.get("competency", previous.competency),
        module_id=kwargs.get("module_id", previous.module_id),
        rubric_id=kwargs.get("rubric_id", previous.rubric_id),
        rubric_version=previous.rubric_version,
        answer_type=kwargs.get("answer_type", previous.answer_type),
        answer_key=kwargs.get("answer_key", previous.answer_key),
        required_elements=kwargs.get("required_elements", previous.required_elements),
        allowed_formulations=kwargs.get("allowed_formulations", previous.allowed_formulations),
        critical_errors=kwargs.get("critical_errors", previous.critical_errors),
    )
    db.add(card)
    db.flush()
    _bind_sources(db, card, chunk_ids)
    _bind_quotes(card, quotes)
    report = validate_card(db, card)
    card.generation_status = "validated" if report.ok else "draft"
    db.add(
        AuditEvent(
            actor=actor_id,
            action="qa_revise",
            object=card.qa_id,
            payload={"parent": previous.qa_id, "version": card.qa_version},
        )
    )
    db.commit()
    db.refresh(card)
    return card, report


def freeze_eval(
    db: Session,
    card: QaCard,
    payload: dict,
    actor_id: str = "qa-eval",
) -> QaEvalSnapshot:
    row = QaEvalSnapshot(
        qa_id=card.qa_id,
        qa_version=card.qa_version,
        corpus_version=card.corpus_version,
        eval_split=card.eval_split,
        payload=payload,
    )
    db.add(row)
    db.add(
        AuditEvent(
            actor=actor_id,
            action="qa_eval_freeze",
            object=card.qa_id,
            payload={"snapshot_id": row.id, "corpus_version": card.corpus_version},
        )
    )
    db.commit()
    db.refresh(row)
    return row


def invalidate_for_chunk(db: Session, chunk_id: str, actor_id: str = "system") -> list[str]:
    links = db.query(QaSourceChunk).filter(QaSourceChunk.chunk_id == chunk_id).all()
    touched: list[str] = []
    for link in links:
        card = _load(db, link.qa_id)
        if card.generation_status == "superseded":
            continue
        changed = False
        if card.golden_status == TRACK_APPROVED:
            card.golden_status = TRACK_STALE
            changed = True
        if card.learning_status == TRACK_APPROVED:
            card.learning_status = TRACK_STALE
            changed = True
        if changed:
            card.generation_status = "stale"
            card.updated_at = datetime.utcnow()
            touched.append(card.qa_id)
            db.add(
                AuditEvent(
                    actor=actor_id,
                    action="qa_invalidate_source",
                    object=card.qa_id,
                    payload={"chunk_id": chunk_id},
                )
            )
    db.commit()
    return touched


def expert_view(db: Session, qa_id: str) -> dict:
    card = _load(db, qa_id)
    chunks = []
    for src in card.sources:
        chunk = db.get(KnowledgeChunk, src.chunk_id)
        doc = db.get(KnowledgeDocument, src.document_id) if src.document_id else None
        chunks.append(
            {
                "chunk_id": src.chunk_id,
                "document_id": src.document_id,
                "document_class": doc.document_class if doc else None,
                "document_version": src.document_version,
                "section": src.section,
                "page": src.page,
                "evidence_text": getattr(chunk, "evidence_text", None) if chunk else None,
                "status": doc.status if doc else None,
            }
        )
    rubric = db.get(Rubric, card.rubric_id) if card.rubric_id else None
    module = db.get(LearningModule, card.module_id) if card.module_id else None
    return {
        "qa_id": card.qa_id,
        "qa_version": card.qa_version,
        "question_text": card.question_text,
        "reference_answer": card.reference_answer,
        "quotes": [{"chunk_id": q.chunk_id, "quote": q.quote} for q in card.quotes],
        "chunks": chunks,
        "golden_status": card.golden_status,
        "learning_status": card.learning_status,
        "golden_test_kind": card.golden_test_kind,
        "competency": card.competency,
        "module": {"id": module.id, "title": module.title} if module else None,
        "rubric": (
            {
                "id": rubric.id,
                "version": rubric.version,
                "required_elements": rubric.required_elements,
                "status": rubric.status,
            }
            if rubric
            else None
        ),
        "eval_split": card.eval_split,
        "corpus_version": card.corpus_version,
    }


def sync_question_row(db: Session, card: QaCard) -> Question | None:
    if card.learning_status != TRACK_APPROVED:
        return None
    row = db.query(Question).filter(Question.qa_id == card.qa_id).first()
    if row is None:
        row = Question(qa_id=card.qa_id, prompt=card.question_text)
        db.add(row)
    row.prompt = card.question_text
    row.type = card.answer_type or card.question_type
    row.competency = card.competency
    row.difficulty = card.difficulty
    row.key = card.answer_key
    row.rubric_version = card.rubric_version
    row.evidence_ids = [s.chunk_id for s in card.sources]
    row.status = "approved"
    return row
