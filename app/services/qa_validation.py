"""Автопроверка Q&A без генеративной LLM."""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.models import (
    DocumentElement,
    KnowledgeChunk,
    KnowledgeDocument,
    QaCard,
    QaEvidenceQuote,
    QaSourceChunk,
    Rubric,
)
from app.services.qa_duplicates import find_exact_duplicates
from app.services.verification import quote_contained

ACTIVE_STATUSES = {"approved"}


@dataclass
class ValidationIssue:
    code: str
    message: str


@dataclass
class ValidationReport:
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)


def _issue(code: str, message: str) -> ValidationIssue:
    return ValidationIssue(code, message)


def validate_card(db: Session, card: QaCard) -> ValidationReport:
    issues: list[ValidationIssue] = []
    if not (card.question_text or "").strip():
        issues.append(_issue("question_required", "Нет текста вопроса."))
    no_basis = (card.golden_test_kind == "no_basis") or (
        card.expected_answer_status == "basis_not_found"
    )
    if not no_basis and not (card.reference_answer or "").strip():
        issues.append(_issue("answer_required", "Нет эталонного ответа."))
    if no_basis and card.expected_answer_status != "basis_not_found":
        issues.append(_issue("no_basis_status", "Для вопроса без основания нужен basis_not_found."))
    if no_basis and card.quotes:
        issues.append(_issue("no_basis_quote", "Вопрос без основания не должен нести искусственную цитату."))

    sources = list(card.sources or [])
    if not no_basis and not sources:
        issues.append(_issue("source_required", "Нет связанных чанков."))

    for src in sources:
        chunk = db.get(KnowledgeChunk, src.chunk_id)
        if chunk is None:
            issues.append(_issue("chunk_missing", f"Чанк {src.chunk_id} не найден."))
            continue
        doc = db.get(KnowledgeDocument, chunk.document_id)
        if src.document_version and doc and doc.version and src.document_version != doc.version:
            issues.append(
                _issue(
                    "version_mismatch",
                    f"Версия документа {doc.version} не совпадает со снимком {src.document_version}.",
                )
            )
        if chunk.document_version and src.document_version and chunk.document_version != src.document_version:
            issues.append(_issue("chunk_version_mismatch", "Версия чанка не совпадает со снимком."))

    for quote_row in list(card.quotes or []):
        ok_quote = _quote_supported(db, quote_row)
        if not ok_quote:
            issues.append(_issue("quote_mismatch", "Цитата не подтверждается evidence_text/элементами."))

    dupes = find_exact_duplicates(db, card.content_hash, exclude_qa_id=card.qa_id)
    if dupes:
        issues.append(_issue("exact_duplicate", f"Точный дубль: {dupes[0].qa_id}"))

    if not card.content_hash:
        issues.append(_issue("meta_hash", "Нет content_hash."))
    if not card.qa_version:
        issues.append(_issue("meta_version", "Нет qa_version."))

    if card.learning_status in ACTIVE_STATUSES or (card.answer_type and card.learning_status != "none"):
        if card.learning_status == "approved":
            if not (card.competency or "").strip():
                issues.append(_issue("competency_required", "Учебный вопрос без компетенции."))
            if (card.answer_type or "") == "free":
                if not card.rubric_id:
                    issues.append(_issue("rubric_required", "Свободный ответ без рубрики."))
                else:
                    rubric = db.get(Rubric, card.rubric_id)
                    if rubric is None or rubric.status != "approved":
                        issues.append(_issue("rubric_not_approved", "Рубрика не утверждена."))
                    elif card.rubric_version and rubric.version != card.rubric_version:
                        issues.append(_issue("rubric_version", "Версия рубрики не совпадает."))

    if card.golden_status == "approved" and card.generation_status not in {"validated", "approved"}:
        if any(i.code in {"quote_mismatch", "chunk_missing"} for i in issues):
            issues.append(_issue("not_for_active_set", "Неутверждённая/битая карточка не входит в Golden-набор."))

    return ValidationReport(ok=not issues, issues=issues)


def _quote_supported(db: Session, quote_row: QaEvidenceQuote) -> bool:
    quote = quote_row.quote or ""
    chunk = db.get(KnowledgeChunk, quote_row.chunk_id) if quote_row.chunk_id else None
    evidence = (getattr(chunk, "evidence_text", None) if chunk else None) or (chunk.text if chunk else "")
    if quote_contained(quote, evidence):
        return True
    ids = list(quote_row.source_element_ids or [])
    if chunk:
        ids = ids or list(getattr(chunk, "source_element_ids", None) or [])
    if not ids:
        return False
    elements = db.query(DocumentElement).filter(DocumentElement.id.in_(ids)).all()
    blob = "\n".join((e.text_normalized or e.text_original or "") for e in elements)
    return quote_contained(quote, blob)


def assert_not_in_active_sets(card: QaCard) -> None:
    if card.golden_status == "approved" and card.generation_status in {"superseded", "rejected", "stale"}:
        raise ValueError("stale golden card")
    if card.learning_status == "approved" and card.generation_status in {"superseded", "rejected", "stale"}:
        raise ValueError("stale learning card")
