"""Пост-проверка цитат: chunk существует, документ утверждён и действует, quote есть в тексте."""

from datetime import datetime

from sqlalchemy.orm import Session

from app.models import DocumentElement, KnowledgeChunk, KnowledgeDocument
from app.services.rag import AskResponse, base_limitations

VERIFICATION_HIDDEN = "Ошибка верификации источников. Ответ скрыт."
LIMITATION_UNVERIFIED = "Цитаты не прошли проверку подлинности."


def quote_contained(quote: str | None, chunk_text: str | None) -> bool:
    """quote (или его ключевая часть) содержится в тексте чанка — проверка через in."""
    source = chunk_text or ""
    text = quote or ""
    if not text.strip() or not source:
        return False
    if text in source:
        return True
    stripped = text.strip()
    if stripped in source:
        return True
    key = " ".join(stripped.split())
    collapsed = " ".join(source.split())
    if key in collapsed:
        return True
    key_part = key[:120] if len(key) > 120 else key
    return bool(key_part) and key_part in collapsed


def _load_chunk_row(db: Session, chunk_id: str):
    if not chunk_id:
        return None
    return (
        db.query(KnowledgeChunk, KnowledgeDocument)
        .join(KnowledgeDocument, KnowledgeChunk.document_id == KnowledgeDocument.id)
        .filter(KnowledgeChunk.id == chunk_id)
        .first()
    )


def _document_is_live(doc: KnowledgeDocument, now: datetime) -> bool:
    if (doc.status or "").strip() != "approved":
        return False
    if doc.valid_to is not None and doc.valid_to <= now:
        return False
    if doc.valid_from is not None and doc.valid_from > now:
        return False
    return True


def verify_citations(response: AskResponse, db: Session, *, now: datetime | None = None) -> dict:
    now = now or datetime.utcnow()
    checked: list[dict] = []
    citations = list(response.citations or [])
    all_ok = True

    for citation in citations:
        chunk_id = (citation.chunk_id or "").strip()
        item = {
            "citation_id": citation.id,
            "chunk_id": chunk_id,
            "ok": False,
            "reason": None,
        }
        row = _load_chunk_row(db, chunk_id)
        if row is None:
            item["reason"] = "chunk_not_found"
            all_ok = False
            checked.append(item)
            continue
        chunk, doc = row
        if (chunk.approval_status or "").strip() != "approved":
            item["reason"] = "chunk_not_approved"
            all_ok = False
            checked.append(item)
            continue
        if not _document_is_live(doc, now):
            item["reason"] = "document_not_live"
            all_ok = False
            checked.append(item)
            continue
        evidence = getattr(chunk, "evidence_text", None) or chunk.text or ""
        element_blob = ""
        ids = list(getattr(chunk, "source_element_ids", None) or [])
        if ids:
            elements = (
                db.query(DocumentElement)
                .filter(DocumentElement.id.in_(ids))
                .all()
            )
            element_blob = "\n".join(e.text_normalized or e.text_original or "" for e in elements)
        if not (
            quote_contained(citation.quote, evidence)
            or quote_contained(citation.quote, element_blob)
        ):
            item["reason"] = "quote_mismatch"
            all_ok = False
            checked.append(item)
            continue
        item["ok"] = True
        checked.append(item)

    return {"is_verified": all_ok, "checked": checked}


def apply_citation_verification(response: AskResponse, db: Session) -> AskResponse:
    report = verify_citations(response, db)
    if report["is_verified"]:
        return response
    limitations = list(response.limitations or [])
    if LIMITATION_UNVERIFIED not in limitations:
        limitations.append(LIMITATION_UNVERIFIED)
    if not limitations:
        limitations = base_limitations()
        limitations.append(LIMITATION_UNVERIFIED)
    return AskResponse(
        answer_status=response.answer_status,
        answer=VERIFICATION_HIDDEN,
        claims=[],
        citations=[],
        limitations=limitations,
        review_required=True,
        review_reason=response.review_reason,
        conflict_detected=response.conflict_detected,
        conflict_details=response.conflict_details,
        reviewed_sources=response.reviewed_sources,
        corpus_version=response.corpus_version,
    )
