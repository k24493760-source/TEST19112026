"""Реестр источников: /api/v1/most/corpus/registry."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, or_
from sqlalchemy.orm import Session, selectinload

from app.config import BASE_DIR
from app.db import get_db
from app.models import AuditEvent, DocumentHistory, KnowledgeDocument, User
from app.services.auth import effective_role, require_permission
from app.services.approval import ApprovalError, finalize_if_experts_complete
from app.services.corpus import apply_related_on_approve
from app.services.publication import PublicationError, current_published_release, publish_corpus, retrieval_contract

router = APIRouter(prefix="/api/v1/most/corpus", tags=["most-registry"])

CLASS_FILTER = {
    "normative_act": ("Нормативный акт", "normative_act"),
    "official_explanation": ("Официальное разъяснение", "official_explanation"),
    "legal_comment": ("Юридический комментарий", "legal_comment"),
    "edu_material": ("Учебный материал", "edu_material"),
    "case": ("Кейс", "case"),
    "method": ("Методика", "method"),
}

STATUS_PUBLIC = {
    "approved": "active",
    "active": "active",
    "draft": "draft",
    "changed": "changed",
    "archived": "revoked",
    "revoked": "revoked",
}

STATUS_FILTER = {
    "active": ("approved", "active"),
    "changed": ("changed",),
    "revoked": ("archived", "revoked"),
    "draft": ("draft",),
}


class StatusRequest(BaseModel):
    status: str
    comment: str = ""
    actor_id: str
    actor_role: str
    effective_date: datetime | None = None


class ApproveRequest(BaseModel):
    actor_id: str
    actor_role: str
    comment: str = ""
    signature: str | None = None


class PublishRequest(BaseModel):
    document_ids: list[str]
    actor_id: str
    actor_role: str
    notes: str = ""


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat() + ("Z" if value.tzinfo is None else "")


def _file_formats(doc: KnowledgeDocument) -> list[str]:
    raw = (doc.file_format or "").strip().lower().lstrip(".")
    if raw in {"docx", "pdf", "txt", "md"}:
        return [raw]
    path = (doc.file_path or "").lower()
    for ext in ("docx", "pdf", "txt", "md"):
        if path.endswith("." + ext):
            return [ext]
    return []


def _file_size(doc: KnowledgeDocument) -> int | None:
    if not doc.file_path:
        return None
    path = BASE_DIR / doc.file_path
    if path.is_file():
        return path.stat().st_size
    return None


def _list_item(doc: KnowledgeDocument) -> dict:
    return {
        "document_id": doc.id,
        "title": doc.title,
        "document_class": doc.document_class,
        "owner": doc.owner,
        "version": doc.version,
        "status": STATUS_PUBLIC.get(doc.status, doc.status),
        "valid_from": _iso(doc.valid_from),
        "valid_to": _iso(doc.valid_to),
        "official_url": doc.official_url,
        "reviewed_at": _iso(doc.reviewed_at or doc.valid_from),
        "hash": doc.hash,
        "file_formats": _file_formats(doc),
        "last_updated_at": _iso(doc.last_updated_at or doc.created_at),
    }


def _get_doc(db: Session, document_id: str) -> KnowledgeDocument:
    doc = (
        db.query(KnowledgeDocument)
        .options(selectinload(KnowledgeDocument.chunks), selectinload(KnowledgeDocument.history))
        .filter(KnowledgeDocument.id == document_id)
        .first()
    )
    if doc is None:
        raise HTTPException(status_code=404, detail="document_not_found")
    return doc


def _history_row(
    db: Session,
    doc: KnowledgeDocument,
    action: str,
    old_status: str | None,
    new_status: str | None,
    actor_id: str,
    actor_role: str,
    comment: str,
) -> DocumentHistory:
    row = DocumentHistory(
        document_id=doc.id,
        action=action,
        old_status=old_status,
        new_status=new_status,
        actor_id=actor_id,
        actor_role=actor_role,
        comment=comment,
    )
    db.add(row)
    return row


def _audit(db: Session, actor: str, action: str, object_id: str, payload: dict) -> AuditEvent:
    event = AuditEvent(actor=actor, action=action, object=object_id, payload=payload)
    db.add(event)
    db.flush()
    return event


@router.get("/registry")
def registry_list(
    document_class: str | None = None,
    status: str | None = None,
    owner: str | None = None,
    search: str | None = None,
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    query = db.query(KnowledgeDocument)
    if document_class and document_class not in ("", "all"):
        aliases = CLASS_FILTER.get(document_class, (document_class,))
        query = query.filter(KnowledgeDocument.document_class.in_(aliases))
    if status and status not in ("", "all"):
        aliases = STATUS_FILTER.get(status, (status,))
        query = query.filter(KnowledgeDocument.status.in_(aliases))
    if owner and owner not in ("", "all"):
        query = query.filter(KnowledgeDocument.owner == owner)
    if search:
        like = f"%{search.strip()}%"
        query = query.filter(
            or_(
                KnowledgeDocument.title.ilike(like),
                KnowledgeDocument.owner.ilike(like),
            )
        )
    total = query.count()
    rows = (
        query.order_by(
            func.coalesce(KnowledgeDocument.last_updated_at, KnowledgeDocument.created_at).desc()
        )
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )
    return {"total": total, "page": page, "items": [_list_item(row) for row in rows]}


@router.get("/registry/{document_id}")
def registry_card(document_id: str, db: Session = Depends(get_db)):
    doc = _get_doc(db, document_id)
    chunks = list(doc.chunks or [])
    approved = sum(1 for c in chunks if (c.approval_status or "") == "approved")
    pending = sum(1 for c in chunks if (c.approval_status or "") in ("draft", "pending", None, ""))
    formats = _file_formats(doc)
    representations = []
    for fmt in formats or ["unknown"]:
        representations.append(
            {
                "format": fmt,
                "file_size": _file_size(doc),
                "uploaded_at": _iso(doc.created_at),
                "is_primary": True,
            }
        )
    history = [
        {
            "action": row.action,
            "old_status": row.old_status,
            "new_status": row.new_status,
            "actor": row.actor_id,
            "comment": row.comment,
            "timestamp": _iso(row.created_at),
        }
        for row in sorted(doc.history or [], key=lambda r: r.created_at or datetime.min)
    ]
    return {
        "document_id": doc.id,
        "metadata": {
            "title": doc.title,
            "document_class": doc.document_class,
            "owner": doc.owner,
            "version": doc.version,
            "status": STATUS_PUBLIC.get(doc.status, doc.status),
            "valid_from": _iso(doc.valid_from),
            "valid_to": _iso(doc.valid_to),
            "official_url": doc.official_url,
            "reviewed_at": _iso(doc.reviewed_at or doc.valid_from),
            "hash": doc.hash,
            "last_updated_at": _iso(doc.last_updated_at or doc.created_at),
        },
        "representations": representations,
        "chunking_stats": {
            "total_chunks": len(chunks),
            "approved_chunks": approved,
            "pending_approval_chunks": pending if chunks else 0,
        },
        "dependencies": {
            "dependent_answers_count": 0,
            "dependent_tests_count": 0,
            "dependent_modules": [],
        },
        "history": history,
        "conflict_flags": {"has_conflicts": False, "conflicting_document_ids": []},
    }


@router.post("/registry/{document_id}/status")
def registry_status(
    document_id: str,
    payload: StatusRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("manage_corpus")),
):
    payload.actor_id = user.login
    payload.actor_role = effective_role(user)
    doc = _get_doc(db, document_id)
    old = doc.status
    new_internal = payload.status
    if payload.status == "active":
        new_internal = "approved"
    elif payload.status == "revoked":
        new_internal = "archived"
    now = payload.effective_date or datetime.utcnow()
    doc.status = new_internal
    doc.last_updated_at = now
    if new_internal == "archived":
        doc.valid_to = now
    _history_row(
        db, doc, "status_change", old, new_internal,
        payload.actor_id, payload.actor_role, payload.comment,
    )
    event = _audit(
        db,
        payload.actor_id,
        "registry_status",
        doc.id,
        {"old_status": old, "new_status": new_internal, "comment": payload.comment},
    )
    invalidated_answers = 0
    invalidated_tests = 0
    reindex = payload.status in ("changed", "revoked") or new_internal in (
        "changed",
        "archived",
        "revoked",
    )
    if reindex:
        invalidated_answers = 0
        invalidated_tests = 0
    db.commit()
    public = STATUS_PUBLIC.get(doc.status, doc.status)
    message = "Статус обновлён."
    if reindex:
        message = (
            f"Инвалидировано {invalidated_answers + invalidated_tests} зависимых объектов"
        )
    return {
        "document_id": doc.id,
        "new_status": public,
        "affected_dependencies": {
            "invalidated_answers_count": invalidated_answers,
            "invalidated_tests_count": invalidated_tests,
        },
        "audit_event_id": event.id,
        "requires_reindex": reindex,
        "message": message,
    }


@router.post("/registry/{document_id}/approve")
def registry_approve(
    document_id: str,
    payload: ApproveRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("manage_corpus")),
):
    payload.actor_id = user.login
    payload.actor_role = effective_role(user)
    doc = _get_doc(db, document_id)
    try:
        result = finalize_if_experts_complete(
            db, document_id, payload.actor_id, payload.actor_role
        )
    except ApprovalError as exc:
        raise HTTPException(status_code=exc.http_status, detail=exc.code) from exc
    now = datetime.utcnow()
    if result.status == "approved":
        apply_related_on_approve(db, doc)
        db.commit()
    return {
        "document_id": result.document_id,
        "status": "active" if result.status == "approved" else result.status,
        "approved_by": payload.actor_id,
        "approved_at": _iso(now) if result.status == "approved" else None,
        "audit_event_id": result.audit_event_id,
        "reindex_triggered": result.status == "approved",
    }


@router.post("/releases/publish")
def corpus_publish(
    payload: PublishRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("publish_release")),
):
    payload.actor_id = user.login
    payload.actor_role = effective_role(user)
    try:
        result = publish_corpus(
            db,
            payload.document_ids,
            payload.actor_id,
            payload.actor_role,
            notes=payload.notes,
        )
    except PublicationError as exc:
        raise HTTPException(
            status_code=exc.http_status,
            detail={"code": exc.code, "errors": exc.errors},
        ) from exc
    return {
        "release_id": result.release_id,
        "version": result.version,
        "status": result.status,
        "previous_version": result.previous_version,
        "duplicate": result.duplicate,
        "audit_event_id": result.audit_event_id,
        "n_documents": result.n_documents,
        "n_chunks": result.n_chunks,
        "n_embeddings": result.n_embeddings,
        "rag_contract": retrieval_contract(),
    }


@router.get("/releases/current")
def corpus_current(db: Session = Depends(get_db)):
    row = current_published_release(db)
    if row is None:
        return {"release": None, "rag_contract": retrieval_contract()}
    return {
        "release": {
            "id": row.id,
            "version": row.version,
            "status": row.status,
            "published_at": _iso(row.published_at),
            "published_by": row.published_by,
            "embedding_model": row.embedding_model,
            "model_revision": row.model_revision,
            "previous_version": row.previous_version,
        },
        "rag_contract": retrieval_contract(),
    }
