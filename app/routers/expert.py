"""Очередь эксперта: контракт GET/POST /api/v1/most/expert/*."""

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import or_
from sqlalchemy.orm import Session, selectinload

from app.db import get_db
from app.models import AuditEvent, ExpertComment, ExpertReview, User, generate_uuid
from app.services.auth import effective_role, require_permission
from app.services.approval import ApprovalError, apply_expert_decision, submit_for_review
from app.services.qa_approval import OBJECT_QA, apply_qa_expert_decision

router = APIRouter(prefix="/api/v1/most/expert", tags=["most-expert"])

EXPERT_ROLES = {"lawyer", "medical_methodologist", "methodist"}
REVIEW_REASONS = {
    "no_basis",
    "source_conflict",
    "low_confidence",
    "appeal",
    "new_document",
    "qa_golden",
    "qa_learning",
}
DECISIONS = {"approve", "reject", "request_revision", "escalate"}
STATUS_PENDING = {"pending", "open"}
OBJECT_TYPE_ALIASES = {
    "document": ("document", "knowledge_document"),
    "chunk": ("chunk",),
    "answer": ("answer", "ask"),
    "assessment": ("assessment",),
    "appeal": ("appeal",),
    "qa_card": ("qa_card", "qa"),
}


def _is_all(value: str | None) -> bool:
    return value is None or str(value).strip() in ("", "all")
MEDICAL_KEYS = {
    "diagnosis",
    "diagnoses",
    "vital",
    "vitals",
    "prmo",
    "medical_data",
    "health",
}


class ResolutionAction(BaseModel):
    action_type: str = "close_task"
    payload: dict = Field(default_factory=dict)


class ResolveRequest(BaseModel):
    expert_id: str
    expert_role: str
    decision: str
    comment: str = ""
    resolution_action: ResolutionAction = Field(default_factory=ResolutionAction)
    signature: str | None = None


class CommentRequest(BaseModel):
    expert_id: str
    expert_role: str
    comment: str
    internal_only: bool = False


class SubmitReviewRequest(BaseModel):
    actor_id: str
    actor_role: str


def require_expert_role(expert_role: str | None) -> str:
    role = (expert_role or "").strip()
    if role not in EXPERT_ROLES:
        raise HTTPException(status_code=400, detail="invalid_role")
    return role


def _strip_medical(value):
    if isinstance(value, dict):
        return {
            key: _strip_medical(item)
            for key, item in value.items()
            if key.lower() not in MEDICAL_KEYS
        }
    if isinstance(value, list):
        return [_strip_medical(item) for item in value]
    return value


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat() + ("Z" if value.tzinfo is None else "")


def _public_status(status: str | None) -> str:
    if (status or "").strip() in STATUS_PENDING:
        return "pending"
    return status or "pending"


def _summary(review: ExpertReview) -> str:
    evidence = review.evidence if isinstance(review.evidence, dict) else {}
    question = str(evidence.get("question") or "").strip()
    if question:
        return question[:180]
    return f"{review.object_type}:{review.reason}"


def _queue_item(review: ExpertReview) -> dict:
    evidence = review.evidence if isinstance(review.evidence, dict) else {}
    requested = evidence.get("requested_by") or evidence.get("user_context") or "listener"
    if isinstance(requested, dict):
        requested = requested.get("user_id") or requested.get("role") or "listener"
    return {
        "review_id": review.id,
        "object_type": review.object_type,
        "object_id": review.object_id,
        "reason": review.reason,
        "status": _public_status(review.status),
        "priority": review.priority or "normal",
        "created_at": _iso(review.created_at),
        "sla_deadline": _iso(review.sla_deadline),
        "assigned_expert_role": review.expert_role,
        "summary": _summary(review),
        "requested_by": str(requested),
    }


def _evidence_items(evidence: dict) -> list[dict]:
    items = []
    raw = evidence.get("evidence")
    if isinstance(raw, list):
        for row in raw:
            if not isinstance(row, dict):
                continue
            items.append(
                {
                    "chunk_id": row.get("chunk_id"),
                    "document_id": row.get("document_id") or row.get("id"),
                    "text_fragment": row.get("text_fragment") or row.get("quote") or "",
                    "metadata": {
                        "section": row.get("section"),
                        "page": row.get("page"),
                        "version": row.get("version"),
                        "status": row.get("status"),
                    },
                }
            )
    for src in evidence.get("reviewed_sources") or []:
        if not isinstance(src, dict):
            continue
        items.append(
            {
                "chunk_id": src.get("chunk_id"),
                "document_id": src.get("document_id") or src.get("id"),
                "text_fragment": src.get("title") or "",
                "metadata": {
                    "section": src.get("section"),
                    "page": src.get("page"),
                    "version": src.get("version"),
                    "status": src.get("status"),
                },
            }
        )
    return _strip_medical(items)


def _card(review: ExpertReview) -> dict:
    evidence = review.evidence if isinstance(review.evidence, dict) else {}
    snapshot = evidence.get("object_snapshot")
    if not isinstance(snapshot, dict):
        snapshot = {
            "question": evidence.get("question"),
            "answer": evidence.get("answer"),
            "claims": evidence.get("claims") or [],
        }
    return _strip_medical(
        {
            "review_id": review.id,
            "status": _public_status(review.status),
            "reason": review.reason,
            "object_type": review.object_type,
            "object_snapshot": snapshot,
            "evidence": _evidence_items(evidence),
            "conflict_details": evidence.get("conflict_details"),
            "user_appeal_context": evidence.get("user_appeal_context"),
            "corpus_version": evidence.get("corpus_version"),
            "created_at": _iso(review.created_at),
        }
    )


def _get_review(db: Session, review_id: str) -> ExpertReview:
    review = (
        db.query(ExpertReview)
        .options(selectinload(ExpertReview.comments))
        .filter(ExpertReview.id == review_id)
        .first()
    )
    if review is None:
        raise HTTPException(status_code=404, detail="review_not_found")
    return review


def _audit(db: Session, actor: str, action: str, object_id: str, payload: dict) -> AuditEvent:
    event = AuditEvent(
        actor=actor,
        action=action,
        object=object_id,
        payload=_strip_medical(payload),
    )
    db.add(event)
    db.flush()
    return event


@router.post("/documents/{document_id}/submit")
def submit_document_review(
    document_id: str,
    payload: SubmitReviewRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("submit_review")),
):
    try:
        result = submit_for_review(db, document_id, user.login, effective_role(user))
    except ApprovalError as exc:
        raise HTTPException(status_code=exc.http_status, detail=exc.code) from exc
    return {
        "document_id": result.document_id,
        "status": result.status,
        "review_ids": result.reviews,
        "missing_roles": result.missing_roles,
        "audit_event_id": result.audit_event_id,
    }


@router.get("/queue")
def expert_queue(
    expert_role: str | None = None,
    expert_id: str | None = None,
    status: str | None = None,
    reason: str | None = None,
    object_type: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("view_queue")),
):
    session_role = effective_role(user)
    if session_role in EXPERT_ROLES:
        expert_role = session_role
    elif expert_role:
        require_expert_role(expert_role)
    else:
        expert_role = None
    query = db.query(ExpertReview)
    if expert_id:
        query = query.filter(ExpertReview.expert_id == expert_id)
    if expert_role:
        query = query.filter(
            or_(
                ExpertReview.expert_role == expert_role,
                ExpertReview.expert_role.is_(None),
            )
        )
    if not _is_all(status):
        if status == "pending":
            query = query.filter(ExpertReview.status.in_(STATUS_PENDING))
        else:
            query = query.filter(ExpertReview.status == status)
    if not _is_all(reason):
        query = query.filter(ExpertReview.reason == reason)
    if not _is_all(object_type):
        aliases = OBJECT_TYPE_ALIASES.get(object_type, (object_type,))
        query = query.filter(ExpertReview.object_type.in_(aliases))
    total = query.count()
    rows = (
        query.order_by(ExpertReview.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return {"total": total, "items": [_queue_item(row) for row in rows]}


@router.get("/queue/{review_id}")
def expert_queue_card(
    review_id: str,
    expert_role: str | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("view_queue")),
):
    return _card(_get_review(db, review_id))


@router.post("/queue/{review_id}/resolve")
def expert_resolve(
    review_id: str,
    payload: ResolveRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("expert_approve")),
):
    payload.expert_role = effective_role(user)
    payload.expert_id = user.id
    require_expert_role(payload.expert_role)
    if payload.decision not in DECISIONS:
        raise HTTPException(status_code=400, detail="invalid_decision")
    review = _get_review(db, review_id)
    if review.reason == "source_conflict" and payload.decision == "approve":
        if not (payload.comment or "").strip():
            raise HTTPException(status_code=400, detail="comment_required")
    now = datetime.utcnow()
    try:
        if (review.object_type or "") in OBJECT_QA:
            card = apply_qa_expert_decision(
                db,
                review,
                expert_id=payload.expert_id,
                expert_role=payload.expert_role,
                decision=payload.decision,
                comment=payload.comment,
            )

            class _R:
                document_id = card.qa_id
                status = card.golden_status if review.reason == "qa_golden" else card.learning_status
                missing_roles = []

            result = _R()
        else:
            result = apply_expert_decision(
                db,
                review,
                expert_id=payload.expert_id,
                expert_role=payload.expert_role,
                decision=payload.decision,
                comment=payload.comment,
            )
    except ApprovalError as exc:
        raise HTTPException(status_code=exc.http_status, detail=exc.code)
    review.resolution_action = payload.resolution_action.model_dump()
    if review.created_at:
        review.time_spent_seconds = int((now - review.created_at).total_seconds())
    invalidated: list[str] = []
    reindex_required = result.status == "approved"
    if result.document_id and result.status == "approved":
        invalidated.append(result.document_id)

    event = _audit(
        db,
        payload.expert_id,
        "expert_resolve_http",
        review.id,
        {
            "review_id": review.id,
            "decision": payload.decision,
            "signature": payload.signature,
            "document_status": result.status,
            "missing_roles": result.missing_roles,
        },
    )
    db.commit()
    db.refresh(review)
    return {
        "review_id": review.id,
        "status": review.status,
        "decision": review.decision,
        "resolved_at": _iso(review.resolved_at),
        "resolved_by": payload.expert_id,
        "audit_event_id": event.id,
        "next_steps": {
            "status_changed_to": review.status,
            "dependent_objects_invalidated": invalidated,
            "reindex_required": reindex_required,
        },
    }


@router.post("/queue/{review_id}/comment")
def expert_comment(
    review_id: str,
    payload: CommentRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("view_queue")),
):
    payload.expert_role = effective_role(user)
    payload.expert_id = user.id
    require_expert_role(payload.expert_role)
    review = _get_review(db, review_id)
    comment = ExpertComment(
        id=generate_uuid(),
        review_id=review.id,
        expert_id=payload.expert_id,
        comment=payload.comment,
        internal_only=payload.internal_only,
    )
    db.add(comment)
    event = _audit(
        db,
        payload.expert_id,
        "expert_comment",
        review.id,
        {"comment_id": comment.id, "internal_only": payload.internal_only},
    )
    db.commit()
    db.refresh(comment)
    return {
        "comment_id": comment.id,
        "review_id": review.id,
        "created_at": _iso(comment.created_at),
        "audit_event_id": event.id,
    }


@router.get("/history")
def expert_history(
    expert_role: str | None = None,
    expert_id: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    object_type: str | None = None,
    db: Session = Depends(get_db),
):
    if expert_role:
        require_expert_role(expert_role)
    query = db.query(ExpertReview).filter(ExpertReview.resolved_at.isnot(None))
    if expert_id:
        query = query.filter(ExpertReview.expert_id == expert_id)
    if expert_role:
        query = query.filter(ExpertReview.expert_role == expert_role)
    if object_type:
        query = query.filter(ExpertReview.object_type == object_type)
    if date_from:
        query = query.filter(ExpertReview.resolved_at >= date_from)
    if date_to:
        query = query.filter(ExpertReview.resolved_at <= date_to)
    rows = query.order_by(ExpertReview.resolved_at.desc()).limit(200).all()
    review_ids = [row.id for row in rows]
    audit_by_object: dict[str, str] = {}
    if review_ids:
        events = (
            db.query(AuditEvent)
            .filter(AuditEvent.action == "expert_resolve")
            .filter(AuditEvent.object.in_(review_ids))
            .order_by(AuditEvent.timestamp.desc())
            .all()
        )
        for event in events:
            if event.object and event.object not in audit_by_object:
                audit_by_object[event.object] = event.id
    items = []
    for row in rows:
        items.append(
            {
                "review_id": row.id,
                "object_type": row.object_type,
                "decision": row.decision,
                "comment": row.comment,
                "resolved_at": _iso(row.resolved_at),
                "time_spent_seconds": row.time_spent_seconds,
                "audit_event_id": audit_by_object.get(row.id),
            }
        )
    return {"items": items}


def default_sla(now: datetime | None = None) -> datetime:
    return (now or datetime.utcnow()) + timedelta(hours=48)
