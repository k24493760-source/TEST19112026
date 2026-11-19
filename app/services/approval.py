"""Экспертное утверждение документов. Без автоутверждения и без LLM-решения."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache

import yaml
from sqlalchemy.orm import Session, selectinload

from app.config import BASE_DIR
from app.models import (
    AuditEvent,
    DocumentHistory,
    ExpertReview,
    KnowledgeChunk,
    KnowledgeDocument,
)
from app.services.corpus import normalize_document_class

logger = logging.getLogger("most.approval")
POLICY_PATH = BASE_DIR / "contracts" / "approval_policy.yaml"
PENDING_REVIEW = {"pending", "open"}
OBJECT_DOC = ("document", "knowledge_document")


class ApprovalError(Exception):
    def __init__(self, code: str, message: str, http_status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


@dataclass
class ApprovalResult:
    document_id: str
    status: str
    reviews: list[str] = field(default_factory=list)
    missing_roles: list[str] = field(default_factory=list)
    audit_event_id: str | None = None


@lru_cache(maxsize=1)
def load_policy() -> dict:
    with POLICY_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def pending_status() -> str:
    return str(load_policy().get("pending_document_status") or "ready_for_review")


def expert_roles() -> set[str]:
    return set((load_policy().get("expert_roles") or {}).keys())


def admin_roles() -> set[str]:
    return set(load_policy().get("admin_roles") or [])


def can(actor_role: str, action: str) -> bool:
    allowed = (load_policy().get("permissions") or {}).get(action) or []
    return actor_role in allowed


def granted_document_approvers() -> set[str]:
    raw = load_policy().get("document_approvers") or []
    return {str(item).strip() for item in raw if str(item).strip()}


def is_granted_document_approver(login: str | None) -> bool:
    return (login or "").strip() in granted_document_approvers()


def required_expert_roles(doc: KnowledgeDocument) -> list[str]:
    policy = load_policy()
    label = normalize_document_class(doc.document_class)
    roles = list((policy.get("class_roles") or {}).get(label) or [])
    extra = policy.get("clinical_org_extra_role")
    markers = [re.compile(re.escape(m), re.I) for m in (policy.get("clinical_org_markers") or [])]
    blob = " ".join(
        [
            doc.title or "",
            label,
            " ".join((c.retrieval_text or c.text or "") for c in (doc.chunks or [])),
        ]
    )
    if extra and any(pat.search(blob) for pat in markers) and extra not in roles:
        roles.append(extra)
    if not roles:
        raise ApprovalError("no_roles", f"Не заданы экспертные роли для класса «{label}».")
    return roles


def _audit(db: Session, actor: str, action: str, object_id: str, payload: dict) -> AuditEvent:
    safe = {k: v for k, v in payload.items() if k not in {"text", "content", "preview", "segment_text"}}
    event = AuditEvent(actor=actor, action=action, object=object_id, payload=safe)
    db.add(event)
    db.flush()
    return event


def _history(
    db: Session,
    doc: KnowledgeDocument,
    action: str,
    old: str | None,
    new: str | None,
    actor_id: str,
    actor_role: str,
    comment: str,
) -> None:
    row = DocumentHistory(
        document_id=doc.id,
        action=action,
        old_status=old,
        new_status=new,
        actor_id=actor_id,
        actor_role=actor_role,
        comment=comment[:500] if comment else None,
    )
    db.add(row)
    if "history" in doc.__dict__:
        doc.history.append(row)


def document_reviews(db: Session, document_id: str, reason: str = "new_document") -> list[ExpertReview]:
    return (
        db.query(ExpertReview)
        .filter(ExpertReview.object_id == document_id)
        .filter(ExpertReview.object_type.in_(OBJECT_DOC))
        .filter(ExpertReview.reason == reason)
        .all()
    )


def missing_approval_roles(db: Session, doc: KnowledgeDocument) -> list[str]:
    needed = required_expert_roles(doc)
    ok: set[str] = set()
    for review in document_reviews(db, doc.id):
        if (
            review.expert_role in needed
            and review.decision == "approve"
            and review.status in {"resolved", "approved"}
            and review.resolved_at is not None
        ):
            ok.add(review.expert_role)
    return [role for role in needed if role not in ok]


def submit_for_review(
    db: Session,
    document_id: str,
    actor_id: str,
    actor_role: str,
) -> ApprovalResult:
    if not can(actor_role, "submit_review"):
        raise ApprovalError("forbidden", "Недостаточно прав для передачи на экспертизу.", 403)
    doc = (
        db.query(KnowledgeDocument)
        .options(selectinload(KnowledgeDocument.chunks), selectinload(KnowledgeDocument.history))
        .filter(KnowledgeDocument.id == document_id)
        .one_or_none()
    )
    if doc is None:
        raise ApprovalError("not_found", "Документ не найден.", 404)
    if (doc.status or "draft") not in {"draft", "rejected", pending_status()}:
        raise ApprovalError("bad_status", f"Документ в статусе {doc.status} нельзя отправить на экспертизу.", 409)
    chunks = list(doc.chunks or [])
    if not chunks:
        raise ApprovalError("no_chunks", "Нет чанков для экспертизы.")
    old = doc.status
    doc.status = pending_status()
    roles = required_expert_roles(doc)
    created: list[str] = []
    existing = document_reviews(db, doc.id)
    by_role = {r.expert_role: r for r in existing if r.expert_role}
    unassigned = [r for r in existing if not r.expert_role and r.status in PENDING_REVIEW]
    for role in roles:
        row = by_role.get(role)
        if row is None and unassigned:
            row = unassigned.pop(0)
            row.expert_role = role
        if row is None:
            row = ExpertReview(
                object_type="knowledge_document",
                object_id=doc.id,
                reason="new_document",
                status="pending",
                expert_role=role,
                priority="normal",
                sla_deadline=datetime.utcnow() + timedelta(hours=48),
                evidence={
                    "document_id": doc.id,
                    "document_class": doc.document_class,
                    "owner": doc.owner,
                    "version": doc.version,
                    "hash": doc.hash or doc.file_hash,
                    "required_role": role,
                },
            )
            db.add(row)
            db.flush()
        elif row.status in PENDING_REVIEW:
            row.expert_role = role
        created.append(row.id)
    for chunk in chunks:
        if chunk.approval_status != "approved":
            chunk.approval_status = "draft"
    _history(db, doc, "submit_review", old, doc.status, actor_id, actor_role, "")
    event = _audit(
        db,
        actor_id,
        "document_submit_review",
        doc.id,
        {"roles": roles, "status": doc.status, "actor_role": actor_role},
    )
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
    return ApprovalResult(
        document_id=doc.id,
        status=doc.status,
        reviews=created,
        missing_roles=roles,
        audit_event_id=event.id,
    )


def apply_expert_decision(
    db: Session,
    review: ExpertReview,
    *,
    expert_id: str,
    expert_role: str,
    decision: str,
    comment: str = "",
) -> ApprovalResult:
    if expert_role not in expert_roles() or not can(expert_role, "expert_approve"):
        raise ApprovalError("forbidden", "Роль не вправе выносить экспертное решение.", 403)
    if expert_role in admin_roles():
        raise ApprovalError("admin_cannot_replace_expert", "Администратор не заменяет эксперта.", 403)
    if review.expert_role and review.expert_role != expert_role:
        raise ApprovalError("role_mismatch", "Эксперт другой предметной роли.", 403)
    if review.status not in PENDING_REVIEW:
        raise ApprovalError("not_pending", "Задача уже закрыта.", 409)
    if decision == "approve" and not (comment or "").strip() and review.reason == "source_conflict":
        raise ApprovalError("comment_required", "Для конфликта источников нужен комментарий.")

    now = datetime.utcnow()
    review.expert_id = expert_id
    review.expert_role = expert_role
    review.decision = decision
    review.expert_decision = decision
    review.comment = comment
    review.resolved_at = now
    if decision == "approve":
        review.status = "resolved"
    elif decision == "reject":
        review.status = "rejected"
    elif decision == "request_revision":
        review.status = "revision_requested"
    else:
        review.status = "escalated"

    doc = None
    if review.object_id:
        doc = (
            db.query(KnowledgeDocument)
            .options(selectinload(KnowledgeDocument.chunks), selectinload(KnowledgeDocument.history))
            .filter(KnowledgeDocument.id == review.object_id)
            .one_or_none()
        )

    missing: list[str] = []
    if doc is not None and review.reason == "new_document":
        old = doc.status
        if decision == "reject":
            doc.status = "rejected"
            _history(db, doc, "reject", old, doc.status, expert_id, expert_role, comment)
        elif decision == "approve":
            missing = missing_approval_roles(db, doc)
            if not missing:
                _finalize_document(db, doc, expert_id, expert_role)
            else:
                if doc.status == "draft":
                    doc.status = pending_status()
        event = _audit(
            db,
            expert_id,
            "expert_resolve",
            review.id,
            {
                "decision": decision,
                "expert_role": expert_role,
                "document_id": doc.id,
                "document_status": doc.status,
                "missing_roles": missing,
            },
        )
        db.flush()
        return ApprovalResult(
            document_id=doc.id,
            status=doc.status,
            reviews=[review.id],
            missing_roles=missing,
            audit_event_id=event.id,
        )

    event = _audit(
        db,
        expert_id,
        "expert_resolve",
        review.id,
        {"decision": decision, "expert_role": expert_role},
    )
    db.flush()
    return ApprovalResult(
        document_id=review.object_id or "",
        status=review.status,
        reviews=[review.id],
        audit_event_id=event.id,
    )


def _finalize_document(db: Session, doc: KnowledgeDocument, actor_id: str, actor_role: str) -> None:
    old = doc.status
    now = datetime.utcnow()
    doc.status = "approved"
    doc.reviewed_at = now
    if doc.valid_from is None:
        doc.valid_from = now
    for chunk in doc.chunks or []:
        chunk.approval_status = "approved"
    _history(db, doc, "approve", old, "approved", actor_id, actor_role, "complete_expert_set")


def approve_document_as_grant(
    db: Session,
    document_id: str,
    actor_id: str,
    actor_role: str,
) -> ApprovalResult:
    """Именной допуск утверждает документ и пишет это в аудит."""
    doc = (
        db.query(KnowledgeDocument)
        .options(selectinload(KnowledgeDocument.chunks), selectinload(KnowledgeDocument.history))
        .filter(KnowledgeDocument.id == document_id)
        .one_or_none()
    )
    if doc is None:
        raise ApprovalError("not_found", "Документ не найден.", 404)
    if not is_granted_document_approver(actor_id):
        raise ApprovalError("forbidden", "Недостаточно прав.", 403)
    if doc.status != "approved":
        _finalize_document(db, doc, actor_id, actor_role)
    event = _audit(
        db,
        actor_id,
        "document_approve_granted",
        doc.id,
        {"actor_role": actor_role, "status": doc.status},
    )
    db.commit()
    return ApprovalResult(document_id=doc.id, status=doc.status, audit_event_id=event.id)


def finalize_if_experts_complete(
    db: Session,
    document_id: str,
    actor_id: str,
    actor_role: str,
) -> ApprovalResult:
    """Админ/реестр не утверждают сами: только фиксируют уже полный комплект решений."""
    doc = (
        db.query(KnowledgeDocument)
        .options(selectinload(KnowledgeDocument.chunks), selectinload(KnowledgeDocument.history))
        .filter(KnowledgeDocument.id == document_id)
        .one_or_none()
    )
    if doc is None:
        raise ApprovalError("not_found", "Документ не найден.", 404)
    if actor_role in admin_roles() and actor_role not in expert_roles():
        missing = missing_approval_roles(db, doc)
        if missing:
            raise ApprovalError(
                "experts_incomplete",
                "Администратор не заменяет экспертизу. Не хватает ролей: " + ", ".join(missing),
                403,
            )
        if doc.status != "approved":
            _finalize_document(db, doc, actor_id, actor_role)
        event = _audit(
            db,
            actor_id,
            "document_finalize_after_experts",
            doc.id,
            {"actor_role": actor_role, "status": doc.status},
        )
        db.commit()
        return ApprovalResult(document_id=doc.id, status=doc.status, audit_event_id=event.id)
    if actor_role in expert_roles():
        raise ApprovalError(
            "use_queue",
            "Эксперт утверждает через очередь задач, а не через административный approve.",
            403,
        )
    raise ApprovalError("forbidden", "Недостаточно прав.", 403)
