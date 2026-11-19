"""Независимое утверждение Golden RAG и учебного банка. Без автоутверждения."""
from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.models import AuditEvent, ExpertReview, QaCard
from app.services.approval import ApprovalError, admin_roles, can, expert_roles
from app.services.qa_service import (
    TRACK_APPROVED,
    TRACK_NONE,
    TRACK_PENDING,
    TRACK_REJECTED,
    TRACK_REVISION,
    expert_view,
    sync_question_row,
)
from app.services.qa_validation import validate_card

OBJECT_QA = ("qa_card", "qa")
REASON_GOLDEN = "qa_golden"
REASON_LEARNING = "qa_learning"
TRACK_REASONS = {"golden": REASON_GOLDEN, "learning": REASON_LEARNING}


def _set_track(card: QaCard, track: str, status: str) -> None:
    if track == "golden":
        card.golden_status = status
    elif track == "learning":
        card.learning_status = status
    else:
        raise ApprovalError("bad_track", "Назначение: golden или learning.")


def submit_qa_review(
    db: Session,
    qa_id: str,
    track: str,
    actor_id: str,
    actor_role: str,
) -> ExpertReview:
    if not can(actor_role, "submit_review") and not can(actor_role, "qa_submit"):
        raise ApprovalError("forbidden", "Недостаточно прав для передачи Q&A на экспертизу.", 403)
    card = db.get(QaCard, qa_id)
    if card is None:
        raise ApprovalError("not_found", "Карточка Q&A не найдена.", 404)
    reason = TRACK_REASONS.get(track)
    if reason is None:
        raise ApprovalError("bad_track", "Назначение: golden или learning.")
    if track == "golden" and card.golden_status == TRACK_APPROVED:
        raise ApprovalError("already_approved", "Golden уже утверждён.", 409)
    if track == "learning" and card.learning_status == TRACK_APPROVED:
        raise ApprovalError("already_approved", "Учебный контур уже утверждён.", 409)
    report = validate_card(db, card)
    if track == "golden" and card.golden_test_kind != "no_basis" and not report.ok:
        raise ApprovalError("invalid_card", "Карточка не проходит автопроверку.", 409)
    if track == "learning" and not (card.competency or "").strip():
        raise ApprovalError("competency_required", "Для обучения нужна компетенция.", 409)
    review = ExpertReview(
        object_type="qa_card",
        object_id=card.qa_id,
        reason=reason,
        status="pending",
        expert_role="methodist" if track == "learning" else "lawyer",
        evidence={
            "track": track,
            "question": card.question_text,
            "answer": card.reference_answer,
            **expert_view(db, card.qa_id),
        },
        sla_deadline=datetime.utcnow() + timedelta(hours=48),
    )
    db.add(review)
    _set_track(card, track, TRACK_PENDING)
    db.add(
        AuditEvent(
            actor=actor_id,
            action="qa_submit_review",
            object=card.qa_id,
            payload={"track": track, "actor_role": actor_role},
        )
    )
    db.commit()
    db.refresh(review)
    return review


def apply_qa_expert_decision(
    db: Session,
    review: ExpertReview,
    *,
    expert_id: str,
    expert_role: str,
    decision: str,
    comment: str = "",
):
    if expert_role not in expert_roles() or not can(expert_role, "expert_approve"):
        raise ApprovalError("forbidden", "Роль не вправе выносить экспертное решение.", 403)
    if expert_role in admin_roles():
        raise ApprovalError("admin_cannot_replace_expert", "Администратор не заменяет эксперта.", 403)
    if review.object_type not in OBJECT_QA:
        raise ApprovalError("not_qa", "Это не задача Q&A.")
    if review.status not in {"pending", "open"}:
        raise ApprovalError("not_pending", "Задача уже закрыта.", 409)
    track = "golden" if review.reason == REASON_GOLDEN else "learning"
    if track == "golden" and expert_role not in {"lawyer", "methodist", "medical_methodologist"}:
        raise ApprovalError("role_mismatch", "Неподходящая роль для Golden RAG.", 403)
    if track == "learning" and expert_role != "methodist":
        raise ApprovalError("role_mismatch", "Учебный банк утверждает методист.", 403)

    card = db.get(QaCard, review.object_id)
    if card is None:
        raise ApprovalError("not_found", "Карточка Q&A не найдена.", 404)

    now = datetime.utcnow()
    review.expert_id = expert_id
    review.expert_role = expert_role
    review.decision = decision
    review.expert_decision = decision
    review.comment = comment
    review.resolved_at = now

    if decision == "approve":
        review.status = "resolved"
        _set_track(card, track, TRACK_APPROVED)
        if track == "learning":
            sync_question_row(db, card)
    elif decision == "reject":
        review.status = "rejected"
        _set_track(card, track, TRACK_REJECTED)
    elif decision == "request_revision":
        review.status = "revision_requested"
        _set_track(card, track, TRACK_REVISION)
    else:
        review.status = "escalated"

    other_after = card.learning_status if track == "golden" else card.golden_status
    db.add(
        AuditEvent(
            actor=expert_id,
            action="qa_expert_resolve",
            object=card.qa_id,
            payload={
                "track": track,
                "decision": decision,
                "qa_version": card.qa_version,
                "other_track_status": other_after,
                "comment": (comment or "")[:300],
            },
        )
    )
    db.flush()
    return card
