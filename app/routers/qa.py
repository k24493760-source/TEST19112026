"""API карточек Q&A. Без автопубликации и без LLM."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import ExpertReview, QaCard, User
from app.services.auth import effective_role, require_permission
from app.services.approval import ApprovalError
from app.services.qa_approval import apply_qa_expert_decision, submit_qa_review
from app.services.qa_service import (
    QaServiceError,
    create_candidate,
    expert_view,
    revise_candidate,
)
from app.services.qa_validation import validate_card
from sqlalchemy.orm import selectinload

router = APIRouter(prefix="/api/v1/most/qa", tags=["most-qa"])


class CandidateBody(BaseModel):
    question_text: str
    reference_answer: str = ""
    chunk_ids: list[str] = Field(default_factory=list)
    quotes: list[dict] = Field(default_factory=list)
    question_type: str = "factual"
    difficulty: str | None = None
    golden_test_kind: str | None = "single"
    expected_answer_status: str | None = "answered"
    competency: str | None = None
    module_id: str | None = None
    rubric_id: str | None = None
    answer_type: str | None = None
    answer_key: dict | list | None = None
    actor_id: str = "qa-author"


class SubmitBody(BaseModel):
    track: str
    actor_id: str
    actor_role: str


class DecideBody(BaseModel):
    expert_id: str
    expert_role: str
    decision: str
    comment: str = ""


def _http(exc: Exception):
    if isinstance(exc, (QaServiceError, ApprovalError)):
        raise HTTPException(status_code=exc.http_status, detail=exc.code) from exc
    raise exc


@router.post("/candidates")
def api_create(
    payload: CandidateBody,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("qa_submit")),
):
    try:
        card, report = create_candidate(
            db,
            question_text=payload.question_text,
            reference_answer=payload.reference_answer,
            chunk_ids=payload.chunk_ids,
            quotes=payload.quotes,
            question_type=payload.question_type,
            difficulty=payload.difficulty,
            golden_test_kind=payload.golden_test_kind,
            expected_answer_status=payload.expected_answer_status,
            competency=payload.competency,
            module_id=payload.module_id,
            rubric_id=payload.rubric_id,
            answer_type=payload.answer_type,
            answer_key=payload.answer_key,
            actor_id=user.login,
        )
    except QaServiceError as exc:
        _http(exc)
    return {
        "qa_id": card.qa_id,
        "qa_version": card.qa_version,
        "generation_status": card.generation_status,
        "golden_status": card.golden_status,
        "learning_status": card.learning_status,
        "validation_ok": report.ok,
        "issues": [i.code for i in report.issues],
        "auto_published": False,
    }


@router.get("/{qa_id}")
def api_get(qa_id: str, db: Session = Depends(get_db), user: User = Depends(require_permission("qa_submit"))):
    try:
        return expert_view(db, qa_id)
    except QaServiceError as exc:
        _http(exc)


@router.post("/{qa_id}/validate")
def api_validate(qa_id: str, db: Session = Depends(get_db)):
    card = (
        db.query(QaCard)
        .options(selectinload(QaCard.sources), selectinload(QaCard.quotes))
        .filter(QaCard.qa_id == qa_id)
        .first()
    )
    if card is None:
        raise HTTPException(status_code=404, detail="not_found")
    report = validate_card(db, card)
    return {"ok": report.ok, "issues": [{"code": i.code, "message": i.message} for i in report.issues]}


@router.post("/{qa_id}/submit")
def api_submit(
    qa_id: str,
    payload: SubmitBody,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("qa_submit")),
):
    try:
        review = submit_qa_review(db, qa_id, payload.track, user.login, effective_role(user))
    except ApprovalError as exc:
        _http(exc)
    return {"review_id": review.id, "status": review.status, "track": payload.track}


@router.post("/{qa_id}/revise")
def api_revise(
    qa_id: str,
    payload: CandidateBody,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("qa_submit")),
):
    try:
        card, report = revise_candidate(
            db,
            qa_id,
            question_text=payload.question_text,
            reference_answer=payload.reference_answer,
            chunk_ids=payload.chunk_ids,
            quotes=payload.quotes,
            actor_id=user.login,
        )
    except QaServiceError as exc:
        _http(exc)
    return {"qa_id": card.qa_id, "qa_version": card.qa_version, "parent_qa_id": card.parent_qa_id, "ok": report.ok}


@router.post("/reviews/{review_id}/decide")
def api_decide(
    review_id: str,
    payload: DecideBody,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("expert_approve")),
):
    review = db.get(ExpertReview, review_id)
    if review is None:
        raise HTTPException(status_code=404, detail="review_not_found")
    try:
        card = apply_qa_expert_decision(
            db,
            review,
            expert_id=user.id,
            expert_role=effective_role(user),
            decision=payload.decision,
            comment=payload.comment,
        )
        db.commit()
    except ApprovalError as exc:
        db.rollback()
        _http(exc)
    return {
        "qa_id": card.qa_id,
        "golden_status": card.golden_status,
        "learning_status": card.learning_status,
        "decision": payload.decision,
        "auto_published": False,
    }
