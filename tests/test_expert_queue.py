from fastapi import HTTPException

from app.routers.expert import EXPERT_ROLES, REVIEW_REASONS, require_expert_role
from app.models import ExpertReview


def test_require_expert_role_rejects_missing():
    try:
        require_expert_role(None)
        assert False
    except HTTPException as exc:
        assert exc.status_code == 400
        assert exc.detail == "invalid_role"


def test_require_expert_role_accepts_lawyer():
    assert require_expert_role("lawyer") == "lawyer"
    assert "medical_methodologist" in EXPERT_ROLES
    assert "methodist" in EXPERT_ROLES


def test_review_reason_enum_includes_contract_values():
    expected = {
        "no_basis",
        "source_conflict",
        "low_confidence",
        "appeal",
        "new_document",
    }
    assert expected == REVIEW_REASONS
    assert ExpertReview.__tablename__ == "expert_reviews"
