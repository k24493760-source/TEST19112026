"""Детерминированная оценка закрытых вопросов. Свободный ответ — только по рубрике."""
from __future__ import annotations

from app.models import QaCard
from app.services.qa_service import QaServiceError


def score_closed(card: QaCard, answer: str) -> tuple[float, str]:
    if (card.answer_type or "") != "closed":
        raise QaServiceError("not_closed", "Это не закрытый вопрос.")
    key = card.answer_key
    if isinstance(key, dict):
        expected = str(key.get("correct") or key.get("key") or "")
    elif isinstance(key, list) and key:
        expected = str(key[0])
    else:
        expected = str(key or "")
    ok = (answer or "").strip() == expected.strip()
    return (1.0 if ok else 0.0, "closed_match" if ok else "closed_mismatch")
