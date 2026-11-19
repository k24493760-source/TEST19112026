"""Точные дубли в коде; семантика — отдельный интерфейс без LLM-истины."""
from __future__ import annotations

from typing import Protocol

from sqlalchemy.orm import Session

from app.models import QaCard


class SemanticDuplicateChecker(Protocol):
    def similar(self, question_text: str, existing: list[str]) -> list[str]: ...


class NoOpSemanticDuplicateChecker:
    def similar(self, question_text: str, existing: list[str]) -> list[str]:
        return []


def find_exact_duplicates(db: Session, content_hash: str, exclude_qa_id: str | None = None) -> list[QaCard]:
    query = db.query(QaCard).filter(QaCard.content_hash == content_hash)
    query = query.filter(QaCard.generation_status != "superseded")
    if exclude_qa_id:
        query = query.filter(QaCard.qa_id != exclude_qa_id)
    return query.all()
