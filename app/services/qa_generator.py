"""Контракт генератора Q&A. Без вызова внешней LLM."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import yaml

from app.config import BASE_DIR

CONFIG_PATH = BASE_DIR / "contracts" / "qa_generation.yaml"


def load_qa_generation_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}


@dataclass
class ChunkContext:
    chunk_id: str
    evidence_text: str
    section_path: str | None
    page: int | None
    document_id: str
    document_version: str | None
    document_class: str | None
    source_element_ids: list[str] = field(default_factory=list)


@dataclass
class GenerateRequest:
    primary: ChunkContext
    related: list[ChunkContext] = field(default_factory=list)
    config: dict = field(default_factory=dict)


@dataclass
class GeneratedCandidate:
    question_text: str
    reference_answer: str
    quotes: list[dict]
    question_type: str
    golden_test_kind: str
    expected_answer_status: str
    difficulty: str | None = None
    ambiguity_flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class GenerateResult:
    candidates: list[GeneratedCandidate] = field(default_factory=list)
    skipped_reason: str | None = None


class QaGenerator(Protocol):
    def generate(self, request: GenerateRequest) -> GenerateResult: ...


def is_contentful(text: str | None, min_chars: int | None = None) -> bool:
    cfg = load_qa_generation_config()
    limit = min_chars if min_chars is not None else int(cfg.get("min_evidence_chars") or 80)
    blob = " ".join((text or "").split())
    if len(blob) < limit:
        return False
    if blob.lower() in {"см. таблицу", "см. рисунок"}:
        return False
    return True


class StubQaGenerator:
    """Заглушка: не вызывает LLM и не выдумывает вопросы."""

    def generate(self, request: GenerateRequest) -> GenerateResult:
        cfg = load_qa_generation_config()
        text = request.primary.evidence_text or ""
        if not is_contentful(text, int(cfg.get("min_evidence_chars") or 80)):
            return GenerateResult(skipped_reason="empty_or_technical_chunk")
        return GenerateResult(
            skipped_reason="llm_not_connected",
            candidates=[],
        )


def default_generator() -> QaGenerator:
    from app.services.qa_llm_client import build_generator_from_config

    return build_generator_from_config()
