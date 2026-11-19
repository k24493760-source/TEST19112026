from types import SimpleNamespace

from app.services.rag import (
    CLAIM_TYPE_BY_CLASS,
    LIMITATION_DEMO,
    LIMITATION_EXTRACTIVE,
    LIMITATION_LEGAL,
    LIMITATION_MEDICAL,
    NO_BASIS,
    SYSTEM_PROMPT,
    NoEvidenceResponse,
    claim_type_for,
    compose_rag_answer,
    evidence_from_chunks,
)

CLAIM_TYPE_ENUM = {
    "legal_norm",
    "official_explanation",
    "legal_interpretation",
    "edu_methodical",
}


def _row(chunk_id="chunk-1", text="Статья 1. Текст нормы.", status="approved", cls="Нормативный акт"):
    chunk = SimpleNamespace(id=chunk_id, text=text, section="1", page=1)
    doc = SimpleNamespace(
        id="doc-1",
        title="Закон",
        version="1.0",
        status=status,
        document_class=cls,
        official_url=None,
    )
    return chunk, doc


def test_system_prompt_requires_basis():
    assert "Основание не найдено" in SYSTEM_PROMPT
    assert "legal_norm" in SYSTEM_PROMPT


def test_claim_type_by_document_class():
    assert claim_type_for("Нормативный акт") == "legal_norm"
    assert claim_type_for("Официальное разъяснение") == "official_explanation"
    assert claim_type_for("Юридический комментарий") == "legal_interpretation"
    assert claim_type_for("Учебный материал") == "edu_methodical"


def test_basis_not_found_requires_review_and_sources():
    payload = NoEvidenceResponse(
        reviewed_sources=[{"id": "doc-1", "title": "Источник"}]
    ).model_dump()
    assert payload["answer_status"] == "basis_not_found"
    assert payload["review_required"] is True
    assert payload["review_reason"] == "no_basis"
    assert payload["reviewed_sources"]


def test_claim_type_in_enum():
    for document_class in CLAIM_TYPE_BY_CLASS:
        assert claim_type_for(document_class) in CLAIM_TYPE_ENUM
    assert evidence_from_chunks([_row()])[0][0].claim_type in CLAIM_TYPE_ENUM


def test_compose_no_chunks_returns_strict_basis():
    result = compose_rag_answer("вопрос", [], approved_count=0, draft_count=0)
    assert result.answer == NO_BASIS
    assert result.claims == []
    assert result.review_required is True
    assert LIMITATION_LEGAL in result.limitations
    assert LIMITATION_MEDICAL in result.limitations
    assert LIMITATION_DEMO in result.limitations


def test_evidence_links_claim_to_citation():
    claims, citations = evidence_from_chunks([_row()])
    assert claims[0].claim_id == "c1"
    assert claims[0].citation_ids == ["cit1"]
    assert claims[0].claim_type == "legal_norm"
    assert citations[0].id == "cit1"
    assert citations[0].chunk_id == "chunk-1"
    assert citations[0].quote.startswith("Статья 1")
    assert citations[0].page == "1"
    assert citations[0].document_title == "Закон"


def test_limitations_always_present_on_hit():
    result = compose_rag_answer("Текст нормы", [_row()], approved_count=5, draft_count=0)
    assert result.answer_mode == "extractive"
    assert result.answer.startswith("Статья 1")
    assert LIMITATION_LEGAL in result.limitations
    assert LIMITATION_MEDICAL in result.limitations
    assert LIMITATION_EXTRACTIVE in result.limitations
    assert LIMITATION_DEMO not in result.limitations


def test_extractive_answer_quotes_lecture_chunk():
    chunk = SimpleNamespace(
        id="chk-lecture-2",
        text="Лекция 2 / Срок составляет четырнадцать рабочих дней.",
        evidence_text="Лекция 2 / Срок составляет четырнадцать рабочих дней.",
        section="Лекция 2",
        page=None,
        source_element_ids=["e1"],
    )
    other = SimpleNamespace(
        id="chk-lecture-1",
        text="Лекция 1 / Слушатель изучает порядок направления уведомления.",
        evidence_text="Лекция 1 / Слушатель изучает порядок направления уведомления.",
        section="Лекция 1",
        page=None,
        source_element_ids=["e2"],
    )
    doc = SimpleNamespace(
        id="doc-demo",
        title="Демонстрационный учебный материал",
        version="1.0",
        status="approved",
        document_class="Учебный материал",
        file_format="md",
    )
    result = compose_rag_answer(
        "Какой срок указан в лекции 2?",
        [(other, doc), (chunk, doc)],
        approved_count=1,
        draft_count=0,
        allowed_corpus_version="corpus-1",
    )
    assert result.answer_status == "answered"
    assert result.answer_mode == "extractive"
    assert result.answer == chunk.evidence_text
    assert "четырнадцать рабочих дней" in result.answer
    assert result.citations[0].document_title == "Демонстрационный учебный материал"
    assert result.citations[0].section == "Лекция 2"
    assert result.citations[0].quote == chunk.evidence_text
    assert len(result.citations) == 1


def test_unrelated_question_is_not_answered_from_chunks():
    result = compose_rag_answer(
        "Какая ставка налога на прибыль?",
        [_row()],
        approved_count=1,
        draft_count=0,
    )
    assert result.answer == NO_BASIS
    assert result.answer_status == "basis_not_found"
    assert result.claims == []
    assert result.review_required is True
