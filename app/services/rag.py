"""Сборка ответа по утверждённым чанкам: системный промт, claims, citations, limitations."""

import re
from typing import Literal

from pydantic import BaseModel, Field

from app.config import LLM_API_KEY, LLM_MODEL, ROLE_PROFILES
from app.models import KnowledgeChunk, KnowledgeDocument

NO_BASIS = "Основание не найдено"

SYSTEM_PROMPT = (
    "Ты — нейро-сотрудник «Эксперт МОСТ» (УКЗ-15): экспертно-обучающий помощник "
    "по управлению корпоративным здоровьем в ИС УКЗ. Ты не являешься аттестацией, "
    "юридическим заключением, медицинской консультацией или допуском к работе.\n"
    "\n"
    "ЦЕЛЬ ОТВЕТА: дать проверяемое утверждение, полностью выводимое из переданных "
    "утверждённых фрагментов, с цитатами (документ + раздел/страница + версия + статус) "
    "и разделением типов утверждений.\n"
    "\n"
    "ИСТОЧНИКИ: только переданный контекст (утверждённые фрагменты). Общие знания, "
    "память диалога и история чата источником не являются. Высокий score поиска не "
    "разрешает ответ по неутверждённому материалу.\n"
    "\n"
    "ТИПЫ УТВЕРЖДЕНИЙ (помечай каждый claim): legal_norm | official_explanation | "
    "legal_interpretation | edu_methodical.\n"
    "\n"
    "ПРАВИЛА ОТВЕТА:\n"
    "1) Каждое содержательное утверждение — со ссылкой на конкретный фрагмент "
    "(document_id, раздел/страница, версия, статус).\n"
    "2) Нет достаточного основания — ответ строго: «Основание не найдено», перечисли "
    "просмотренные источники, review_required=true, reason=no_basis.\n"
    "3) Конфликт редакций — покажи противоречие, не выбирай юридическую позицию, "
    "conflict_detected=true, reason=source_conflict.\n"
    "4) Цитата не прошла проверку соответствия фрагменту — утверждение не выдавай.\n"
    "5) Не выходи за роль слушателя: не делай кадровых, медицинских выводов, "
    "не оценивай здоровье, дисциплину, благонадёжность.\n"
    "\n"
    "ЗАПРЕЩЕНО: придумывать норму, ссылку, статус, версию, расшифровку «МОСТ»; "
    "разрешать юридическую коллизию; выдавать квалификацию или допуск; использовать "
    "медицинские данные работников, диагнозы, результаты ПрМО.\n"
    "\n"
    "ФОРМАТ ВЫВОДА (JSON, без markdown): {\"answer_status\":\"answered|basis_not_found|"
    "conflict_detected\",\"answer\":\"string\",\"claims\":[{\"claim_id\":\"string\","
    "\"text\":\"string\",\"claim_type\":\"legal_norm|official_explanation|"
    "legal_interpretation|edu_methodical\",\"citations\":[{\"document_id\":\"string\","
    "\"section\":\"string|null\",\"page\":\"integer|null\",\"version\":\"string\","
    "\"status\":\"string\",\"quote_check\":\"passed|failed\"}]}],"
    "\"conflict_detected\":false,\"review_required\":false,\"review_reason\":null,"
    "\"limitations\":[\"string\"]}\n"
    "\n"
    "САМОПРОВЕРКА ПЕРЕД ВЫДАЧЕЙ: все claims имеют непустые citations; "
    "quote_check=passed; версия и статус совпадают с контекстом; нет медданных; "
    "нет неподтверждённых юридических утверждений; нет расшифровки «МОСТ».\n"
    "\n"
    "АДАПТАЦИЯ ПОД РОЛЬ (learner_role): Строго следуй профилю роли, переданному "
    "в запросе: {role_profile}. Базовые правила (доказательность, цитаты, "
    "abstention, отсутствие медданных) неизменны для всех ролей."
)

DEFAULT_ROLE_PROFILE = (
    "Фокус: нейтральное пояснение нормы. Стиль: спокойный, без пошаговых "
    "инструкций и без управленческих выводов. Формат: краткие проверяемые "
    "утверждения с цитатами. Простой язык без чек-листов. "
    "Доказательность, «Основание не найдено» при отсутствии фактов, нейтральный "
    "тон; без медданных и оценок здоровья."
)

LIMITATION_UNKNOWN_ROLE = "Стиль ответа: нейтральный (роль не распознана)"

LIMITATION_LEGAL = (
    "Ответ не является юридической консультацией или официальным заключением."
)
LIMITATION_MEDICAL = "Не является медицинской рекомендацией."
LIMITATION_DEMO = (
    "Внимание: некоторые источники не прошли экспертную проверку (demo)."
)
LIMITATION_NO_BASIS = (
    "Среди утверждённых и действующих источников основание не найдено."
)
LIMITATION_EXTRACTIVE = (
    "Ответ извлечён дословно из найденного фрагмента. Генеративная модель не использовалась."
)

_QUESTION_STOPWORDS = {
    "какой",
    "какая",
    "какие",
    "какое",
    "каким",
    "каких",
    "что",
    "где",
    "когда",
    "как",
    "кто",
    "это",
    "для",
    "или",
    "при",
    "над",
    "под",
    "из",
    "на",
    "по",
    "не",
    "да",
    "нет",
    "указан",
    "указана",
    "указано",
    "указаны",
}


def _content_tokens(text: str) -> set[str]:
    raw = re.findall(r"[0-9]+|[a-zа-яё]{3,}", (text or "").lower())
    return {token for token in raw if token not in _QUESTION_STOPWORDS}


def _evidence_text(chunk) -> str:
    return (getattr(chunk, "evidence_text", None) or getattr(chunk, "text", None) or "").strip()


def rows_with_question_overlap(
    question: str,
    rows: list[tuple[KnowledgeChunk, KnowledgeDocument]],
) -> list[tuple[KnowledgeChunk, KnowledgeDocument]]:
    """Оставляет фрагменты, в которых есть слова вопроса. Слабый cosine не считается основанием."""
    asked = _content_tokens(question)
    if not asked:
        return []
    kept: list[tuple[KnowledgeChunk, KnowledgeDocument]] = []
    for chunk, doc in rows:
        if asked & _content_tokens(_evidence_text(chunk)):
            kept.append((chunk, doc))
    return kept

ClaimType = Literal[
    "legal_norm",
    "official_explanation",
    "legal_interpretation",
    "edu_methodical",
]

CLAIM_TYPE_BY_CLASS = {
    "Нормативный акт": "норма права",
    "Официальное разъяснение": "разъяснение",
    "Юридический комментарий": "интерпретация",
    "Учебный материал": "учебное объяснение",
    "Кейс": "учебное объяснение",
    "Методика": "учебное объяснение",
}

CLAIM_TYPE_CANONICAL = {
    "норма права": "legal_norm",
    "разъяснение": "official_explanation",
    "интерпретация": "legal_interpretation",
    "учебное объяснение": "edu_methodical",
    "legal_norm": "legal_norm",
    "official_explanation": "official_explanation",
    "legal_interpretation": "legal_interpretation",
    "edu_methodical": "edu_methodical",
}


class Citation(BaseModel):
    id: str
    chunk_id: str
    section: str | None = None
    page: str | None = None
    quote: str
    document_title: str | None = None
    document_id: str | None = None
    version: str | None = None
    status: str | None = None
    quote_check: Literal["passed", "failed"] = "passed"
    document_class: str | None = None
    document_version: str | None = None
    corpus_version: str | None = None
    source_element_ids: list | None = None


class Claim(BaseModel):
    claim_id: str
    claim_type: ClaimType
    citation_ids: list[str] = Field(default_factory=list)
    text: str = ""
    citations: list[Citation] = Field(default_factory=list)


class AskResponse(BaseModel):
    status: str = "ok"
    answer_status: Literal["answered", "basis_not_found", "conflict_detected"] = (
        "answered"
    )
    answer: str
    answer_mode: Literal["extractive"] | None = None
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    limitations: list[str]
    review_required: bool = False
    review_reason: str | None = None
    conflict_detected: bool = False
    conflict_details: list[dict] | None = None
    reviewed_sources: list[dict] = Field(default_factory=list)
    corpus_version: str | None = None


class NoEvidenceResponse(BaseModel):
    status: Literal["no_evidence"] = "no_evidence"
    answer_status: Literal["basis_not_found"] = "basis_not_found"
    answer: str = NO_BASIS
    review_required: bool = True
    review_reason: Literal["no_basis"] = "no_basis"
    reviewed_sources: list[dict] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    conflict_detected: bool = False
    corpus_version: str | None = None


def normalize_claim_type(value: str | None) -> ClaimType:
    mapped = CLAIM_TYPE_CANONICAL.get((value or "").strip())
    if mapped:
        return mapped  # type: ignore[return-value]
    return "edu_methodical"


def claim_type_for(document_class: str | None) -> ClaimType:
    mapped = CLAIM_TYPE_BY_CLASS.get((document_class or "").strip())
    return normalize_claim_type(mapped)


def base_limitations(
    *,
    no_basis: bool = False,
    demo: bool = False,
) -> list[str]:
    items = [LIMITATION_LEGAL, LIMITATION_MEDICAL]
    if no_basis:
        items.append(LIMITATION_NO_BASIS)
    if demo:
        items.append(LIMITATION_DEMO)
    return items


def is_demo_corpus(
    rows: list[tuple[KnowledgeChunk, KnowledgeDocument]],
    *,
    approved_count: int = 0,
    draft_count: int = 0,
) -> bool:
    found_draft = any((doc.status or "").strip() == "draft" for _chunk, doc in rows)
    sparse = approved_count < 3
    return found_draft or draft_count > 0 or sparse


def format_context(rows: list[tuple[KnowledgeChunk, KnowledgeDocument]]) -> str:
    if not rows:
        return ""
    blocks = []
    for chunk, doc in rows:
        claim_type = claim_type_for(doc.document_class)
        evidence = _evidence_text(chunk)
        fmt = (getattr(doc, "file_format", None) or "").lower().lstrip(".")
        page = "" if fmt == "docx" and getattr(chunk, "page", None) is None else (getattr(chunk, "page", None) or "")
        blocks.append(
            f"chunk_id={chunk.id}\n"
            f"document_id={doc.id}\n"
            f"document_class={doc.document_class}\n"
            f"claim_type={claim_type}\n"
            f"section={chunk.section or ''}\n"
            f"page={page}\n"
            f"{evidence}"
        )
    return "\n\n---\n\n".join(blocks)


def build_llm_messages(
    question: str,
    context: str,
    learner_role: str = "unknown",
    allowed_corpus_version: str = "",
    role_profile: str = "",
) -> list[dict[str, str]]:
    profile = role_profile or DEFAULT_ROLE_PROFILE
    user_content = (
        f"Вопрос слушателя:\n{question.strip()}\n\n"
        f"Роль слушателя: {learner_role}\n"
        f"Профиль роли: {profile}\n"
        f"Версия корпуса (allowed_corpus_version): {allowed_corpus_version}\n\n"
        "Контекст (только утверждённые фрагменты; каждому фрагменту соответствуют "
        "document_id, section, page, version, status):\n"
        f"{context or '(контекст пуст)'}\n\n"
        "Требование: отвечай по SYSTEM_PROMPT. Если контекста недостаточно — "
        "«Основание не найдено» и review_required=true. Не используй историю "
        "диалога как источник."
    )
    system_content = SYSTEM_PROMPT.replace("{role_profile}", profile)
    print(
        f"[rag] learner_role={learner_role} "
        f"role_profile={profile[:160]}",
        flush=True,
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def _optional_str(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def evidence_from_chunks(
    rows: list[tuple[KnowledgeChunk, KnowledgeDocument]],
    corpus_version: str | None = None,
) -> tuple[list[Claim], list[Citation]]:
    claims: list[Claim] = []
    citations: list[Citation] = []
    for index, (chunk, doc) in enumerate(rows, start=1):
        quote = _evidence_text(chunk)
        if not quote:
            continue
        citation_id = f"cit{index}"
        fmt = (getattr(doc, "file_format", None) or "").lower().lstrip(".")
        page_value = chunk.page
        if fmt == "docx" and page_value is None:
            page_str = None
        else:
            page_str = _optional_str(page_value)
        citation = Citation(
            id=citation_id,
            chunk_id=chunk.id,
            section=_optional_str(chunk.section),
            page=page_str,
            quote=quote,
            document_title=_optional_str(getattr(doc, "title", None)),
            document_id=_optional_str(getattr(doc, "id", None)),
            version=_optional_str(getattr(doc, "version", None)),
            status=_optional_str(getattr(doc, "status", None)),
            quote_check="passed",
            document_class=_optional_str(getattr(doc, "document_class", None)),
            document_version=_optional_str(getattr(doc, "version", None)),
            corpus_version=_optional_str(corpus_version) or _optional_str(
                getattr(chunk, "corpus_version", None)
            ),
            source_element_ids=list(getattr(chunk, "source_element_ids", None) or []) or None,
        )
        citations.append(citation)
        claims.append(
            Claim(
                claim_id=f"c{index}",
                claim_type=claim_type_for(doc.document_class),
                citation_ids=[citation_id],
                text=quote,
                citations=[citation],
            )
        )
    return claims, citations


def _call_llm(messages: list[dict[str, str]]) -> AskResponse | None:
    """Внешний LLM не подключён: ключ/модель не заданы или клиента нет."""
    if not (LLM_API_KEY and LLM_MODEL):
        return None
    return None


def compose_rag_answer(
    question: str,
    rows: list[tuple[KnowledgeChunk, KnowledgeDocument]],
    *,
    approved_count: int = 0,
    draft_count: int = 0,
    learner_role: str = "unknown",
    allowed_corpus_version: str = "",
) -> AskResponse:
    context = format_context(rows)
    role_known = learner_role in ROLE_PROFILES
    role_profile = ROLE_PROFILES.get(learner_role, DEFAULT_ROLE_PROFILE)
    messages = build_llm_messages(
        question,
        context,
        learner_role=learner_role,
        allowed_corpus_version=allowed_corpus_version,
        role_profile=role_profile,
    )
    demo = is_demo_corpus(
        rows, approved_count=approved_count, draft_count=draft_count
    )

    extra_limits = [] if role_known else [LIMITATION_UNKNOWN_ROLE]

    if not rows:
        return AskResponse(
            answer_status="basis_not_found",
            answer=NO_BASIS,
            claims=[],
            citations=[],
            limitations=base_limitations(no_basis=True, demo=demo) + extra_limits,
            review_required=True,
            review_reason="no_basis",
            corpus_version=allowed_corpus_version or None,
        )

    llm_response = _call_llm(messages)
    if llm_response is not None:
        return llm_response

    supported = rows_with_question_overlap(question, rows)
    claims, citations = evidence_from_chunks(supported, allowed_corpus_version or None)
    if not claims:
        return AskResponse(
            answer_status="basis_not_found",
            answer=NO_BASIS,
            claims=[],
            citations=[],
            limitations=base_limitations(no_basis=True, demo=demo) + extra_limits,
            review_required=True,
            review_reason="no_basis",
            corpus_version=allowed_corpus_version or None,
        )

    return AskResponse(
        answer_status="answered",
        answer_mode="extractive",
        answer=claims[0].text,
        claims=claims,
        citations=citations,
        limitations=base_limitations(demo=demo) + extra_limits + [LIMITATION_EXTRACTIVE],
        review_required=False,
        corpus_version=allowed_corpus_version or None,
    )
