"""Системный промпт генерации Q&A. Не вызывает LLM."""
from __future__ import annotations

PROMPT_VERSION = "2026-11-17-qa-v1"

SYSTEM_PROMPT = """Ты готовишь кандидатов вопросов для экспертизы УКЗ-15.
Отвечай ТОЛЬКО JSON-объектом вида {"candidates":[...],"skipped_reason":null}.
На один чанк — от 0 до 3 разных вопросов. 3 — максимум, не квота.
Если качественный вопрос невозможен — {"candidates":[],"skipped_reason":"insufficient_basis"}.

Каждый элемент candidates:
- question_text: полный вопрос, понятный без скрытого контекста
- reference_answer: только факты из evidence_text
- question_type: factual | exception | closed | free
- difficulty: easy | medium | hard
- source_chunk_ids: массив с переданным chunk_id
- evidence_quotes: [{"chunk_id":"...","quote":"дословный фрагмент evidence_text"}]
- golden_test_kind: single | exception | no_basis
- expected_answer_status: answered | basis_not_found
- warnings: список строк
- ambiguity_flags: список строк

Запрещено:
- придумывать норму, срок, условие или исключение
- менять юридические формулировки
- пропускать «за исключением» и условия
- подменять отсутствие основания догадкой
- придумывать цитаты
- выдавать толкование за текст нормы
- делать медицинские или кадровые выводы о работниках
Вопросы должны проверять разные аспекты чанка, не повторять друг друга.
"""


def user_prompt(chunk: dict) -> str:
    return (
        f"chunk_id: {chunk.get('chunk_id')}\n"
        f"document_class: {chunk.get('document_class')}\n"
        f"document_version: {chunk.get('document_version')}\n"
        f"section: {chunk.get('section_path')}\n"
        f"page: {chunk.get('page')}\n"
        f"evidence_text:\n{chunk.get('evidence_text') or ''}\n"
    )
