"""Разбор и отбраковка ответа LLM. Выход модели — недоверенный вход."""
from __future__ import annotations

import json
import re

from app.config import BASE_DIR
from app.services.qa_generator import GenerateRequest, GenerateResult, GeneratedCandidate, load_qa_generation_config
from app.services.qa_llm_client import LlmTransportError
from app.services.verification import quote_contained

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
ALLOWED_TYPES = {"factual", "exception", "closed", "free"}
ALLOWED_KIND = {"single", "multi", "exception", "no_basis", "conflict"}
ALLOWED_STATUS = {"answered", "basis_not_found"}
SCHEMA_PATH = BASE_DIR / "contracts" / "qa_llm_response.schema.json"


def load_response_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def extract_json_object(text: str) -> dict:
    blob = (text or "").strip()
    blob = _FENCE.sub("", blob).strip()
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        start = blob.find("{")
        end = blob.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(blob[start : end + 1])
            except json.JSONDecodeError as inner:
                raise LlmTransportError("invalid_json", "Ответ LLM не является JSON.") from inner
        else:
            raise LlmTransportError("invalid_json", "Ответ LLM не является JSON.") from exc
    if not isinstance(data, dict):
        raise LlmTransportError("invalid_json", "Корень JSON должен быть объектом.")
    return data


def validate_schema(data: dict) -> None:
    schema = load_response_schema()
    required_root = list(schema.get("required") or ["candidates"])
    for key in required_root:
        if key not in data:
            raise LlmTransportError("invalid_json", f"Нет поля {key}.")
    cands = data["candidates"]
    if not isinstance(cands, list):
        raise LlmTransportError("invalid_json", "candidates должен быть массивом.")
    max_items = int(((schema.get("properties") or {}).get("candidates") or {}).get("maxItems") or 3)
    if len(cands) > max_items:
        # лишние отсекаются позже; схема фиксирует максимум
        pass
    item_schema = ((schema.get("properties") or {}).get("candidates") or {}).get("items") or {}
    required = tuple(item_schema.get("required") or (
        "question_text",
        "reference_answer",
        "question_type",
        "difficulty",
        "source_chunk_ids",
        "evidence_quotes",
    ))
    for item in cands:
        if not isinstance(item, dict):
            raise LlmTransportError("invalid_json", "Элемент candidates не объект.")
        for key in required:
            if key not in item:
                raise LlmTransportError("invalid_json", f"Нет поля {key}.")


def _norm_q(text: str) -> str:
    return " ".join((text or "").lower().split())


def parse_and_filter(raw_text: str, request: GenerateRequest) -> GenerateResult:
    cfg = load_qa_generation_config()
    limit = int(cfg.get("max_questions_per_chunk") or 3)
    data = extract_json_object(raw_text)
    validate_schema(data)
    raw_list = data.get("candidates") or []
    skipped = data.get("skipped_reason")
    if not raw_list:
        return GenerateResult(skipped_reason=skipped or "empty_result")

    overflow = False
    if len(raw_list) > limit:
        overflow = True
        raw_list = raw_list[:limit]

    evidence = request.primary.evidence_text or ""
    expected_id = request.primary.chunk_id
    seen: set[str] = set()
    kept: list[GeneratedCandidate] = []
    for item in raw_list:
        question = str(item.get("question_text") or "").strip()
        answer = str(item.get("reference_answer") or "").strip()
        qtype = str(item.get("question_type") or "factual").strip()
        kind = str(item.get("golden_test_kind") or "single").strip()
        status = str(item.get("expected_answer_status") or "answered").strip()
        difficulty = str(item.get("difficulty") or "").strip() or None
        ids = [str(x) for x in (item.get("source_chunk_ids") or []) if str(x).strip()]
        quotes_in = item.get("evidence_quotes") or []
        warnings = [str(w) for w in (item.get("warnings") or [])]
        flags = [str(w) for w in (item.get("ambiguity_flags") or [])]
        no_basis = kind == "no_basis" or status == "basis_not_found"

        if qtype not in ALLOWED_TYPES:
            continue
        if kind not in ALLOWED_KIND:
            continue
        if status not in ALLOWED_STATUS:
            continue
        if not question:
            continue
        if not no_basis and not answer:
            continue
        if expected_id not in ids:
            continue
        if any(i != expected_id for i in ids):
            continue
        if no_basis and quotes_in:
            continue

        quotes: list[dict] = []
        quotes_ok = True
        if not no_basis:
            if not quotes_in:
                quotes_ok = False
            for q in quotes_in:
                if not isinstance(q, dict):
                    quotes_ok = False
                    break
                qid = str(q.get("chunk_id") or expected_id)
                quote = str(q.get("quote") or "").strip()
                if qid != expected_id or not quote or not quote_contained(quote, evidence):
                    quotes_ok = False
                    break
                quotes.append({"chunk_id": qid, "quote": quote})
        if not quotes_ok:
            continue
        key = _norm_q(question)
        if key in seen:
            continue
        seen.add(key)
        if overflow:
            warnings = list(warnings) + ["candidate_limit"]
        kept.append(
            GeneratedCandidate(
                question_text=question,
                reference_answer=answer,
                quotes=quotes,
                question_type=qtype,
                golden_test_kind=kind,
                expected_answer_status=status,
                difficulty=difficulty,
                ambiguity_flags=flags,
                warnings=warnings,
                source_chunk_ids=[expected_id],
            )
        )

    if not kept:
        return GenerateResult(skipped_reason=skipped or "all_rejected")
    return GenerateResult(candidates=kept, skipped_reason=None)
