"""Шаг 2.9Б. Адаптер LLM для Golden Q&A. Без настоящей модели и без most_db."""
from __future__ import annotations

import json

from tests import isolation

isolation.activate()
from app.db import Base
from app.models import (
    AuditEvent,
    DocumentElement,
    DocumentParseRun,
    KnowledgeChunk,
    KnowledgeDocument,
    QaCard,
    QaGenerationJob,
    generate_uuid,
)
from app.services.qa_generator import ChunkContext, GenerateRequest, StubQaGenerator, default_generator
from app.services.qa_jobs import enqueue_for_chunk, persist_candidates, run_job
from app.services.qa_llm_client import (
    HttpQaLlmClient,
    LlmQaGenerator,
    LlmTransportError,
    MockQaLlmClient,
)
from app.services.qa_llm_config import LlmConfigError, assert_url_allowed, resolved_llm_config
from app.services.qa_llm_parse import parse_and_filter
from tests.isolation import assert_prod_corpus_untouched, assert_test_database

assert_test_database()
Base.metadata.create_all(bind=isolation.engine)

EVIDENCE = (
    "Уведомление уполномоченного органа направляется не позднее десяти рабочих дней, "
    "за исключением случая, предусмотренного частью 2."
)


def _chunk():
    db = isolation.SessionLocal()
    try:
        doc = KnowledgeDocument(
            title=f"LLM src {generate_uuid()[:8]}",
            document_class="Нормативный акт",
            owner="qa-llm",
            version="1.0",
            status="approved",
        )
        db.add(doc)
        db.flush()
        run = DocumentParseRun(document_id=doc.id, parser_version="qa-llm", status="parsed")
        db.add(run)
        db.flush()
        el = DocumentElement(
            id=generate_uuid(),
            document_id=doc.id,
            parse_run_id=run.id,
            element_type="paragraph",
            order_index=0,
            text_original=EVIDENCE,
            text_normalized=EVIDENCE,
            page_number=1,
        )
        db.add(el)
        db.flush()
        chunk = KnowledgeChunk(
            document_id=doc.id,
            text=EVIDENCE,
            evidence_text=EVIDENCE,
            section="Статья 8",
            page=1,
            approval_status="approved",
            document_version="1.0",
            chunking_version="c1",
            source_element_ids=[el.id],
        )
        db.add(chunk)
        db.commit()
        return doc.id, chunk.id
    finally:
        db.close()


def _req(chunk_id: str) -> GenerateRequest:
    return GenerateRequest(
        primary=ChunkContext(
            chunk_id=chunk_id,
            evidence_text=EVIDENCE,
            section_path="Статья 8",
            page=1,
            document_id="doc",
            document_version="1.0",
            document_class="Нормативный акт",
        )
    )


def _item(chunk_id: str, question: str, quote: str, **extra) -> dict:
    row = {
        "question_text": question,
        "reference_answer": extra.pop("reference_answer", quote),
        "question_type": extra.pop("question_type", "factual"),
        "difficulty": extra.pop("difficulty", "easy"),
        "source_chunk_ids": extra.pop("source_chunk_ids", [chunk_id]),
        "evidence_quotes": extra.pop("evidence_quotes", [{"chunk_id": chunk_id, "quote": quote}]),
        "golden_test_kind": extra.pop("golden_test_kind", "single"),
        "expected_answer_status": extra.pop("expected_answer_status", "answered"),
        "warnings": extra.pop("warnings", []),
        "ambiguity_flags": extra.pop("ambiguity_flags", []),
    }
    row.update(extra)
    return row


def test_three_candidates():
    _, cid = _chunk()
    payload = {
        "candidates": [
            _item(cid, "В какой срок направляется уведомление?", "не позднее десяти рабочих дней"),
            _item(
                cid,
                "Какое исключение указано в фрагменте?",
                "за исключением случая, предусмотренного частью 2",
                difficulty="medium",
                question_type="exception",
                golden_test_kind="exception",
            ),
            _item(cid, "Куда направляется уведомление?", "уполномоченного органа"),
        ]
    }
    result = parse_and_filter(json.dumps(payload, ensure_ascii=False), _req(cid))
    assert len(result.candidates) == 3


def test_one_question_and_empty():
    _, cid = _chunk()
    one = parse_and_filter(
        json.dumps({"candidates": [_item(cid, "Срок?", "десяти рабочих дней")]}),
        _req(cid),
    )
    assert len(one.candidates) == 1
    empty = parse_and_filter('{"candidates":[],"skipped_reason":"insufficient_basis"}', _req(cid))
    assert empty.candidates == []
    assert empty.skipped_reason == "insufficient_basis"


def test_invalid_json_and_bad_quote_and_chunk():
    _, cid = _chunk()
    req = _req(cid)
    try:
        parse_and_filter("{not-json", req)
        raise AssertionError("json")
    except LlmTransportError as exc:
        assert exc.code == "invalid_json"
    invented = parse_and_filter(
        json.dumps({"candidates": [_item(cid, "Срок?", "три года тюрьмы")]}),
        req,
    )
    assert invented.skipped_reason == "all_rejected"
    wrong = parse_and_filter(
        json.dumps({"candidates": [_item("missing-id", "Срок?", "десяти рабочих дней")]}),
        req,
    )
    assert wrong.skipped_reason == "all_rejected"


def test_duplicates_and_limit():
    _, cid = _chunk()
    dup = parse_and_filter(
        json.dumps(
            {
                "candidates": [
                    _item(cid, "Один и тот же вопрос?", "десяти рабочих дней"),
                    _item(cid, "Один и тот же вопрос?", "уполномоченного органа"),
                ]
            }
        ),
        _req(cid),
    )
    assert len(dup.candidates) == 1
    many = [
        _item(cid, f"Вопрос номер {i}?", "десяти рабочих дней") for i in range(4)
    ]
    limited = parse_and_filter(json.dumps({"candidates": many}), _req(cid))
    assert len(limited.candidates) == 3
    assert "candidate_limit" in limited.candidates[0].warnings


def test_timeout_and_unavailable_jobs():
    _, cid = _chunk()
    db = isolation.SessionLocal()
    try:
        chunk = db.get(KnowledgeChunk, cid)
        job = enqueue_for_chunk(db, chunk)
        db.commit()
        gen = LlmQaGenerator(MockQaLlmClient(errors=[LlmTransportError("timeout", "t")]))
        out = run_job(db, job.id, generator=gen)
        assert out.status == "failed" and out.error_code == "timeout"
        job2 = QaGenerationJob(
            document_id=chunk.document_id,
            chunk_id=cid,
            status="queued",
            idempotency_key=generate_uuid(),
            prompt_version="2026-11-17-qa-v1",
            adapter="mock",
            document_version="1.0",
        )
        db.add(job2)
        db.commit()
        gen2 = LlmQaGenerator(MockQaLlmClient(errors=[LlmTransportError("server_unavailable", "down")]))
        out2 = run_job(db, job2.id, generator=gen2)
        assert out2.status == "failed" and out2.error_code == "server_unavailable"
    finally:
        db.close()


def test_retry_idempotent_persist_versions_no_auto_approve():
    _, cid = _chunk()
    payload = json.dumps(
        {"candidates": [_item(cid, "Срок направления уведомления?", "не позднее десяти рабочих дней")]},
        ensure_ascii=False,
    )
    db = isolation.SessionLocal()
    try:
        chunk = db.get(KnowledgeChunk, cid)
        job = enqueue_for_chunk(db, chunk)
        again = enqueue_for_chunk(db, chunk)
        assert job.id == again.id
        db.commit()
        fail_gen = LlmQaGenerator(MockQaLlmClient(errors=[LlmTransportError("timeout", "t")]))
        run_job(db, job.id, generator=fail_gen)
        db.refresh(job)
        assert job.status == "failed" and job.attempt == 1
        ok_gen = LlmQaGenerator(MockQaLlmClient(payloads=[payload]))
        run_job(db, job.id, generator=ok_gen)
        db.refresh(job)
        assert job.status == "done"
        assert job.n_persisted == 1
        assert job.prompt_version
        cards = db.query(QaCard).filter(QaCard.generation_prompt_version == job.prompt_version).all()
        saved = [c for c in cards if c.sources and c.sources[0].chunk_id == cid]
        assert saved
        card = saved[-1]
        assert card.golden_status == "none"
        assert card.learning_status == "none"
        assert card.generation_status in {"validated", "draft"}
        assert card.document_version == "1.0"
        run_job(db, job.id, generator=ok_gen)
        db.refresh(job)
        assert job.status == "done"
        n = db.query(QaCard).filter(QaCard.question_text == "Срок направления уведомления?").count()
        assert n == 1
        events = db.query(AuditEvent).filter(AuditEvent.object == job.id).all()
        blob = json.dumps([e.payload for e in events], ensure_ascii=False)
        assert EVIDENCE not in blob
        assert "диагноз" not in blob.lower()
    finally:
        db.close()


def test_empty_answer_rejected_and_stub_default():
    _, cid = _chunk()
    empty_ans = parse_and_filter(
        json.dumps({"candidates": [_item(cid, "Срок?", "десяти рабочих дней", reference_answer="")]}),
        _req(cid),
    )
    assert empty_ans.skipped_reason == "all_rejected"
    gen = default_generator()
    assert isinstance(gen, StubQaGenerator)
    cfg = resolved_llm_config()
    assert cfg["enabled"] is False
    try:
        assert_url_allowed("https://api.openai.com/v1", ["127.0.0.1", "localhost"])
        raise AssertionError("host")
    except LlmConfigError as exc:
        assert exc.code == "llm_host_forbidden"
    try:
        HttpQaLlmClient({"enabled": False, "base_url": "http://127.0.0.1:8080", "allowed_hosts": ["127.0.0.1"]})
        raise AssertionError("disabled")
    except LlmConfigError as exc:
        assert exc.code == "llm_disabled"


def test_persist_does_not_approve():
    _, cid = _chunk()
    db = isolation.SessionLocal()
    try:
        chunk = db.get(KnowledgeChunk, cid)
        result = parse_and_filter(
            json.dumps({"candidates": [_item(cid, "Кто уведомляется?", "уполномоченного органа")]}),
            _req(cid),
        )
        saved = persist_candidates(db, candidates=result.candidates, chunk=chunk, cfg={"model": "mock", "prompt_version": "2026-11-17-qa-v1"})
        assert saved[0].golden_status == "none"
        assert saved[0].learning_status == "none"
        reviews = [e for e in db.query(AuditEvent).all() if e.action == "qa_create"]
        assert reviews
    finally:
        db.close()
    assert_prod_corpus_untouched()


if __name__ == "__main__":
    tests = [
        test_three_candidates,
        test_one_question_and_empty,
        test_invalid_json_and_bad_quote_and_chunk,
        test_duplicates_and_limit,
        test_timeout_and_unavailable_jobs,
        test_retry_idempotent_persist_versions_no_auto_approve,
        test_empty_answer_rejected_and_stub_default,
        test_persist_does_not_approve,
    ]
    for fn in tests:
        fn()
        print("OK", fn.__name__)
    print(f"{len(tests)}/{len(tests)} PASS")
