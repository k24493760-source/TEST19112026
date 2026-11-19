"""Шаг 2.8. Golden Q&A и учебный банк. most_db_test / tests/_tmp."""
from __future__ import annotations

from tests import isolation

isolation.activate()
from app.db import Base
from app.models import (
    AuditEvent,
    DocumentElement,
    DocumentParseRun,
    KnowledgeChunk,
    KnowledgeDocument,
    LearningModule,
    QaEvalSnapshot,
    Rubric,
    generate_uuid,
)
from app.services.qa_approval import apply_qa_expert_decision, submit_qa_review
from app.services.approval import ApprovalError
from app.services.qa_generator import ChunkContext, GenerateRequest, StubQaGenerator, is_contentful
from app.services.qa_scoring import score_closed
from app.services.qa_service import (
    QaServiceError,
    create_candidate,
    freeze_eval,
    invalidate_for_chunk,
    revise_candidate,
)
from app.services.qa_validation import validate_card
from app.services.search import retrieve_chunks
from tests.isolation import assert_prod_corpus_untouched, assert_test_database

assert_test_database()
Base.metadata.create_all(bind=isolation.engine)

EVIDENCE = (
    "Уведомление уполномоченного органа направляется не позднее десяти рабочих дней, "
    "за исключением случая, предусмотренного частью 2."
)


def _chunk(text: str = EVIDENCE, page: int | None = 1, approved: bool = True) -> tuple[str, str]:
    db = isolation.SessionLocal()
    try:
        doc = KnowledgeDocument(
            title=f"QA src {generate_uuid()[:8]}",
            document_class="Нормативный акт",
            owner="qa-bank",
            version="1.0",
            status="approved" if approved else "draft",
        )
        db.add(doc)
        db.flush()
        run = DocumentParseRun(
            document_id=doc.id,
            parser_version="qa-test",
            status="parsed",
        )
        db.add(run)
        db.flush()
        el = DocumentElement(
            id=generate_uuid(),
            document_id=doc.id,
            parse_run_id=run.id,
            element_type="paragraph",
            order_index=0,
            text_original=text,
            text_normalized=text,
            page_number=page,
        )
        db.add(el)
        db.flush()
        chunk = KnowledgeChunk(
            document_id=doc.id,
            text=text,
            evidence_text=text,
            section="Статья 8",
            page=page,
            approval_status="approved" if approved else "draft",
            document_version="1.0",
            source_element_ids=[el.id],
        )
        db.add(chunk)
        db.commit()
        return doc.id, chunk.id, el.id
    finally:
        db.close()


def test_create_and_bind():
    _, cid, eid = _chunk()
    db = isolation.SessionLocal()
    try:
        card, report = create_candidate(
            db,
            question_text="В какой срок направляется уведомление?",
            reference_answer="Не позднее десяти рабочих дней.",
            chunk_ids=[cid],
            quotes=[{"chunk_id": cid, "quote": "не позднее десяти рабочих дней", "source_element_ids": [eid]}],
        )
        assert report.ok, [i.code for i in report.issues]
        assert card.sources and card.sources[0].chunk_id == cid
        assert card.golden_status == "none" and card.learning_status == "none"
        assert card.generation_status == "validated"
    finally:
        db.close()


def test_multi_chunk_and_bad_quote():
    _, a, ea = _chunk()
    _, b, _eb = _chunk("Карта пробелов компетенций слушателя без кадровых выводов. " * 2, page=None)
    db = isolation.SessionLocal()
    try:
        card, report = create_candidate(
            db,
            question_text="Как связаны срок и карта пробелов?",
            reference_answer="Это разные нормы; ответ только по найденным фрагментам.",
            chunk_ids=[a, b],
            quotes=[{"chunk_id": a, "quote": "не позднее десяти рабочих дней", "source_element_ids": [ea]}],
            golden_test_kind="multi",
        )
        assert len(card.sources) == 2
        assert report.ok
        bad, bad_report = create_candidate(
            db,
            question_text="Выдуманная цитата?",
            reference_answer="нет",
            chunk_ids=[a],
            quotes=[{"chunk_id": a, "quote": "этой фразы нет в источнике вообще"}],
        )
        assert not bad_report.ok
        assert any(i.code == "quote_mismatch" for i in bad_report.issues)
    finally:
        db.close()


def test_duplicate_and_no_basis():
    _, cid, eid = _chunk()
    db = isolation.SessionLocal()
    try:
        kwargs = dict(
            question_text="Срок уведомления?",
            reference_answer="Десять рабочих дней.",
            chunk_ids=[cid],
            quotes=[{"chunk_id": cid, "quote": "десяти рабочих дней", "source_element_ids": [eid]}],
        )
        create_candidate(db, **kwargs)
        try:
            create_candidate(db, **kwargs)
            raise AssertionError("дубль должен быть отклонён")
        except QaServiceError as exc:
            assert exc.code == "exact_duplicate"
        nb, report = create_candidate(
            db,
            question_text="Каков диагноз работника Иванова?",
            reference_answer="",
            chunk_ids=[],
            quotes=[],
            golden_test_kind="no_basis",
            expected_answer_status="basis_not_found",
        )
        assert nb.expected_answer_status == "basis_not_found"
        assert report.ok
    finally:
        db.close()


def test_independent_approval_and_reject():
    _, cid, eid = _chunk()
    db = isolation.SessionLocal()
    try:
        module = LearningModule(title="Сроки уведомления", competencies=["comp-notify"], status="approved")
        rubric = Rubric(
            version="1.0",
            competency="comp-notify",
            required_elements=["срок", "рабочих дней"],
            status="approved",
        )
        db.add_all([module, rubric])
        db.commit()
        card, _ = create_candidate(
            db,
            question_text="Срок направления уведомления?",
            reference_answer="Не позднее десяти рабочих дней.",
            chunk_ids=[cid],
            quotes=[{"chunk_id": cid, "quote": "не позднее десяти рабочих дней", "source_element_ids": [eid]}],
            competency="comp-notify",
            module_id=module.id,
            rubric_id=rubric.id,
            answer_type="closed",
            answer_key={"correct": "10 рабочих дней"},
        )
        g = submit_qa_review(db, card.qa_id, "golden", "admin-1", "admin")
        apply_qa_expert_decision(
            db, g, expert_id="test-expert-lawyer", expert_role="lawyer",
            decision="approve", comment="golden ok",
        )
        db.commit()
        db.refresh(card)
        assert card.golden_status == "approved"
        assert card.learning_status == "pending" or card.learning_status == "none"
        # golden не утверждает обучение
        assert card.learning_status != "approved"
        l = submit_qa_review(db, card.qa_id, "learning", "admin-1", "admin")
        apply_qa_expert_decision(
            db, l, expert_id="test-expert-methodist", expert_role="methodist",
            decision="approve", comment="learning ok",
        )
        db.commit()
        db.refresh(card)
        assert card.golden_status == "approved" and card.learning_status == "approved"
        score, why = score_closed(card, "10 рабочих дней")
        assert score == 1.0 and why == "closed_match"
        assert score_closed(card, "месяц")[0] == 0.0
        try:
            apply_qa_expert_decision(
                db, l, expert_id="admin-1", expert_role="admin",
                decision="approve", comment="no",
            )
            raise AssertionError("админ не утверждает")
        except ApprovalError as exc:
            assert exc.code in {"forbidden", "admin_cannot_replace_expert"}
    finally:
        db.close()

    _, cid2, eid2 = _chunk()
    db = isolation.SessionLocal()
    try:
        card, _ = create_candidate(
            db,
            question_text="Отклоняемый вопрос?",
            reference_answer="нет",
            chunk_ids=[cid2],
            quotes=[{"chunk_id": cid2, "quote": "десяти рабочих дней", "source_element_ids": [eid2]}],
        )
        rev = submit_qa_review(db, card.qa_id, "golden", "admin-1", "admin")
        apply_qa_expert_decision(
            db, rev, expert_id="test-expert-lawyer", expert_role="lawyer",
            decision="reject", comment="неточно",
        )
        db.commit()
        db.refresh(card)
        assert card.golden_status == "rejected"
        events = db.query(AuditEvent).filter(AuditEvent.object == card.qa_id).all()
        assert any(e.action == "qa_expert_resolve" for e in events)
    finally:
        db.close()


def test_revise_invalidate_history_generator_rag():
    _, cid, eid = _chunk()
    db = isolation.SessionLocal()
    try:
        card, _ = create_candidate(
            db,
            question_text="Срок?",
            reference_answer="10 дней.",
            chunk_ids=[cid],
            quotes=[{"chunk_id": cid, "quote": "десяти рабочих дней", "source_element_ids": [eid]}],
            corpus_version="corpus-1",
        )
        snap = freeze_eval(db, card, {"hit": True, "corpus_version": "corpus-1"})
        g = submit_qa_review(db, card.qa_id, "golden", "admin-1", "admin")
        apply_qa_expert_decision(
            db, g, expert_id="test-expert-lawyer", expert_role="lawyer",
            decision="approve", comment="ok",
        )
        db.commit()
        touched = invalidate_for_chunk(db, cid)
        db.refresh(card)
        assert card.qa_id in touched
        assert card.golden_status == "stale"
        assert db.get(QaEvalSnapshot, snap.id).payload["hit"] is True
        new, _ = revise_candidate(
            db,
            card.qa_id,
            question_text="В какой срок направляется уведомление уполномоченного органа?",
            reference_answer="Не позднее десяти рабочих дней.",
            chunk_ids=[cid],
            quotes=[{"chunk_id": cid, "quote": "не позднее десяти рабочих дней", "source_element_ids": [eid]}],
        )
        db.refresh(card)
        assert card.generation_status == "superseded"
        assert new.parent_qa_id == card.qa_id
        assert new.golden_status == "none"
        kept = db.get(QaEvalSnapshot, snap.id)
        assert kept.payload["hit"] is True
        assert db.get(QaEvalSnapshot, snap.id).payload["hit"] is True
        gen = StubQaGenerator().generate(
            GenerateRequest(primary=ChunkContext(cid, "коротко", None, 1, "d", "1.0", "Нормативный акт"))
        )
        assert gen.skipped_reason
        assert not is_contentful("стр. 1")
        found = retrieve_chunks(db, "срок уведомления")
        assert found.rows == [] or found.error_code in {"no_published_release", "search_error", None}
    finally:
        db.close()
    assert_prod_corpus_untouched()


def test_learning_methodist_only():
    _, cid, eid = _chunk()
    db = isolation.SessionLocal()
    try:
        card, _ = create_candidate(
            db,
            question_text="Учебный?",
            reference_answer="да",
            chunk_ids=[cid],
            quotes=[{"chunk_id": cid, "quote": "десяти рабочих дней", "source_element_ids": [eid]}],
            competency="comp-notify",
            answer_type="closed",
            answer_key={"correct": "да"},
        )
        rev = submit_qa_review(db, card.qa_id, "learning", "admin-1", "admin")
        try:
            apply_qa_expert_decision(
                db, rev, expert_id="test-expert-lawyer", expert_role="lawyer",
                decision="approve", comment="нет",
            )
            raise AssertionError("юрист не утверждает учебный банк")
        except ApprovalError as exc:
            assert exc.code == "role_mismatch"
    finally:
        db.close()


if __name__ == "__main__":
    tests = [
        test_create_and_bind,
        test_multi_chunk_and_bad_quote,
        test_duplicate_and_no_basis,
        test_independent_approval_and_reject,
        test_revise_invalidate_history_generator_rag,
        test_learning_methodist_only,
    ]
    for fn in tests:
        fn()
        print("OK", fn.__name__)
    print(f"passed {len(tests)}/{len(tests)}")
