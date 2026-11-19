"""Шаг 2.5А: экспертиза и публикация корпуса. Только most_db_test."""
from __future__ import annotations

import uuid

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from tests import isolation

isolation.activate()
from app.config import DATABASE_URL
from app.db import Base
from app.models import (
    AuditEvent,
    ChunkEmbedding,
    CorpusRelease,
    CorpusReleaseItem,
    ExpertReview,
    KnowledgeChunk,
    KnowledgeDocument,
)
from app.services.approval import (
    ApprovalError,
    apply_expert_decision,
    missing_approval_roles,
    submit_for_review,
)
from app.services.chunking import chunk_parsed_document
from app.services.embedding_pipeline import vectorize_draft_chunks
from app.services.intake import intake_document
from app.services.publication import (
    PublicationError,
    current_published_release,
    publish_corpus,
    retrieval_contract,
)
from app.services.search import retrieve_chunks
from tests.isolation import assert_prod_corpus_untouched, assert_test_database
from tests.test_intake import _cleanup

assert_test_database()
Base.metadata.create_all(bind=isolation.engine)

NORM = (
    "# Статья 8\n\n"
    "Уведомление уполномоченного органа направляется не позднее десяти рабочих дней.\n"
)
CLINICAL = (
    "# Клинико-организационный порядок оказания помощи\n\n"
    "Учебный модуль описывает организационный порядок допуска к программе обучения.\n"
)
LONG = NORM + ("Положение описывает порядок цитирования утверждённого источника. " * 40)


def _intake(body: str, cls: str = "Нормативный акт"):
    db = isolation.SessionLocal()
    try:
        return intake_document(
            db,
            body.encode("utf-8"),
            f"appr-{uuid.uuid4().hex[:8]}.md",
            title=f"Approval {uuid.uuid4().hex[:8]}",
            document_class=cls,
            corpus_source="database_edu_lectcii",
            owner="qa-approval",
            actor_id="qa-approval",
            content_type="text/markdown",
        )
    finally:
        db.close()


def _prepare(body: str, cls: str = "Нормативный акт", embed: bool = True) -> str:
    item = _intake(body, cls)
    db = isolation.SessionLocal()
    try:
        chunk_parsed_document(db, item.document_id)
        if embed:
            vectorize_draft_chunks(db, document_id=item.document_id)
    finally:
        db.close()
    return item.document_id


def _submit(doc_id: str, role: str = "admin"):
    db = isolation.SessionLocal()
    try:
        return submit_for_review(db, doc_id, "actor-1", role)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _resolve(doc_id: str, role: str, decision: str = "approve"):
    db = isolation.SessionLocal()
    try:
        review = (
            db.query(ExpertReview)
            .filter(
                ExpertReview.object_id == doc_id,
                ExpertReview.expert_role == role,
                ExpertReview.status.in_(("pending", "open")),
            )
            .first()
        )
        assert review is not None, f"нет задачи для {role}"
        result = apply_expert_decision(
            db,
            review,
            expert_id=f"expert-{role}",
            expert_role=role,
            decision=decision,
            comment="проверка комплекта",
        )
        db.commit()
        return result
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _publish(doc_ids: list[str], role: str = "admin"):
    db = isolation.SessionLocal()
    try:
        return publish_corpus(db, doc_ids, "admin-1", role, notes="test")
    finally:
        db.close()


def test_authorized_expert_approves():
    doc_id = _prepare(NORM)
    try:
        submitted = _submit(doc_id)
        assert submitted.status == "ready_for_review"
        result = _resolve(doc_id, "lawyer")
        assert result.status == "approved"
        assert not result.missing_roles
        db = isolation.SessionLocal()
        try:
            doc = db.get(KnowledgeDocument, doc_id)
            assert doc.status == "approved"
            assert all(c.approval_status == "approved" for c in doc.chunks)
        finally:
            db.close()
    finally:
        _cleanup(doc_id)


def test_wrong_role_rejected():
    doc_id = _prepare(NORM)
    try:
        _submit(doc_id)
        try:
            _resolve(doc_id, "methodist")
            raise AssertionError("методист не должен закрывать задачу юриста")
        except AssertionError as exc:
            if "не должен" in str(exc):
                raise
        db = isolation.SessionLocal()
        try:
            review = (
                db.query(ExpertReview)
                .filter(ExpertReview.object_id == doc_id, ExpertReview.expert_role == "lawyer")
                .one()
            )
            try:
                apply_expert_decision(
                    db,
                    review,
                    expert_id="m1",
                    expert_role="methodist",
                    decision="approve",
                    comment="нет",
                )
                raise AssertionError("ожидался role_mismatch")
            except ApprovalError as exc:
                assert exc.code == "role_mismatch"
            doc = db.get(KnowledgeDocument, doc_id)
            assert doc.status == "ready_for_review"
        finally:
            db.close()
    finally:
        _cleanup(doc_id)


def test_reject_document():
    doc_id = _prepare(NORM)
    try:
        _submit(doc_id)
        result = _resolve(doc_id, "lawyer", decision="reject")
        assert result.status == "rejected"
        db = isolation.SessionLocal()
        try:
            doc = db.get(KnowledgeDocument, doc_id)
            assert doc.status == "rejected"
            assert all(c.approval_status == "draft" for c in doc.chunks)
        finally:
            db.close()
    finally:
        _cleanup(doc_id)


def test_incomplete_expert_set():
    doc_id = _prepare(CLINICAL, "Учебный материал")
    try:
        submitted = _submit(doc_id)
        assert "methodist" in submitted.missing_roles
        assert "medical_methodologist" in submitted.missing_roles
        result = _resolve(doc_id, "methodist")
        assert result.status == "ready_for_review"
        assert "medical_methodologist" in result.missing_roles
        db = isolation.SessionLocal()
        try:
            doc = db.get(KnowledgeDocument, doc_id)
            assert doc.status != "approved"
            assert missing_approval_roles(db, doc) == ["medical_methodologist"]
        finally:
            db.close()
    finally:
        _cleanup(doc_id)


def test_publish_blocks_no_source_and_no_embedding_and_draft_and_conflict():
    doc_id = _prepare(NORM, embed=False)
    try:
        try:
            _publish([doc_id])
            raise AssertionError("draft не публикуется")
        except PublicationError as exc:
            assert exc.code == "preflight_failed"
            joined = " ".join(exc.errors)
            assert "draft_not_publishable" in joined or "document_not_approved" in joined
        _submit(doc_id)
        _resolve(doc_id, "lawyer")
        try:
            _publish([doc_id])
            raise AssertionError("без векторов нельзя")
        except PublicationError as exc:
            assert any("embeddings_missing" in e for e in exc.errors)
        db = isolation.SessionLocal()
        try:
            chunks = db.query(KnowledgeChunk).filter_by(document_id=doc_id).all()
            for chunk in chunks:
                chunk.approval_status = "draft"
            db.commit()
            vectorize_draft_chunks(db, document_id=doc_id)
            chunks = db.query(KnowledgeChunk).filter_by(document_id=doc_id).all()
            for chunk in chunks:
                chunk.approval_status = "approved"
                chunk.source_element_ids = []
            db.add(
                ExpertReview(
                    object_type="knowledge_document",
                    object_id=doc_id,
                    reason="source_conflict",
                    status="pending",
                    evidence={"note": "две редакции"},
                )
            )
            db.commit()
        finally:
            db.close()
        try:
            _publish([doc_id])
            raise AssertionError("конфликт и пустые связи должны блокировать")
        except PublicationError as exc:
            blob = " ".join(exc.errors)
            assert "no_source_elements" in blob
            assert "unresolved_source_conflict" in blob
    finally:
        _cleanup(doc_id)


def test_successful_publish_idempotent_audit_and_old_versions():
    doc_id = _prepare(LONG)
    extra = _prepare(NORM)
    try:
        _submit(doc_id)
        _resolve(doc_id, "lawyer")
        first = _publish([doc_id])
        assert first.status == "published"
        assert first.n_chunks >= 1
        assert first.n_embeddings >= 1
        again = _publish([doc_id])
        assert again.duplicate is True
        assert again.version == first.version
        _submit(extra)
        _resolve(extra, "lawyer")
        second = _publish([doc_id, extra])
        assert second.version != first.version
        db = isolation.SessionLocal()
        try:
            old = db.query(CorpusRelease).filter_by(version=first.version).one()
            assert old.status == "superseded"
            live = current_published_release(db)
            assert live.version == second.version
            old_items = db.query(CorpusReleaseItem).filter_by(release_id=old.id).count()
            assert old_items >= 1
            chunks = db.query(KnowledgeChunk).filter_by(document_id=doc_id).all()
            assert all(c.approval_status == "approved" for c in chunks)
            audits = (
                db.query(AuditEvent)
                .filter(AuditEvent.action.in_(("corpus_publish", "expert_resolve", "document_submit_review")))
                .all()
            )
            assert audits
            for event in audits:
                payload = event.payload or {}
                dumped = str(payload)
                assert "Уведомление уполномоченного" not in dumped
            working = db.query(CorpusRelease).filter_by(version="working-draft").one_or_none()
            if working:
                assert working.status == "draft"
        finally:
            db.close()
    finally:
        _cleanup(doc_id)
        _cleanup(extra)


def test_publish_error_keeps_previous():
    doc_id = _prepare(NORM)
    extra = _prepare(LONG)
    try:
        _submit(doc_id)
        _resolve(doc_id, "lawyer")
        first = _publish([doc_id])
        _submit(extra)
        _resolve(extra, "lawyer")
        import app.services.publication as pub

        def boom(*_a, **_k):
            raise RuntimeError("simulated publish fail")

        orig = pub._activate_published
        pub._activate_published = boom
        try:
            try:
                _publish([doc_id, extra])
                raise AssertionError("ожидался сбой публикации")
            except PublicationError as exc:
                assert exc.code == "publish_failed"
        finally:
            pub._activate_published = orig
        db = isolation.SessionLocal()
        try:
            live = current_published_release(db)
            assert live is not None
            assert live.version == first.version
            assert live.status == "published"
        finally:
            db.close()
    finally:
        _cleanup(doc_id)
        _cleanup(extra)


def test_prod_rag_and_most_db_untouched():
    doc_id = _prepare(NORM)
    try:
        _submit(doc_id)
        _resolve(doc_id, "lawyer")
        _publish([doc_id])
        assert retrieval_contract()["wired_to_retrieve_chunks"] is True
        db = isolation.SessionLocal()
        try:
            found = retrieve_chunks(db, "срок уведомления")
            _ = found.rows
        finally:
            db.close()
        url = DATABASE_URL
        if make_url(url).database == "most_db":
            engine = create_engine(
                "postgresql+psycopg://" + url[len("postgresql://") :]
                if url.startswith("postgresql://")
                else url
            )
            with engine.connect() as conn:
                n = conn.execute(
                    text("SELECT count(*) FROM knowledge_documents WHERE owner = 'qa-approval'")
                ).scalar()
                assert n == 0
                has_items = conn.execute(
                    text(
                        "SELECT 1 FROM information_schema.tables "
                        "WHERE table_name = 'corpus_release_items'"
                    )
                ).scalar()
                if has_items:
                    assert conn.execute(text("SELECT count(*) FROM corpus_release_items")).scalar() == 0
            engine.dispose()
        assert_test_database()
        assert_prod_corpus_untouched()
    finally:
        _cleanup(doc_id)


def run_all() -> int:
    tests = [
        test_authorized_expert_approves,
        test_wrong_role_rejected,
        test_reject_document,
        test_incomplete_expert_set,
        test_publish_blocks_no_source_and_no_embedding_and_draft_and_conflict,
        test_successful_publish_idempotent_audit_and_old_versions,
        test_publish_error_keeps_previous,
        test_prod_rag_and_most_db_untouched,
    ]
    failed = []
    for fn in tests:
        try:
            fn()
            print(f"OK {fn.__name__}")
        except Exception as exc:
            failed.append((fn.__name__, exc))
            print(f"FAIL {fn.__name__}: {exc}")
    if failed:
        raise SystemExit(f"failed {len(failed)}/{len(tests)}")
    print(f"passed {len(tests)}/{len(tests)}")
    return len(tests)


if __name__ == "__main__":
    run_all()
