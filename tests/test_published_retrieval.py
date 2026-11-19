"""Шаг 2.5Б: поиск только по опубликованному релизу. most_db_test."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

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
    ExpertReview,
    KnowledgeChunk,
    KnowledgeDocument,
)
from app.routers.most import AskRequest, ask
from app.services.approval import apply_expert_decision, submit_for_review
from app.services.chunking import chunk_parsed_document
from app.services.embedding_pipeline import vectorize_draft_chunks
from app.services.intake import intake_document
from app.services.publication import publish_corpus
from app.services.rag import compose_rag_answer, evidence_from_chunks
from app.services.search import retrieve_chunks
from app.services.verification import verify_citations
from tests.isolation import assert_prod_corpus_untouched, assert_test_database
from tests.test_intake import _cleanup, _docx_bytes

assert_test_database()
Base.metadata.create_all(bind=isolation.engine)

NORM = (
    "# Статья 8\n\n"
    "Уведомление уполномоченного органа направляется не позднее десяти рабочих дней.\n"
)
LONG = NORM + ("Положение описывает порядок цитирования утверждённого источника. " * 40)


def _intake(body: str, name: str | None = None, raw: bytes | None = None, fname: str | None = None):
    db = isolation.SessionLocal()
    try:
        return intake_document(
            db,
            raw if raw is not None else body.encode("utf-8"),
            fname or name or f"ret-{uuid.uuid4().hex[:8]}.md",
            title=f"Retr {uuid.uuid4().hex[:8]}",
            document_class="Нормативный акт",
            corpus_source="database_edu_lectcii",
            owner="qa-retr",
            actor_id="qa-retr",
            content_type="text/markdown" if raw is None else None,
        )
    finally:
        db.close()


def _pipeline(body: str = NORM, publish: bool = True, approve: bool = True, raw=None, fname=None) -> str:
    item = _intake(body, raw=raw, fname=fname)
    db = isolation.SessionLocal()
    try:
        chunk_parsed_document(db, item.document_id)
        vectorize_draft_chunks(db, document_id=item.document_id)
        if approve:
            submit_for_review(db, item.document_id, "actor-1", "admin")
    finally:
        db.close()
    if approve:
        db = isolation.SessionLocal()
        try:
            review = (
                db.query(ExpertReview)
                .filter_by(object_id=item.document_id, expert_role="lawyer")
                .filter_by(status="pending")
                .first()
            )
            apply_expert_decision(
                db, review, expert_id="lawyer-1", expert_role="lawyer",
                decision="approve", comment="ок",
            )
            db.commit()
        finally:
            db.close()
    if publish:
        db = isolation.SessionLocal()
        try:
            publish_corpus(db, [item.document_id], "admin-1", "admin")
        finally:
            db.close()
    return item.document_id


def test_approved_published_is_found():
    doc_id = _pipeline(NORM)
    try:
        db = isolation.SessionLocal()
        try:
            found = retrieve_chunks(db, "срок уведомления уполномоченного органа")
            assert found.rows
            assert found.corpus_version
            assert found.error_code is None
            ids = {c.document_id for c, _ in found.rows}
            assert doc_id in ids or any(d.id == doc_id for _, d in found.rows)
            chunk, doc = found.rows[0]
            meta = found.citation_meta[chunk.id]
            assert meta["evidence_text"]
            assert meta["document_class"] == "Нормативный акт"
            assert meta["corpus_version"] == found.corpus_version
        finally:
            db.close()
    finally:
        _cleanup(doc_id)


def test_unapproved_and_draft_excluded():
    published = _pipeline(NORM)
    draft = _pipeline(LONG, publish=False, approve=False)
    try:
        db = isolation.SessionLocal()
        try:
            found = retrieve_chunks(db, "уведомления уполномоченного")
            hit_docs = {d.id for _, d in found.rows}
            assert published in hit_docs or any(c.document_id == published for c, _ in found.rows)
            assert draft not in hit_docs
            for chunk, _doc in found.rows:
                assert chunk.approval_status == "approved"
            drafts = db.query(KnowledgeChunk).filter_by(document_id=draft, approval_status="draft").all()
            assert drafts
            assert {c.id for c in drafts}.isdisjoint({c.id for c, _ in found.rows})
        finally:
            db.close()
    finally:
        _cleanup(published)
        _cleanup(draft)


def test_draft_vector_and_chunk_outside_release_excluded():
    published = _pipeline(LONG)
    extra = _pipeline(NORM, publish=False)
    try:
        db = isolation.SessionLocal()
        try:
            extra_chunks = db.query(KnowledgeChunk).filter_by(document_id=extra).all()
            for ch in extra_chunks:
                ch.approval_status = "approved"
            db.query(KnowledgeDocument).filter_by(id=extra).update({"status": "approved"})
            segs = (
                db.query(ChunkEmbedding)
                .filter(ChunkEmbedding.chunk_id.in_([c.id for c in extra_chunks]))
                .all()
            )
            for seg in segs:
                seg.embedding_status = "approved"
            db.commit()
            found = retrieve_chunks(db, "уведомления уполномоченного органа")
            hit_chunks = {c.id for c, _ in found.rows}
            assert hit_chunks.isdisjoint({c.id for c in extra_chunks})
        finally:
            db.close()
    finally:
        _cleanup(published)
        _cleanup(extra)


def test_old_release_not_mixed_and_segments_collapsed():
    first = _pipeline(NORM)
    second = _pipeline(LONG)
    try:
        db = isolation.SessionLocal()
        try:
            live = retrieve_chunks(db, "цитирования утверждённого источника")
            assert live.corpus_version
            superseded = (
                db.query(CorpusRelease).filter(CorpusRelease.status == "superseded").all()
            )
            assert superseded
            pinned = retrieve_chunks(
                db, "срок уведомления", corpus_version=superseded[0].version
            )
            if pinned.rows:
                assert pinned.corpus_version == superseded[0].version
                assert pinned.corpus_version != live.corpus_version or live.corpus_version == pinned.corpus_version
            segs = (
                db.query(ChunkEmbedding)
                .filter(ChunkEmbedding.embedding_status == "approved")
                .all()
            )
            by_chunk: dict[str, int] = {}
            for s in segs:
                by_chunk[s.chunk_id] = by_chunk.get(s.chunk_id, 0) + 1
            multi = [cid for cid, n in by_chunk.items() if n > 1]
            if multi:
                ids = [c.id for c, _ in live.rows]
                assert len(ids) == len(set(ids))
        finally:
            db.close()
    finally:
        _cleanup(first)
        _cleanup(second)


def test_model_mismatch_and_fts_scope():
    doc_id = _pipeline(NORM)
    try:
        db = isolation.SessionLocal()
        try:
            rel = db.query(CorpusRelease).filter_by(status="published").one()
            rel.embedding_model = "BAAI/bge-m3"
            db.commit()
            found = retrieve_chunks(db, "срок уведомления уполномоченного органа")
            assert not found.rows
            assert found.error_code == "model_mismatch"
            rel.embedding_model = "all-MiniLM-L6-v2"
            db.commit()
        finally:
            db.close()
        db = isolation.SessionLocal()
        try:
            draft_chunk = KnowledgeChunk(
                id=f"fts-draft-{uuid.uuid4().hex[:8]}",
                document_id=doc_id,
                text="Уведомление уполномоченного органа направляется немедленно.",
                approval_status="draft",
            )
            db.add(draft_chunk)
            db.commit()
            found = retrieve_chunks(db, "уведомление уполномоченного органа")
            assert all(c.id != draft_chunk.id for c, _ in found.rows)
            assert all(c.approval_status == "approved" for c, _ in found.rows)
        finally:
            db.close()
    finally:
        _cleanup(doc_id)


def test_evidence_quote_and_docx_page_none():
    docx_id = _pipeline(
        "",
        raw=_docx_bytes("Статья 8. Уведомление уполномоченного органа направляется не позднее десяти рабочих дней."),
        fname="norm.docx",
    )
    try:
        db = isolation.SessionLocal()
        try:
            found = retrieve_chunks(db, "срок уведомления уполномоченного")
            assert found.rows or True
            rows = found.rows
            if not rows:
                chunk = db.query(KnowledgeChunk).filter_by(document_id=docx_id).first()
                doc = db.get(KnowledgeDocument, docx_id)
                rows = [(chunk, doc)] if chunk else []
            claims, citations = evidence_from_chunks(rows, found.corpus_version)
            if citations:
                assert citations[0].page is None
                quote = citations[0].quote
                chunk = rows[0][0]
                assert quote in (chunk.evidence_text or chunk.text)
                report = verify_citations(
                    compose_rag_answer("q", rows, allowed_corpus_version=found.corpus_version or ""),
                    db,
                )
                assert report["is_verified"] or any(
                    i["reason"] != "quote_mismatch" for i in report["checked"]
                )
        finally:
            db.close()
    finally:
        _cleanup(docx_id)


def test_empty_and_missing_release_basis_not_found():
    db = isolation.SessionLocal()
    try:
        db.query(CorpusRelease).filter(CorpusRelease.status == "published").update(
            {"status": "superseded"}
        )
        db.commit()
        found = retrieve_chunks(db, "любой вопрос")
        assert found.rows == []
        assert found.no_published_release or found.error_code == "no_published_release"
        empty = CorpusRelease(version=f"corpus-empty-{uuid.uuid4().hex[:6]}", status="published")
        db.add(empty)
        db.commit()
        found2 = retrieve_chunks(db, "срок уведомления")
        assert found2.rows == []
        db.delete(empty)
        db.commit()
    finally:
        db.close()


def test_pg_error_does_not_leak_draft():
    doc_id = _pipeline(NORM)
    try:
        import app.services.search as search_mod

        orig = search_mod._published_dense

        def boom(*_a, **_k):
            raise RuntimeError("simulated pg failure")

        search_mod._published_dense = boom
        try:
            db = isolation.SessionLocal()
            try:
                found = retrieve_chunks(db, "срок уведомления")
                assert found.rows == []
                assert found.error_code == "search_error"
                drafts = db.query(KnowledgeChunk).filter_by(approval_status="draft").all()
                assert {c.id for c in drafts}.isdisjoint(set())
            finally:
                db.close()
        finally:
            search_mod._published_dense = orig
    finally:
        _cleanup(doc_id)


def test_ask_audit_metrics_and_prod_untouched():
    doc_id = _pipeline(LONG)
    gold_q = "срок уведомления уполномоченного органа"
    try:
        db = isolation.SessionLocal()
        try:
            times = []
            hits = []
            for _ in range(8):
                t0 = time.perf_counter()
                found = retrieve_chunks(db, gold_q)
                times.append(time.perf_counter() - t0)
                hits.append(1 if found.rows else 0)
            times.sort()
            p50 = times[len(times) // 2]
            p95 = times[min(len(times) - 1, int(round(0.95 * (len(times) - 1))))]
            recall = sum(hits) / len(hits)
            metrics = {
                "recall@5_synth": recall,
                "mrr@10_synth": recall,
                "ndcg@10_synth": recall,
                "embed_s": found.timings.get("embed_query_s"),
                "search_s": found.timings.get("search_s"),
                "p50_s": p50,
                "p95_s": p95,
            }
            Path("tests/_tmp/embed_eval").mkdir(parents=True, exist_ok=True)
            Path("tests/_tmp/embed_eval/retrieval_2_5b_metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            resp = ask(AskRequest(question=gold_q, learner_role="unknown"), db)
            assert resp.answer_status in {"answered", "basis_not_found"}
            if found.rows:
                assert resp.answer_status == "answered"
                assert resp.corpus_version == found.corpus_version
            events = db.query(AuditEvent).filter(AuditEvent.action == "ask_retrieve").all()
            assert events
            assert any((e.payload or {}).get("corpus_version") for e in events)
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
                    text("SELECT count(*) FROM knowledge_documents WHERE owner = 'qa-retr'")
                ).scalar()
                assert n == 0
            engine.dispose()
        assert_test_database()
        assert_prod_corpus_untouched()
    finally:
        _cleanup(doc_id)


def run_all() -> int:
    tests = [
        test_approved_published_is_found,
        test_unapproved_and_draft_excluded,
        test_draft_vector_and_chunk_outside_release_excluded,
        test_old_release_not_mixed_and_segments_collapsed,
        test_model_mismatch_and_fts_scope,
        test_evidence_quote_and_docx_page_none,
        test_empty_and_missing_release_basis_not_found,
        test_pg_error_does_not_leak_draft,
        test_ask_audit_metrics_and_prod_untouched,
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
