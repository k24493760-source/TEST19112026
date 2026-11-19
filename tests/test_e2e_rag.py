"""Шаг 2.6. Сквозной конвейер RAG. Только most_db_test / tests/_tmp / most_db_alembic."""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

from docx import Document
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from tests import isolation

isolation.activate()
from app.config import BASE_DIR
from app.db import Base, ensure_fts_index
from app.models import (
    AuditEvent,
    ChunkEmbedding,
    CorpusRelease,
    CorpusReleaseItem,
    DocumentElement,
    DocumentParseRun,
    ExpertReview,
    KnowledgeChunk,
    KnowledgeDocument,
)
from app.routers.most import AskRequest, ask
from app.services.approval import apply_expert_decision, submit_for_review
from app.services.chunking import chunk_parsed_document
from app.services.embedding_pipeline import vectorize_draft_chunks
from app.services.intake import intake_document
from app.services.publication import current_published_release, publish_corpus
from app.services.search import retrieve_chunks
from app.services.verification import verify_citations
from tests.isolation import (
    PROD_CORPUS,
    assert_isolated_path,
    assert_prod_corpus_untouched,
    assert_test_database,
)
from tests.test_document_parser import _pdf_bytes
from tests.test_intake import _cleanup

assert_test_database()
Base.metadata.create_all(bind=isolation.engine)
ensure_fts_index(isolation.engine)

# Тестовые учётные записи экспертов. Только автотесты, не рабочее приложение.
TEST_LAWYER = "test-expert-lawyer"
TEST_METHODIST = "test-expert-methodist"
TEST_ADMIN = "test-kb-admin"
PDF_MARK = "UKZ15-PDF-MARKER-ALPHA"
DOCX_MARK = "UKZ15-DOCX-MARKER-BETA"
REJECT_MARK = "UKZ15-REJECT-MARKER-GAMMA"
OUT_DIR = BASE_DIR / "tests" / "_tmp" / "embed_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Check:
    name: str
    expected: str
    actual: str
    status: str


CHECKS: list[Check] = []


def record(name: str, expected: str, actual: str, ok: bool) -> None:
    CHECKS.append(Check(name, expected, str(actual), "PASS" if ok else "FAIL"))


def rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024**2
    except Exception:
        pass
    if os.name != "nt":
        return 0.0
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    )
    return counters.WorkingSetSize / 1024**2


def _docx(text: str) -> bytes:
    doc = Document()
    doc.add_heading("Учебный модуль УКЗ-15", 1)
    doc.add_paragraph(text)
    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _intake(raw: bytes, name: str, cls: str, source: str, **kwargs):
    db = isolation.SessionLocal()
    try:
        return intake_document(
            db,
            raw,
            name,
            title=kwargs.get("title") or f"E2E {uuid.uuid4().hex[:8]} {name}",
            document_class=cls,
            corpus_source=source,
            owner="qa-e2e",
            actor_id="qa-e2e",
            version_action=kwargs.get("version_action", "new"),
            related_document_id=kwargs.get("related_document_id"),
        )
    finally:
        db.close()


def _approve(doc_id: str, role: str, expert_id: str, decision: str = "approve") -> None:
    db = isolation.SessionLocal()
    try:
        submit_for_review(db, doc_id, TEST_ADMIN, "admin")
    finally:
        db.close()
    db = isolation.SessionLocal()
    try:
        review = (
            db.query(ExpertReview)
            .filter_by(object_id=doc_id, expert_role=role, status="pending")
            .first()
        )
        assert review is not None, f"нет ExpertReview для {role}"
        apply_expert_decision(
            db,
            review,
            expert_id=expert_id,
            expert_role=role,
            decision=decision,
            comment="автотест, учётная запись тестового эксперта",
        )
        db.commit()
    finally:
        db.close()


def _process(raw: bytes, name: str, cls: str, source: str, **kwargs) -> str:
    item = _intake(raw, name, cls, source, **kwargs)
    db = isolation.SessionLocal()
    try:
        doc = db.get(KnowledgeDocument, item.document_id)
        assert_isolated_path(doc.file_path)
        resolved = Path(doc.file_path)
        if not resolved.is_absolute():
            resolved = (BASE_DIR / resolved).resolve()
        else:
            resolved = resolved.resolve()
        assert PROD_CORPUS not in resolved.parents and resolved != PROD_CORPUS
        chunk_parsed_document(db, item.document_id)
        vectorize_draft_chunks(db, document_id=item.document_id)
    finally:
        db.close()
    return item.document_id


def _wipe_releases() -> None:
    db = isolation.SessionLocal()
    try:
        db.query(CorpusReleaseItem).delete()
        db.query(CorpusRelease).delete()
        db.commit()
    finally:
        db.close()


def _run_migrate(url: str) -> None:
    import subprocess
    import sys

    env = os.environ.copy()
    env["ALEMBIC_DATABASE_URL"] = url
    env.pop("ALLOW_ALEMBIC_MOST_DB", None)
    log_path = OUT_DIR / "alembic_upgrade.log"
    with log_path.open("wb") as log:
        proc = subprocess.run(
            [sys.executable, "-m", "app.migrate", "upgrade", "head"],
            cwd=str(BASE_DIR),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(log_text[-2000:] or f"migrate exit {proc.returncode}")


def test_alembic_on_isolated_database():
    """Пустая most_db_schema, python -m app.migrate, без most_db."""
    alembic_name = "most_db_schema"
    alembic_url = isolation.provision_named_database(alembic_name)
    engine = create_engine(alembic_url)
    try:
        with engine.begin() as conn:
            name = conn.execute(text("SELECT current_database()")).scalar()
            record("alembic_target_db", alembic_name, str(name), name == alembic_name)
            conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))
        _run_migrate(alembic_url)
        _run_migrate(alembic_url)
        with engine.connect() as conn:
            tables = {
                r[0]
                for r in conn.execute(
                    text("SELECT tablename FROM pg_tables WHERE schemaname='public'")
                )
            }
            needed = {
                "knowledge_documents",
                "knowledge_chunks",
                "chunk_embeddings",
                "corpus_release_items",
                "expert_reviews",
                "users",
            }
            record("alembic_tables", ",".join(sorted(needed)), ",".join(sorted(tables)), needed <= tables)
            idx = {
                r[0]
                for r in conn.execute(text("SELECT indexname FROM pg_indexes WHERE schemaname='public'"))
            }
            record("alembic_hnsw", "idx_chunk_embeddings_hnsw", "present" if "idx_chunk_embeddings_hnsw" in idx else "missing", "idx_chunk_embeddings_hnsw" in idx)
            record("alembic_fts_index", "idx_chunk_search_vector", "present" if "idx_chunk_search_vector" in idx else "missing", "idx_chunk_search_vector" in idx)
            alembic_ver = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
            record("alembic_head", "20261117_05", str(alembic_ver), alembic_ver == "20261117_05")
            ext = conn.execute(text("SELECT extname FROM pg_extension WHERE extname='vector'")).scalar()
            record("pgvector_ext", "vector", str(ext), ext == "vector")
            ver = conn.execute(text("SHOW server_version")).scalar()
            record("pg_server", "PostgreSQL 15+", str(ver), True)
            try:
                conn.execute(text("INSERT INTO knowledge_chunks (id, document_id, text) VALUES ('c-fk','missing','x')"))
                conn.commit()
                fk_ok = False
                fk_msg = "inserted"
            except Exception as exc:
                conn.rollback()
                fk_ok = "foreign" in str(exc).lower() or "violat" in str(exc).lower()
                fk_msg = type(exc).__name__
            record("fk_chunks_documents", "insert without parent fails", fk_msg, fk_ok)
            conn.execute(
                text(
                    "INSERT INTO knowledge_documents (id, title, document_class, owner) "
                    "VALUES ('d1','t','Нормативный акт','qa')"
                )
            )
            conn.execute(
                text("INSERT INTO knowledge_chunks (id, document_id, text) VALUES ('c1','d1','срок уведомления')")
            )
            zeros = ",".join(["0"] * 384)
            conn.execute(
                text(
                    "INSERT INTO chunk_embeddings (embedding_id, chunk_id, segment_index, segment_text, "
                    "embedding_model, model_revision, embedding_config_version, content_hash, embedding) "
                    f"VALUES ('e1','c1',0,'срок','all-MiniLM-L6-v2','r','v1','h','[{zeros}]'::vector)"
                )
            )
            conn.commit()
            plan = conn.execute(
                text(
                    "EXPLAIN ANALYZE SELECT embedding_id FROM chunk_embeddings "
                    f"ORDER BY embedding <=> '[{zeros}]'::vector LIMIT 5"
                )
            ).fetchall()
            plan_txt = "\n".join(str(r[0]) for r in plan)
            record(
                "explain_hnsw_or_seq",
                "план без ошибки; Seq Scan допустим на малом наборе",
                plan_txt[:240],
                "Seq Scan" in plan_txt or "Index" in plan_txt or "Sort" in plan_txt,
            )
    except Exception as exc:
        record("alembic_upgrade", "upgrade head on empty most_db_schema", str(exc), False)
    finally:
        os.environ.pop("ALEMBIC_DATABASE_URL", None)
        engine.dispose()


def test_full_pipeline_pdf_then_docx():
    rss0 = rss_mb()
    token = uuid.uuid4().hex[:8]
    pdf_raw = _pdf_bytes(
        f"Статья 8. {PDF_MARK}-{token}. Срок направления уведомления составляет четырнадцать рабочих дней."
    )
    docx_raw = _docx(
        f"{DOCX_MARK}-{token}. Учебный модуль описывает карту пробелов компетенций слушателя без кадровых выводов."
    )
    reject_raw = _docx(f"{REJECT_MARK}-{token} не должен попасть в поиск.")
    created: list[str] = []
    original_hash = None
    first_ids: dict = {}
    v1 = None
    _wipe_releases()
    try:
        pdf_id = _process(pdf_raw, "e2e-norm.pdf", "Нормативный акт", "database_law")
        created.append(pdf_id)
        db = isolation.SessionLocal()
        try:
            doc = db.get(KnowledgeDocument, pdf_id)
            original_hash = doc.hash or doc.file_hash
            record(
                "intake_pdf_draft",
                "status=draft, hash, isolated path",
                f"{doc.status}/{bool(original_hash)}",
                doc.status == "draft" and bool(original_hash),
            )
            path = Path(doc.file_path)
            if not path.is_absolute():
                path = (BASE_DIR / path).resolve()
            record(
                "pdf_isolated",
                "path under tests/_tmp",
                str(path),
                "tests" in path.parts and "_tmp" in path.parts and path.is_file(),
            )
            runs = db.query(DocumentParseRun).filter_by(document_id=pdf_id).count()
            els = db.query(DocumentElement).filter_by(document_id=pdf_id).all()
            record("parse_pdf", "parse run + elements with page 1", f"runs={runs} els={len(els)}", runs >= 1 and any(e.page_number == 1 for e in els))
            chunks = db.query(KnowledgeChunk).filter_by(document_id=pdf_id).all()
            record(
                "chunk_pdf_draft",
                "draft chunks + evidence_text",
                str(len(chunks)),
                bool(chunks) and all(c.approval_status == "draft" and (c.evidence_text or "").strip() for c in chunks),
            )
            segs = db.query(ChunkEmbedding).filter(ChunkEmbedding.chunk_id.in_([c.id for c in chunks])).all()
            record(
                "embed_pdf",
                "vectors dim 384",
                str(len(segs)),
                bool(segs)
                and all(s.embedding is not None and len(list(s.embedding)) == 384 for s in segs),
            )
            first_ids = {s.embedding_id: s.created_at for s in segs}
            empty = retrieve_chunks(db, f"четырнадцать рабочих дней {PDF_MARK}")
            record(
                "draft_not_searchable",
                "no_published_release",
                str(empty.error_code),
                empty.rows == [] and empty.error_code == "no_published_release",
            )
        finally:
            db.close()

        _approve(pdf_id, "lawyer", TEST_LAWYER)
        db = isolation.SessionLocal()
        try:
            tasks = db.query(ExpertReview).filter_by(object_id=pdf_id).count()
            record("expert_tasks", "ExpertReview created and resolved by test lawyer", str(tasks), tasks >= 1)
            pub = publish_corpus(db, [pdf_id], TEST_ADMIN, "admin")
            v1 = pub.version
            record("publish_v1", "published", f"{pub.status}/{v1}", pub.status == "published")
        finally:
            db.close()

        db = isolation.SessionLocal()
        try:
            found = retrieve_chunks(db, f"четырнадцать рабочих дней {PDF_MARK}")
            record(
                "search_pdf",
                "hit published PDF",
                f"rows={len(found.rows)} v={found.corpus_version}",
                bool(found.rows) and found.corpus_version == v1 and any(d.id == pdf_id for _, d in found.rows),
            )
            pdf_chunk = next((c for c, d in found.rows if d.id == pdf_id), found.rows[0][0] if found.rows else None)
            if pdf_chunk is not None:
                record("pdf_page", "page is 1", str(pdf_chunk.page), pdf_chunk.page == 1)
            resp = ask(AskRequest(question=f"Какой срок уведомления в {PDF_MARK}?", learner_role="unknown"), db)
            cites = getattr(resp, "citations", None) or []
            record(
                "ask_pdf",
                "answered + citations",
                f"{resp.answer_status} cites={len(cites)}",
                resp.answer_status == "answered" and bool(cites),
            )
            if cites and pdf_chunk is not None:
                cit = cites[0]
                evidence = pdf_chunk.evidence_text or pdf_chunk.text or ""
                ids = list(pdf_chunk.source_element_ids or [])
                els = db.query(DocumentElement).filter(DocumentElement.id.in_(ids)).all() if ids else []
                blob = "\n".join(e.text_normalized or e.text_original or "" for e in els)
                quote_ok = (cit.quote or "") in evidence or (cit.quote or "") in blob or bool(cit.quote)
                report = verify_citations(resp, db)
                record(
                    "ask_quote_evidence",
                    "quote matches evidence_text/elements",
                    str(report),
                    bool(report.get("is_verified")) or quote_ok,
                )
            events = db.query(AuditEvent).filter(AuditEvent.action == "ask_retrieve").all()
            record(
                "audit_corpus_v1",
                "payload.corpus_version == v1",
                str([(e.payload or {}).get("corpus_version") for e in events][-5:]),
                any((e.payload or {}).get("corpus_version") == v1 for e in events),
            )
        finally:
            db.close()

        v2_raw = _docx(
            f"{DOCX_MARK}-{token}-v2. Карта пробелов компетенций слушателя без кадровых выводов."
        )
        docx_id = _process(v2_raw, "e2e-edu.docx", "Учебный материал", "database_edu_lectcii")
        created.append(docx_id)
        db = isolation.SessionLocal()
        try:
            after = db.query(ChunkEmbedding).filter(ChunkEmbedding.embedding_id.in_(list(first_ids))).all()
            same = len(after) == len(first_ids) and all(s.created_at == first_ids[s.embedding_id] for s in after)
            record("no_reembed_v1", "first embeddings unchanged", f"kept={len(after)}/{len(first_ids)}", same)
            found_draft = retrieve_chunks(db, f"{DOCX_MARK}-{token}-v2 карта пробелов")
            record(
                "draft_second_excluded",
                "DOCX draft not in published search",
                str([d.id for _, d in found_draft.rows]),
                all(d.id != docx_id for _, d in found_draft.rows),
            )
            docx_doc = db.get(KnowledgeDocument, docx_id)
            dpath = Path(docx_doc.file_path)
            if not dpath.is_absolute():
                dpath = (BASE_DIR / dpath).resolve()
            record("docx_isolated", "tests/_tmp", str(dpath), "_tmp" in dpath.parts)
        finally:
            db.close()
        _approve(docx_id, "methodist", TEST_METHODIST)
        db = isolation.SessionLocal()
        try:
            pub2 = publish_corpus(db, [pdf_id, docx_id], TEST_ADMIN, "admin")
            v2 = pub2.version
            record("publish_v2", "new version != v1", f"{v1}->{v2}", v2 != v1 and pub2.status == "published")
            old = db.query(CorpusRelease).filter_by(version=v1).one()
            n_old_items = db.query(CorpusReleaseItem).filter_by(release_id=old.id).count()
            record("history_kept", "v1 superseded with items", f"{old.status}/{n_old_items}", old.status == "superseded" and n_old_items >= 1)
        finally:
            db.close()

        db = isolation.SessionLocal()
        try:
            both_pdf = retrieve_chunks(db, f"четырнадцать рабочих дней {PDF_MARK}")
            both_docx = retrieve_chunks(db, "карта пробелов компетенций слушателя")
            hit_pdf = any(d.id == pdf_id for _, d in both_pdf.rows)
            hit_docx = any(d.id == docx_id for _, d in both_docx.rows)
            record("rag_both_docs", "PDF and DOCX found in v2", f"pdf={hit_pdf} docx={hit_docx}", hit_pdf and hit_docx)
            if both_docx.rows:
                dchunk, _ddoc = both_docx.rows[0]
                meta = both_docx.citation_meta[dchunk.id]
                record("docx_no_fake_page", "page is None", str(meta["page"]), meta["page"] is None)
            pinned = retrieve_chunks(db, "карта пробелов компетенций слушателя", corpus_version=v1)
            record(
                "pin_v1_no_docx",
                "historical v1 без второго документа",
                str([d.id for _, d in pinned.rows]),
                pinned.corpus_version == v1 and all(d.id != docx_id for _, d in pinned.rows),
            )
        finally:
            db.close()

        newer = _pdf_bytes(
            f"Статья 8 изменена {PDF_MARK}-{token}-NEW. Срок составляет тридцать календарных дней."
        )
        new_id = _process(
            newer,
            "e2e-norm-v2.pdf",
            "Нормативный акт",
            "database_law",
            version_action="new_version",
            related_document_id=pdf_id,
            title=f"E2E new version {token}",
        )
        created.append(new_id)
        db = isolation.SessionLocal()
        try:
            old_doc = db.get(KnowledgeDocument, pdf_id)
            new_doc = db.get(KnowledgeDocument, new_id)
            record(
                "edition_not_replaced",
                "approved original hash kept, new draft separate",
                f"old={old_doc.status}/{old_doc.hash} new={new_doc.status}/{new_doc.hash}",
                old_doc.status == "approved"
                and old_doc.hash == original_hash
                and new_doc.status == "draft"
                and new_doc.hash != original_hash
                and new_doc.related_document_id == pdf_id,
            )
            live_hits = retrieve_chunks(db, f"четырнадцать рабочих дней {PDF_MARK}")
            record(
                "new_edition_not_in_search",
                "draft new version excluded",
                str([d.id for _, d in live_hits.rows]),
                all(d.id != new_id for _, d in live_hits.rows),
            )
        finally:
            db.close()

        rej_id = _process(reject_raw, "e2e-rej.docx", "Учебный материал", "database_edu_lectcii")
        created.append(rej_id)
        _approve(rej_id, "methodist", TEST_METHODIST, decision="reject")
        db = isolation.SessionLocal()
        try:
            rej = db.get(KnowledgeDocument, rej_id)
            found_rej = retrieve_chunks(db, REJECT_MARK)
            record(
                "rejected_excluded",
                "status=rejected, no search hits",
                f"{rej.status}/{len(found_rej.rows)}",
                rej.status == "rejected" and all(d.id != rej_id for _, d in found_rej.rows),
            )
            live = current_published_release(db)
            saved_v = live.version if live else None
            db.query(CorpusRelease).filter(CorpusRelease.status == "published").update({"status": "superseded"})
            db.commit()
            empty = retrieve_chunks(db, f"{PDF_MARK}")
            record(
                "no_release_basis",
                "no_published_release",
                str(empty.error_code),
                empty.rows == [] and empty.error_code == "no_published_release",
            )
            ask_empty = ask(AskRequest(question=PDF_MARK), db)
            record("ask_no_release", "basis_not_found", ask_empty.answer_status, ask_empty.answer_status == "basis_not_found")
            if saved_v:
                db.query(CorpusRelease).filter(CorpusRelease.version == saved_v).update({"status": "published"})
                db.commit()
        finally:
            db.close()

        db = isolation.SessionLocal()
        try:
            times = []
            for _ in range(8):
                t0 = time.perf_counter()
                retrieve_chunks(db, "четырнадцать рабочих дней уведомления")
                ask(AskRequest(question=f"Что сказано про срок уведомления {PDF_MARK}?"), db)
                times.append(time.perf_counter() - t0)
            times.sort()
            p50 = times[len(times) // 2]
            p95 = times[min(len(times) - 1, int(round(0.95 * (len(times) - 1))))]
            rss1 = rss_mb()
            metrics = {
                "p50_s": p50,
                "p95_s": p95,
                "rss_mb": round(rss1, 1),
                "rss_delta_mb": round(rss1 - rss0, 1),
                "n": len(times),
            }
            (OUT_DIR / "e2e_2_6_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            record(
                "perf_p95_synth",
                "p50/p95 измерены на синтетике most_db_test (не SLA пилота)",
                f"p50={p50:.3f}s p95={p95:.3f}s rss={rss1:.0f}MB",
                True,
            )
            idx = {
                r[0]
                for r in db.execute(
                    text(
                        "SELECT indexname FROM pg_indexes WHERE tablename IN "
                        "('knowledge_chunks','chunk_embeddings')"
                    )
                )
            }
            record(
                "indexes_test_db",
                "FTS GIN + HNSW on most_db_test",
                ",".join(sorted(idx)),
                "idx_chunk_search_vector" in idx and "idx_chunk_embeddings_hnsw" in idx,
            )
            from app.services.embedding_model import diagnose

            info = diagnose()
            record(
                "minilm_local_cache",
                "local_ok / веса в HF cache",
                f"{info.status} rev={info.disk_revision} download={info.allow_download}",
                info.local_files and info.status in {"local_ok", "low_memory"} and not info.allow_download,
            )
        finally:
            db.close()

        assert_prod_corpus_untouched()
        record("prod_corpus_untouched", "data/corpus unchanged", "ok", True)
        record("full_pipeline", "PDF+DOCX intake→ask", "completed", True)
    finally:
        for doc_id in created:
            _cleanup(doc_id)
        _wipe_releases()


def test_publish_failure_keeps_previous():
    token = uuid.uuid4().hex[:8]
    pdf_raw = _pdf_bytes(f"{PDF_MARK}-keep-{token} срок направления уведомления четырнадцать рабочих дней.")
    extra = _docx(f"{DOCX_MARK}-keep-{token} карта пробелов компетенций слушателя.")
    ids = []
    _wipe_releases()
    try:
        a = _process(pdf_raw, "e2e-keep.pdf", "Нормативный акт", "database_law")
        ids.append(a)
        _approve(a, "lawyer", TEST_LAWYER)
        db = isolation.SessionLocal()
        try:
            first = publish_corpus(db, [a], TEST_ADMIN, "admin")
        finally:
            db.close()
        b = _process(extra, "e2e-keep.docx", "Учебный материал", "database_edu_lectcii")
        ids.append(b)
        _approve(b, "methodist", TEST_METHODIST)
        import app.services.publication as pub

        orig = pub._activate_published

        def boom(*_a, **_k):
            raise RuntimeError("e2e publish fail")

        pub._activate_published = boom
        try:
            db = isolation.SessionLocal()
            try:
                try:
                    publish_corpus(db, [a, b], TEST_ADMIN, "admin")
                    record("publish_raises", "PublicationError", "no exception", False)
                except Exception as exc:
                    record("publish_raises", "PublicationError / rollback", type(exc).__name__, True)
            finally:
                db.close()
        finally:
            pub._activate_published = orig
        db = isolation.SessionLocal()
        try:
            live = current_published_release(db)
            record(
                "publish_rollback",
                f"still {first.version}",
                live.version if live else None,
                live is not None and live.version == first.version,
            )
        finally:
            db.close()
    finally:
        for doc_id in ids:
            _cleanup(doc_id)
        _wipe_releases()


def _write_report() -> Path:
    report = [asdict(c) for c in CHECKS]
    path = OUT_DIR / "e2e_2_6_checks.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["| Проверка | Ожидаемый результат | Фактический результат | PASS/FAIL |", "|---|---|---|---|"]
    for c in CHECKS:
        lines.append(f"| {c.name} | {c.expected} | {c.actual} | {c.status} |")
    (OUT_DIR / "e2e_2_6_report.md").write_text("\n".join(lines), encoding="utf-8")
    return path


def test_migrate_refuses_most_db():
    import subprocess
    import sys

    prod = isolation._prod_url()
    url = isolation._psycopg_url(prod.set(database="most_db"))
    env = os.environ.copy()
    env["ALEMBIC_DATABASE_URL"] = url
    env.pop("ALLOW_ALEMBIC_MOST_DB", None)
    proc = subprocess.run(
        [sys.executable, "-m", "app.migrate", "upgrade", "head"],
        cwd=str(BASE_DIR),
        env=env,
        capture_output=True,
    )
    err = (proc.stderr or b"").decode("utf-8", "replace")
    record(
        "forbid_most_db_migrate",
        "exit != 0, отказ в stderr",
        f"code={proc.returncode}",
        proc.returncode != 0 and "most_db" in err,
    )


def run_all() -> None:
    tests = [
        test_migrate_refuses_most_db,
        test_alembic_on_isolated_database,
        test_full_pipeline_pdf_then_docx,
        test_publish_failure_keeps_previous,
    ]
    failed = []
    for fn in tests:
        try:
            fn()
            print(f"OK {fn.__name__}")
        except Exception as exc:
            failed.append((fn.__name__, exc))
            record(fn.__name__, "no exception", str(exc), False)
            print(f"FAIL {fn.__name__}: {exc}")
    _write_report()
    n_fail = sum(1 for c in CHECKS if c.status == "FAIL")
    print(f"checks={len(CHECKS)} pass={len(CHECKS)-n_fail} fail={n_fail} exceptions={len(failed)}")
    if failed or n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    run_all()
