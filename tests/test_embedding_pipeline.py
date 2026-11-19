"""Шаг 2.4Б: сегментная векторизация draft-чанков. Только most_db_test."""
from __future__ import annotations

import inspect
import math
import uuid

from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.engine import make_url

from tests import isolation

isolation.activate()
from app.db import Base
from app.models import ChunkEmbedding, CorpusRelease, KnowledgeChunk, KnowledgeDocument
from app.services.chunking import chunk_parsed_document
from app.services.embedding_pipeline import (
    EmbeddingPipelineError,
    count_tokens_with_special,
    embed_segments_strict,
    load_embedding_config,
    max_content_tokens,
    refuse_prod_database,
    retrieval_not_wired,
    split_retrieval_text,
    vectorize_draft_chunks,
)
from app.services.intake import intake_document
from app.services.search import retrieve_chunks
from tests.isolation import assert_prod_corpus_untouched, assert_test_database
from tests.test_intake import _cleanup

assert_test_database()
Base.metadata.create_all(bind=isolation.engine)

SHORT = "Статья 8. Уведомление направляется не позднее десяти рабочих дней."
LONG_UNIT = (
    "Положение описывает порядок действий уполномоченного лица при проверке источника и цитаты. "
)


def _intake(body: str, name: str | None = None):
    db = isolation.SessionLocal()
    try:
        return intake_document(
            db,
            body.encode("utf-8"),
            name or f"emb-{uuid.uuid4().hex[:8]}.md",
            title=f"Embed {uuid.uuid4().hex[:8]}",
            document_class="Нормативный акт",
            corpus_source="database_edu_lectcii",
            owner="qa-embed",
            actor_id="qa-embed",
            content_type="text/markdown",
        )
    finally:
        db.close()


def _chunk(doc_id: str):
    db = isolation.SessionLocal()
    try:
        return chunk_parsed_document(db, doc_id)
    finally:
        db.close()


def _vectorize(doc_id: str, **kwargs):
    db = isolation.SessionLocal()
    try:
        result = vectorize_draft_chunks(db, document_id=doc_id, **kwargs)
        return result
    finally:
        db.close()


def _rows(doc_id: str) -> list[ChunkEmbedding]:
    db = isolation.SessionLocal()
    try:
        chunk_ids = [
            c.id
            for c in db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id == doc_id)
        ]
        return (
            db.query(ChunkEmbedding)
            .filter(ChunkEmbedding.chunk_id.in_(chunk_ids))
            .order_by(ChunkEmbedding.chunk_id, ChunkEmbedding.segment_index)
            .all()
        )
    finally:
        db.close()


def _long_body() -> str:
    text = SHORT + " "
    while count_tokens_with_special(text) < 400:
        text += LONG_UNIT
    return "# Статья 40\n\n" + text


def _assert_vectors(rows: list[ChunkEmbedding]) -> None:
    max_seq = int(load_embedding_config()["max_seq_length"])
    for row in rows:
        if row.embedding_status not in {"ready", "superseded"}:
            continue
        if row.embedding_status != "ready":
            continue
        values = list(row.embedding)
        assert len(values) == 384
        assert all(math.isfinite(float(v)) for v in values)
        assert count_tokens_with_special(row.segment_text) <= max_seq
        assert count_tokens_with_special(row.segment_text) == row.token_count


def test_refuse_prod_name():
    db = isolation.SessionLocal()
    try:
        assert refuse_prod_database(db) == "most_db_test"
    finally:
        db.close()


def test_short_chunk_one_vector():
    item = _intake(f"# Статья 8\n\n{SHORT}\n")
    try:
        _chunk(item.document_id)
        result = _vectorize(item.document_id)
        rows = [r for r in _rows(item.document_id) if r.embedding_status == "ready"]
        assert rows
        assert all(r.segment_index == 0 for r in rows) or any(
            sum(1 for x in rows if x.chunk_id == cid) == 1
            for cid in {r.chunk_id for r in rows}
        )
        _assert_vectors(rows)
        db = isolation.SessionLocal()
        try:
            doc = db.query(KnowledgeDocument).filter_by(id=item.document_id).one()
            assert doc.status == "draft"
            for chunk in db.query(KnowledgeChunk).filter_by(document_id=item.document_id):
                assert chunk.approval_status == "draft"
                assert chunk.embedding is None
            rel = db.query(CorpusRelease).filter_by(version="working-draft").one()
            assert rel.status == "draft"
            assert rel.published_at is None
        finally:
            db.close()
        assert result.created >= 1
    finally:
        _cleanup(item.document_id)


def test_long_chunk_multiple_vectors_full_coverage():
    item = _intake(_long_body())
    try:
        _chunk(item.document_id)
        db = isolation.SessionLocal()
        try:
            chunks = db.query(KnowledgeChunk).filter_by(document_id=item.document_id).all()
            long_chunks = [c for c in chunks if count_tokens_with_special(c.retrieval_text or c.text) > 256]
            assert long_chunks, "нужен чанк длиннее 256 токенов MiniLM"
        finally:
            db.close()
        result = _vectorize(item.document_id, batch_size=2)
        rows = [r for r in _rows(item.document_id) if r.embedding_status == "ready"]
        by_chunk: dict[str, list[ChunkEmbedding]] = {}
        for row in rows:
            by_chunk.setdefault(row.chunk_id, []).append(row)
        multi = [cid for cid, items in by_chunk.items() if len(items) > 1]
        assert multi, f"ожидались несколько векторов, created={result.created}"
        db = isolation.SessionLocal()
        try:
            for cid in multi:
                chunk = db.get(KnowledgeChunk, cid)
                segs = sorted(by_chunk[cid], key=lambda r: r.segment_index)
                rebuilt = "".join(chunk.retrieval_text[s.char_start : s.char_end] for s in segs)
                assert rebuilt == chunk.retrieval_text
                assert segs[0].char_start == 0
                assert segs[-1].char_end == len(chunk.retrieval_text)
                assert [s.segment_index for s in segs] == list(range(len(segs)))
                assert len({s.embedding_id for s in segs}) == len(segs)
            for chunk in db.query(KnowledgeChunk).filter_by(document_id=item.document_id):
                assert chunk.approval_status == "draft"
                assert chunk.embedding is None
        finally:
            db.close()
        _assert_vectors(rows)
        windows = split_retrieval_text(long_chunks[0].retrieval_text)
        assert len(windows) > 1
        assert all(w[3] <= 256 for w in windows)
    finally:
        _cleanup(item.document_id)


def test_rerun_stable_ids_no_duplicates():
    item = _intake(_long_body())
    try:
        _chunk(item.document_id)
        first = _vectorize(item.document_id)
        ids1 = sorted(r.embedding_id for r in _rows(item.document_id) if r.embedding_status == "ready")
        second = _vectorize(item.document_id)
        ids2 = sorted(r.embedding_id for r in _rows(item.document_id) if r.embedding_status == "ready")
        assert ids1 == ids2
        assert second.created == 0
        assert second.skipped >= first.created
        assert len(ids2) == len(set(ids2))
    finally:
        _cleanup(item.document_id)


def test_content_change_supersedes_not_approved():
    item = _intake(f"# Статья 8\n\n{SHORT}\n")
    try:
        _chunk(item.document_id)
        _vectorize(item.document_id)
        db = isolation.SessionLocal()
        try:
            chunk = db.query(KnowledgeChunk).filter_by(document_id=item.document_id).first()
            chunk.retrieval_text = (chunk.retrieval_text or "") + " Дополнение о сроке в рабочих днях."
            chunk.text = chunk.retrieval_text
            db.commit()
            cid = chunk.id
        finally:
            db.close()
        result = _vectorize(item.document_id)
        rows = _rows(item.document_id)
        statuses = {r.embedding_status for r in rows if r.chunk_id == cid}
        assert "ready" in statuses
        assert "superseded" in statuses or result.superseded >= 1
        assert not any(r.embedding_status == "approved" and r.chunk_id == cid for r in rows)
    finally:
        _cleanup(item.document_id)


def test_model_revision_change_keeps_old_rows():
    item = _intake(f"# Статья 8\n\n{SHORT}\n")
    try:
        _chunk(item.document_id)
        _vectorize(item.document_id, model_revision="rev-a")
        _vectorize(item.document_id, model_revision="rev-b")
        rows = _rows(item.document_id)
        revs = {r.model_revision for r in rows if r.embedding_status == "ready"}
        assert "rev-a" in revs and "rev-b" in revs
        assert len(rows) >= 2
    finally:
        _cleanup(item.document_id)


def test_interrupt_and_resume():
    item = _intake(_long_body())
    try:
        _chunk(item.document_id)
        import app.services.embedding_pipeline as ep

        real = ep.embed_segments_strict
        state = {"n": 0}

        def flaky(texts):
            state["n"] += 1
            if state["n"] == 2:
                raise RuntimeError("simulated interrupt")
            return real(texts)

        ep.embed_segments_strict = flaky
        try:
            try:
                _vectorize(item.document_id, batch_size=1)
                raise AssertionError("ожидался сбой второго пакета")
            except EmbeddingPipelineError as exc:
                assert exc.code == "batch_failed"
        finally:
            ep.embed_segments_strict = real
        after_fail = [r for r in _rows(item.document_id) if r.embedding_status == "ready"]
        assert after_fail, "первый пакет должен быть зафиксирован"
        resumed = _vectorize(item.document_id, batch_size=1)
        ready = [r for r in _rows(item.document_id) if r.embedding_status == "ready"]
        assert len(ready) > len(after_fail) or resumed.skipped >= 1
        ids = [r.embedding_id for r in ready]
        assert len(ids) == len(set(ids))
        _assert_vectors(ready)
    finally:
        _cleanup(item.document_id)


def test_postgres_integrity_rollback():
    item = _intake(f"# Статья 8\n\n{SHORT}\n")
    try:
        _chunk(item.document_id)
        _vectorize(item.document_id)
        db = isolation.SessionLocal()
        try:
            row = db.query(ChunkEmbedding).first()
            payload = {
                "embedding_id": row.embedding_id,
                "chunk_id": row.chunk_id,
                "segment_index": row.segment_index,
                "segment_text": row.segment_text,
                "char_start": row.char_start,
                "char_end": row.char_end,
                "embedding_model": row.embedding_model,
                "model_revision": row.model_revision,
                "embedding_config_version": row.embedding_config_version,
                "content_hash": row.content_hash,
                "token_count": row.token_count,
                "embedding_status": "ready",
            }
            db.expunge_all()
            try:
                db.execute(
                    text(
                        "INSERT INTO chunk_embeddings ("
                        "embedding_id, chunk_id, segment_index, segment_text, char_start, char_end, "
                        "embedding_model, model_revision, embedding_config_version, content_hash, "
                        "token_count, embedding_status"
                        ") VALUES ("
                        ":embedding_id, :chunk_id, :segment_index, :segment_text, :char_start, :char_end, "
                        ":embedding_model, :model_revision, :embedding_config_version, :content_hash, "
                        ":token_count, :embedding_status)"
                    ),
                    payload,
                )
                db.commit()
                raise AssertionError("ожидался IntegrityError")
            except IntegrityError:
                db.rollback()
            count = db.query(ChunkEmbedding).count()
            assert count >= 1
            refuse_prod_database(db)
        finally:
            db.close()
        again = _vectorize(item.document_id)
        assert again.created == 0
    finally:
        _cleanup(item.document_id)


def test_draft_not_used_by_prod_retrieve_and_most_db_untouched():
    item = _intake(_long_body())
    try:
        _chunk(item.document_id)
        _vectorize(item.document_id)
        db = isolation.SessionLocal()
        try:
            found = retrieve_chunks(db, "Каков срок уведомления уполномоченного органа?")
            chunk_ids = {c.id for c, _d in found.rows}
            draft_ids = {
                c.id
                for c in db.query(KnowledgeChunk).filter_by(
                    document_id=item.document_id, approval_status="draft"
                )
            }
            assert chunk_ids.isdisjoint(draft_ids)
            assert all(c.embedding is None for c in db.query(KnowledgeChunk).filter_by(document_id=item.document_id))
        finally:
            db.close()
        assert retrieval_not_wired()["wired_to_retrieve_chunks"] is True

        from app.config import DATABASE_URL

        url = make_url(DATABASE_URL)
        if url.database == "most_db":
            engine = create_engine(
                "postgresql+psycopg://" + DATABASE_URL[len("postgresql://") :]
                if DATABASE_URL.startswith("postgresql://")
                else DATABASE_URL
            )
            with engine.connect() as conn:
                exists = conn.execute(
                    text(
                        "SELECT 1 FROM information_schema.tables "
                        "WHERE table_name = 'chunk_embeddings'"
                    )
                ).scalar()
                if exists:
                    n = conn.execute(text("SELECT count(*) FROM chunk_embeddings")).scalar()
                    assert n == 0
                stolen = conn.execute(
                    text(
                        "SELECT count(*) FROM knowledge_documents WHERE owner = 'qa-embed'"
                    )
                ).scalar()
                assert stolen == 0
            engine.dispose()
        assert_prod_corpus_untouched()
        assert_test_database()
    finally:
        _cleanup(item.document_id)


def test_no_hidden_truncation_unit():
    too_long = SHORT + LONG_UNIT * 40
    assert count_tokens_with_special(too_long) > 256
    try:
        embed_segments_strict([too_long])
        raise AssertionError("длинный текст не должен кодироваться целиком")
    except EmbeddingPipelineError as exc:
        assert exc.code == "would_truncate"
    windows = split_retrieval_text(too_long)
    texts = [w[2] for w in windows]
    vecs = embed_segments_strict(texts)
    assert len(vecs) == len(windows)
    assert all(len(v) == 384 for v in vecs)
    assert max_content_tokens() < 256


def run_all() -> int:
    tests = [
        test_refuse_prod_name,
        test_short_chunk_one_vector,
        test_long_chunk_multiple_vectors_full_coverage,
        test_rerun_stable_ids_no_duplicates,
        test_content_change_supersedes_not_approved,
        test_model_revision_change_keeps_old_rows,
        test_interrupt_and_resume,
        test_postgres_integrity_rollback,
        test_draft_not_used_by_prod_retrieve_and_most_db_untouched,
        test_no_hidden_truncation_unit,
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
