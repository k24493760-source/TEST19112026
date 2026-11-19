"""Read-only snapshot of most_db. No writes."""
from __future__ import annotations

import urllib.request
from sqlalchemy import inspect, text
from app.db import DATABASE_URL, engine, SessionLocal
from app.models import (
    KnowledgeDocument,
    KnowledgeChunk,
    ChunkEmbedding,
    CorpusRelease,
    ExpertReview,
    QaCard,
    User,
    AuditEvent,
)


def main() -> None:
    url = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")
    from sqlalchemy.engine import make_url
    u = make_url(url)
    print("db", u.database, "host", u.host, "port", u.port)
    insp = inspect(engine)
    tables = sorted(insp.get_table_names())
    print("tables", len(tables), ",".join(tables))
    try:
        with engine.connect() as conn:
            print("alembic", conn.execute(text("SELECT version_num FROM alembic_version")).fetchall())
    except Exception as e:
        print("alembic", type(e).__name__)
    try:
        with engine.connect() as conn:
            print("vector_ext", conn.execute(text("SELECT extversion FROM pg_extension WHERE extname='vector'")).scalar())
    except Exception as e:
        print("vector_ext", type(e).__name__)
    indexes = []
    for t in ("knowledge_chunks", "chunk_embeddings"):
        if t in tables:
            for ix in insp.get_indexes(t):
                indexes.append(f"{t}.{ix.get('name')}")
    print("indexes", indexes)
    if "corpus_releases" in tables:
        cols = [c["name"] for c in insp.get_columns("corpus_releases")]
        print("corpus_releases_cols", cols)
    db = SessionLocal()
    try:
        print("users", db.query(User).count())
        print("docs", db.query(KnowledgeDocument).count())
        for d in db.query(KnowledgeDocument).all():
            print("doc", d.status, d.parse_status, d.owner, bool(d.file_path))
        print("chunks", db.query(KnowledgeChunk).count())
        print("embeddings", db.query(ChunkEmbedding).count())
        print("releases", db.query(CorpusRelease).count())
        print("reviews", db.query(ExpertReview).count())
        print("qa_cards", db.query(QaCard).count() if "qa_cards" in tables else "no_table")
        print("audit", db.query(AuditEvent).count())
        if "document_parse_runs" in tables:
            print("parse_runs", conn_count := db.execute(text("SELECT count(*) FROM document_parse_runs")).scalar())
            print("elements", db.execute(text("SELECT count(*) FROM document_elements")).scalar())
    finally:
        db.close()
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/api/v1/health", timeout=5) as r:
            print("health", r.status, r.read()[:200].decode())
    except Exception as e:
        print("health_fail", type(e).__name__)


if __name__ == "__main__":
    main()
