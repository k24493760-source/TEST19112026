import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base

# Строка подключения берётся из .env (DATABASE_URL)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("В .env не задан DATABASE_URL (postgresql://user:password@localhost:5433/most_db)")
# Python 3.14: psycopg2-binary нет готового колеса, используем psycopg v3.
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = "postgresql+psycopg://" + DATABASE_URL[len("postgresql://") :]

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def ensure_vector_extension(bind) -> None:
    if not DATABASE_URL.startswith("postgresql"):
        return
    with bind.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


def ensure_fts_index(bind) -> None:
    if not DATABASE_URL.startswith("postgresql"):
        return
    with bind.begin() as conn:
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_chunk_search_vector "
                "ON knowledge_chunks USING gin (search_vector)"
            )
        )


def ensure_hnsw_index(bind) -> None:
    if not DATABASE_URL.startswith("postgresql"):
        return
    tables = set(inspect(bind).get_table_names())
    with bind.begin() as conn:
        if "knowledge_chunks" in tables:
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_chunk_embedding "
                    "ON knowledge_chunks USING hnsw (embedding vector_cosine_ops)"
                )
            )
        if "chunk_embeddings" in tables:
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_hnsw "
                    "ON chunk_embeddings USING hnsw (embedding vector_cosine_ops)"
                )
            )


def ensure_search_indexes(bind) -> None:
    """FTS GIN + HNSW для тестового bootstrap. Прод-схема — через Alembic 05."""
    inspector = inspect(bind)
    tables = set(inspector.get_table_names())
    if "knowledge_chunks" in tables:
        ensure_fts_index(bind)
        ensure_hnsw_index(bind)
    if "knowledge_chunks" in tables:
        with bind.begin() as conn:
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_document_id "
                    "ON knowledge_chunks (document_id)"
                )
            )
    if "chunk_embeddings" in tables:
        with bind.begin() as conn:
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_chunk_embeddings_chunk_id "
                    "ON chunk_embeddings (chunk_id)"
                )
            )
    if "corpus_release_items" in tables:
        with bind.begin() as conn:
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_corpus_release_items_chunk_id "
                    "ON corpus_release_items (chunk_id)"
                )
            )


def ensure_knowledge_schema(bind) -> None:
    inspector = inspect(bind)
    if "knowledge_documents" not in inspector.get_table_names():
        return
    existing = {col["name"] for col in inspector.get_columns("knowledge_documents")}
    statements = []
    timestamp_type = "TIMESTAMP" if DATABASE_URL.startswith("postgresql") else "DATETIME"
    if "valid_from" not in existing:
        statements.append(f"ALTER TABLE knowledge_documents ADD COLUMN valid_from {timestamp_type}")
    if "valid_to" not in existing:
        statements.append(f"ALTER TABLE knowledge_documents ADD COLUMN valid_to {timestamp_type}")
    if "official_url" not in existing:
        statements.append("ALTER TABLE knowledge_documents ADD COLUMN official_url VARCHAR")
    if "file_path" not in existing:
        statements.append("ALTER TABLE knowledge_documents ADD COLUMN file_path VARCHAR")
    if "file_format" not in existing:
        statements.append("ALTER TABLE knowledge_documents ADD COLUMN file_format VARCHAR")
    if "reviewed_at" not in existing:
        statements.append(f"ALTER TABLE knowledge_documents ADD COLUMN reviewed_at {timestamp_type}")
    if "last_updated_at" not in existing:
        statements.append(
            f"ALTER TABLE knowledge_documents ADD COLUMN last_updated_at {timestamp_type}"
        )
    if "file_hash" not in existing:
        statements.append("ALTER TABLE knowledge_documents ADD COLUMN file_hash VARCHAR")
    if "version_action" not in existing:
        statements.append("ALTER TABLE knowledge_documents ADD COLUMN version_action VARCHAR")
    if "related_document_id" not in existing:
        statements.append(
            "ALTER TABLE knowledge_documents ADD COLUMN related_document_id VARCHAR"
        )
    if "original_filename" not in existing:
        statements.append("ALTER TABLE knowledge_documents ADD COLUMN original_filename VARCHAR")
    if "corpus_source" not in existing:
        statements.append("ALTER TABLE knowledge_documents ADD COLUMN corpus_source VARCHAR")
    if "parse_status" not in existing:
        statements.append("ALTER TABLE knowledge_documents ADD COLUMN parse_status VARCHAR")
    if "parser_version" not in existing:
        statements.append("ALTER TABLE knowledge_documents ADD COLUMN parser_version VARCHAR")
    if "knowledge_chunks" in inspector.get_table_names():
        chunk_cols = {col["name"] for col in inspector.get_columns("knowledge_chunks")}
        if "embedding" not in chunk_cols:
            if DATABASE_URL.startswith("postgresql"):
                statements.append(
                    "ALTER TABLE knowledge_chunks ADD COLUMN embedding vector(384)"
                )
            else:
                statements.append("ALTER TABLE knowledge_chunks ADD COLUMN embedding JSON")
        if "search_vector" not in chunk_cols and DATABASE_URL.startswith("postgresql"):
            statements.append(
                "ALTER TABLE knowledge_chunks ADD COLUMN search_vector tsvector "
                "GENERATED ALWAYS AS (to_tsvector('russian', coalesce(text, ''))) STORED"
            )
        if "embedding_status" not in chunk_cols:
            statements.append(
                "ALTER TABLE knowledge_chunks ADD COLUMN embedding_status VARCHAR"
            )
        extras = {
            "document_version": "VARCHAR",
            "corpus_version": "VARCHAR",
            "chunking_version": "VARCHAR",
            "source_element_ids": "JSONB" if DATABASE_URL.startswith("postgresql") else "JSON",
            "token_count": "INTEGER",
            "parent_chunk_id": "VARCHAR",
            "applicability": "VARCHAR",
            "retrieval_text": "TEXT",
            "evidence_text": "TEXT",
        }
        for name, coltype in extras.items():
            if name not in chunk_cols:
                statements.append(
                    f"ALTER TABLE knowledge_chunks ADD COLUMN {name} {coltype}"
                )

    if not statements:
        return
    with bind.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


def ensure_expert_schema(bind) -> None:
    inspector = inspect(bind)
    tables = inspector.get_table_names()
    if "expert_reviews" not in tables:
        return
    existing = {col["name"] for col in inspector.get_columns("expert_reviews")}
    ts = "TIMESTAMP" if DATABASE_URL.startswith("postgresql") else "DATETIME"
    js = "JSON"
    bool_t = "BOOLEAN" if DATABASE_URL.startswith("postgresql") else "INTEGER"
    adds = []
    if "expert_id" not in existing:
        adds.append("ALTER TABLE expert_reviews ADD COLUMN expert_id VARCHAR")
    if "expert_role" not in existing:
        adds.append("ALTER TABLE expert_reviews ADD COLUMN expert_role VARCHAR")
    if "priority" not in existing:
        adds.append("ALTER TABLE expert_reviews ADD COLUMN priority VARCHAR")
    if "sla_deadline" not in existing:
        adds.append(f"ALTER TABLE expert_reviews ADD COLUMN sla_deadline {ts}")
    if "resolution_action" not in existing:
        adds.append(f"ALTER TABLE expert_reviews ADD COLUMN resolution_action {js}")
    if "resolved_at" not in existing:
        adds.append(f"ALTER TABLE expert_reviews ADD COLUMN resolved_at {ts}")
    if "time_spent_seconds" not in existing:
        adds.append("ALTER TABLE expert_reviews ADD COLUMN time_spent_seconds INTEGER")
    if "expert_decision" not in existing:
        adds.append("ALTER TABLE expert_reviews ADD COLUMN expert_decision VARCHAR")
    if "expert_comment" not in existing:
        adds.append("ALTER TABLE expert_reviews ADD COLUMN expert_comment TEXT")
    if not adds:
        return
    with bind.begin() as conn:
        for stmt in adds:
            conn.execute(text(stmt))


def ensure_storage_schema(bind) -> None:
    inspector = inspect(bind)
    tables = inspector.get_table_names()
    ts = "TIMESTAMP" if DATABASE_URL.startswith("postgresql") else "DATETIME"
    bool_t = "BOOLEAN" if DATABASE_URL.startswith("postgresql") else "INTEGER"
    js = "JSONB" if DATABASE_URL.startswith("postgresql") else "JSON"
    stmts = []
    if "users" in tables:
        cols = {c["name"] for c in inspector.get_columns("users")}
        if "global_role" not in cols:
            stmts.append("ALTER TABLE users ADD COLUMN global_role VARCHAR")
        if "learner_subrole" not in cols:
            stmts.append("ALTER TABLE users ADD COLUMN learner_subrole VARCHAR")
    if "chat_sessions" in tables:
        cols = {c["name"] for c in inspector.get_columns("chat_sessions")}
        if "learner_role" not in cols:
            stmts.append("ALTER TABLE chat_sessions ADD COLUMN learner_role VARCHAR")
        if "allowed_corpus_version" not in cols:
            stmts.append("ALTER TABLE chat_sessions ADD COLUMN allowed_corpus_version VARCHAR")
        if "started_at" not in cols:
            stmts.append(f"ALTER TABLE chat_sessions ADD COLUMN started_at {ts}")
        if "last_active_at" not in cols:
            stmts.append(f"ALTER TABLE chat_sessions ADD COLUMN last_active_at {ts}")
        if "is_active" not in cols:
            stmts.append(
                f"ALTER TABLE chat_sessions ADD COLUMN is_active {bool_t} DEFAULT TRUE"
            )
    if "chat_messages" in tables:
        cols = {c["name"] for c in inspector.get_columns("chat_messages")}
        if "review_required" not in cols:
            stmts.append(
                f"ALTER TABLE chat_messages ADD COLUMN review_required {bool_t} DEFAULT FALSE"
            )
        if "is_appealed" not in cols:
            stmts.append(
                f"ALTER TABLE chat_messages ADD COLUMN is_appealed {bool_t} DEFAULT FALSE"
            )
    with bind.begin() as conn:
        for stmt in stmts:
            conn.execute(text(stmt))
        if "chat_messages" in tables and DATABASE_URL.startswith("postgresql"):
            col = next(
                (
                    c
                    for c in inspector.get_columns("chat_messages")
                    if c["name"] == "content"
                ),
                None,
            )
            col_type = str((col or {}).get("type", "")).lower()
            if col is not None and "json" not in col_type:
                conn.execute(
                    text(
                        "ALTER TABLE chat_messages ALTER COLUMN content TYPE jsonb "
                        "USING CASE WHEN content IS NULL THEN '{}'::jsonb "
                        "WHEN left(trim(content), 1) IN ('{', '[') THEN content::jsonb "
                        "ELSE to_jsonb(content) END"
                    )
                )


def schema_bootstrap_enabled() -> bool:
    return (os.getenv("MOST_SCHEMA_BOOTSTRAP") or "").strip() == "1"


def maybe_prepare_schema(bind) -> None:
    """Не меняет данные. DDL только при MOST_SCHEMA_BOOTSTRAP=1."""
    if not schema_bootstrap_enabled():
        return
    ensure_vector_extension(bind)
    Base.metadata.create_all(bind=bind)
    ensure_knowledge_schema(bind)
    ensure_expert_schema(bind)
    ensure_storage_schema(bind)
    ensure_hnsw_index(bind)
    ensure_fts_index(bind)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
