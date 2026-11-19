"""Полная схема приложения, FTS GIN и поисковые индексы.

Revision ID: 20261117_05
Revises: 20261117_04

Не применяет изменения к most_db без ALLOW_ALEMBIC_MOST_DB=1 (см. env.py).
Идемпотентна: CREATE TABLE / INDEX IF NOT EXISTS.
"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy import text  # noqa: F401  used in downgrade

revision: str = "20261117_05"
down_revision: Union[str, Sequence[str], None] = "20261117_04"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS version VARCHAR;
ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS status VARCHAR;
ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS valid_from TIMESTAMP;
ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS valid_to TIMESTAMP;
ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS official_url VARCHAR;
ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS hash VARCHAR;
ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS file_hash VARCHAR;
ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS version_action VARCHAR;
ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS related_document_id VARCHAR;
ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS file_path VARCHAR;
ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS file_format VARCHAR;
ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS original_filename VARCHAR;
ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS corpus_source VARCHAR;
ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS parse_status VARCHAR;
ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS parser_version VARCHAR;
ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS created_at TIMESTAMP;
ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMP;
ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS last_updated_at TIMESTAMP;

ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS section VARCHAR;
ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS page INTEGER;
ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS approval_status VARCHAR;
ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS embedding_status VARCHAR;
ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS embedding vector(384);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'knowledge_chunks' AND column_name = 'search_vector'
    ) THEN
        ALTER TABLE knowledge_chunks
            ADD COLUMN search_vector tsvector
            GENERATED ALWAYS AS (to_tsvector('russian', coalesce(text, ''))) STORED;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS document_parse_runs (
    id VARCHAR PRIMARY KEY,
    document_id VARCHAR NOT NULL REFERENCES knowledge_documents(id),
    parser_version VARCHAR NOT NULL,
    document_version VARCHAR,
    file_format VARCHAR,
    status VARCHAR NOT NULL DEFAULT 'parsed',
    needs_review BOOLEAN DEFAULT FALSE,
    is_current BOOLEAN DEFAULT TRUE,
    warnings JSONB,
    errors JSONB,
    element_count INTEGER DEFAULT 0,
    artifact_path VARCHAR,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS document_elements (
    id VARCHAR PRIMARY KEY,
    document_id VARCHAR NOT NULL REFERENCES knowledge_documents(id),
    parse_run_id VARCHAR NOT NULL REFERENCES document_parse_runs(id),
    element_type VARCHAR NOT NULL,
    order_index INTEGER NOT NULL,
    text_original TEXT NOT NULL DEFAULT '',
    text_normalized TEXT NOT NULL DEFAULT '',
    section_path VARCHAR,
    page_number INTEGER,
    source_location JSONB,
    normalization_actions JSONB
);

CREATE TABLE IF NOT EXISTS document_history (
    id VARCHAR PRIMARY KEY,
    document_id VARCHAR NOT NULL REFERENCES knowledge_documents(id),
    action VARCHAR NOT NULL,
    old_status VARCHAR,
    new_status VARCHAR,
    actor_id VARCHAR,
    actor_role VARCHAR,
    comment TEXT,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS upload_history (
    id VARCHAR PRIMARY KEY,
    document_id VARCHAR REFERENCES knowledge_documents(id),
    action VARCHAR NOT NULL,
    actor_id VARCHAR,
    timestamp TIMESTAMP,
    details JSONB
);

CREATE TABLE IF NOT EXISTS users (
    id VARCHAR PRIMARY KEY,
    email VARCHAR UNIQUE NOT NULL,
    login VARCHAR UNIQUE NOT NULL,
    password_hash VARCHAR NOT NULL,
    role VARCHAR,
    global_role VARCHAR,
    learner_subrole VARCHAR,
    name VARCHAR,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL REFERENCES users(id),
    title VARCHAR NOT NULL DEFAULT 'Диалог',
    learner_role VARCHAR,
    allowed_corpus_version VARCHAR,
    started_at TIMESTAMP,
    last_active_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id VARCHAR PRIMARY KEY,
    session_id VARCHAR NOT NULL REFERENCES chat_sessions(id),
    role VARCHAR NOT NULL,
    content JSONB NOT NULL,
    review_required BOOLEAN DEFAULT FALSE,
    is_appealed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rag_audit_logs (
    id VARCHAR PRIMARY KEY,
    request_id VARCHAR NOT NULL,
    user_id VARCHAR REFERENCES users(id),
    question_text TEXT NOT NULL,
    retrieved_chunk_ids JSONB,
    search_score DOUBLE PRECISION,
    error_code VARCHAR,
    latency_ms INTEGER,
    timestamp TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analytics_events (
    id VARCHAR PRIMARY KEY,
    user_id VARCHAR,
    event_type VARCHAR NOT NULL,
    payload JSONB,
    timestamp TIMESTAMP
);

CREATE TABLE IF NOT EXISTS expert_reviews (
    id VARCHAR PRIMARY KEY,
    object_type VARCHAR NOT NULL,
    object_id VARCHAR,
    reason VARCHAR NOT NULL,
    reviewer VARCHAR,
    expert_id VARCHAR,
    expert_role VARCHAR,
    priority VARCHAR,
    sla_deadline TIMESTAMP,
    resolution_action JSON,
    decision VARCHAR,
    expert_decision VARCHAR,
    comment TEXT,
    expert_comment TEXT,
    evidence JSON,
    created_at TIMESTAMP,
    resolved_at TIMESTAMP,
    time_spent_seconds INTEGER,
    status VARCHAR
);

CREATE TABLE IF NOT EXISTS expert_comments (
    id VARCHAR PRIMARY KEY,
    review_id VARCHAR NOT NULL REFERENCES expert_reviews(id),
    expert_id VARCHAR,
    comment TEXT NOT NULL,
    internal_only BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_events (
    id VARCHAR PRIMARY KEY,
    actor VARCHAR NOT NULL,
    action VARCHAR NOT NULL,
    object VARCHAR,
    timestamp TIMESTAMP,
    payload JSON
);

CREATE TABLE IF NOT EXISTS attempts (
    id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL REFERENCES users(id),
    module_id VARCHAR NOT NULL,
    question_version VARCHAR NOT NULL,
    answer TEXT,
    score DOUBLE PRECISION,
    rationale TEXT,
    started_at TIMESTAMP NOT NULL,
    finished_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_document_id ON knowledge_chunks (document_id);
CREATE INDEX IF NOT EXISTS ix_document_parse_runs_document_id ON document_parse_runs (document_id);
CREATE INDEX IF NOT EXISTS ix_document_elements_document_id ON document_elements (document_id);
CREATE INDEX IF NOT EXISTS ix_document_elements_parse_run_id ON document_elements (parse_run_id);
CREATE INDEX IF NOT EXISTS ix_document_history_document_id ON document_history (document_id);
CREATE INDEX IF NOT EXISTS ix_upload_history_document_id ON upload_history (document_id);
CREATE INDEX IF NOT EXISTS ix_chat_sessions_user_id ON chat_sessions (user_id);
CREATE INDEX IF NOT EXISTS ix_chat_messages_session_id ON chat_messages (session_id);
CREATE INDEX IF NOT EXISTS ix_expert_comments_review_id ON expert_comments (review_id);
CREATE INDEX IF NOT EXISTS ix_attempts_user_id ON attempts (user_id);
CREATE INDEX IF NOT EXISTS ix_rag_audit_logs_user_id ON rag_audit_logs (user_id);
CREATE INDEX IF NOT EXISTS ix_corpus_release_items_chunk_id ON corpus_release_items (chunk_id);
CREATE INDEX IF NOT EXISTS ix_corpus_release_items_embedding_id ON corpus_release_items (embedding_id);
CREATE INDEX IF NOT EXISTS ix_chunk_embeddings_chunk_id ON chunk_embeddings (chunk_id);
CREATE INDEX IF NOT EXISTS ix_corpus_release_items_release_id ON corpus_release_items (release_id);
CREATE INDEX IF NOT EXISTS ix_corpus_release_items_document_id ON corpus_release_items (document_id);

CREATE INDEX IF NOT EXISTS idx_chunk_search_vector
    ON knowledge_chunks USING gin (search_vector);
CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_hnsw
    ON chunk_embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_chunk_embedding
    ON knowledge_chunks USING hnsw (embedding vector_cosine_ops);
"""


def upgrade() -> None:
    op.get_bind().exec_driver_sql(DDL)


def downgrade() -> None:
    op.execute(text("DROP INDEX IF EXISTS idx_chunk_embeddings_hnsw"))
    op.execute(text("DROP INDEX IF EXISTS idx_chunk_search_vector"))
    op.execute(text("DROP INDEX IF EXISTS idx_chunk_embedding"))
