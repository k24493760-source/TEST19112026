"""Задания генерации Golden Q&A. Не применять к most_db без ALLOW_ALEMBIC_MOST_DB=1.

Revision ID: 20261117_07
Revises: 20261117_06
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20261117_07"
down_revision: Union[str, Sequence[str], None] = "20261117_06"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

DDL = """
CREATE TABLE IF NOT EXISTS qa_generation_jobs (
    id VARCHAR PRIMARY KEY,
    document_id VARCHAR,
    chunk_id VARCHAR NOT NULL,
    status VARCHAR NOT NULL DEFAULT 'queued',
    attempt INTEGER NOT NULL DEFAULT 0,
    error_code VARCHAR,
    skipped_reason VARCHAR,
    n_candidates INTEGER NOT NULL DEFAULT 0,
    n_persisted INTEGER NOT NULL DEFAULT 0,
    idempotency_key VARCHAR NOT NULL,
    model VARCHAR,
    revision VARCHAR,
    prompt_version VARCHAR,
    adapter VARCHAR,
    document_version VARCHAR,
    chunking_version VARCHAR,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_qa_generation_jobs_idempotency ON qa_generation_jobs (idempotency_key);
CREATE INDEX IF NOT EXISTS ix_qa_generation_jobs_chunk_id ON qa_generation_jobs (chunk_id);
CREATE INDEX IF NOT EXISTS ix_qa_generation_jobs_document_id ON qa_generation_jobs (document_id);
"""


def upgrade() -> None:
    op.execute(DDL)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS qa_generation_jobs")
