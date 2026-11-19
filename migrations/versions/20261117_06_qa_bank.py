"""Golden Q&A и учебный банк. Не применять к most_db без ALLOW_ALEMBIC_MOST_DB=1.

Revision ID: 20261117_06
Revises: 20261117_05
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20261117_06"
down_revision: Union[str, Sequence[str], None] = "20261117_05"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

DDL = """
CREATE TABLE IF NOT EXISTS learning_modules (
    id VARCHAR PRIMARY KEY,
    title VARCHAR NOT NULL,
    competencies JSONB,
    prerequisites JSONB,
    corpus_version VARCHAR,
    owner VARCHAR,
    status VARCHAR NOT NULL DEFAULT 'draft',
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rubrics (
    id VARCHAR PRIMARY KEY,
    version VARCHAR NOT NULL DEFAULT '1.0',
    competency VARCHAR,
    title VARCHAR,
    required_elements JSONB,
    allowed_formulations JSONB,
    critical_errors JSONB,
    owner VARCHAR,
    status VARCHAR NOT NULL DEFAULT 'draft',
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS qa_cards (
    qa_id VARCHAR PRIMARY KEY,
    lineage_id VARCHAR NOT NULL,
    parent_qa_id VARCHAR REFERENCES qa_cards(qa_id),
    qa_version VARCHAR NOT NULL DEFAULT '1.0',
    question_text TEXT NOT NULL,
    reference_answer TEXT NOT NULL DEFAULT '',
    question_type VARCHAR NOT NULL DEFAULT 'factual',
    difficulty VARCHAR,
    document_version VARCHAR,
    corpus_version VARCHAR,
    generation_model VARCHAR,
    generation_prompt_version VARCHAR,
    content_hash VARCHAR NOT NULL,
    generation_status VARCHAR NOT NULL DEFAULT 'draft',
    golden_status VARCHAR NOT NULL DEFAULT 'none',
    learning_status VARCHAR NOT NULL DEFAULT 'none',
    golden_test_kind VARCHAR,
    expected_answer_status VARCHAR,
    expected_chunk_ids JSONB,
    eval_split VARCHAR,
    competency VARCHAR,
    module_id VARCHAR REFERENCES learning_modules(id),
    rubric_id VARCHAR REFERENCES rubrics(id),
    rubric_version VARCHAR,
    answer_type VARCHAR,
    answer_key JSONB,
    required_elements JSONB,
    allowed_formulations JSONB,
    critical_errors JSONB,
    warnings JSONB,
    ambiguity_flags JSONB,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS questions (
    id VARCHAR PRIMARY KEY,
    type VARCHAR NOT NULL DEFAULT 'free',
    competency VARCHAR,
    difficulty VARCHAR,
    prompt TEXT NOT NULL,
    key JSONB,
    rubric_version VARCHAR,
    evidence_ids JSONB,
    status VARCHAR NOT NULL DEFAULT 'draft',
    qa_id VARCHAR REFERENCES qa_cards(qa_id),
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS qa_source_chunks (
    id VARCHAR PRIMARY KEY,
    qa_id VARCHAR NOT NULL REFERENCES qa_cards(qa_id) ON DELETE CASCADE,
    chunk_id VARCHAR NOT NULL REFERENCES knowledge_chunks(id),
    document_id VARCHAR,
    document_version VARCHAR,
    section VARCHAR,
    page INTEGER,
    role VARCHAR NOT NULL DEFAULT 'primary'
);

CREATE TABLE IF NOT EXISTS qa_evidence_quotes (
    id VARCHAR PRIMARY KEY,
    qa_id VARCHAR NOT NULL REFERENCES qa_cards(qa_id) ON DELETE CASCADE,
    chunk_id VARCHAR,
    quote TEXT NOT NULL,
    source_element_ids JSONB
);

CREATE TABLE IF NOT EXISTS qa_eval_snapshots (
    id VARCHAR PRIMARY KEY,
    qa_id VARCHAR NOT NULL,
    qa_version VARCHAR,
    corpus_version VARCHAR,
    eval_split VARCHAR,
    payload JSONB,
    created_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_qa_cards_lineage_id ON qa_cards (lineage_id);
CREATE INDEX IF NOT EXISTS ix_qa_cards_content_hash ON qa_cards (content_hash);
CREATE INDEX IF NOT EXISTS ix_qa_source_chunks_qa_id ON qa_source_chunks (qa_id);
CREATE INDEX IF NOT EXISTS ix_qa_source_chunks_chunk_id ON qa_source_chunks (chunk_id);
CREATE INDEX IF NOT EXISTS ix_qa_evidence_quotes_qa_id ON qa_evidence_quotes (qa_id);
CREATE INDEX IF NOT EXISTS ix_qa_eval_snapshots_qa_id ON qa_eval_snapshots (qa_id);
"""


def upgrade() -> None:
    op.get_bind().exec_driver_sql(DDL)


def downgrade() -> None:
    op.get_bind().exec_driver_sql(
        """
        DROP TABLE IF EXISTS qa_eval_snapshots;
        DROP TABLE IF EXISTS qa_evidence_quotes;
        DROP TABLE IF EXISTS qa_source_chunks;
        DROP TABLE IF EXISTS questions;
        DROP TABLE IF EXISTS qa_cards;
        DROP TABLE IF EXISTS rubrics;
        DROP TABLE IF EXISTS learning_modules;
        """
    )
