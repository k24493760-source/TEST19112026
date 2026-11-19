"""chunking fields and corpus_releases

Revision ID: 20261117_01
Revises:
Create Date: 2026-11-17

Идемпотентно: на пустой БД создаёт knowledge_documents / knowledge_chunks,
на уже применённой ревизии не запускается повторно.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect, text
from sqlalchemy.dialects import postgresql

revision: str = "20261117_01"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _tables() -> set[str]:
    return set(inspect(op.get_bind()).get_table_names())


def _columns(table: str) -> set[str]:
    bind = op.get_bind()
    if table not in inspect(bind).get_table_names():
        return set()
    return {c["name"] for c in inspect(bind).get_columns(table)}


def upgrade() -> None:
    op.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    tables = _tables()
    if "knowledge_documents" not in tables:
        op.create_table(
            "knowledge_documents",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("title", sa.String(), nullable=False),
            sa.Column("document_class", sa.String(), nullable=False),
            sa.Column("owner", sa.String(), nullable=False),
            sa.Column("version", sa.String(), nullable=True),
            sa.Column("status", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
        )
    if "knowledge_chunks" not in _tables():
        op.create_table(
            "knowledge_chunks",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column(
                "document_id",
                sa.String(),
                sa.ForeignKey("knowledge_documents.id"),
                nullable=False,
            ),
            sa.Column("text", sa.Text(), nullable=False),
        )
    if "corpus_releases" not in _tables():
        op.create_table(
            "corpus_releases",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("version", sa.String(), nullable=False, unique=True),
            sa.Column("status", sa.String(), nullable=False, server_default="draft"),
            sa.Column("notes", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("published_at", sa.DateTime(), nullable=True),
        )
    chunk_cols = {
        "document_version": sa.String(),
        "corpus_version": sa.String(),
        "chunking_version": sa.String(),
        "source_element_ids": postgresql.JSONB(astext_type=sa.Text()),
        "token_count": sa.Integer(),
        "parent_chunk_id": sa.String(),
        "applicability": sa.String(),
        "retrieval_text": sa.Text(),
        "evidence_text": sa.Text(),
    }
    existing = _columns("knowledge_chunks")
    for name, coltype in chunk_cols.items():
        if name not in existing:
            op.add_column("knowledge_chunks", sa.Column(name, coltype, nullable=True))


def downgrade() -> None:
    existing = _columns("knowledge_chunks")
    for name in (
        "evidence_text",
        "retrieval_text",
        "applicability",
        "parent_chunk_id",
        "token_count",
        "source_element_ids",
        "chunking_version",
        "corpus_version",
        "document_version",
    ):
        if name in existing:
            op.drop_column("knowledge_chunks", name)
    if "corpus_releases" in _tables():
        op.drop_table("corpus_releases")
