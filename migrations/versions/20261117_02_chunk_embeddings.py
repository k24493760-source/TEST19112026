"""chunk_embeddings search segments

Revision ID: 20261117_02
Revises: 20261117_01
Create Date: 2026-11-17

Не применять к рабочей БД most_db на шаге 2.4Б.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "20261117_02"
down_revision: Union[str, Sequence[str], None] = "20261117_01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from sqlalchemy import inspect

    if "chunk_embeddings" in inspect(op.get_bind()).get_table_names():
        return
    op.create_table(
        "chunk_embeddings",
        sa.Column("embedding_id", sa.String(), primary_key=True),
        sa.Column("chunk_id", sa.String(), sa.ForeignKey("knowledge_chunks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("segment_index", sa.Integer(), nullable=False),
        sa.Column("segment_text", sa.Text(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("char_end", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("embedding", Vector(384), nullable=True),
        sa.Column("embedding_model", sa.String(), nullable=False),
        sa.Column("model_revision", sa.String(), nullable=False),
        sa.Column("embedding_config_version", sa.String(), nullable=False),
        sa.Column("content_hash", sa.String(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("embedding_status", sa.String(), nullable=False, server_default="ready"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint(
            "chunk_id",
            "segment_index",
            "embedding_model",
            "model_revision",
            "embedding_config_version",
            "content_hash",
            name="uq_chunk_embedding_identity",
        ),
    )
    op.create_index("ix_chunk_embeddings_chunk_id", "chunk_embeddings", ["chunk_id"])


def downgrade() -> None:
    op.drop_index("ix_chunk_embeddings_chunk_id", table_name="chunk_embeddings")
    op.drop_table("chunk_embeddings")
