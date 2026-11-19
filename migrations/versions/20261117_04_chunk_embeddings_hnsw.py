"""hnsw index for chunk_embeddings

Revision ID: 20261117_04
Revises: 20261117_03
Create Date: 2026-11-17

Не применять к рабочей БД most_db на шаге 2.5Б.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20261117_04"
down_revision: Union[str, Sequence[str], None] = "20261117_03"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_hnsw "
        "ON chunk_embeddings USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_chunk_embeddings_hnsw")
