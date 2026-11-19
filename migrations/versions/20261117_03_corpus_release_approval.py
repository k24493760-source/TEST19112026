"""corpus release snapshot and items

Revision ID: 20261117_03
Revises: 20261117_02
Create Date: 2026-11-17

Не применять к рабочей БД most_db на шаге 2.5А.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20261117_03"
down_revision: Union[str, Sequence[str], None] = "20261117_02"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from sqlalchemy import inspect

    bind = op.get_bind()
    tables = set(inspect(bind).get_table_names())
    rel_cols = (
        {c["name"] for c in inspect(bind).get_columns("corpus_releases")}
        if "corpus_releases" in tables
        else set()
    )
    extras = {
        "published_by": sa.String(),
        "embedding_model": sa.String(),
        "model_revision": sa.String(),
        "embedding_config_version": sa.String(),
        "previous_version": sa.String(),
        "manifest_hash": sa.String(),
        "snapshot": postgresql.JSONB(astext_type=sa.Text()),
    }
    for name, coltype in extras.items():
        if name not in rel_cols:
            op.add_column("corpus_releases", sa.Column(name, coltype, nullable=True))
    if "corpus_release_items" not in tables:
        op.create_table(
            "corpus_release_items",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column(
                "release_id",
                sa.String(),
                sa.ForeignKey("corpus_releases.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("document_id", sa.String(), nullable=False),
            sa.Column("document_version", sa.String(), nullable=True),
            sa.Column("chunk_id", sa.String(), nullable=False),
            sa.Column("embedding_id", sa.String(), nullable=True),
            sa.Column("chunking_version", sa.String(), nullable=True),
        )
        op.create_index("ix_corpus_release_items_release_id", "corpus_release_items", ["release_id"])
        op.create_index("ix_corpus_release_items_document_id", "corpus_release_items", ["document_id"])


def downgrade() -> None:
    op.drop_index("ix_corpus_release_items_document_id", table_name="corpus_release_items")
    op.drop_index("ix_corpus_release_items_release_id", table_name="corpus_release_items")
    op.drop_table("corpus_release_items")
    for name in (
        "snapshot",
        "manifest_hash",
        "previous_version",
        "embedding_config_version",
        "model_revision",
        "embedding_model",
        "published_by",
    ):
        op.drop_column("corpus_releases", name)
