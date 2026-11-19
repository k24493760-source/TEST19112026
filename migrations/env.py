import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import make_url

from app.db import Base
from app import models  # noqa: F401

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _database_name(url: str | None) -> str:
    if not url:
        return ""
    raw = url.replace("postgresql+psycopg://", "postgresql://")
    try:
        return make_url(raw).database or ""
    except Exception:
        return url.rsplit("/", 1)[-1].split("?")[0]


def _resolve_url() -> str:
    env_url = (os.getenv("ALEMBIC_DATABASE_URL") or "").strip()
    try:
        ini_url = (config.get_main_option("sqlalchemy.url") or "").strip()
    except Exception:
        ini_url = ""
    url = env_url or ini_url
    if not url:
        raise RuntimeError(
            "Задайте ALEMBIC_DATABASE_URL. Alembic не подставляет most_db по умолчанию."
        )
    name = _database_name(url)
    if name == "most_db" and os.getenv("ALLOW_ALEMBIC_MOST_DB") != "1":
        raise RuntimeError("Отказ: миграции на most_db без ALLOW_ALEMBIC_MOST_DB=1")
    return url


migrate_url = _resolve_url()
config.set_main_option("sqlalchemy.url", migrate_url)
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = config.get_main_option("sqlalchemy.url")
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        url=url,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
