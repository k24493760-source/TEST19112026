"""Запуск Alembic без конфликта имени каталога и без молчаливого most_db.

    python -m app.migrate upgrade head
    python -m app.migrate current

Адрес БД только из ALEMBIC_DATABASE_URL. Пароли в код не записываются.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROD_DB = "most_db"


def _database_name(url: str) -> str:
    from sqlalchemy.engine import make_url

    raw = url.replace("postgresql+psycopg://", "postgresql://")
    return make_url(raw).database or ""


def require_url() -> str:
    url = (os.getenv("ALEMBIC_DATABASE_URL") or "").strip()
    if not url:
        raise SystemExit(
            "Задайте ALEMBIC_DATABASE_URL (например postgresql+psycopg://user:pass@localhost:5433/most_db_test). "
            "DATABASE_URL рабочего контура не используется."
        )
    name = _database_name(url)
    if name == PROD_DB and os.getenv("ALLOW_ALEMBIC_MOST_DB") != "1":
        raise SystemExit("Отказ: миграции на most_db без ALLOW_ALEMBIC_MOST_DB=1")
    if name == PROD_DB:
        print("ВНИМАНИЕ: миграции на most_db разрешены явно ALLOW_ALEMBIC_MOST_DB=1", file=sys.stderr)
    return url


def alembic_config(url: str):
    from alembic.config import Config

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Миграции УКЗ-15")
    parser.add_argument("command", choices=("upgrade", "downgrade", "current", "heads", "history"))
    parser.add_argument("target", nargs="?", default="head")
    args = parser.parse_args(argv)
    url = require_url()
    os.environ["ALEMBIC_DATABASE_URL"] = url
    from alembic import command

    cfg = alembic_config(url)
    if args.command == "upgrade":
        command.upgrade(cfg, args.target)
    elif args.command == "downgrade":
        command.downgrade(cfg, args.target)
    elif args.command == "current":
        command.current(cfg)
    elif args.command == "heads":
        command.heads(cfg)
    else:
        command.history(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
