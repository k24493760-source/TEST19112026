"""Изоляция тестов: отдельная БД most_db_test и временный корпус."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from app.config import BASE_DIR, DATABASE_URL as PROD_DATABASE_URL
from app.db import Base, ensure_knowledge_schema, ensure_search_indexes
from app import models  # noqa: F401
from app.services import corpus as corpus_svc
from app.services import document_parser as parser_svc
from app.services import intake as intake_svc

TEST_DB_NAME = "most_db_test"
PROD_DB_NAME = "most_db"
PROD_CORPUS = (BASE_DIR / "data" / "corpus").resolve()
PROD_EDU = PROD_CORPUS / "database_edu_lectcii"
PROD_LAW = PROD_CORPUS / "database_law"

_activated = False
engine = None
SessionLocal = None
TMP_ROOT: Path | None = None
_prod_edu_names: set[str] | None = None
_prod_law_names: set[str] | None = None


def _prod_url():
    raw = PROD_DATABASE_URL
    if raw.startswith("postgresql+psycopg://"):
        raw = "postgresql://" + raw[len("postgresql+psycopg://") :]
    return make_url(raw)


def _psycopg_url(url) -> str:
    rendered = url.render_as_string(hide_password=False)
    if rendered.startswith("postgresql://"):
        return "postgresql+psycopg://" + rendered[len("postgresql://") :]
    return rendered


def provision_named_database(name: str) -> str:
    """Создаёт пустую PostgreSQL БД. most_db запрещён."""
    if name == PROD_DB_NAME:
        raise RuntimeError("Отказ создавать/чистить рабочую БД most_db")
    prod = _prod_url()
    admin = prod.set(database="postgres")
    admin_engine = create_engine(_psycopg_url(admin), isolation_level="AUTOCOMMIT")
    try:
        with admin_engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :n"),
                {"n": name},
            ).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{name}"'))
    finally:
        admin_engine.dispose()
    return _psycopg_url(prod.set(database=name))


def _ensure_test_database() -> str:
    prod = _prod_url()
    if prod.database == TEST_DB_NAME:
        return _psycopg_url(prod)
    admin = prod.set(database="postgres")
    admin_engine = create_engine(_psycopg_url(admin), isolation_level="AUTOCOMMIT")
    try:
        with admin_engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :n"),
                {"n": TEST_DB_NAME},
            ).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{TEST_DB_NAME}"'))
    finally:
        admin_engine.dispose()
    return _psycopg_url(prod.set(database=TEST_DB_NAME))


def _current_db_name(bind) -> str:
    with bind.connect() as conn:
        return conn.execute(text("SELECT current_database()")).scalar()


def assert_test_database(bind=None) -> str:
    name = _current_db_name(bind or engine)
    if name != TEST_DB_NAME:
        raise RuntimeError(f"Тесты должны идти в {TEST_DB_NAME}, сейчас {name}")
    if name == PROD_DB_NAME:
        raise RuntimeError("Отказ: подключена рабочая БД most_db")
    return name


def snapshot_prod_corpus() -> None:
    global _prod_edu_names, _prod_law_names
    _prod_edu_names = {p.name for p in PROD_EDU.glob("*") if p.is_file()}
    _prod_law_names = {p.name for p in PROD_LAW.glob("*") if p.is_file()}


def assert_prod_corpus_untouched() -> None:
    if _prod_edu_names is None:
        return
    edu_now = {p.name for p in PROD_EDU.glob("*") if p.is_file()}
    law_now = {p.name for p in PROD_LAW.glob("*") if p.is_file()}
    if edu_now != _prod_edu_names or law_now != _prod_law_names:
        raise AssertionError("Тесты изменили рабочий корпус data/corpus")


def assert_isolated_path(path: Path | str | None) -> None:
    if not path:
        return
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (BASE_DIR / resolved).resolve()
    else:
        resolved = resolved.resolve()
    if PROD_CORPUS == resolved or PROD_CORPUS in resolved.parents:
        raise AssertionError(f"Запись в рабочий корпус запрещена: {resolved}")


def bind_temp_corpus(root: Path) -> Path:
    corpus_root = root / "corpus"
    corpus_svc.configure_corpus_root(corpus_root)
    intake_svc.STASH_DIR = root / "stash"
    parser_svc.PARSED_DIR = root / "parsed"
    corpus_svc.ensure_corpus_layout()
    intake_svc.STASH_DIR.mkdir(parents=True, exist_ok=True)
    parser_svc.PARSED_DIR.mkdir(parents=True, exist_ok=True)
    assert_isolated_path(corpus_svc.EDU_DIR)
    assert_isolated_path(corpus_svc.LAW_DIR)
    return corpus_root


def activate() -> None:
    global _activated, engine, SessionLocal, TMP_ROOT
    if _activated:
        assert_test_database()
        return
    os.environ.setdefault("MOST_EMBEDDING_OFFLINE", "1")
    snapshot_prod_corpus()
    tmp_parent = BASE_DIR / "tests" / "_tmp"
    tmp_parent.mkdir(parents=True, exist_ok=True)
    TMP_ROOT = Path(tempfile.mkdtemp(prefix="ukz15_", dir=tmp_parent))
    bind_temp_corpus(TMP_ROOT)

    test_url = _ensure_test_database()
    test_engine = create_engine(test_url)
    db_name = _current_db_name(test_engine)
    if db_name != TEST_DB_NAME or db_name == PROD_DB_NAME:
        test_engine.dispose()
        raise RuntimeError(f"Отказ инициализировать тесты на БД {db_name}")

    with test_engine.begin() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=test_engine)
    ensure_knowledge_schema(test_engine)
    ensure_search_indexes(test_engine)

    import app.db as dbmod

    dbmod.engine = test_engine
    dbmod.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    engine = test_engine
    SessionLocal = dbmod.SessionLocal
    _activated = True
    atexit.register(_cleanup_tmp)


def _cleanup_tmp() -> None:
    try:
        assert_prod_corpus_untouched()
    except AssertionError:
        pass
    if TMP_ROOT and TMP_ROOT.exists() and "tests" in TMP_ROOT.parts and "_tmp" in TMP_ROOT.parts:
        shutil.rmtree(TMP_ROOT, ignore_errors=True)


def session():
    activate()
    return SessionLocal()
