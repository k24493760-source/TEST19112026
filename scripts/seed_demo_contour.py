"""Изолированный контур most_db_demo. В most_db не пишет и защиты не отключает."""
from __future__ import annotations

import math
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

os.environ["MOST_ALLOW_HF_DOWNLOAD"] = "1"
os.environ["MOST_EMBEDDING_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["MOST_SCHEMA_BOOTSTRAP"] = "0"

DEMO_DB = "most_db_demo"
DEMO_LOGIN = "demo"
DEMO_PASSWORD = "demo-ask-2026"
TITLE = "Демонстрационный учебный материал"
BODY = (
    "# Лекция 1\n\n"
    "Слушатель изучает порядок направления уведомления.\n\n"
    "# Лекция 2\n\n"
    "Срок составляет четырнадцать рабочих дней.\n"
)

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from app.config import BASE_DIR
from app.db import (
    Base,
    engine as prod_engine,
    ensure_expert_schema,
    ensure_knowledge_schema,
    ensure_search_indexes,
    ensure_storage_schema,
)
from app.models import (
    AuditEvent,
    ChunkEmbedding,
    CorpusRelease,
    ExpertReview,
    KnowledgeChunk,
    KnowledgeDocument,
    User,
)
from app.services.approval import apply_expert_decision, submit_for_review
from app.services.embedding_model import EMBEDDING_DIM, diagnose
from app.services.embedding_pipeline import vectorize_draft_chunks
from app.services.intake import intake_document
from app.services.passwords import hash_password
from app.services.publication import publish_corpus
from app.services.rag import compose_rag_answer
from app.services.search import retrieve_chunks
from tests.isolation import bind_temp_corpus, provision_named_database


def _prod_counts() -> tuple[int, int, int]:
    with prod_engine.connect() as conn:
        name = conn.execute(text("SELECT current_database()")).scalar()
        if name != "most_db":
            raise RuntimeError(f"Контрольное подключение не к most_db, а к {name}")
        return (
            conn.execute(text("SELECT count(*) FROM knowledge_documents")).scalar(),
            conn.execute(text("SELECT count(*) FROM knowledge_chunks")).scalar(),
            conn.execute(text("SELECT count(*) FROM audit_events")).scalar(),
        )


def _user(db, login: str, role: str, password: str) -> User:
    row = User(
        email=f"{login}@demo.local",
        login=login,
        password_hash=hash_password(password),
        role=role,
        global_role=role,
        name=login,
    )
    db.add(row)
    db.flush()
    return row


def main() -> None:
    before = _prod_counts()
    info = diagnose()
    print(f"model_cache status={info.status} dim={info.embedding_dim} allow_download={info.allow_download}")
    if info.embedding_dim != EMBEDDING_DIM:
        raise RuntimeError(f"Ожидалась размерность {EMBEDDING_DIM}, в конфиге {info.embedding_dim}")

    demo_root = BASE_DIR / "data" / "demo_contour"
    if demo_root.exists():
        shutil.rmtree(demo_root)
    bind_temp_corpus(demo_root)

    demo_url = provision_named_database(DEMO_DB)
    from sqlalchemy import create_engine

    demo_engine = create_engine(demo_url)
    with demo_engine.begin() as conn:
        current = conn.execute(text("SELECT current_database()")).scalar()
        if current != DEMO_DB:
            raise RuntimeError(f"Отказ: сессия открыта в {current}")
        conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=demo_engine)
    ensure_knowledge_schema(demo_engine)
    ensure_expert_schema(demo_engine)
    ensure_storage_schema(demo_engine)
    ensure_search_indexes(demo_engine)

    Session = sessionmaker(bind=demo_engine)
    db = Session()
    try:
        if db.execute(text("SELECT current_database()")).scalar() != DEMO_DB:
            raise RuntimeError("Запись отклонена: это не most_db_demo")
        listener = _user(db, DEMO_LOGIN, "listener", DEMO_PASSWORD)
        methodist = _user(db, "methodist-demo", "methodist", "methodist-demo-2026")
        admin = _user(db, "kb-admin-demo", "kb_admin", "kb-admin-demo-2026")
        db.commit()

        item = intake_document(
            db,
            BODY.encode("utf-8"),
            "demo-lectures.md",
            title=TITLE,
            document_class="Учебный материал",
            corpus_source="database_edu_lectcii",
            owner="демонстрационный контур",
            actor_id=admin.login,
            content_type="text/markdown",
        )
        print(f"intake id={item.document_id} chunks={item.n_chunks} status={item.status}")
        if item.n_chunks < 2:
            raise RuntimeError("Чанки не созданы")

        chunks = (
            db.query(KnowledgeChunk)
            .filter(KnowledgeChunk.document_id == item.document_id)
            .order_by(KnowledgeChunk.section)
            .all()
        )
        for chunk in chunks:
            print(
                f"chunk {chunk.id} section={chunk.section} status={chunk.approval_status} "
                f"evidence={chunk.evidence_text}"
            )
            if chunk.approval_status != "draft":
                raise RuntimeError("Векторизация допустима только для draft-чанков")

        vectorized = vectorize_draft_chunks(db, document_id=item.document_id)
        print(
            f"embeddings created={vectorized.created} skipped={vectorized.skipped} "
            f"revision={vectorized.model_revision} config={vectorized.embedding_config_version}"
        )
        stored = (
            db.query(ChunkEmbedding)
            .join(KnowledgeChunk, ChunkEmbedding.chunk_id == KnowledgeChunk.id)
            .filter(KnowledgeChunk.document_id == item.document_id)
            .all()
        )
        if not stored:
            raise RuntimeError("Embeddings не записаны")
        for row in stored:
            values = list(row.embedding)
            if len(values) != 384 or not all(math.isfinite(float(v)) for v in values):
                raise RuntimeError(f"Плохой вектор {row.embedding_id}")
            if (row.token_count or 0) > 256:
                raise RuntimeError(f"Сегмент длиннее 256 токенов: {row.embedding_id}")
            print(
                f"embedding {row.embedding_id} model={row.embedding_model} "
                f"revision={row.model_revision} tokens={row.token_count} dim={len(values)}"
            )

        submitted = submit_for_review(db, item.document_id, admin.id, "kb_admin")
        print(f"submit status={submitted.status} missing={submitted.missing_roles}")
        review = (
            db.query(ExpertReview)
            .filter(
                ExpertReview.object_id == item.document_id,
                ExpertReview.expert_role == "methodist",
                ExpertReview.status.in_(("pending", "open")),
            )
            .one()
        )
        decided = apply_expert_decision(
            db,
            review,
            expert_id=methodist.id,
            expert_role="methodist",
            decision="approve",
            comment="Методист подтвердил синтетический учебный материал демонстрационного контура.",
        )
        db.commit()
        print(f"expert status={decided.status} missing={decided.missing_roles}")
        if decided.status != "approved" or decided.missing_roles:
            raise RuntimeError("Экспертное утверждение не завершено")

        doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == item.document_id).one()
        approved_chunks = [c for c in doc.chunks if c.approval_status == "approved"]
        if doc.status != "approved" or len(approved_chunks) != len(list(doc.chunks)):
            raise RuntimeError("Документ или чанки не утверждены")
        for chunk in approved_chunks:
            if (chunk.document_version or doc.version) != doc.version:
                raise RuntimeError("Версия чанка не совпадает с документом")

        published = publish_corpus(
            db,
            [doc.id],
            admin.id,
            "kb_admin",
            notes="Демонстрационный релиз синтетического учебного материала",
        )
        print(
            f"release {published.version} status={published.status} "
            f"docs={published.n_documents} chunks={published.n_chunks} embeddings={published.n_embeddings}"
        )
        release = db.query(CorpusRelease).filter(CorpusRelease.id == published.release_id).one()
        if release.status != "published":
            raise RuntimeError("Релиз не опубликован")

        questions = [
            "Какой срок указан в лекции 2?",
            "Что изучает слушатель в лекции 1?",
            "Какая ставка налога на прибыль организаций?",
        ]
        for question in questions:
            found = retrieve_chunks(db, question, corpus_version=release.version)
            answer = compose_rag_answer(
                question,
                found.rows,
                approved_count=1,
                draft_count=0,
                learner_role="employee_enterprise",
                allowed_corpus_version=release.version,
            )
            top = found.rows[0] if found.rows else None
            print("---")
            print("Q", question)
            print("search_error", found.error_code, "hits", [c.id for c, _d in found.rows])
            print("answer_status", answer.answer_status, "mode", answer.answer_mode)
            print("answer", answer.answer)
            if top:
                print("top_section", top[0].section, "top_doc", top[1].title)
            if answer.citations:
                cit = answer.citations[0]
                print("citation", cit.document_title, cit.section, cit.quote, cit.chunk_id)

        audit = (
            db.query(AuditEvent)
            .filter(AuditEvent.action.in_(("expert_resolve", "corpus_publish", "document_submit_review")))
            .count()
        )
        print(f"audit_rows {audit}")
        print(f"login {DEMO_LOGIN}")
        print(f"listener_id {listener.id}")
    finally:
        db.close()
        demo_engine.dispose()

    after = _prod_counts()
    print(f"most_db_unchanged {before == after} before={before} after={after}")
    if before != after:
        raise RuntimeError("Рабочая база most_db изменилась")


if __name__ == "__main__":
    main()
