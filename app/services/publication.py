"""Публикация CorpusRelease. Не переключает retrieve_chunks на chunk_embeddings."""
from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.orm import Session, selectinload

from app.config import BASE_DIR, EMBEDDING_DIM
from app.models import (
    AuditEvent,
    ChunkEmbedding,
    CorpusRelease,
    CorpusReleaseItem,
    DocumentParseRun,
    ExpertReview,
    KnowledgeChunk,
    KnowledgeDocument,
    generate_uuid,
)
from app.services.approval import admin_roles, can, missing_approval_roles
from app.services.corpus import sha256_bytes
from app.services.embedding_pipeline import count_tokens_with_special, load_embedding_config
from app.services.intake import _privacy_hit

logger = logging.getLogger("most.publication")
PROD_DB = "most_db"


class PublicationError(Exception):
    def __init__(self, code: str, message: str, errors: list[str] | None = None, http_status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.errors = errors or [message]
        self.http_status = http_status


@dataclass
class PublishResult:
    release_id: str
    version: str
    status: str
    previous_version: str | None
    duplicate: bool = False
    audit_event_id: str | None = None
    n_documents: int = 0
    n_chunks: int = 0
    n_embeddings: int = 0


def refuse_prod_database(db: Session) -> str:
    name = db.execute(text("SELECT current_database()")).scalar()
    if name == PROD_DB:
        raise PublicationError("prod_forbidden", f"Публикация 2.5А запрещена на {name}.")
    return str(name)


def current_published_release(db: Session) -> CorpusRelease | None:
    return (
        db.query(CorpusRelease)
        .filter(CorpusRelease.status == "published")
        .order_by(CorpusRelease.published_at.desc().nullslast(), CorpusRelease.created_at.desc())
        .first()
    )


def resolve_search_release(db: Session, corpus_version: str | None = None) -> CorpusRelease | None:
    """Обычный поиск — единственный published. Явная версия — снимок published/superseded."""
    requested = (corpus_version or "").strip()
    if not requested:
        rel = current_published_release(db)
        if rel is None or rel.status != "published":
            return None
        return rel
    if requested == "working-draft":
        return None
    rel = db.query(CorpusRelease).filter(CorpusRelease.version == requested).one_or_none()
    if rel is None or rel.status not in {"published", "superseded"}:
        return None
    return rel


def _audit(db: Session, actor: str, action: str, object_id: str, payload: dict) -> AuditEvent:
    event = AuditEvent(actor=actor, action=action, object=object_id, payload=payload)
    db.add(event)
    db.flush()
    return event


def _next_version(db: Session) -> str:
    rows = db.query(CorpusRelease.version).all()
    numbers = []
    for (ver,) in rows:
        if ver and ver.startswith("corpus-") and ver[7:].isdigit():
            numbers.append(int(ver[7:]))
    nxt = (max(numbers) + 1) if numbers else 1
    return f"corpus-{nxt}"


def collect_preflight_errors(db: Session, document_ids: list[str]) -> list[str]:
    errors: list[str] = []
    cfg = load_embedding_config()
    max_seq = int(cfg.get("max_seq_length") or 256)
    expected_model = str(cfg.get("embedding_model") or "")
    models: set[tuple[str, str, str]] = set()
    if not document_ids:
        return ["empty_release: нет документов для публикации"]

    for doc_id in document_ids:
        doc = (
            db.query(KnowledgeDocument)
            .options(selectinload(KnowledgeDocument.chunks))
            .filter(KnowledgeDocument.id == doc_id)
            .one_or_none()
        )
        if doc is None:
            errors.append(f"{doc_id}: document_not_found")
            continue
        if (doc.status or "") == "draft":
            errors.append(f"{doc.id}: draft_not_publishable")
        if (doc.status or "") != "approved":
            errors.append(f"{doc.id}: document_not_approved ({doc.status})")
        missing = missing_approval_roles(db, doc)
        if missing:
            errors.append(f"{doc.id}: experts_incomplete ({','.join(missing)})")
        conflicts = (
            db.query(ExpertReview)
            .filter(
                ExpertReview.object_id == doc.id,
                ExpertReview.reason == "source_conflict",
                ExpertReview.status.in_(("pending", "open")),
            )
            .first()
        )
        if conflicts:
            errors.append(f"{doc.id}: unresolved_source_conflict")
        privacy = (
            db.query(ExpertReview)
            .filter(
                ExpertReview.object_id == doc.id,
                ExpertReview.reason == "privacy_hold",
                ExpertReview.status.in_(("pending", "open")),
            )
            .first()
        )
        if privacy:
            errors.append(f"{doc.id}: privacy_hold")
        if doc.file_path:
            path = Path(doc.file_path)
            if not path.is_absolute():
                path = BASE_DIR / doc.file_path
            if path.is_file():
                digest = sha256_bytes(path.read_bytes())
                stored = (doc.hash or doc.file_hash or "").strip()
                if stored and digest != stored:
                    errors.append(f"{doc.id}: hash_mismatch")
            else:
                errors.append(f"{doc.id}: original_missing")
        run = (
            db.query(DocumentParseRun)
            .filter(
                DocumentParseRun.document_id == doc.id,
                DocumentParseRun.is_current.is_(True),
            )
            .first()
        )
        if run is None:
            errors.append(f"{doc.id}: parse_missing")
        else:
            if (run.status or "") in {"error", "failed"}:
                errors.append(f"{doc.id}: parse_error")
            if run.errors:
                errors.append(f"{doc.id}: parse_blocking_errors")
        chunks = [c for c in (doc.chunks or []) if c.approval_status == "approved"]
        if not chunks:
            errors.append(f"{doc.id}: no_approved_chunks")
        for chunk in chunks:
            if not chunk.source_element_ids:
                errors.append(f"{chunk.id}: no_source_elements")
            retrieval = chunk.retrieval_text or chunk.text or ""
            if _privacy_hit(retrieval):
                errors.append(f"{chunk.id}: sensitive_data")
            segs = (
                db.query(ChunkEmbedding)
                .filter(
                    ChunkEmbedding.chunk_id == chunk.id,
                    ChunkEmbedding.embedding_status.in_(("ready", "approved")),
                )
                .order_by(ChunkEmbedding.segment_index)
                .all()
            )
            if not segs:
                errors.append(f"{chunk.id}: embeddings_missing")
                continue
            rebuilt = "".join((chunk.retrieval_text or chunk.text or "")[s.char_start : s.char_end] for s in segs)
            source_text = chunk.retrieval_text or chunk.text or ""
            if segs[0].char_start == 0 and segs[-1].char_end == len(source_text):
                if rebuilt != source_text:
                    errors.append(f"{chunk.id}: text_loss")
            for seg in segs:
                models.add((seg.embedding_model, seg.model_revision, seg.embedding_config_version))
                if expected_model and seg.embedding_model != expected_model:
                    errors.append(f"{seg.embedding_id}: model_mismatch")
                ntok = seg.token_count or count_tokens_with_special(seg.segment_text)
                if ntok > max_seq:
                    errors.append(f"{seg.embedding_id}: truncated_or_overlong")
                values = list(seg.embedding) if seg.embedding is not None else []
                if len(values) != EMBEDDING_DIM or not all(math.isfinite(float(v)) for v in values):
                    errors.append(f"{seg.embedding_id}: bad_vector")
    if len(models) > 1:
        errors.append("mixed_embedding_models")
    return errors


def build_manifest(db: Session, document_ids: list[str]) -> dict:
    docs = []
    for doc_id in sorted(document_ids):
        doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == doc_id).one()
        chunks = [c for c in doc.chunks if c.approval_status == "approved"]
        chunk_rows = []
        for chunk in sorted(chunks, key=lambda c: c.id):
            segs = (
                db.query(ChunkEmbedding)
                .filter(
                    ChunkEmbedding.chunk_id == chunk.id,
                    ChunkEmbedding.embedding_status.in_(("ready", "approved")),
                )
                .order_by(ChunkEmbedding.segment_index)
                .all()
            )
            chunk_rows.append(
                {
                    "chunk_id": chunk.id,
                    "chunking_version": chunk.chunking_version,
                    "embedding_ids": [s.embedding_id for s in segs],
                }
            )
        docs.append(
            {
                "document_id": doc.id,
                "document_version": doc.version,
                "hash": doc.hash or doc.file_hash,
                "chunks": chunk_rows,
            }
        )
    cfg = load_embedding_config()
    return {
        "documents": docs,
        "embedding_model": cfg.get("embedding_model"),
        "embedding_config_version": cfg.get("embedding_config_version"),
    }


def _manifest_hash(manifest: dict) -> str:
    blob = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def publish_corpus(
    db: Session,
    document_ids: list[str],
    actor_id: str,
    actor_role: str,
    notes: str = "",
) -> PublishResult:
    refuse_prod_database(db)
    if not can(actor_role, "publish_release") or actor_role not in admin_roles():
        raise PublicationError("forbidden", "Публиковать корпус может только администратор базы знаний.", 403)
    errors = collect_preflight_errors(db, document_ids)
    if errors:
        raise PublicationError("preflight_failed", "Публикация заблокирована.", errors=errors, http_status=409)

    manifest = build_manifest(db, document_ids)
    digest = _manifest_hash(manifest)
    existing = (
        db.query(CorpusRelease)
        .filter(CorpusRelease.manifest_hash == digest, CorpusRelease.status.in_(("published", "superseded")))
        .order_by(CorpusRelease.published_at.desc().nullslast())
        .first()
    )
    current = current_published_release(db)
    if existing is not None:
        event = _audit(
            db,
            actor_id,
            "corpus_publish_idempotent",
            existing.id,
            {"version": existing.version, "manifest_hash": digest, "actor_role": actor_role},
        )
        db.commit()
        return PublishResult(
            release_id=existing.id,
            version=existing.version,
            status=existing.status,
            previous_version=existing.previous_version,
            duplicate=True,
            audit_event_id=event.id,
            n_documents=len(manifest["documents"]),
        )

    cfg = load_embedding_config()
    revision = ""
    for doc_row in manifest["documents"]:
        for chunk_row in doc_row["chunks"]:
            if chunk_row["embedding_ids"]:
                seg0 = (
                    db.query(ChunkEmbedding)
                    .filter(ChunkEmbedding.embedding_id == chunk_row["embedding_ids"][0])
                    .one()
                )
                revision = seg0.model_revision
                break
        if revision:
            break
    now = datetime.utcnow()
    version = _next_version(db)
    release = CorpusRelease(
        id=generate_uuid(),
        version=version,
        status="draft",
        notes=notes or None,
        published_by=actor_id,
        embedding_model=str(cfg.get("embedding_model")),
        model_revision=revision,
        embedding_config_version=str(cfg.get("embedding_config_version")),
        previous_version=current.version if current else None,
        manifest_hash=digest,
        snapshot=manifest,
    )
    db.add(release)
    db.flush()
    n_chunks = 0
    n_emb = 0
    for doc_row in manifest["documents"]:
        for chunk_row in doc_row["chunks"]:
            n_chunks += 1
            emb_ids = chunk_row["embedding_ids"] or [None]
            for emb_id in emb_ids:
                if emb_id:
                    n_emb += 1
                    seg = db.query(ChunkEmbedding).filter(ChunkEmbedding.embedding_id == emb_id).one()
                    if seg.embedding_status != "approved":
                        seg.embedding_status = "approved"
                db.add(
                    CorpusReleaseItem(
                        release_id=release.id,
                        document_id=doc_row["document_id"],
                        document_version=doc_row["document_version"],
                        chunk_id=chunk_row["chunk_id"],
                        embedding_id=emb_id,
                        chunking_version=chunk_row["chunking_version"],
                    )
                )

    try:
        _activate_published(db, release, current, now)
        event = _audit(
            db,
            actor_id,
            "corpus_publish",
            release.id,
            {
                "version": version,
                "previous_version": release.previous_version,
                "manifest_hash": digest,
                "n_documents": len(document_ids),
                "actor_role": actor_role,
            },
        )
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("publish failed, previous release kept: %s", exc)
        raise PublicationError("publish_failed", f"Публикация не применена: {exc}") from exc

    return PublishResult(
        release_id=release.id,
        version=release.version,
        status="published",
        previous_version=release.previous_version,
        audit_event_id=event.id,
        n_documents=len(document_ids),
        n_chunks=n_chunks,
        n_embeddings=n_emb,
    )


def _activate_published(
    db: Session,
    release: CorpusRelease,
    current: CorpusRelease | None,
    now: datetime,
) -> None:
    if current is not None:
        current.status = "superseded"
    release.status = "published"
    release.published_at = now
    working = db.query(CorpusRelease).filter(CorpusRelease.version == "working-draft").one_or_none()
    if working is not None and working.id != release.id:
        working.status = "draft"


def retrieval_contract() -> dict:
    return {
        "wired_to_retrieve_chunks": True,
        "select": {
            "corpus_release.status": "published",
            "knowledge_documents.status": "approved",
            "knowledge_chunks.approval_status": "approved",
            "chunk_embeddings.embedding_status": "approved",
            "chunk_embeddings.embedding_model": "<release.embedding_model>",
            "chunk_embeddings.model_revision": "<release.model_revision>",
        },
        "exclude": ["draft", "ready_for_review", "rejected", "working-draft"],
    }
