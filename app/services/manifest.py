"""Демонстрационный индекс корпуса для утверждения."""

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session, selectinload

from app.config import BASE_DIR
from app.models import KnowledgeDocument
from app.services.corpus import document_class_label, document_is_valid

MANIFEST_PATH = BASE_DIR / "data" / "corpus_manifest.json"


def generate_manifest(db: Session) -> Path:
    documents = (
        db.query(KnowledgeDocument)
        .options(selectinload(KnowledgeDocument.chunks))
        .order_by(KnowledgeDocument.created_at.desc())
        .all()
    )
    entries = []
    for doc in documents:
        if (doc.status or "") == "archived":
            continue
        if not document_is_valid(doc) and doc.status != "approved":
            continue
        entries.append(
            {
                "id": doc.id,
                "title": doc.title,
                "document_class": document_class_label(doc.document_class),
                "version": doc.version or "1.0",
                "status": doc.status,
                "hash": doc.hash,
                "file_format": doc.file_format,
                "chunk_count": len(doc.chunks),
            }
        )
    payload = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "total_documents": len(entries),
        "total_chunks": sum(item["chunk_count"] for item in entries),
        "documents": entries,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return MANIFEST_PATH
