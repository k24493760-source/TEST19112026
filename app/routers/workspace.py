"""Логи пайплайна и чанки для рабочего пространства."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import KnowledgeChunk, KnowledgeDocument, UploadHistory

router = APIRouter(prefix="/api/v1/most/workspace", tags=["most-workspace"])


@router.get("/document/{document_id}/pipeline_log")
def pipeline_log(document_id: str, db: Session = Depends(get_db)):
    doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == document_id).first()
    if doc is None:
        raise HTTPException(status_code=404, detail="document_not_found")
    rows = (
        db.query(UploadHistory)
        .filter(UploadHistory.document_id == document_id)
        .order_by(UploadHistory.timestamp.asc())
        .all()
    )
    items = []
    for row in rows:
        details = row.details if isinstance(row.details, dict) else {}
        items.append(
            {
                "action": row.action,
                "status": details.get("status") or "success",
                "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                "details": details,
            }
        )
    return {"document_id": document_id, "items": items}


@router.get("/document/{document_id}/chunks")
def workspace_chunks(document_id: str, db: Session = Depends(get_db)):
    doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == document_id).first()
    if doc is None:
        raise HTTPException(status_code=404, detail="document_not_found")
    rows = (
        db.query(KnowledgeChunk)
        .filter(KnowledgeChunk.document_id == document_id)
        .order_by(KnowledgeChunk.page.asc())
        .all()
    )
    items = []
    for chunk in rows:
        text = chunk.text or ""
        items.append(
            {
                "id": chunk.id,
                "section": chunk.section,
                "page": chunk.page,
                "text_preview": text[:300],
                "embedding_status": chunk.embedding_status or "pending",
            }
        )
    return {"document_id": document_id, "items": items}
