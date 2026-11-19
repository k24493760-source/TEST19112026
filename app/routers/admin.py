"""Админ загрузки корпуса: validate/commit/status/url/relationships."""

from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import KnowledgeChunk, KnowledgeDocument
from app.services.corpus import find_live_duplicate
from app.services.intake import (
    IntakeError,
    inspect_payload,
    intake_document,
    read_stash,
    stash_bytes,
)
from app.services.auth import require_permission

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

ui_router = APIRouter(tags=["admin-ui"])
router = APIRouter(prefix="/api/v1/most/admin", tags=["most-admin"])

MAX_DOWNLOAD = 10 * 1024 * 1024


class CommitMetadata(BaseModel):
    title: str
    document_class: str | None = None
    owner: str = "admin"
    valid_from: datetime | None = None
    official_url: str | None = None

    model_config = {"populate_by_name": True}


class CommitRequest(BaseModel):
    file_hash: str
    file_name: str = "document.txt"
    metadata: dict = Field(default_factory=dict)
    version_action: str = "new"
    related_document_id: str | None = None
    actor_id: str = "admin-ui"


class UrlUploadRequest(BaseModel):
    url: str
    metadata: dict = Field(default_factory=dict)
    version_action: str = "new"
    related_document_id: str | None = None
    actor_id: str = "admin-ui"


def _http_intake(exc: IntakeError) -> HTTPException:
    return HTTPException(status_code=exc.http_status, detail=exc.message)


def _parse_valid_from(value) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", ""))
        except ValueError:
            return None
    return None


def _intake_from_bytes(
    db: Session,
    raw: bytes,
    file_name: str,
    metadata: dict,
    version_action: str,
    related_document_id: str | None,
    actor_id: str,
    content_type: str | None = None,
):
    meta_class = metadata.get("class") or metadata.get("document_class") or ""
    try:
        result = intake_document(
            db,
            raw,
            file_name,
            title=(metadata.get("title") or Path(file_name).stem or "Документ").strip(),
            document_class=str(meta_class),
            corpus_source=metadata.get("corpus_source"),
            owner=(metadata.get("owner") or "admin").strip(),
            official_url=metadata.get("official_url"),
            valid_from=_parse_valid_from(metadata.get("valid_from")),
            version_action=version_action,
            related_document_id=related_document_id,
            actor_id=actor_id,
            content_type=content_type,
        )
    except IntakeError as exc:
        raise _http_intake(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@ui_router.get("/ui/admin/upload", response_class=HTMLResponse)
def admin_upload_page(request: Request, user=Depends(require_permission("upload_document"))):
    return templates.TemplateResponse("admin_upload.html", {"request": request, "user_login": user.login})


@router.post("/upload/validate")
async def admin_validate(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user=Depends(require_permission("upload_document")),
):
    raw = await file.read()
    try:
        inspection = inspect_payload(raw, file.filename or "doc.bin", file.content_type)
    except IntakeError as exc:
        raise _http_intake(exc) from exc
    existing = find_live_duplicate(db, inspection.digest)
    if existing is not None:
        return {
            "status": "duplicate",
            "existing_document_id": existing.id,
            "title": existing.title,
        }
    stash_bytes(raw, inspection.digest)
    return {
        "status": "unique",
        "hash": inspection.digest,
        "text_preview": inspection.preview,
        "needs_ocr": inspection.needs_ocr,
        "privacy_hold": inspection.privacy_hold,
        "format": inspection.format,
    }


@router.post("/upload/commit")
def admin_commit(
    payload: CommitRequest,
    db: Session = Depends(get_db),
    user=Depends(require_permission("upload_document")),
):
    try:
        raw = read_stash(payload.file_hash)
    except IntakeError as exc:
        raise _http_intake(exc) from exc
    result = _intake_from_bytes(
        db,
        raw,
        payload.file_name,
        payload.metadata,
        payload.version_action,
        payload.related_document_id,
        user.login,
    )
    if result.duplicate:
        raise HTTPException(
            status_code=409,
            detail={"status": "duplicate", "existing_document_id": result.document_id},
        )
    return {
        "document_id": result.document_id,
        "status": result.status,
        "message": result.message,
        "review_task_id": result.review_task_id,
        "needs_ocr": result.needs_ocr,
        "privacy_hold": result.privacy_hold,
        "n_chunks": result.n_chunks,
    }


@router.get("/upload/status/{document_id}")
def admin_status(
    document_id: str,
    db: Session = Depends(get_db),
    user=Depends(require_permission("upload_document")),
):
    doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == document_id).first()
    if doc is None:
        raise HTTPException(status_code=404, detail="document_not_found")
    chunks = (
        db.query(KnowledgeChunk)
        .filter(KnowledgeChunk.document_id == document_id)
        .count()
    )
    target = 5
    percent = min(100, int(chunks / target * 100)) if target else 0
    if doc.status in {"ready_for_review", "approved", "active"}:
        percent = 100
    return {
        "document_id": document_id,
        "status": doc.status,
        "chunks_created": chunks,
        "progress_percent": percent,
    }


@router.post("/upload/url")
def admin_upload_url(
    payload: UrlUploadRequest,
    db: Session = Depends(get_db),
    user=Depends(require_permission("upload_document")),
):
    try:
        req = Request(payload.url, method="GET")
        with urlopen(req, timeout=15) as resp:
            raw = resp.read(MAX_DOWNLOAD + 1)
    except (HTTPError, URLError, TimeoutError, ValueError, OSError):
        raise HTTPException(status_code=400, detail="Не удалось скачать по ссылке")
    if not raw or len(raw) > MAX_DOWNLOAD:
        raise HTTPException(status_code=400, detail="Не удалось скачать по ссылке")
    name = Path(payload.url.split("?")[0]).name or "remote.bin"
    if not (payload.metadata or {}).get("corpus_source"):
        try:
            inspection = inspect_payload(raw, name)
        except IntakeError as exc:
            raise _http_intake(exc) from exc
        existing = find_live_duplicate(db, inspection.digest)
        if existing is not None:
            return {
                "status": "duplicate",
                "existing_document_id": existing.id,
                "title": existing.title,
            }
        stash_bytes(raw, inspection.digest)
        return {
            "status": "unique",
            "hash": inspection.digest,
            "text_preview": inspection.preview,
            "file_name": name,
            "needs_ocr": inspection.needs_ocr,
            "privacy_hold": inspection.privacy_hold,
        }
    result = _intake_from_bytes(
        db,
        raw,
        name,
        payload.metadata,
        payload.version_action,
        payload.related_document_id,
        user.login,
    )
    if result.duplicate:
        return {
            "status": "duplicate",
            "existing_document_id": result.document_id,
            "title": result.existing_title,
        }
    return {
        "document_id": result.document_id,
        "status": result.status,
        "message": result.message,
        "review_task_id": result.review_task_id,
        "hash": result.file_hash,
        "needs_ocr": result.needs_ocr,
        "privacy_hold": result.privacy_hold,
        "n_chunks": result.n_chunks,
    }


@router.get("/documents/{document_id}/relationships")
def admin_relationships(
    document_id: str,
    db: Session = Depends(get_db),
    user=Depends(require_permission("manage_corpus")),
):
    doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == document_id).first()
    if doc is None:
        raise HTTPException(status_code=404, detail="document_not_found")
    newer = (
        db.query(KnowledgeDocument)
        .filter(KnowledgeDocument.related_document_id == document_id)
        .order_by(KnowledgeDocument.created_at.desc())
        .first()
    )
    return {
        "is_referenced_in_answers": 0,
        "is_referenced_in_tests": 0,
        "has_newer_version": newer is not None,
        "newer_version_id": newer.id if newer else None,
    }


@router.get("/uploads")
def admin_uploads(db: Session = Depends(get_db), user=Depends(require_permission("upload_document"))):
    rows = (
        db.query(KnowledgeDocument)
        .order_by(KnowledgeDocument.created_at.desc())
        .limit(100)
        .all()
    )
    items = []
    for doc in rows:
        items.append(
            {
                "document_id": doc.id,
                "title": doc.title,
                "status": doc.status,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "version_action": doc.version_action,
            }
        )
    return {"items": items}
