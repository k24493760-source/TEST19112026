from datetime import datetime, timedelta
from pathlib import Path
import time
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import func, or_
from sqlalchemy.orm import Session, selectinload

from app.db import get_db
from app.embedder import apply_embeddings, embedding_as_list
from app.models import (
    AuditEvent,
    ChatMessage,
    ChatSession,
    ExpertReview,
    KnowledgeChunk,
    KnowledgeDocument,
    RagAuditLog,
    User,
)
from app.services.manifest import generate_manifest
from app.services.rag import AskResponse, NoEvidenceResponse, compose_rag_answer
from app.services.search import retrieve_chunks
from app.services.verification import apply_citation_verification
from app.config import BASE_DIR, EMBEDDING_DIM
from app.services.corpus import (
    apply_related_on_approve,
    audit_documents,
    document_audit_rows,
    extract_and_normalize,
    normalize_text,
)
from app.services.intake import IntakeError, intake_document
from app.services.auth import require_document_approval, require_permission

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

ui_router = APIRouter(tags=["most"])
router = APIRouter(prefix="/api/v1/most", tags=["most"])

UNSUPPORTED_UPLOAD = "Поддерживаются файлы .txt, .md, .docx и .pdf"

ORIGINAL_MEDIA_TYPES = {
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pdf": "application/pdf",
}


LEARNER_ROLES = {
    "employee_enterprise",
    "employee_medorg",
    "director_enterprise",
    "director_medorg",
    "sro",
}


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    user_context: dict | None = None
    learner_role: str = "unknown"
    allowed_corpus_version: str | None = ""
    corpus_version: str | None = None


def _archive_document(doc: KnowledgeDocument, when: datetime | None = None) -> None:
    now = when or datetime.utcnow()
    doc.status = "archived"
    doc.valid_to = now
    for chunk in doc.chunks:
        chunk.approval_status = "archived"


def _previous_approved(db: Session, title: str, exclude_id: str | None = None) -> list[KnowledgeDocument]:
    query = (
        db.query(KnowledgeDocument)
        .filter(func.lower(KnowledgeDocument.title) == title.lower())
        .filter(KnowledgeDocument.status == "approved")
    )
    if exclude_id:
        query = query.filter(KnowledgeDocument.id != exclude_id)
    return query.all()


def _get_document(db: Session, doc_id: str) -> KnowledgeDocument:
    doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == doc_id).first()
    if doc is None:
        raise HTTPException(status_code=404, detail="Документ не найден.")
    return doc


def _original_path(doc: KnowledgeDocument) -> Path:
    if not doc.file_path:
        raise HTTPException(status_code=404, detail="Оригинал файла не сохранён.")
    root = BASE_DIR.resolve()
    path = (BASE_DIR / doc.file_path).resolve()
    if path != root and root not in path.parents:
        raise HTTPException(status_code=400, detail="Некорректный путь к файлу.")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Файл не найден в data/corpus.")
    return path


def _normalized_text(doc: KnowledgeDocument) -> str:
    if doc.file_path:
        path = (BASE_DIR / doc.file_path).resolve()
        if path.is_file():
            suffix = path.suffix.lower() or f".{(doc.file_format or 'txt').lower()}"
            return extract_and_normalize(path.read_bytes(), suffix)
    pieces = [chunk.text or "" for chunk in sorted(doc.chunks, key=lambda item: item.page or 0)]
    if pieces:
        return "\n\n".join(pieces)
    raise HTTPException(status_code=404, detail="Нормализованный текст недоступен.")


def _embedding_preview(vector) -> dict:
    values = embedding_as_list(vector) or []
    preview = ", ".join(f"{float(item):.4f}" for item in values[:8])
    if values:
        preview = f"[{preview}{' …' if len(values) > 8 else ''}]"
    return {
        "has_embedding": bool(values),
        "dim": len(values),
        "preview": preview or "вектор не посчитан",
    }


def _log_no_evidence(
    db: Session,
    question: str,
    reviewed_sources: list[dict],
    user_context: dict | None,
) -> ExpertReview:
    review = ExpertReview(
        object_type="ask",
        object_id=None,
        reason="no_basis",
        evidence={
            "question": question,
            "reviewed_sources": reviewed_sources,
            "user_context": user_context or {},
            "object_snapshot": {
                "question": question,
                "answer": "Основание не найдено",
                "claims": [],
            },
        },
        status="pending",
        priority="normal",
        sla_deadline=datetime.utcnow() + timedelta(hours=48),
        created_at=datetime.utcnow(),
    )
    db.add(review)
    db.flush()
    review.object_id = review.id
    db.add(
        AuditEvent(
            actor="system",
            action="ask_no_evidence",
            object=review.id,
            payload={
                "question": question,
                "review_id": review.id,
                "reviewed_sources": reviewed_sources,
            },
        )
    )
    db.commit()
    db.refresh(review)
    return review


def _context_user_id(user_context: dict | None) -> str | None:
    if not isinstance(user_context, dict):
        return None
    value = user_context.get("user_id")
    return str(value) if value else None


def _ensure_listener_user(db: Session, user_id: str | None) -> User:
    if user_id:
        user = db.query(User).filter(User.id == user_id).first()
        if user is not None:
            return user
    user = db.query(User).order_by(User.created_at.asc()).first()
    if user is not None:
        return user
    user = User(
        email="system-listener@local",
        login="system-listener",
        password_hash="!",
        role="listener",
        global_role="listener",
        name="Слушатель",
    )
    db.add(user)
    db.flush()
    return user


def _ensure_chat_session(
    db: Session,
    user: User,
    learner_role: str,
    corpus_version: str,
) -> ChatSession:
    now = datetime.utcnow()
    session = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user.id)
        .filter(or_(ChatSession.is_active.is_(True), ChatSession.is_active.is_(None)))
        .order_by(ChatSession.last_active_at.desc())
        .first()
    )
    if session is None:
        session = ChatSession(
            user_id=user.id,
            title="Диалог",
            learner_role=learner_role,
            allowed_corpus_version=corpus_version,
            started_at=now,
            last_active_at=now,
            is_active=True,
        )
        db.add(session)
        db.flush()
        return session
    session.last_active_at = now
    session.learner_role = learner_role or session.learner_role
    if corpus_version:
        session.allowed_corpus_version = corpus_version
    return session


def _persist_ask_trace(
    db: Session,
    *,
    session: ChatSession,
    user: User,
    request_id: str,
    question: str,
    answer_payload: dict,
    chunk_ids: list[str],
    latency_ms: int,
    error_code: str | None,
    review_required: bool,
) -> None:
    now = datetime.utcnow()
    db.add(
        ChatMessage(
            session_id=session.id,
            role="user",
            content={"text": question},
            review_required=False,
            is_appealed=False,
        )
    )
    db.add(
        ChatMessage(
            session_id=session.id,
            role="assistant",
            content=answer_payload,
            review_required=review_required,
            is_appealed=False,
        )
    )
    db.add(
        RagAuditLog(
            request_id=request_id,
            user_id=user.id,
            question_text=question,
            retrieved_chunk_ids=chunk_ids,
            search_score=None,
            error_code=error_code,
            latency_ms=latency_ms,
            timestamp=now,
        )
    )
    session.last_active_at = now
    db.commit()


@router.post("/ask", response_model=AskResponse | NoEvidenceResponse)
def ask(payload: AskRequest, db: Session = Depends(get_db)) -> AskResponse | NoEvidenceResponse:
    started = time.perf_counter()
    request_id = str(uuid.uuid4())
    question = payload.question.strip()
    if payload.learner_role not in LEARNER_ROLES and payload.learner_role != "unknown":
        raise HTTPException(status_code=400, detail="invalid_role")
    corpus_version = payload.allowed_corpus_version or payload.corpus_version or ""
    user = _ensure_listener_user(db, _context_user_id(payload.user_context))
    session = _ensure_chat_session(db, user, payload.learner_role, corpus_version)
    found = retrieve_chunks(db, question, corpus_version=corpus_version or None)
    live_corpus = found.corpus_version or corpus_version or None
    chunk_ids = [chunk.id for chunk, _doc in found.rows]
    db.add(
        AuditEvent(
            actor=user.id,
            action="ask_retrieve",
            object=request_id,
            payload={
                "corpus_version": live_corpus,
                "release_id": found.release_id,
                "chunk_ids": chunk_ids,
                "error_code": found.error_code,
            },
        )
    )
    if not found.rows:
        _log_no_evidence(db, question, found.reviewed_sources, payload.user_context)
        response = NoEvidenceResponse(
            reviewed_sources=found.reviewed_sources,
            corpus_version=live_corpus,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        try:
            _persist_ask_trace(
                db,
                session=session,
                user=user,
                request_id=request_id,
                question=question,
                answer_payload=response.model_dump(),
                chunk_ids=chunk_ids,
                latency_ms=latency_ms,
                error_code="no_basis",
                review_required=True,
            )
        except Exception:
            db.rollback()
        return response

    approved_count = (
        db.query(KnowledgeDocument)
        .filter(KnowledgeDocument.status == "approved")
        .count()
    )
    draft_count = (
        db.query(KnowledgeDocument)
        .filter(KnowledgeDocument.status == "draft")
        .count()
    )
    response = compose_rag_answer(
        question,
        found.rows,
        approved_count=approved_count,
        draft_count=0,
        learner_role=payload.learner_role,
        allowed_corpus_version=live_corpus or "",
    )
    verified = apply_citation_verification(response, db)
    latency_ms = int((time.perf_counter() - started) * 1000)
    error_code = None
    if getattr(verified, "answer_status", "") == "basis_not_found":
        error_code = "no_basis"
    try:
        _persist_ask_trace(
            db,
            session=session,
            user=user,
            request_id=request_id,
            question=question,
            answer_payload=verified.model_dump(),
            chunk_ids=chunk_ids,
            latency_ms=latency_ms,
            error_code=error_code,
            review_required=bool(getattr(verified, "review_required", False)),
        )
    except Exception:
        db.rollback()
    return verified


@ui_router.get("/ui/upload", response_class=HTMLResponse)
def upload_page(request: Request, user=Depends(require_permission("upload_document"))):
    return templates.TemplateResponse("upload.html", {"request": request, "user_login": user.login})


@ui_router.get("/ui/expert_queue", response_class=HTMLResponse)
def expert_queue(request: Request, db: Session = Depends(get_db)):
    drafts = (
        db.query(KnowledgeDocument)
        .options(selectinload(KnowledgeDocument.chunks))
        .filter(KnowledgeDocument.status == "draft")
        .order_by(KnowledgeDocument.created_at.desc())
        .all()
    )
    approved_by_title: dict[str, KnowledgeDocument] = {}
    for live in (
        db.query(KnowledgeDocument)
        .filter(KnowledgeDocument.status == "approved")
        .all()
    ):
        approved_by_title[(live.title or "").lower()] = live
    for draft in drafts:
        previous = approved_by_title.get((draft.title or "").lower())
        draft.is_version_update = previous is not None
        draft.replaces_version = previous.version if previous is not None else None
    reviews = (
        db.query(ExpertReview)
        .filter(ExpertReview.status.in_(("open", "pending")))
        .order_by(ExpertReview.created_at.desc())
        .all()
    )
    return templates.TemplateResponse(
        "expert.html",
        {"request": request, "documents": drafts, "reviews": reviews},
    )


@ui_router.get("/ui/registry", response_class=HTMLResponse)
def registry_page(request: Request):
    return templates.TemplateResponse("registry.html", {"request": request})


@ui_router.get("/ui/workspace", response_class=HTMLResponse)
def workspace_page(request: Request, db: Session = Depends(get_db)):
    documents = document_audit_rows(db)
    return templates.TemplateResponse(
        "workspace.html",
        {"request": request, "documents": documents},
    )


@router.get("/document/{doc_id}/original")
def document_original(
    doc_id: str,
    db: Session = Depends(get_db),
    user=Depends(require_permission("view_original")),
):
    doc = _get_document(db, doc_id)
    path = _original_path(doc)
    media_type = ORIGINAL_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(
        path=path,
        media_type=media_type,
        filename=path.name,
        content_disposition_type="inline",
    )


@router.get("/document/{doc_id}/normalized", response_class=HTMLResponse)
def document_normalized(doc_id: str, request: Request, db: Session = Depends(get_db)):
    doc = _get_document(db, doc_id)
    return templates.TemplateResponse(
        "document_normalized.html",
        {
            "request": request,
            "document": doc,
            "normalized_text": _normalized_text(doc),
        },
    )


@router.get("/document/{doc_id}/chunks", response_class=HTMLResponse)
def document_chunks(doc_id: str, request: Request, db: Session = Depends(get_db)):
    doc = (
        db.query(KnowledgeDocument)
        .options(selectinload(KnowledgeDocument.chunks))
        .filter(KnowledgeDocument.id == doc_id)
        .first()
    )
    if doc is None:
        raise HTTPException(status_code=404, detail="Документ не найден.")
    chunks = []
    for chunk in sorted(doc.chunks, key=lambda item: item.page or 0):
        info = _embedding_preview(chunk.embedding)
        chunks.append(
            {
                "page": chunk.page,
                "section": chunk.section,
                "text": chunk.text,
                "has_embedding": info["has_embedding"],
                "dim": info["dim"],
                "preview": info["preview"],
            }
        )
    embedded = sum(1 for item in chunks if item["has_embedding"])
    return templates.TemplateResponse(
        "document_chunks.html",
        {
            "request": request,
            "document": doc,
            "chunks": chunks,
            "embedded": embedded,
            "embedding_dim": EMBEDDING_DIM,
        },
    )


@router.get("/manifest")
def download_manifest(db: Session = Depends(get_db)):
    path = generate_manifest(db)
    return FileResponse(
        path=path,
        media_type="application/json; charset=utf-8",
        filename="corpus_manifest.json",
        content_disposition_type="attachment",
    )


@router.post("/audit")
def run_audit(db: Session = Depends(get_db)) -> dict:
    """Инвентаризация без пересборки чанков; досчёт только пустых эмбеддингов."""
    embedded = apply_embeddings(db)
    db.commit()
    rows = audit_documents(db)
    return {"documents": rows, "count": len(rows), "embedded": embedded}


def _http_intake(exc: IntakeError) -> HTTPException:
    status = 409 if exc.code in {"conflict"} or "уже загружен" in exc.message else exc.http_status
    if exc.code == "conflict":
        status = 409
    return HTTPException(status_code=status, detail=exc.message)


@router.post("/upload")
async def upload_document(
    title: str = Form(""),
    text: str = Form(""),
    document_class: str = Form("Учебный материал"),
    corpus_source: str = Form(""),
    related_document_id: str | None = Form(None),
    version_action: str = Form("new"),
    official_url: str | None = Form(None),
    file: UploadFile | None = File(None),
    db: Session = Depends(get_db),
    user=Depends(require_permission("upload_document")),
) -> dict:
    title = (title or "").strip()
    body = (text or "").strip()
    uploaded_name = file.filename if file is not None else ""
    raw = b""
    content_type = file.content_type if file is not None else "text/markdown"

    if uploaded_name:
        if not title:
            title = Path(uploaded_name).stem.strip()
        raw = await file.read()
    elif body:
        raw = normalize_text(body).encode("utf-8")
        uploaded_name = f"{title or 'document'}.md"
        content_type = "text/markdown"
    else:
        raise HTTPException(status_code=400, detail="Добавьте текст документа или загрузите файл.")

    try:
        result = intake_document(
            db,
            raw,
            uploaded_name,
            title=title,
            document_class=document_class,
            corpus_source=corpus_source,
            owner="Заказчик",
            official_url=official_url,
            version_action=version_action,
            related_document_id=related_document_id or None,
            actor_id=user.login,
            content_type=content_type,
        )
    except IntakeError as exc:
        raise _http_intake(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if result.duplicate:
        raise HTTPException(status_code=409, detail="Этот документ уже загружен")

    rows = document_audit_rows(db)
    return {
        "document_id": result.document_id,
        "status": result.status,
        "version": result.version,
        "approval_required": True,
        "message": result.message,
        "n_chunks": result.n_chunks,
        "needs_ocr": result.needs_ocr,
        "privacy_hold": result.privacy_hold,
        "documents": rows,
    }


@router.post("/archive/{doc_id}")
def archive_document(doc_id: str, db: Session = Depends(get_db)) -> dict:
    doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == doc_id).first()
    if doc is None:
        raise HTTPException(status_code=404, detail="Документ не найден.")
    if doc.status == "archived":
        raise HTTPException(status_code=409, detail="Документ уже в архиве.")

    _archive_document(doc)
    db.commit()
    return {
        "document_id": doc.id,
        "status": doc.status,
        "documents": document_audit_rows(db),
    }


@router.post("/approve/{doc_id}")
def approve_document(
    doc_id: str,
    actor_id: str = "legacy-ui",
    actor_role: str = "admin",
    db: Session = Depends(get_db),
    user=Depends(require_document_approval),
) -> dict:
    from app.services.approval import (
        ApprovalError,
        approve_document_as_grant,
        finalize_if_experts_complete,
        is_granted_document_approver,
    )

    doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == doc_id).first()
    if doc is None:
        raise HTTPException(status_code=404, detail="Документ не найден.")
    try:
        from app.services.auth import effective_role

        if is_granted_document_approver(user.login):
            result = approve_document_as_grant(db, doc_id, user.login, effective_role(user))
        else:
            result = finalize_if_experts_complete(db, doc_id, user.login, effective_role(user))
    except ApprovalError as exc:
        raise HTTPException(status_code=exc.http_status, detail=exc.code) from exc
    now = datetime.utcnow()
    previous = _previous_approved(db, doc.title, exclude_id=doc.id)
    apply_related_on_approve(db, doc)
    superseded_ids = []
    if result.status == "approved":
        for old in previous:
            _archive_document(old, now)
            superseded_ids.append(old.id)
        db.commit()
    return {
        "document_id": result.document_id,
        "status": result.status,
        "superseded_ids": superseded_ids,
    }
