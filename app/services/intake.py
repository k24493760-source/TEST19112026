"""Единый приём документов (intake). Без парсинга, чанкинга и эмбеддингов."""

from __future__ import annotations

import logging
import re
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import BASE_DIR
from app.models import ExpertReview, KnowledgeChunk, KnowledgeDocument, UploadHistory
from app.services import corpus as corpus_svc
from app.services.corpus import (
    corpus_source_from_path,
    ensure_corpus_layout,
    find_live_duplicate,
    next_document_version,
    normalize_document_class,
    relative_corpus_path,
    safe_filename,
    sha256_bytes,
    write_file_registry,
)

logger = logging.getLogger("most.intake")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
STASH_DIR = BASE_DIR / "data" / "tmp_uploads"
STASH_TTL = timedelta(hours=24)
ALLOWED_SOURCES = {"database_law", "database_edu_lectcii"}
VERSION_ACTIONS = {"new", "new_version", "replace"}

MIME_BY_FORMAT = {
    "pdf": {"application/pdf"},
    "docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
        "application/octet-stream",
    },
    "txt": {"text/plain", "application/octet-stream"},
    "md": {"text/markdown", "text/plain", "application/octet-stream"},
}

PRIVACY_PATTERNS = (
    re.compile(r"снилс\s*[:№]?\s*\d{3}[-\s]?\d{3}[-\s]?\d{3}\s*\d{2}", re.I),
    re.compile(r"табельн\w*\s+номер\s*[:№]?\s*\d{3,}", re.I),
    re.compile(r"полис\s+(омc|омс)\s*[:№]?\s*\d{8,}", re.I),
    re.compile(r"дата\s+рождения\s*[:\s]+\d{1,2}[./]\d{1,2}[./]\d{2,4}", re.I),
)


class IntakeError(Exception):
    def __init__(self, code: str, message: str, http_status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


@dataclass
class FileInspection:
    digest: str
    format: str
    original_filename: str
    size_bytes: int
    needs_ocr: bool = False
    encrypted: bool = False
    privacy_hold: bool = False
    sample_chars: int = 0
    preview: str = ""


@dataclass
class IntakeResult:
    document_id: str
    status: str
    duplicate: bool
    file_hash: str
    file_path: str | None
    version: str | None = None
    review_task_id: str | None = None
    needs_ocr: bool = False
    privacy_hold: bool = False
    message: str = ""
    existing_title: str | None = None
    n_chunks: int = 0


def cleanup_stale_stashes(now: datetime | None = None) -> int:
    STASH_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = (now or datetime.utcnow()) - STASH_TTL
    removed = 0
    for path in STASH_DIR.glob("*.bin"):
        mtime = datetime.utcfromtimestamp(path.stat().st_mtime)
        if mtime < cutoff:
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def stash_bytes(raw: bytes, digest: str | None = None) -> tuple[str, Path]:
    cleanup_stale_stashes()
    digest = digest or sha256_bytes(raw)
    STASH_DIR.mkdir(parents=True, exist_ok=True)
    path = STASH_DIR / f"{digest}.bin"
    path.write_bytes(raw)
    return digest, path


def read_stash(digest: str) -> bytes:
    path = STASH_DIR / f"{digest}.bin"
    if not path.is_file():
        raise IntakeError("file_not_staged", "Файл не прошёл предварительную проверку.")
    return path.read_bytes()


def _safe_original_name(name: str) -> str:
    base = Path(str(name or "document").replace("\\", "/")).name
    if not base or base in {".", ".."}:
        return "document"
    return base[:180]


def _ext(name: str) -> str:
    return Path(_safe_original_name(name)).suffix.lower()


def _is_executable(raw: bytes) -> bool:
    return raw.startswith(b"MZ") or raw.startswith(b"\x7fELF")


def _sample_text(raw: bytes, fmt: str) -> str:
    if fmt in {"txt", "md"}:
        for enc in ("utf-8-sig", "utf-8", "cp1251"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return ""
    if fmt == "docx":
        try:
            from docx import Document

            paras = [p.text for p in Document(BytesIO(raw)).paragraphs[:80]]
            return "\n".join(paras)
        except Exception:
            return ""
    if fmt == "pdf":
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(BytesIO(raw))
            pages = []
            for page in reader.pages[:3]:
                pages.append(page.extract_text() or "")
            return "\n".join(pages)
        except Exception:
            return ""
    return ""


def _privacy_hit(sample: str) -> bool:
    if not sample:
        return False
    return any(pat.search(sample) for pat in PRIVACY_PATTERNS)


def inspect_payload(
    raw: bytes,
    filename: str,
    content_type: str | None = None,
) -> FileInspection:
    if not raw:
        raise IntakeError("empty_file", "Файл пустой.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise IntakeError("too_large", "Файл слишком большой (больше 10 МБ).")
    if _is_executable(raw):
        raise IntakeError("forbidden_type", "Исполняемые файлы запрещены.")

    original = _safe_original_name(filename)
    ext = _ext(original)
    mime = (content_type or "").split(";")[0].strip().lower()
    fmt: str | None = None
    needs_ocr = False

    if raw.startswith(b"%PDF") or ext == ".pdf" or mime == "application/pdf":
        if not raw.startswith(b"%PDF"):
            raise IntakeError("invalid_signature", "Сигнатура не соответствует PDF.")
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(BytesIO(raw))
        except Exception as exc:
            raise IntakeError("corrupt_file", "Повреждённый PDF.") from exc
        if getattr(reader, "is_encrypted", False):
            raise IntakeError("encrypted_pdf", "Зашифрованный PDF не принимается.")
        try:
            texts = [(page.extract_text() or "") for page in reader.pages]
        except Exception as exc:
            raise IntakeError("corrupt_file", "Не удалось прочитать PDF.") from exc
        if not any(t.strip() for t in texts):
            needs_ocr = True
        fmt = "pdf"
    elif raw[:2] == b"PK" or ext == ".docx" or "wordprocessingml" in mime:
        if raw[:2] != b"PK":
            raise IntakeError("invalid_signature", "Сигнатура не соответствует DOCX.")
        try:
            with zipfile.ZipFile(BytesIO(raw)) as archive:
                names = set(archive.namelist())
        except zipfile.BadZipFile as exc:
            raise IntakeError("corrupt_file", "Повреждённый DOCX.") from exc
        if "word/document.xml" not in names:
            raise IntakeError("unsupported_format", "ZIP без документа Word не принимается.")
        try:
            from docx import Document

            Document(BytesIO(raw))
        except Exception as exc:
            raise IntakeError("corrupt_file", "Повреждённый DOCX.") from exc
        fmt = "docx"
    elif ext in {".txt", ".md"} or mime in {"text/plain", "text/markdown"}:
        if b"\x00" in raw[:4096]:
            raise IntakeError("unsupported_format", "Бинарный файл вместо текста.")
        decoded = False
        for enc in ("utf-8-sig", "utf-8", "cp1251"):
            try:
                raw.decode(enc)
                decoded = True
                break
            except UnicodeDecodeError:
                continue
        if not decoded:
            raise IntakeError("corrupt_file", "Не удалось прочитать текстовый файл.")
        if ext == ".md" or mime == "text/markdown":
            fmt = "md"
        else:
            fmt = "txt"
    else:
        raise IntakeError(
            "unsupported_format",
            "Поддерживаются файлы .txt, .md, .docx и .pdf.",
        )

    if mime and mime not in MIME_BY_FORMAT[fmt] and mime not in {"", "application/octet-stream"}:
        if not ((fmt == "pdf" and raw.startswith(b"%PDF")) or (fmt == "docx" and raw[:2] == b"PK")):
            raise IntakeError("unsupported_format", "MIME-тип не соответствует содержимому.")

    sample = _sample_text(raw, fmt)
    privacy_hold = _privacy_hit(sample)
    return FileInspection(
        digest=sha256_bytes(raw),
        format=fmt,
        original_filename=original,
        size_bytes=len(raw),
        needs_ocr=needs_ocr,
        privacy_hold=privacy_hold,
        sample_chars=len(sample),
        preview="" if privacy_hold else (sample or "")[:500],
    )


def require_corpus_source(value: str | None) -> str:
    source = (value or "").strip()
    if not source:
        raise IntakeError("corpus_source_required", "Укажите источник корпуса.")
    if source not in ALLOWED_SOURCES:
        raise IntakeError("invalid_corpus_source", "Источник: database_law или database_edu_lectcii.")
    return source


def _lock_hash(db: Session, digest: str) -> None:
    bind = db.get_bind()
    if bind is None or not str(bind.url).startswith("postgresql"):
        return
    key = int(digest[:16], 16) % (2**31 - 1)
    db.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": key})


def _destination_dir(corpus_source: str, privacy_hold: bool) -> Path:
    ensure_corpus_layout()
    if privacy_hold:
        return corpus_svc.QUARANTINE_DIR
    if corpus_source == "database_law":
        return corpus_svc.LAW_DIR
    return corpus_svc.EDU_DIR


def _store_original(raw: bytes, inspection: FileInspection, corpus_source: str) -> Path:
    folder = _destination_dir(corpus_source, inspection.privacy_hold)
    folder.mkdir(parents=True, exist_ok=True)
    stem = safe_filename(inspection.original_filename)
    name = f"{stem}_{uuid.uuid4().hex[:12]}.{inspection.format}"
    dest = (folder / name).resolve()
    corpus_root = corpus_svc.CORPUS_DIR.resolve()
    if dest.parent != corpus_root and corpus_root not in dest.parents:
        raise IntakeError("unsafe_path", "Некорректный путь сохранения.")
    dest.write_bytes(raw)
    return dest


def intake_document(
    db: Session,
    raw: bytes,
    filename: str,
    *,
    title: str,
    document_class: str,
    corpus_source: str | None,
    owner: str = "загрузка через UI",
    official_url: str | None = None,
    valid_from: datetime | None = None,
    version_action: str = "new",
    related_document_id: str | None = None,
    actor_id: str = "intake",
    content_type: str | None = None,
    create_review: bool = True,
) -> IntakeResult:
    ensure_corpus_layout()
    cleanup_stale_stashes()
    inspection = inspect_payload(raw, filename, content_type)
    source = require_corpus_source(corpus_source)
    try:
        class_value = normalize_document_class(document_class)
    except ValueError as exc:
        raise IntakeError("invalid_class", str(exc)) from exc
    title = (title or Path(inspection.original_filename).stem or "Документ").strip()
    if not title:
        raise IntakeError("title_required", "Укажите название документа.")

    action = (version_action or "new").strip() or "new"
    if action not in VERSION_ACTIONS:
        raise IntakeError("invalid_version_action", "Некорректное действие версии.")
    related = None
    if action in {"new_version", "replace"}:
        if not related_document_id:
            raise IntakeError("related_document_id_required", "Укажите связанный документ.")
        related = (
            db.query(KnowledgeDocument)
            .filter(KnowledgeDocument.id == related_document_id)
            .first()
        )
        if related is None:
            raise IntakeError("related_document_not_found", "Связанный документ не найден.")

    _lock_hash(db, inspection.digest)
    duplicate = find_live_duplicate(db, inspection.digest)
    if duplicate is not None:
        logger.info(
            "intake duplicate hash=%s existing_id=%s",
            inspection.digest[:12],
            duplicate.id,
        )
        return IntakeResult(
            document_id=duplicate.id,
            status=duplicate.status or "draft",
            duplicate=True,
            file_hash=inspection.digest,
            file_path=duplicate.file_path,
            version=duplicate.version,
            message="Этот документ уже загружен",
            existing_title=duplicate.title,
            n_chunks=db.query(KnowledgeChunk)
            .filter(KnowledgeChunk.document_id == duplicate.id)
            .count(),
        )

    same_title = (
        db.query(KnowledgeDocument)
        .filter(KnowledgeDocument.title == title)
        .all()
    )
    version = "1.0"
    if action == "new_version" and related is not None:
        version = next_document_version([related.version or "1.0"])
    elif same_title and action == "new":
        version = next_document_version([item.version or "1.0" for item in same_title])

    saved: Path | None = None
    try:
        saved = _store_original(raw, inspection, source)
        if corpus_source_from_path(saved) == "unknown" and not inspection.privacy_hold:
            raise IntakeError("unsafe_path", "Файл должен сохраняться в каталоге корпуса.")
        doc = KnowledgeDocument(
            title=title,
            document_class=class_value,
            owner=(owner or "загрузка через UI").strip(),
            version=version,
            status="draft",
            hash=inspection.digest,
            file_hash=inspection.digest,
            file_path=relative_corpus_path(saved),
            file_format=inspection.format,
            original_filename=inspection.original_filename,
            corpus_source=source,
            official_url=official_url or None,
            valid_from=valid_from,
            version_action=action,
            related_document_id=related.id if related is not None and action != "new" else None,
            reviewed_at=None,
        )
        db.add(doc)
        db.flush()
        review_id = None
        reason = "privacy_hold" if inspection.privacy_hold else "new_document"
        if create_review:
            review = ExpertReview(
                object_type="knowledge_document",
                object_id=doc.id,
                reason=reason,
                status="pending",
                evidence={
                    "title": title,
                    "file_hash": inspection.digest,
                    "needs_ocr": inspection.needs_ocr,
                    "privacy_hold": inspection.privacy_hold,
                    "corpus_source": source,
                },
            )
            db.add(review)
            db.flush()
            review_id = review.id
        db.add(
            UploadHistory(
                document_id=doc.id,
                action="intake",
                actor_id=actor_id,
                details={
                    "status": "success",
                    "format": inspection.format,
                    "size_bytes": inspection.size_bytes,
                    "needs_ocr": inspection.needs_ocr,
                    "privacy_hold": inspection.privacy_hold,
                    "corpus_source": source,
                    "original_filename": inspection.original_filename,
                },
            )
        )
        if action == "replace":
            db.add(
                UploadHistory(
                    document_id=doc.id,
                    action="replace_requested",
                    actor_id=actor_id,
                    details={"status": "success", "related_document_id": related_document_id},
                )
            )
        db.commit()
        db.refresh(doc)
        write_file_registry(db)
        logger.info(
            "intake stored document_id=%s hash=%s format=%s corpus=%s ocr=%s privacy=%s",
            doc.id,
            inspection.digest[:12],
            inspection.format,
            source,
            inspection.needs_ocr,
            inspection.privacy_hold,
        )
        stash = STASH_DIR / f"{inspection.digest}.bin"
        stash.unlink(missing_ok=True)
        n_chunks = 0
        if not inspection.privacy_hold:
            parsed = None
            try:
                from app.services.document_parser import parse_registered_document

                parsed = parse_registered_document(db, doc.id)
                db.refresh(doc)
            except Exception:
                logger.exception("parse after intake failed document_id=%s", doc.id)
            if parsed is not None and parsed.status not in {"error", "privacy_blocked"} and parsed.elements:
                try:
                    from app.services.chunking import ChunkingError, chunk_parsed_document

                    chunked = chunk_parsed_document(db, doc.id)
                    n_chunks = len(chunked.chunks)
                except ChunkingError as exc:
                    logger.exception(
                        "chunk after intake failed document_id=%s code=%s",
                        doc.id,
                        exc.code,
                    )
                except Exception:
                    logger.exception("chunk after intake failed document_id=%s", doc.id)
        message = "Документ сохранён как черновик и ожидает экспертизу."
        if inspection.privacy_hold:
            message = "Обнаружены признаки персональных данных. Черновик заблокирован до проверки эксперта."
        elif inspection.needs_ocr:
            message = "PDF без текстового слоя принят. OCR будет выполнен на следующем этапе."
        return IntakeResult(
            document_id=doc.id,
            status="draft",
            duplicate=False,
            file_hash=inspection.digest,
            file_path=doc.file_path,
            version=doc.version,
            review_task_id=review_id,
            needs_ocr=inspection.needs_ocr,
            privacy_hold=inspection.privacy_hold,
            message=message,
            n_chunks=n_chunks,
        )
    except IntakeError:
        db.rollback()
        if saved is not None and saved.exists():
            saved.unlink(missing_ok=True)
        raise
    except IntegrityError:
        db.rollback()
        if saved is not None and saved.exists():
            saved.unlink(missing_ok=True)
        again = find_live_duplicate(db, inspection.digest)
        if again is not None:
            return IntakeResult(
                document_id=again.id,
                status=again.status or "draft",
                duplicate=True,
                file_hash=inspection.digest,
                file_path=again.file_path,
                version=again.version,
                message="Этот документ уже загружен",
                existing_title=again.title,
            )
        raise IntakeError("conflict", "Конфликт записи документа.")
    except Exception:
        db.rollback()
        if saved is not None and saved.exists():
            saved.unlink(missing_ok=True)
        logger.exception("intake failed hash=%s", inspection.digest[:12])
        raise IntakeError("intake_failed", "Не удалось сохранить документ.")
