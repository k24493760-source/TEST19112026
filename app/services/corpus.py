import hashlib
import json
import re
import uuid
from io import BytesIO
from pathlib import Path

from docx import Document as DocxDocument
from sqlalchemy import or_
from sqlalchemy.orm import Session, selectinload

from app.config import BASE_DIR
from app.models import KnowledgeChunk, KnowledgeDocument, UploadHistory

CORPUS_DIR = BASE_DIR / "data" / "corpus"
LAW_DIR = CORPUS_DIR / "database_law"
EDU_DIR = CORPUS_DIR / "database_edu_lectcii"
QUARANTINE_DIR = CORPUS_DIR / "quarantine"
FILE_REGISTRY_PATH = CORPUS_DIR / "file_registry.json"


def configure_corpus_root(root: Path) -> Path:
    """Переключает каталоги корпуса (для тестов). Не используется в проде."""
    global CORPUS_DIR, LAW_DIR, EDU_DIR, QUARANTINE_DIR, FILE_REGISTRY_PATH
    CORPUS_DIR = Path(root)
    LAW_DIR = CORPUS_DIR / "database_law"
    EDU_DIR = CORPUS_DIR / "database_edu_lectcii"
    QUARANTINE_DIR = CORPUS_DIR / "quarantine"
    FILE_REGISTRY_PATH = CORPUS_DIR / "file_registry.json"
    return CORPUS_DIR
CHUNK_MAX_CHARS = 1000
CHUNK_OVERLAP = 120
SUPPORTED_SUFFIXES = {".txt", ".md", ".docx", ".pdf"}
KNOWN_CORPUS_SOURCES = {
    "database_law",
    "database_edu_lectcii",
    "quarantine",
}
DOCUMENT_CLASSES = (
    "Нормативный акт",
    "Официальное разъяснение",
    "Юридический комментарий",
    "Учебный материал",
    "Кейс",
    "Методика",
)
DOCUMENT_CLASS_ALIASES = {
    "нормативный акт": "Нормативный акт",
    "закон": "Нормативный акт",
    "нормативный_акт": "Нормативный акт",
    "официальное разъяснение": "Официальное разъяснение",
    "юридический комментарий": "Юридический комментарий",
    "учебный материал": "Учебный материал",
    "учебный_материал": "Учебный материал",
    "учебник": "Учебный материал",
    "лекция": "Учебный материал",
    "кейс": "Кейс",
    "практический кейс": "Кейс",
    "методика": "Методика",
}
LAW_CLASSES = {"нормативный акт", "закон", "нормативный_акт"}
LECTURE_CLASSES = {"учебный материал", "учебный_материал", "учебник", "лекция"}
DOCUMENT_STATUSES = {
    "draft",
    "processing",
    "ready_for_review",
    "error",
    "approved",
    "archived",
    "changed",
    "revoked",
    "rejected",
}
CHUNK_APPROVAL_STATUSES = {"draft", "approved", "archived", "invalidated", "pending"}

ARTICLE_SPLIT = re.compile(r"(?im)(?=^\s*статья\s+\d+(?:\.\d+)*)")
ARTICLE_HEADING = re.compile(
    r"(?im)^\s*(статья\s+\d+(?:\.\d+)*\.?(?:[ \t]+[^\n]+)?)"
)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def parse_version(value: str | None) -> tuple[int, int]:
    text = (value or "1.0").strip()
    parts = text.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return major, minor
    except ValueError:
        return 1, 0


def next_document_version(existing_versions: list[str]) -> str:
    if not existing_versions:
        return "1.0"
    major, minor = max(parse_version(item) for item in existing_versions)
    return f"{major}.{minor + 1}"


def normalize_text(text: str) -> str:
    """Убирает лишние пробелы и пустые строки, абзацы через пустую строку (Markdown)."""
    unified = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in unified.split("\n"):
        collapsed = re.sub(r"[ \t]+", " ", line).strip()
        if collapsed:
            lines.append(collapsed)
    return "\n\n".join(lines)


def normalize_document_class(value: str | None) -> str:
    key = re.sub(r"\s+", " ", (value or "").strip().lower())
    if key in DOCUMENT_CLASS_ALIASES:
        return DOCUMENT_CLASS_ALIASES[key]
    raise ValueError(
        "Укажите класс документа: Нормативный акт, Официальное разъяснение, "
        "Юридический комментарий, Учебный материал, Кейс или Методика."
    )


def document_class_label(value: str | None) -> str:
    if not value:
        return "—"
    key = re.sub(r"\s+", " ", value.strip().lower())
    return DOCUMENT_CLASS_ALIASES.get(key, value)


def is_law_class(document_class: str | None) -> bool:
    return document_class_label(document_class) == "Нормативный акт"


def is_lecture_class(document_class: str | None) -> bool:
    return document_class_label(document_class) == "Учебный материал"


def validate_document(doc: KnowledgeDocument) -> list[str]:
    """Проверка целостности документа и чанков. Пустой список — ошибок нет."""
    errors: list[str] = []
    class_value = (doc.document_class or "").strip()
    if not class_value:
        errors.append("Не заполнен класс документа (document_class).")
    else:
        try:
            normalize_document_class(class_value)
        except ValueError:
            errors.append(f"Недопустимый класс документа: {class_value}.")

    status = (doc.status or "").strip()
    if not status:
        errors.append("Не заполнен статус документа (status).")
    elif status not in DOCUMENT_STATUSES:
        errors.append(f"Недопустимый статус документа: {status}.")

    chunks = list(getattr(doc, "chunks", None) or [])
    if not chunks:
        errors.append("Нет связанных чанков.")
        errors.append("Не заполнен approval_status.")
        return errors

    missing_approval = []
    seen_ids: set[str] = set()
    for index, chunk in enumerate(chunks, start=1):
        approval = (chunk.approval_status or "").strip()
        if not approval:
            missing_approval.append(str(index))
        elif approval not in CHUNK_APPROVAL_STATUSES:
            errors.append(
                f"Чанк #{index}: недопустимый approval_status «{chunk.approval_status}»."
            )

        chunk_id = (chunk.id or "").strip()
        if not chunk_id:
            errors.append(f"Чанк #{index}: нет уникального chunk_id.")
        elif chunk_id in seen_ids:
            errors.append(f"Чанк #{index}: chunk_id «{chunk_id}» повторяется.")
        else:
            seen_ids.add(chunk_id)

        text = chunk.text if chunk.text is not None else ""
        if not str(text).strip():
            label = chunk_id or f"#{index}"
            errors.append(f"Чанк {label}: текст пустой или состоит только из пробелов.")

    if missing_approval:
        errors.append(
            "Не заполнен approval_status у чанков: " + ", ".join(missing_approval) + "."
        )
    return errors


def document_is_valid(doc: KnowledgeDocument) -> bool:
    return not validate_document(doc)


def apply_related_on_approve(db: Session, doc: KnowledgeDocument) -> None:
    related_id = getattr(doc, "related_document_id", None)
    action = (getattr(doc, "version_action", None) or "").strip()
    if not related_id:
        return
    old = (
        db.query(KnowledgeDocument)
        .options(selectinload(KnowledgeDocument.chunks))
        .filter(KnowledgeDocument.id == related_id)
        .first()
    )
    if old is None:
        return
    if action == "new_version":
        old.status = "changed"
    elif action == "replace":
        old.status = "revoked"
    else:
        return
    for chunk in old.chunks:
        chunk.approval_status = "invalidated"


def _hard_cut(text: str, max_chars: int, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    overlap = max(0, min(overlap, max_chars - 1))
    step = max(max_chars - overlap, 1)
    pieces = []
    start = 0
    while start < len(text):
        piece = text[start : start + max_chars].strip()
        if piece:
            pieces.append(piece)
        if start + max_chars >= len(text):
            break
        start += step
    return pieces


def split_by_sentences(
    text: str,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    overlap = max(0, min(overlap, max_chars - 1))
    sentences = [part.strip() for part in SENTENCE_SPLIT.split(text) if part.strip()]
    if not sentences:
        return _hard_cut(text, max_chars, overlap)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_cut(sentence, max_chars, overlap))
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail} {sentence}".strip() if tail else sentence
            if len(current) > max_chars:
                current = sentence
        else:
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def chunk_lecture(text: str, max_chars: int = CHUNK_MAX_CHARS) -> list[tuple[str, str]]:
    paragraphs = [part.strip() for part in (text or "").split("\n\n") if part.strip()]
    if not paragraphs and (text or "").strip():
        paragraphs = [text.strip()]
    result: list[tuple[str, str]] = []
    index = 0
    for paragraph in paragraphs:
        pieces = (
            split_by_sentences(paragraph, max_chars)
            if len(paragraph) > max_chars
            else [paragraph]
        )
        for piece in pieces:
            index += 1
            result.append((f"Абзац {index}", piece))
    return result


def _article_title(block: str) -> str:
    first_line = (block or "").split("\n", 1)[0].strip()
    match = ARTICLE_HEADING.match(first_line)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    return re.sub(r"\s+", " ", first_line)[:200] or "Статья"


def chunk_law(text: str, max_chars: int = CHUNK_MAX_CHARS) -> list[tuple[str, str]]:
    unified = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    blocks = [part.strip() for part in ARTICLE_SPLIT.split(unified) if part.strip()]
    has_articles = bool(ARTICLE_HEADING.search(unified))
    if not has_articles:
        return []
    result: list[tuple[str, str]] = []
    for block in blocks:
        heading_match = ARTICLE_HEADING.match(block)
        if heading_match is None:
            for piece in split_by_sentences(block, max_chars):
                result.append(("Преамбула", piece))
            continue
        title = _article_title(block)
        body = block[heading_match.end() :].strip()
        if not body:
            result.append((title, title))
            continue
        budget = max(max_chars - len(title) - 2, 200)
        for piece in split_by_sentences(body, budget):
            chunk = piece if piece.startswith(title) else f"{title}\n\n{piece}"
            result.append((title, chunk))
    return result


def chunk_document(text: str, document_class: str | None, max_chars: int = CHUNK_MAX_CHARS) -> list[tuple[str, str]]:
    if is_law_class(document_class):
        law_chunks = chunk_law(text, max_chars)
        if law_chunks:
            return law_chunks
    return chunk_lecture(text, max_chars)


def chunk_paragraphs(text: str, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    return [piece for _, piece in chunk_lecture(text, max_chars)]


def extract_plain_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Не удалось прочитать файл как текст (UTF-8 или Windows-1251).")


def extract_docx_text(raw: bytes) -> str:
    try:
        document = DocxDocument(BytesIO(raw))
    except Exception as exc:
        raise ValueError("Не удалось прочитать файл Word.") from exc
    paragraphs = [p.text for p in document.paragraphs]
    text = normalize_text("\n".join(paragraphs))
    if not text:
        raise ValueError("В файле Word нет текстовых абзацев.")
    return text


def extract_pdf_text(raw: bytes) -> str:
    try:
        from PyPDF2 import PdfReader
    except ImportError as exc:
        raise ValueError("PDF parser not installed") from exc
    try:
        reader = PdfReader(BytesIO(raw))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:
        raise ValueError("Не удалось прочитать PDF.") from exc
    text = normalize_text("\n\n".join(pages))
    if not text:
        raise ValueError("В PDF нет извлекаемого текста.")
    return text


def extract_and_normalize(raw: bytes, suffix: str) -> str:
    suffix = (suffix or "").lower()
    if suffix == ".docx":
        return extract_docx_text(raw)
    if suffix == ".pdf":
        return extract_pdf_text(raw)
    if suffix in {".txt", ".md"}:
        return normalize_text(extract_plain_text(raw))
    raise ValueError("Поддерживаются файлы .txt, .md, .docx и .pdf")


def safe_filename(name: str) -> str:
    stem = Path(name).stem
    cleaned = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE).strip("_")
    return (cleaned[:80] or "document")


def ensure_corpus_layout() -> None:
    """Создаёт канонические каталоги корпуса, не перемещая существующие файлы."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for folder in (LAW_DIR, EDU_DIR, QUARANTINE_DIR):
        folder.mkdir(parents=True, exist_ok=True)
        keep = folder / ".gitkeep"
        if not keep.exists():
            keep.write_text("", encoding="utf-8")


def corpus_source_from_path(path: Path | str | None) -> str:
    """Источник только по каноническому каталогу, не по имени файла."""
    if not path:
        return "unknown"
    resolved = Path(path)
    try:
        relative = resolved.resolve().relative_to(CORPUS_DIR.resolve())
    except (ValueError, OSError):
        parts = {part.lower() for part in Path(path).parts}
        for source in KNOWN_CORPUS_SOURCES:
            if source in parts:
                return source
        return "unknown"
    parts = [part.lower() for part in relative.parts[:-1]]
    for source in ("database_law", "database_edu_lectcii", "quarantine"):
        if source in parts:
            return source
    return "unknown"


def iter_corpus_files() -> list[Path]:
    ensure_corpus_layout()
    files: list[Path] = []
    for path in CORPUS_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.name == FILE_REGISTRY_PATH.name:
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        files.append(path)
    return sorted(files)


def find_document_for_file(
    db: Session,
    *,
    digest: str | None = None,
    relative_path: str | None = None,
) -> KnowledgeDocument | None:
    filters = []
    if digest:
        filters.append(KnowledgeDocument.hash == digest)
        filters.append(KnowledgeDocument.file_hash == digest)
    if relative_path:
        filters.append(KnowledgeDocument.file_path == relative_path)
    if not filters:
        return None
    return db.query(KnowledgeDocument).filter(or_(*filters)).first()


def find_live_duplicate(db: Session, digest: str) -> KnowledgeDocument | None:
    return (
        db.query(KnowledgeDocument)
        .filter(
            or_(
                KnowledgeDocument.hash == digest,
                KnowledgeDocument.file_hash == digest,
            )
        )
        .filter(KnowledgeDocument.status != "archived")
        .first()
    )


def build_file_registry(db: Session | None = None) -> list[dict]:
    """Реестр файлов на диске. Не меняет БД и не режет чанки."""
    ensure_corpus_layout()
    docs_by_hash: dict[str, KnowledgeDocument] = {}
    docs_by_path: dict[str, KnowledgeDocument] = {}
    if db is not None:
        for doc in db.query(KnowledgeDocument).all():
            if doc.hash:
                docs_by_hash[doc.hash] = doc
            if getattr(doc, "file_hash", None):
                docs_by_hash[doc.file_hash] = doc
            if doc.file_path:
                docs_by_path[doc.file_path.replace("\\", "/")] = doc

    rows = []
    for path in iter_corpus_files():
        digest = sha256_bytes(path.read_bytes())
        rel = relative_corpus_path(path)
        doc = docs_by_path.get(rel) or docs_by_hash.get(digest)
        rows.append(
            {
                "file_path": rel,
                "file_name": path.name,
                "format": path.suffix.lower().lstrip("."),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
                "corpus_source": corpus_source_from_path(path),
                "document_id": doc.id if doc is not None else None,
                "document_class": doc.document_class if doc is not None else None,
                "processing_status": (doc.status if doc is not None else "unregistered"),
            }
        )
    return rows


def write_file_registry(db: Session | None = None) -> Path:
    rows = build_file_registry(db)
    FILE_REGISTRY_PATH.write_text(
        json.dumps({"count": len(rows), "files": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return FILE_REGISTRY_PATH


def save_corpus_file(raw: bytes, original_name: str) -> Path:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(original_name).suffix.lower() or ".txt"
    if suffix not in SUPPORTED_SUFFIXES:
        suffix = ".txt"
    filename = f"{safe_filename(original_name)}_{uuid.uuid4().hex[:8]}{suffix}"
    path = CORPUS_DIR / filename
    path.write_bytes(raw)
    return path


def relative_corpus_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(BASE_DIR)).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def replace_chunks(db: Session, doc: KnowledgeDocument, text: str) -> int:
    from app.embedder import apply_embeddings, embedding_is_ready

    db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id == doc.id).delete()
    pieces = chunk_document(text, doc.document_class)
    status = "approved" if doc.status == "approved" else "draft"
    for index, (section, piece) in enumerate(pieces, start=1):
        db.add(
            KnowledgeChunk(
                document_id=doc.id,
                text=piece,
                section=section,
                page=index,
                approval_status=status,
                embedding_status="pending",
            )
        )
    db.flush()
    apply_embeddings(db, document_id=doc.id, force=True)
    for chunk in (
        db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id == doc.id).all()
    ):
        chunk.embedding_status = "done" if embedding_is_ready(chunk.embedding) else "error"
    return len(pieces)


def _log_pipeline(db: Session, document_id: str, action: str, status: str, details: dict | None = None) -> None:
    payload = dict(details or {})
    payload["status"] = status
    db.add(
        UploadHistory(
            document_id=document_id,
            action=action,
            actor_id="pipeline",
            details=payload,
        )
    )


def _resolve_source_path(file_path: str | None, stored: str | None) -> Path:
    candidates = []
    for value in (file_path, stored):
        if not value:
            continue
        path = Path(value)
        candidates.append(path)
        if not path.is_absolute():
            candidates.append(BASE_DIR / path)
    for path in candidates:
        if path.is_file():
            return path
    raise ValueError("Исходный файл документа не найден.")


def process_document_pipeline(document_id: str, file_path: str | None = None) -> dict:
    """Парсинг → чанки (max 1000, overlap 120) → эмбеддинги. Свой сеанс БД для BackgroundTasks."""
    from app.db import SessionLocal
    from app.embedder import apply_embeddings, embedding_is_ready

    db = SessionLocal()
    result = {"document_id": document_id, "status": "error"}
    try:
        doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == document_id).first()
        if doc is None:
            return result
        doc.status = "processing"
        db.commit()
        try:
            source = _resolve_source_path(file_path, doc.file_path)
            raw = source.read_bytes()
            suffix = source.suffix.lower() or f".{(doc.file_format or 'txt')}"
            text = extract_and_normalize(raw, suffix)
            text = normalize_text(text)
            _log_pipeline(
                db,
                document_id,
                "parsed",
                "success",
                {"chars": len(text), "suffix": suffix},
            )
            db.commit()
        except Exception as exc:
            message = str(exc) or "parse_error"
            doc.status = "error"
            _log_pipeline(db, document_id, "parsed", "error", {"message": message})
            db.commit()
            result["message"] = message
            return result

        try:
            db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id == document_id).delete()
            pieces = chunk_document(text, doc.document_class)
            if not pieces:
                raise ValueError("Не удалось нарезать документ на чанки.")
            for index, (section, piece) in enumerate(pieces, start=1):
                db.add(
                    KnowledgeChunk(
                        document_id=document_id,
                        text=piece,
                        section=section,
                        page=index,
                        approval_status="draft",
                        embedding_status="pending",
                    )
                )
            db.flush()
            _log_pipeline(
                db,
                document_id,
                "chunked",
                "success",
                {"chunks": len(pieces), "max_chars": CHUNK_MAX_CHARS, "overlap": CHUNK_OVERLAP},
            )
            db.commit()
        except Exception as exc:
            message = str(exc) or "chunk_error"
            doc.status = "error"
            _log_pipeline(db, document_id, "chunked", "error", {"message": message})
            db.commit()
            result["message"] = message
            return result

        try:
            apply_embeddings(db, document_id=document_id, force=True)
            chunks = (
                db.query(KnowledgeChunk)
                .filter(KnowledgeChunk.document_id == document_id)
                .all()
            )
            done = 0
            for chunk in chunks:
                ready = embedding_is_ready(chunk.embedding)
                chunk.embedding_status = "done" if ready else "error"
                if ready:
                    done += 1
            if done == 0:
                raise ValueError("Эмбеддинги не записаны.")
            doc.status = "ready_for_review"
            _log_pipeline(
                db,
                document_id,
                "embedded",
                "success",
                {"embedded": done, "total": len(chunks)},
            )
            db.commit()
            result["status"] = "ready_for_review"
            result["chunks"] = len(chunks)
            return result
        except Exception as exc:
            message = str(exc) or "embed_error"
            doc.status = "error"
            _log_pipeline(db, document_id, "embedded", "error", {"message": message})
            db.commit()
            result["message"] = message
            return result
    except Exception as exc:
        db.rollback()
        try:
            doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == document_id).first()
            if doc is not None:
                doc.status = "error"
                _log_pipeline(db, document_id, "parsed", "error", {"message": str(exc)})
                db.commit()
        except Exception:
            db.rollback()
        result["message"] = str(exc)
        return result
    finally:
        db.close()


def _process_file(
    db: Session,
    path: Path,
    doc: KnowledgeDocument | None,
    *,
    mutate_existing: bool = False,
    create_missing: bool = False,
) -> KnowledgeDocument | None:
    """Регистрация файла. По умолчанию не меняет уже сохранённые документы и чанки."""
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    suffix = path.suffix.lower()
    rel = relative_corpus_path(path)
    if doc is None:
        doc = find_document_for_file(db, digest=digest, relative_path=rel)
    if doc is not None:
        if not mutate_existing:
            return doc
        if (doc.status or "").strip() == "approved":
            return doc
        doc.hash = digest
        doc.file_path = rel
        doc.file_format = suffix.lstrip(".")
        if not doc.title:
            doc.title = path.stem
        text = extract_and_normalize(raw, suffix)
        replace_chunks(db, doc, text)
        return doc
    if not create_missing:
        return None
    text = extract_and_normalize(raw, suffix)
    doc = KnowledgeDocument(
        title=path.stem,
        document_class="Учебный материал",
        owner="аудит корпуса",
        version="1.0",
        status="draft",
        hash=digest,
        file_path=rel,
        file_format=suffix.lstrip("."),
    )
    db.add(doc)
    db.flush()
    replace_chunks(db, doc, text)
    return doc


def audit_documents(db: Session) -> list[dict]:
    """Инвентаризация корпуса без пересборки чанков и без смены метаданных."""
    ensure_corpus_layout()
    write_file_registry(db)
    return document_audit_rows(db)


def document_audit_rows(db: Session) -> list[dict]:
    documents = (
        db.query(KnowledgeDocument)
        .options(selectinload(KnowledgeDocument.chunks))
        .order_by(KnowledgeDocument.created_at.desc())
        .all()
    )
    rows = []
    for doc in documents:
        size = 0
        if doc.file_path:
            path = BASE_DIR / doc.file_path
            if path.is_file():
                size = path.stat().st_size
        if not size:
            size = sum(len((chunk.text or "").encode("utf-8")) for chunk in doc.chunks)
        status = doc.status or "draft"
        status_labels = {
            "draft": "черновик",
            "processing": "в обработке",
            "ready_for_review": "готов к экспертизе",
            "error": "ошибка",
            "approved": "утверждён",
            "archived": "архив",
            "changed": "изменён",
            "revoked": "отменён",
        }
        validation_errors = validate_document(doc)
        rows.append(
            {
                "id": doc.id,
                "title": doc.title,
                "document_class": document_class_label(doc.document_class),
                "file_format": (doc.file_format or "—").upper(),
                "size_kb": round(size / 1024, 1) if size else 0,
                "hash_short": (doc.hash or "")[:8] or "—",
                "n_chunks": len(doc.chunks),
                "version": doc.version or "1.0",
                "status": status,
                "status_label": status_labels.get(status, status),
                "has_original": bool(doc.file_path and (BASE_DIR / doc.file_path).is_file()),
                "validation_ok": not validation_errors,
                "validation_errors": validation_errors,
            }
        )
    return rows
