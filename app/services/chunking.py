"""Структурный чанкинг по классу документа. Без embeddings и публикации RAG."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml
from sqlalchemy.orm import Session

from app.config import BASE_DIR, EMBEDDING_MODEL
from app.models import (
    CorpusRelease,
    DocumentElement,
    DocumentParseRun,
    KnowledgeChunk,
    KnowledgeDocument,
    UploadHistory,
)
from app.services.corpus import normalize_document_class

logger = logging.getLogger("most.chunking")

CONFIG_PATH = BASE_DIR / "contracts" / "chunking_config.yaml"
SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
EXCEPTION_HEAD = re.compile(
    r"(?is)^(за исключением|кроме случая|если не|при условии|если\b|однако\b)"
)
NEGATION = re.compile(r"(?i)\bне\s+\w+")
CONDITION = re.compile(r"(?i)\b(если|при условии|за исключением|кроме)\b")
SKIP_TYPES = {"table_row"}


@dataclass
class ChunkSpec:
    chunk_id: str
    evidence_text: str
    retrieval_text: str
    section_path: str
    page_number: int | None
    source_element_ids: list[str]
    token_count: int
    parent_chunk_id: str | None = None
    applicability: str | None = None


@dataclass
class ChunkingResult:
    document_id: str
    chunking_version: str
    corpus_version: str
    chunks: list[ChunkSpec] = field(default_factory=list)
    oversized_kept: int = 0


class ChunkingError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@lru_cache(maxsize=1)
def load_chunking_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@lru_cache(maxsize=1)
def _tokenizer():
    """Токенизатор той же модели, что и эмбеддинги. Векторы чанков здесь не считаются."""
    from app.services.embedding_model import load_sentence_transformer

    cfg = load_chunking_config()
    name = cfg.get("tokenizer_model") or EMBEDDING_MODEL
    if isinstance(name, str) and name.startswith("sentence-transformers/"):
        name = name.split("/", 1)[1]
    model = load_sentence_transformer(name)
    tokenizer = model.tokenizer
    tokenizer.model_max_length = 100_000
    return tokenizer


def count_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        tokenizer = _tokenizer()
    except Exception:
        logger.warning("tokenizer_unavailable")
        return max(1, len(text.split()))
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
    )


def _class_cfg(document_class: str) -> dict:
    cfg = load_chunking_config()
    mapped = normalize_document_class(document_class)
    classes = cfg.get("classes") or {}
    if mapped not in classes:
        raise ChunkingError("unknown_class", f"Нет стратегии чанкинга для «{mapped}».")
    merged = {
        "soft_limit_chars": cfg.get("soft_limit_chars"),
        "hard_limit_chars": cfg.get("hard_limit_chars"),
        "overlap_chars": cfg.get("overlap_chars") or 0,
    }
    merged.update(classes[mapped])
    return {"name": mapped, **merged}


STRATEGY_ALIASES = {
    "article_clause": "normative_act",
    "structure_aware": "edu_material",
    "hierarchical": "methodology",
}
ARTICLE_RE = re.compile(r"(?i)^статья\s+\d+")
CLAUSE_RE = re.compile(r"(?i)^(?:пункт\s+\d+|часть\s+\d+|\d+[.)]\s|[а-яё]\))")
QA_Q_RE = re.compile(r"(?i)^вопрос\b")
THEME_RE = re.compile(r"(?i)^(тема|раздел|лекция)\b")
STEP_RE = re.compile(r"(?i)^(шаг|этап)\s+\d+")
CASE_RE = re.compile(r"(?i)^(ситуация|решение|анализ)\b")


def _canonical(strategy: str) -> str:
    return STRATEGY_ALIASES.get(strategy, strategy)


def _el_text(element: DocumentElement) -> str:
    return (element.text_normalized or element.text_original or "").strip()


def _use_chars(cfg: dict) -> bool:
    return bool(cfg.get("hard_limit_chars"))


def _size(text: str, cfg: dict) -> int:
    if _use_chars(cfg):
        return len(text or "")
    return count_tokens(text)


def _hard(cfg: dict) -> int:
    if _use_chars(cfg):
        return int(cfg["hard_limit_chars"])
    return int(cfg["max_tokens"])


def _soft(cfg: dict) -> int:
    if _use_chars(cfg):
        return int(cfg.get("soft_limit_chars") or cfg["hard_limit_chars"])
    return int(cfg["max_tokens"])


def _overlap_prefix_chars(text: str, n: int) -> str:
    if n <= 0 or not text:
        return ""
    return text[-n:].lstrip()


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENTENCE_SPLIT.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _must_keep_together(left: str, right: str) -> bool:
    if NEGATION.search(left) and EXCEPTION_HEAD.match(right):
        return True
    if left.rstrip().lower().endswith("не") and right:
        return True
    if CONDITION.search(left) and len(right) < 400:
        return True
    if EXCEPTION_HEAD.match(right):
        return True
    return False


def _glue_sentences(sentences: list[str]) -> list[str]:
    glued: list[str] = []
    for sentence in sentences:
        if glued and _must_keep_together(glued[-1], sentence):
            glued[-1] = f"{glued[-1]} {sentence}".strip()
        else:
            glued.append(sentence)
    return glued


def _overlap_prefix(text: str, pct: int) -> str:
    if pct <= 0 or not text:
        return ""
    tokens = _tokenizer().encode(text, add_special_tokens=False)
    n = max(1, int(len(tokens) * pct / 100))
    piece = tokens[-n:]
    return _tokenizer().decode(piece).strip()


def _chunk_id(document_id: str, version: str, element_ids: list[str], seq: int) -> str:
    raw = f"{document_id}|{version}|{seq}|{','.join(element_ids)}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"chk_{digest}"


def _table_chunks(element: DocumentElement, cfg: dict, heading: str) -> list[tuple[str, list[str]]]:
    location = element.source_location or {}
    rows = location.get("rows") or []
    if not rows:
        text = (element.text_normalized or element.text_original or "").strip()
        return [(text, [element.id])] if text else []
    header = rows[0]
    header_line = " | ".join(str(c) for c in header)
    limit = _hard(cfg)
    bodies = rows[1:] or rows
    groups: list[list[list[str]]] = [[]]
    for row in bodies:
        trial = groups[-1] + [row]
        body_txt = "\n".join(" | ".join(str(c) for c in r) for r in trial)
        prefixed = f"{heading}\n{header_line}\n{body_txt}".strip() if heading else f"{header_line}\n{body_txt}"
        if groups[-1] and _size(prefixed, cfg) > limit:
            groups.append([row])
        else:
            groups[-1] = trial
    out = []
    for group in groups:
        body_txt = "\n".join(" | ".join(str(c) for c in r) for r in group)
        text = f"{header_line}\n{body_txt}".strip()
        out.append((text, [element.id]))
    return out


def _is_article(element: DocumentElement) -> bool:
    return element.element_type == "article" or bool(ARTICLE_RE.match(_el_text(element)))


def _is_clause(element: DocumentElement) -> bool:
    if _is_article(element):
        return False
    return element.element_type == "clause" or bool(CLAUSE_RE.match(_el_text(element)))


def _is_question(element: DocumentElement) -> bool:
    return bool(QA_Q_RE.match(_el_text(element)))


def _is_theme(element: DocumentElement) -> bool:
    return bool(THEME_RE.match(_el_text(element)))


def _is_step(element: DocumentElement) -> bool:
    return bool(STEP_RE.match(_el_text(element)))


def _is_case_part(element: DocumentElement) -> bool:
    return bool(CASE_RE.match(_el_text(element)))


def _split_on(elements: list[DocumentElement], predicate) -> list[list[DocumentElement]]:
    blocks: list[list[DocumentElement]] = []
    current: list[DocumentElement] = []
    for element in elements:
        if predicate(element) and current:
            blocks.append(current)
            current = [element]
        else:
            current.append(element)
    if current:
        blocks.append(current)
    return blocks


def _join_blocks(blocks: list[list[DocumentElement]]) -> list[DocumentElement]:
    joined: list[DocumentElement] = []
    for block in blocks:
        joined.extend(block)
    return joined


def _blocks(elements: list[DocumentElement], strategy: str, cfg: dict | None = None) -> list[list[DocumentElement]]:
    strategy = _canonical(strategy)
    cfg = cfg or {}
    items = [element for element in elements if element.element_type not in SKIP_TYPES]
    if not items:
        return []
    if strategy == "normative_act":
        return _split_on(items, _is_article)
    if strategy == "official_explanation":
        if any(_is_question(element) for element in items):
            return _split_on(items, _is_question)
        return [items]
    if strategy == "edu_material":
        return _split_on(items, _is_theme)
    if strategy == "methodology":
        return _group_steps(items, _hard(cfg) if cfg else 1500)
    if strategy == "case":
        return _blocks_case(items, int(cfg.get("whole_limit_chars") or 2000))
    if strategy == "legal_commentary":
        return [items]
    return _split_on(items, lambda element: element.element_type == "heading")


def _group_steps(elements: list[DocumentElement], limit: int) -> list[list[DocumentElement]]:
    steps = _split_on(elements, _is_step)
    grouped: list[list[DocumentElement]] = []
    buf: list[list[DocumentElement]] = []

    def length(parts: list[list[DocumentElement]]) -> int:
        return len("\n".join(_evidence_of(part) for part in parts))

    for step in steps:
        if length([step]) > limit:
            if buf:
                grouped.append(_join_blocks(buf))
                buf = []
            grouped.append(step)
            continue
        trial = buf + [step]
        if buf and (len(trial) > 3 or length(trial) > limit):
            grouped.append(_join_blocks(buf))
            buf = [step]
        else:
            buf = trial
    if buf:
        grouped.append(_join_blocks(buf))
    return grouped


def _blocks_case(elements: list[DocumentElement], whole_limit: int) -> list[list[DocumentElement]]:
    whole = "\n".join(_el_text(element) for element in elements if _el_text(element))
    if len(whole) <= whole_limit:
        return [elements]
    if any(_is_case_part(element) for element in elements):
        return _split_on(elements, _is_case_part)
    return [[element] for element in elements]


def _evidence_of(elements: list[DocumentElement]) -> str:
    parts = []
    for element in elements:
        if element.element_type == "table":
            location = element.source_location or {}
            rows = location.get("rows")
            if rows:
                parts.append("\n".join(" | ".join(str(c) for c in row) for row in rows))
                continue
        text = (element.text_normalized or element.text_original or "").strip()
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _heading_prefix(elements: list[DocumentElement]) -> str:
    for element in elements:
        if element.element_type in {"heading", "article", "clause", "appendix"}:
            return (element.text_normalized or element.text_original or "").strip()
    return (elements[0].section_path or "").strip() if elements else ""


def _pack_text(
    text: str,
    element_ids: list[str],
    cfg: dict,
    heading: str,
    section: str,
    page: int | None,
    version: str,
    document_id: str,
    seq_start: int,
    overlap_pct: int,
    overlap_chars: int = 0,
) -> list[ChunkSpec]:
    hard = _hard(cfg)
    soft = _soft(cfg)
    sentences = _glue_sentences(_split_sentences(text))
    specs: list[ChunkSpec] = []
    buf: list[str] = []
    parent_id = None
    seq = seq_start

    def emit(piece: str, oversized: bool = False, parent: str | None = None) -> None:
        nonlocal seq, parent_id
        evidence = piece.strip()
        if not evidence:
            return
        retrieval = f"{heading}\n{evidence}".strip() if heading and heading not in evidence else evidence
        tokens = count_tokens(retrieval)
        applicability = "oversized_unsplit" if oversized else None
        chunk_id = _chunk_id(document_id, version, element_ids, seq)
        spec = ChunkSpec(
            chunk_id=chunk_id,
            evidence_text=evidence,
            retrieval_text=retrieval,
            section_path=section,
            page_number=page,
            source_element_ids=list(element_ids),
            token_count=tokens,
            parent_chunk_id=parent,
            applicability=applicability,
        )
        specs.append(spec)
        if parent_id is None:
            parent_id = chunk_id
        seq += 1

    if _size(text, cfg) <= hard:
        emit(text)
        return specs

    for sentence in sentences:
        trial = " ".join(buf + [sentence]).strip()
        if buf and _size(trial, cfg) > soft:
            piece = " ".join(buf).strip()
            emit(piece, parent=parent_id if specs else None)
            if overlap_chars:
                prefix = _overlap_prefix_chars(piece, overlap_chars)
            else:
                prefix = _overlap_prefix(piece, overlap_pct)
            buf = [prefix, sentence] if prefix else [sentence]
            if _size(" ".join(buf), cfg) > hard:
                emit(sentence, oversized=_size(sentence, cfg) > hard, parent=parent_id if specs else None)
                buf = []
        else:
            buf.append(sentence)
    if buf:
        piece = " ".join(buf).strip()
        emit(piece, oversized=_size(piece, cfg) > hard, parent=parent_id if specs else None)

    if not specs:
        emit(text, oversized=True)
    return specs


def _article_title(block: list[DocumentElement], heading: str) -> str:
    for element in block:
        if _is_article(element):
            return _el_text(element) or heading or "Статья"
    return heading or "Статья"


def _specs_article_clauses(
    block: list[DocumentElement],
    cfg: dict,
    document_id: str,
    version: str,
    seq_start: int,
    heading: str,
    page: int | None,
) -> tuple[list[ChunkSpec], int]:
    """Длинную статью режем по пунктам. Статьи между собой не смешиваем и не перекрываем."""
    title = _article_title(block, heading)
    parts = _split_on(block, _is_clause)
    if len(parts) <= 1:
        evidence = _evidence_of(block)
        packed = _pack_text(
            evidence,
            [element.id for element in block],
            cfg,
            title,
            title,
            page,
            version,
            document_id,
            seq_start,
            0,
            0,
        )
        return packed, seq_start + len(packed)
    specs: list[ChunkSpec] = []
    seq = seq_start
    for part in parts:
        label = _el_text(part[0]) if _is_clause(part[0]) else title
        section = title if label == title else f"{title} / {label}"
        evidence = _evidence_of(part)
        packed = _pack_text(
            evidence,
            [element.id for element in part],
            cfg,
            title,
            section,
            page,
            version,
            document_id,
            seq,
            0,
            0,
        )
        specs.extend(packed)
        seq += len(packed)
    return specs, seq


def _specs_for_block(
    block: list[DocumentElement],
    cfg: dict,
    document_id: str,
    version: str,
    seq_start: int,
) -> tuple[list[ChunkSpec], int]:
    strategy = _canonical(cfg["strategy"])
    heading = _heading_prefix(block)
    if strategy == "normative_act" and not any(_is_article(element) for element in block):
        heading = heading or "Преамбула"
        section = "Преамбула"
    elif strategy == "official_explanation" and any(_is_question(element) for element in block):
        heading = next(_el_text(element) for element in block if _is_question(element))
        section = heading[:180]
    else:
        section = (block[0].section_path if block else "") or heading
    page = next((el.page_number for el in block if el.page_number is not None), None)
    overlap_chars = int(cfg.get("overlap_chars") or 0)
    if strategy in {"normative_act", "case", "methodology"}:
        overlap_chars = 0
    if strategy == "official_explanation" and any(_is_question(element) for element in block):
        overlap_chars = 0
    overlap = 0 if overlap_chars else int(cfg.get("overlap_pct_when_split") or 0)
    if strategy == "normative_act" and _size(_evidence_of(block), cfg) > _hard(cfg):
        return _specs_article_clauses(block, cfg, document_id, version, seq_start, heading, page)
    specs: list[ChunkSpec] = []
    seq = seq_start

    tables = [el for el in block if el.element_type == "table"]
    others = [el for el in block if el.element_type != "table"]

    if others:
        evidence = _evidence_of(others)
        ids = [el.id for el in others]
        packed = _pack_text(
            evidence,
            ids,
            cfg,
            heading,
            section,
            page,
            version,
            document_id,
            seq,
            overlap,
            overlap_chars,
        )
        specs.extend(packed)
        seq += len(packed)

    for table in tables:
        tpage = table.page_number if table.page_number is not None else page
        for text, ids in _table_chunks(table, cfg, heading):
            packed = _pack_text(
                text,
                ids,
                cfg,
                heading,
                section,
                tpage,
                version,
                document_id,
                seq,
                0,
            )
            if packed and _size(packed[0].evidence_text, cfg) > _hard(cfg):
                packed[0].applicability = packed[0].applicability or "oversized_unsplit"
            specs.extend(packed)
            seq += len(packed)

    return specs, seq


def _ensure_working_release(db: Session) -> CorpusRelease:
    cfg = load_chunking_config()
    version = cfg.get("working_corpus_release") or "working-draft"
    row = db.query(CorpusRelease).filter(CorpusRelease.version == version).first()
    if row is None:
        row = CorpusRelease(
            version=version,
            status="draft",
            notes="Рабочая версия корпуса, не опубликована в RAG",
        )
        db.add(row)
        db.flush()
    return row


def current_elements(db: Session, document_id: str) -> tuple[DocumentParseRun, list[DocumentElement]]:
    run = (
        db.query(DocumentParseRun)
        .filter(
            DocumentParseRun.document_id == document_id,
            DocumentParseRun.is_current.is_(True),
        )
        .first()
    )
    if run is None:
        raise ChunkingError("parse_missing", "Нет текущего результата парсинга.")
    elements = (
        db.query(DocumentElement)
        .filter(
            DocumentElement.document_id == document_id,
            DocumentElement.parse_run_id == run.id,
        )
        .order_by(DocumentElement.order_index)
        .all()
    )
    if not elements:
        raise ChunkingError("elements_missing", "Структурные элементы не найдены.")
    return run, elements


def chunk_parsed_document(db: Session, document_id: str) -> ChunkingResult:
    """Создаёт только draft-чанки. Не трогает approved и не считает embeddings."""
    doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == document_id).one_or_none()
    if doc is None:
        raise ChunkingError("not_found", "Документ не найден.")
    original_status = doc.status
    run, elements = current_elements(db, document_id)
    cfg = _class_cfg(doc.document_class)
    version = load_chunking_config()["chunking_version"]
    release = _ensure_working_release(db)
    blocks = _blocks(elements, cfg["strategy"], cfg)
    specs: list[ChunkSpec] = []
    seq = 1
    for block in blocks:
        packed, seq = _specs_for_block(block, cfg, doc.id, version, seq)
        specs.extend(packed)

    merged: list[ChunkSpec] = []
    for spec in specs:
        if (
            merged
            and spec.section_path == merged[-1].section_path
            and _size(merged[-1].evidence_text, cfg) < _soft(cfg)
            and spec.parent_chunk_id is None
            and merged[-1].parent_chunk_id is None
            and _size(merged[-1].evidence_text, cfg) + _size(spec.evidence_text, cfg) <= _hard(cfg)
        ):
            prev = merged[-1]
            prev.evidence_text = f"{prev.evidence_text}\n{spec.evidence_text}".strip()
            prev.retrieval_text = f"{prev.retrieval_text}\n{spec.evidence_text}".strip()
            prev.source_element_ids = list(dict.fromkeys(prev.source_element_ids + spec.source_element_ids))
            prev.token_count = count_tokens(prev.retrieval_text)
            prev.chunk_id = _chunk_id(doc.id, version, prev.source_element_ids, len(merged))
        else:
            merged.append(spec)
    specs = merged
    oversized = sum(1 for s in specs if s.applicability == "oversized_unsplit")

    db.query(KnowledgeChunk).filter(
        KnowledgeChunk.document_id == doc.id,
        KnowledgeChunk.chunking_version == version,
        KnowledgeChunk.approval_status == "draft",
    ).delete(synchronize_session=False)

    for spec in specs:
        if not spec.evidence_text.strip():
            raise ChunkingError("empty_evidence", "Пустой evidence_text недопустим.")
        db.add(
            KnowledgeChunk(
                id=spec.chunk_id,
                document_id=doc.id,
                text=spec.retrieval_text,
                section=spec.section_path,
                page=spec.page_number,
                approval_status="draft",
                embedding_status="pending",
                embedding=None,
                document_version=doc.version,
                corpus_version=release.version,
                chunking_version=version,
                source_element_ids=spec.source_element_ids,
                token_count=spec.token_count,
                parent_chunk_id=spec.parent_chunk_id,
                applicability=spec.applicability,
                retrieval_text=spec.retrieval_text,
                evidence_text=spec.evidence_text,
            )
        )
    db.add(
        UploadHistory(
            document_id=doc.id,
            action="chunk",
            actor_id="chunking",
            details={
                "status": "draft",
                "chunking_version": version,
                "corpus_version": release.version,
                "n_chunks": len(specs),
                "parse_run_id": run.id,
                "oversized_unsplit": oversized,
            },
        )
    )
    doc.status = original_status
    db.commit()
    try:
        from app.services.qa_jobs import enqueue_for_document

        enqueue_for_document(db, doc.id)
    except Exception:
        logger.exception("qa_jobs_enqueue_failed document_id=%s", doc.id)
    logger.info(
        "chunked document_id=%s n=%s version=%s oversized=%s",
        doc.id,
        len(specs),
        version,
        oversized,
    )
    return ChunkingResult(
        document_id=doc.id,
        chunking_version=version,
        corpus_version=release.version,
        chunks=specs,
        oversized_kept=oversized,
    )
