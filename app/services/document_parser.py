"""Структурный парсинг документов после intake. Без чанков и embeddings."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import BASE_DIR
from app.models import (
    DocumentElement,
    DocumentParseRun,
    ExpertReview,
    KnowledgeDocument,
    UploadHistory,
)
from app.services.intake import PRIVACY_PATTERNS, _privacy_hit
from app.services.text_normalizer import (
    classify_legal_block,
    looks_like_heading,
    looks_like_list_item,
    normalize_element_text,
    strip_repeated_headers,
)

logger = logging.getLogger("most.parser")

PARSER_VERSION = "ukz15-parse-1.0"
PARSED_DIR = BASE_DIR / "data" / "parsed"
MD_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
MD_TABLE_ROW = re.compile(r"^\s*\|.+\|\s*$")
MD_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
MD_LIST = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


@dataclass
class ParsedElement:
    element_type: str
    order_index: int
    text_original: str
    text_normalized: str
    section_path: str
    page_number: int | None
    source_location: dict
    normalization_actions: list[str] = field(default_factory=list)
    element_id: str = ""


@dataclass
class ParseResult:
    document_id: str
    document_version: str | None
    file_format: str | None
    parser_version: str
    elements: list[ParsedElement]
    warnings: list[str]
    errors: list[str]
    needs_review: bool
    parse_run_id: str | None = None
    status: str = "parsed"
    artifact_path: str | None = None


class ParseError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def resolve_document_path(doc: KnowledgeDocument) -> Path:
    if not doc.file_path:
        raise ParseError("missing_file", "У документа нет пути к файлу.")
    path = Path(doc.file_path)
    if not path.is_absolute():
        path = BASE_DIR / doc.file_path
    return path.resolve()


def parse_registered_document(db: Session, document_id: str) -> ParseResult:
    doc = (
        db.query(KnowledgeDocument)
        .filter(KnowledgeDocument.id == document_id)
        .first()
    )
    if doc is None:
        raise ParseError("not_found", "Документ не найден.")

    original_status = doc.status
    original_path = doc.file_path
    warnings: list[str] = []
    errors: list[str] = []
    needs_review = False

    if _is_privacy_blocked(db, doc):
        result = _store_result(
            db,
            doc,
            [],
            warnings=["privacy_hold: парсинг отложен до проверки эксперта"],
            errors=[],
            needs_review=True,
            status="privacy_blocked",
        )
        _keep_draft(doc, original_status)
        db.commit()
        return result

    try:
        path = resolve_document_path(doc)
        if not path.is_file():
            raise ParseError("missing_file", "Исходный файл не найден.")
        raw = path.read_bytes()
        fmt = (doc.file_format or path.suffix.lstrip(".")).lower()
        elements, extra_warnings, extra_errors, extra_review = _parse_bytes(raw, fmt)
        warnings.extend(extra_warnings)
        errors.extend(extra_errors)
        needs_review = needs_review or extra_review
        joined = "\n".join(el.text_original for el in elements)
        if _privacy_hit(joined):
            needs_review = True
            warnings.append("privacy_markers_in_text")
            _ensure_review(db, doc.id, "privacy_hold", {"source": "parser"})
        for index, element in enumerate(elements, start=1):
            element.order_index = index
            element.element_id = f"{doc.id}:{PARSER_VERSION}:{index:05d}"
        status = "error" if errors and not elements else ("needs_review" if needs_review else "parsed")
        result = _store_result(
            db,
            doc,
            elements,
            warnings,
            errors,
            needs_review=needs_review,
            status=status,
        )
        _keep_draft(doc, original_status)
        db.add(
            UploadHistory(
                document_id=doc.id,
                action="parse",
                actor_id="parser",
                details={
                    "status": status,
                    "parser_version": PARSER_VERSION,
                    "element_count": len(elements),
                    "needs_review": needs_review,
                    "warning_count": len(warnings),
                    "error_count": len(errors),
                },
            )
        )
        db.commit()
        if original_path != doc.file_path:
            doc.file_path = original_path
            db.commit()
        logger.info(
            "parse document_id=%s elements=%s status=%s review=%s",
            doc.id,
            len(elements),
            status,
            needs_review,
        )
        return result
    except ParseError as exc:
        db.rollback()
        errors.append(exc.code)
        result = _store_failed(db, document_id, warnings, errors, original_status, original_path)
        logger.info("parse failed document_id=%s code=%s", document_id, exc.code)
        return result
    except Exception:
        db.rollback()
        errors.append("parser_exception")
        result = _store_failed(db, document_id, warnings, errors, original_status, original_path)
        logger.exception("parse failed document_id=%s", document_id)
        return result


def _keep_draft(doc: KnowledgeDocument, original_status: str | None) -> None:
    if original_status:
        doc.status = original_status
    if doc.status != "archived":
        if original_status == "draft" or not original_status:
            doc.status = "draft"


def _is_privacy_blocked(db: Session, doc: KnowledgeDocument) -> bool:
    path = (doc.file_path or "").replace("\\", "/")
    if "quarantine" in path:
        return True
    pending = (
        db.query(ExpertReview)
        .filter(
            ExpertReview.object_id == doc.id,
            ExpertReview.reason == "privacy_hold",
            ExpertReview.status.in_(("pending", "open")),
        )
        .first()
    )
    return pending is not None


def _ensure_review(db: Session, document_id: str, reason: str, evidence: dict) -> None:
    existing = (
        db.query(ExpertReview)
        .filter(
            ExpertReview.object_id == document_id,
            ExpertReview.reason == reason,
            ExpertReview.status.in_(("pending", "open")),
        )
        .first()
    )
    if existing is not None:
        return
    db.add(
        ExpertReview(
            object_type="knowledge_document",
            object_id=document_id,
            reason=reason,
            status="pending",
            evidence=evidence,
        )
    )


def _store_failed(
    db: Session,
    document_id: str,
    warnings: list[str],
    errors: list[str],
    original_status: str | None,
    original_path: str | None,
) -> ParseResult:
    doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == document_id).one()
    _ensure_review(db, doc.id, "parse_error", {"errors": errors})
    result = _store_result(
        db,
        doc,
        [],
        warnings,
        errors,
        needs_review=True,
        status="error",
    )
    _keep_draft(doc, original_status)
    if original_path:
        doc.file_path = original_path
    db.add(
        UploadHistory(
            document_id=doc.id,
            action="parse",
            actor_id="parser",
            details={"status": "error", "error_count": len(errors)},
        )
    )
    db.commit()
    return result


def _store_result(
    db: Session,
    doc: KnowledgeDocument,
    elements: list[ParsedElement],
    warnings: list[str],
    errors: list[str],
    needs_review: bool,
    status: str,
) -> ParseResult:
    db.query(DocumentElement).filter(DocumentElement.document_id == doc.id).delete()
    db.query(DocumentParseRun).filter(DocumentParseRun.document_id == doc.id).update(
        {"is_current": False}
    )
    artifact = _write_artifact(doc, elements, warnings, errors, needs_review, status)
    run = DocumentParseRun(
        document_id=doc.id,
        parser_version=PARSER_VERSION,
        document_version=doc.version,
        file_format=doc.file_format,
        status=status,
        needs_review=needs_review,
        is_current=True,
        warnings=warnings,
        errors=errors,
        element_count=len(elements),
        artifact_path=artifact,
    )
    db.add(run)
    db.flush()
    for element in elements:
        db.add(
            DocumentElement(
                id=element.element_id,
                document_id=doc.id,
                parse_run_id=run.id,
                element_type=element.element_type,
                order_index=element.order_index,
                text_original=element.text_original,
                text_normalized=element.text_normalized,
                section_path=element.section_path or None,
                page_number=element.page_number,
                source_location=element.source_location,
                normalization_actions=element.normalization_actions,
            )
        )
    doc.parse_status = status
    doc.parser_version = PARSER_VERSION
    db.flush()
    return ParseResult(
        document_id=doc.id,
        document_version=doc.version,
        file_format=doc.file_format,
        parser_version=PARSER_VERSION,
        elements=elements,
        warnings=warnings,
        errors=errors,
        needs_review=needs_review,
        parse_run_id=run.id,
        status=status,
        artifact_path=artifact,
    )


def _write_artifact(
    doc: KnowledgeDocument,
    elements: list[ParsedElement],
    warnings: list[str],
    errors: list[str],
    needs_review: bool,
    status: str,
) -> str:
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "document_id": doc.id,
        "document_version": doc.version,
        "file_format": doc.file_format,
        "parser_version": PARSER_VERSION,
        "status": status,
        "needs_review": needs_review,
        "warnings": warnings,
        "errors": errors,
        "created_at": datetime.utcnow().isoformat(),
        "elements": [
            {
                "element_id": el.element_id,
                "element_type": el.element_type,
                "order_index": el.order_index,
                "text_original": el.text_original,
                "text_normalized": el.text_normalized,
                "section_path": el.section_path,
                "page_number": el.page_number,
                "source_location": el.source_location,
                "normalization_actions": el.normalization_actions,
            }
            for el in elements
        ],
    }
    path = PARSED_DIR / f"{doc.id}_{PARSER_VERSION}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        return str(path.relative_to(BASE_DIR)).replace("\\", "/")
    except ValueError:
        return str(path)


def _parse_bytes(raw: bytes, fmt: str) -> tuple[list[ParsedElement], list[str], list[str], bool]:
    if fmt == "docx":
        return _parse_docx(raw)
    if fmt == "pdf":
        return _parse_pdf(raw)
    if fmt == "md":
        return _parse_markdown(raw)
    if fmt == "txt":
        return _parse_txt(raw)
    return [], [], [f"unsupported_format:{fmt}"], True


def _new_element(
    element_type: str,
    text: str,
    order_index: int,
    section_path: str,
    page_number: int | None,
    source_location: dict,
    ambiguous: bool = False,
) -> tuple[ParsedElement, list[str]]:
    warnings: list[str] = []
    normalized, actions = normalize_element_text(text)
    legal = classify_legal_block(text)
    if legal:
        element_type = legal
    if ambiguous:
        warnings.append("ambiguous_structure")
        normalized = text
        actions = ["keep_original_ambiguous"]
    return (
        ParsedElement(
            element_type=element_type,
            order_index=order_index,
            text_original=text,
            text_normalized=normalized,
            section_path=section_path,
            page_number=page_number,
            source_location=source_location,
            normalization_actions=actions,
        ),
        warnings,
    )


def _heading_path(stack: list[str], title: str, level: int) -> list[str]:
    while len(stack) >= level:
        stack.pop()
    while len(stack) < level - 1:
        stack.append("")
    stack.append(title)
    return [part for part in stack if part]


def _parse_docx(raw: bytes) -> tuple[list[ParsedElement], list[str], list[str], bool]:
    from io import BytesIO

    from docx import Document
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    warnings: list[str] = []
    errors: list[str] = []
    elements: list[ParsedElement] = []
    try:
        document = Document(BytesIO(raw))
    except Exception as exc:
        raise ParseError("corrupt_docx", "Не удалось разобрать DOCX.") from exc

    section_stack: list[str] = []
    order = 0

    def emit(kind: str, text: str, location: dict, level: int | None = None) -> None:
        nonlocal order
        if text is None:
            return
        raw_text = text if kind == "table" else text.strip()
        if kind != "table" and not raw_text:
            return
        path = " / ".join(section_stack)
        if kind == "heading" and level:
            path = " / ".join(_heading_path(section_stack, raw_text, level))
        order += 1
        item, extra = _new_element(
            kind,
            raw_text,
            order,
            path,
            None,
            location,
        )
        warnings.extend(extra)
        elements.append(item)

    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            paragraph = Paragraph(child, document)
            text, links = _docx_paragraph(paragraph)
            level = _docx_heading_level(paragraph)
            if level:
                emit(
                    "heading",
                    text,
                    {"kind": "paragraph", "style": paragraph.style.name if paragraph.style else None, "hyperlinks": links},
                    level=level,
                )
                continue
            if _docx_is_list(paragraph) or looks_like_list_item(text):
                emit(
                    "list_item",
                    text,
                    {"kind": "list_item", "hyperlinks": links},
                )
                continue
            emit("paragraph", text, {"kind": "paragraph", "hyperlinks": links})
        elif child.tag == qn("w:tbl"):
            table = Table(child, document)
            rows_payload = []
            table_lines = []
            for r_idx, row in enumerate(table.rows):
                cells = [cell.text.strip() for cell in row.cells]
                rows_payload.append(cells)
                table_lines.append(" | ".join(cells))
                order += 1
                row_text = " | ".join(cells)
                item, extra = _new_element(
                    "table_row",
                    row_text,
                    order,
                    " / ".join(section_stack),
                    None,
                    {"kind": "table_row", "row": r_idx, "cells": cells},
                )
                warnings.extend(extra)
                elements.append(item)
            order += 1
            item, extra = _new_element(
                "table",
                "\n".join(table_lines),
                order,
                " / ".join(section_stack),
                None,
                {"kind": "table", "rows": rows_payload},
            )
            warnings.extend(extra)
            elements.append(item)

    footnotes = _docx_footnotes(document)
    for idx, note in enumerate(footnotes):
        emit("footnote", note, {"kind": "footnote", "index": idx})

    warnings.append("docx_page_number_unavailable")
    if not elements:
        warnings.append("empty_document")
        return elements, warnings, errors, True
    return elements, warnings, errors, False


def _docx_paragraph(paragraph) -> tuple[str, list[dict]]:
    from docx.oxml.ns import qn

    links = []
    for node in paragraph._p.iter(qn("w:hyperlink")):
        rid = node.get(qn("r:id"))
        url = None
        if rid and rid in paragraph.part.rels:
            url = paragraph.part.rels[rid].target_ref
        link_text = "".join((t.text or "") for t in node.iter(qn("w:t")))
        links.append({"text": link_text, "url": url})
    return paragraph.text or "", links


def _docx_heading_level(paragraph) -> int | None:
    from docx.oxml.ns import qn

    style = paragraph.style.name if paragraph.style else ""
    if style.startswith("Heading"):
        tail = style.replace("Heading", "").strip()
        try:
            return max(1, int(tail))
        except ValueError:
            return 1
    pPr = paragraph._p.pPr
    if pPr is None:
        return None
    outline = pPr.find(qn("w:outlineLvl"))
    if outline is not None:
        try:
            return int(outline.get(qn("w:val")) or 0) + 1
        except (TypeError, ValueError):
            return 1
    return None


def _docx_is_list(paragraph) -> bool:
    from docx.oxml.ns import qn

    style = paragraph.style.name if paragraph.style else ""
    if "List" in (style or ""):
        return True
    pPr = paragraph._p.pPr
    if pPr is None:
        return False
    return pPr.find(qn("w:numPr")) is not None


def _docx_footnotes(document) -> list[str]:
    from docx.oxml.ns import qn

    notes = []
    try:
        part = document.part.footnotes_part
    except Exception:
        return notes
    if part is None:
        return notes
    for fn in part.element.findall(qn("w:footnote")):
        fn_type = fn.get(qn("w:type"))
        if fn_type in {"separator", "continuationSeparator"}:
            continue
        texts = [t.text or "" for t in fn.iter(qn("w:t"))]
        joined = "".join(texts).strip()
        if joined:
            notes.append(joined)
    return notes


def _parse_pdf(raw: bytes) -> tuple[list[ParsedElement], list[str], list[str], bool]:
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError as exc:
            raise ParseError("pymupdf_missing", "PyMuPDF не установлен.") from exc

    warnings: list[str] = []
    errors: list[str] = []
    needs_review = False
    try:
        pdf = fitz.open(stream=raw, filetype="pdf")
    except Exception as exc:
        raise ParseError("corrupt_pdf", "Не удалось открыть PDF.") from exc

    page_texts: list[str] = []
    page_blocks: list[list[dict]] = []
    ocr_pages: list[int] = []
    for page in pdf:
        text = (page.get_text("text") or "").strip()
        page_texts.append(text)
        if not text:
            ocr_text, ocr_status = _ocr_pdf_page(page)
            if ocr_status == "ok" and ocr_text.strip():
                page_texts[-1] = ocr_text
                warnings.append(f"ocr_used:page={page.number + 1}")
            else:
                ocr_pages.append(page.number + 1)
                warnings.append(f"ocr_required:page={page.number + 1}:{ocr_status}")
                needs_review = True
        page_blocks.append(_pdf_blocks(page))

    stripped, header_actions = strip_repeated_headers(page_texts)
    if header_actions:
        warnings.extend(header_actions)
        for idx, text in enumerate(stripped):
            page_texts[idx] = text

    elements: list[ParsedElement] = []
    section_stack: list[str] = []
    order = 0
    for page_idx, blocks in enumerate(page_blocks):
        page_no = page_idx + 1
        table_boxes = _pdf_tables(pdf[page_idx], page_no, section_stack)
        for table_el in table_boxes["elements"]:
            order += 1
            table_el.order_index = order
            elements.append(table_el)
        warnings.extend(table_boxes["warnings"])
        for block in blocks:
            if _bbox_in_tables(block.get("bbox"), table_boxes["bboxes"]):
                continue
            text = block.get("text") or ""
            if page_texts[page_idx] and header_actions:
                pass
            text = text.strip()
            if not text:
                continue
            kind = "paragraph"
            level = None
            if looks_like_heading(text.splitlines()[0] if text else ""):
                kind = "heading"
                level = 1
            elif looks_like_list_item(text):
                kind = "list_item"
            if kind == "heading" and level:
                path = " / ".join(_heading_path(section_stack, text.splitlines()[0], level))
            else:
                path = " / ".join(section_stack)
            order += 1
            item, extra = _new_element(
                kind,
                text,
                order,
                path,
                page_no,
                {
                    "kind": "pdf_block",
                    "page": page_no,
                    "bbox": block.get("bbox"),
                },
            )
            warnings.extend(extra)
            elements.append(item)

    if ocr_pages and not any(el.text_original.strip() for el in elements):
        errors.append("ocr_unavailable_or_low_quality")
        needs_review = True
    if not elements:
        warnings.append("empty_pdf_or_scan")
        needs_review = True
    pdf.close()
    return elements, warnings, errors, needs_review


def _pdf_blocks(page) -> list[dict]:
    payload = page.get_text("dict") or {}
    blocks = []
    for block in payload.get("blocks", []):
        if block.get("type") != 0:
            continue
        lines = []
        for line in block.get("lines", []):
            lines.append("".join(span.get("text", "") for span in line.get("spans", [])))
        text = "\n".join(lines).strip()
        if not text:
            continue
        blocks.append({"text": text, "bbox": list(block.get("bbox") or []), "y": (block.get("bbox") or [0, 0])[1]})
    blocks.sort(key=lambda item: (round(item["y"], 1), item["bbox"][0] if item["bbox"] else 0))
    return blocks


def _pdf_tables(page, page_no: int, section_stack: list[str]) -> dict:
    warnings: list[str] = []
    elements: list[ParsedElement] = []
    bboxes: list[list[float]] = []
    finder = getattr(page, "find_tables", None)
    if finder is None:
        return {"elements": elements, "warnings": warnings, "bboxes": bboxes}
    try:
        found = finder()
        tables = found.tables if found is not None else []
    except Exception:
        warnings.append(f"table_extract_failed:page={page_no}")
        return {"elements": elements, "warnings": warnings, "bboxes": bboxes}
    for t_idx, table in enumerate(tables):
        try:
            data = table.extract()
            bbox = list(getattr(table, "bbox", []) or [])
        except Exception:
            warnings.append(f"table_extract_failed:page={page_no}:idx={t_idx}")
            continue
        bboxes.append(bbox)
        rows = [[(cell or "").strip() for cell in row] for row in (data or [])]
        for r_idx, cells in enumerate(rows):
            elements.append(
                ParsedElement(
                    element_type="table_row",
                    order_index=0,
                    text_original=" | ".join(cells),
                    text_normalized=normalize_element_text(" | ".join(cells))[0],
                    section_path=" / ".join(section_stack),
                    page_number=page_no,
                    source_location={"kind": "table_row", "page": page_no, "row": r_idx, "cells": cells, "bbox": bbox},
                    normalization_actions=["technical_whitespace"] if False else [],
                )
            )
        table_text = "\n".join(" | ".join(cells) for cells in rows)
        normalized, actions = normalize_element_text(table_text)
        elements.append(
            ParsedElement(
                element_type="table",
                order_index=0,
                text_original=table_text,
                text_normalized=normalized,
                section_path=" / ".join(section_stack),
                page_number=page_no,
                source_location={"kind": "table", "page": page_no, "index": t_idx, "rows": rows, "bbox": bbox},
                normalization_actions=actions,
            )
        )
    return {"elements": elements, "warnings": warnings, "bboxes": bboxes}


def _bbox_in_tables(bbox: list[float] | None, tables: list[list[float]]) -> bool:
    if not bbox or not tables or len(bbox) < 4:
        return False
    x0, y0, x1, y1 = bbox[:4]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    for tb in tables:
        if len(tb) < 4:
            continue
        if tb[0] <= cx <= tb[2] and tb[1] <= cy <= tb[3]:
            return True
    return False


def _ocr_available() -> bool:
    import shutil

    return shutil.which("tesseract") is not None


def _ocr_pdf_page(page) -> tuple[str, str]:
    if not _ocr_available():
        return "", "ocr_unavailable"
    try:
        if hasattr(page, "get_textpage_ocr"):
            textpage = page.get_textpage_ocr(flags=0, language="rus+eng")
            text = page.get_text("text", textpage=textpage) or ""
            if text.strip():
                return text, "ok"
            return "", "ocr_empty"
    except Exception:
        return "", "ocr_unavailable"
    return "", "ocr_unavailable"


def _decode_text(raw: bytes) -> tuple[str, list[str]]:
    warnings: list[str] = []
    for enc in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return raw.decode(enc), warnings
        except UnicodeDecodeError:
            continue
    warnings.append("decode_replaced")
    return raw.decode("utf-8", errors="replace"), warnings


def _parse_txt(raw: bytes) -> tuple[list[ParsedElement], list[str], list[str], bool]:
    text, warnings = _decode_text(raw)
    errors: list[str] = []
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    elements: list[ParsedElement] = []
    section_stack: list[str] = []
    order = 0
    for block in blocks:
        lines = [ln.rstrip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        buffer: list[str] = []

        def flush_buffer() -> None:
            nonlocal order
            if not buffer:
                return
            joined = "\n".join(buffer)
            kind = "list_item" if all(looks_like_list_item(line) for line in buffer) else "paragraph"
            order += 1
            item, extra = _new_element(
                kind,
                joined,
                order,
                " / ".join(section_stack),
                None,
                {"kind": "txt_block"},
            )
            warnings.extend(extra)
            elements.append(item)
            buffer.clear()

        for line in lines:
            if looks_like_heading(line) and len(lines) == 1:
                flush_buffer()
                order += 1
                path = " / ".join(_heading_path(section_stack, line.strip(), 1))
                item, extra = _new_element(
                    "heading",
                    line.strip(),
                    order,
                    path,
                    None,
                    {"kind": "txt_heading"},
                )
                warnings.extend(extra)
                elements.append(item)
            elif looks_like_heading(line) and not buffer:
                order += 1
                path = " / ".join(_heading_path(section_stack, line.strip(), 1))
                item, extra = _new_element(
                    "heading",
                    line.strip(),
                    order,
                    path,
                    None,
                    {"kind": "txt_heading"},
                )
                warnings.extend(extra)
                elements.append(item)
            else:
                buffer.append(line)
        flush_buffer()
    if not elements:
        warnings.append("empty_txt")
        return elements, warnings, errors, True
    return elements, warnings, errors, False


def _parse_markdown(raw: bytes) -> tuple[list[ParsedElement], list[str], list[str], bool]:
    text, warnings = _decode_text(raw)
    errors: list[str] = []
    lines = text.replace("\r\n", "\n").split("\n")
    elements: list[ParsedElement] = []
    section_stack: list[str] = []
    order = 0
    i = 0
    para: list[str] = []

    def flush_para() -> None:
        nonlocal order, para
        if not para:
            return
        joined = "\n".join(para).strip()
        para = []
        if not joined:
            return
        kind = "list_item" if all(MD_LIST.match(ln) or looks_like_list_item(ln) for ln in joined.split("\n")) else "paragraph"
        order += 1
        item, extra = _new_element(
            kind,
            joined,
            order,
            " / ".join(section_stack),
            None,
            {"kind": "md_block"},
        )
        warnings.extend(extra)
        elements.append(item)

    while i < len(lines):
        line = lines[i]
        heading = MD_HEADING.match(line)
        if heading:
            flush_para()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            path = " / ".join(_heading_path(section_stack, title, level))
            order += 1
            item, extra = _new_element(
                "heading",
                title,
                order,
                path,
                None,
                {"kind": "md_heading", "level": level},
            )
            warnings.extend(extra)
            elements.append(item)
            i += 1
            continue
        if MD_TABLE_ROW.match(line):
            flush_para()
            rows = []
            while i < len(lines) and MD_TABLE_ROW.match(lines[i]):
                if MD_TABLE_SEP.match(lines[i]):
                    i += 1
                    continue
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells)
                i += 1
            for r_idx, cells in enumerate(rows):
                order += 1
                item, extra = _new_element(
                    "table_row",
                    " | ".join(cells),
                    order,
                    " / ".join(section_stack),
                    None,
                    {"kind": "table_row", "row": r_idx, "cells": cells},
                )
                warnings.extend(extra)
                elements.append(item)
            order += 1
            table_text = "\n".join(" | ".join(c) for c in rows)
            item, extra = _new_element(
                "table",
                table_text,
                order,
                " / ".join(section_stack),
                None,
                {"kind": "table", "rows": rows},
            )
            warnings.extend(extra)
            elements.append(item)
            continue
        if MD_LIST.match(line):
            flush_para()
            items = []
            while i < len(lines) and MD_LIST.match(lines[i]):
                items.append(re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", lines[i]).strip())
                i += 1
            for item_text in items:
                order += 1
                el, extra = _new_element(
                    "list_item",
                    item_text,
                    order,
                    " / ".join(section_stack),
                    None,
                    {"kind": "md_list_item"},
                )
                warnings.extend(extra)
                elements.append(el)
            continue
        if not line.strip():
            flush_para()
            i += 1
            continue
        para.append(line)
        i += 1
    flush_para()
    if not elements:
        warnings.append("empty_md")
        return elements, warnings, errors, True
    return elements, warnings, errors, False
