"""Структурный парсинг: форматы, юридический текст, повтор, без чанков."""

from __future__ import annotations

import uuid
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from app.models import DocumentElement, DocumentParseRun, KnowledgeChunk, KnowledgeDocument
from app.services.document_parser import PARSER_VERSION, parse_registered_document
from app.services.intake import inspect_payload, intake_document
from tests import isolation
from tests.isolation import (
    assert_isolated_path,
    assert_prod_corpus_untouched,
    assert_test_database,
)
from tests.test_intake import _cleanup

isolation.activate()
SessionLocal = isolation.SessionLocal
assert_test_database()

LEGAL = (
    "Статья 12.1 не применяется до 01.01.2020. "
    "Сумма 1000 руб. не подлежит возврату, за исключением случая, указанного в пункте 2."
)


def _add_hyperlink(paragraph, text: str, url: str) -> None:
    rel = paragraph.part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), rel)
    run = OxmlElement("w:r")
    text_el = OxmlElement("w:t")
    text_el.text = text
    run.append(text_el)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def _docx_bytes(*, headings=False, table=False, lists=False, links=False, legal=False) -> bytes:
    doc = Document()
    if headings:
        doc.add_heading("Глава 1. Общие положения", 1)
        doc.add_heading("Статья 12.1", 2)
        doc.add_paragraph("Текст под заголовком.")
    if table:
        table_obj = doc.add_table(rows=2, cols=2)
        table_obj.cell(0, 0).text = "Норма"
        table_obj.cell(0, 1).text = "Статья 12.1"
        table_obj.cell(1, 0).text = "Срок"
        table_obj.cell(1, 1).text = "01.01.2020"
    if lists:
        doc.add_paragraph("1. Первый пункт не отменяется.")
        doc.add_paragraph("2. Второй пункт.")
        p = doc.add_paragraph("третий элемент списка")
        p.style = "List Number"
    if links:
        para = doc.add_paragraph("См. ")
        _add_hyperlink(para, "официальный текст", "https://example.test/law")
    if legal:
        doc.add_paragraph(LEGAL)
    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _cyr_fontfile() -> str:
    windir = Path(r"C:\Windows\Fonts")
    for name in ("arial.ttf", "Arial.ttf", "times.ttf", "timesi.ttf", "calibri.ttf"):
        candidate = windir / name
        if candidate.is_file():
            return str(candidate)
    raise AssertionError("Нет системного TTF для кириллицы в тесте PDF")


def _pdf_bytes(*pages: str) -> bytes:
    import pymupdf as fitz

    fontfile = _cyr_fontfile()
    pdf = fitz.open()
    for text in pages:
        page = pdf.new_page()
        page.insert_font(fontname="F0", fontfile=fontfile)
        page.insert_text((72, 72), text, fontsize=12, fontname="F0")
    raw = pdf.tobytes()
    pdf.close()
    return raw


def _scanned_pdf_bytes(text: str) -> bytes:
    import pymupdf as fitz

    fontfile = _cyr_fontfile()
    src = fitz.open()
    page = src.new_page()
    page.insert_font(fontname="F0", fontfile=fontfile)
    page.insert_text((72, 72), text, fontsize=14, fontname="F0")
    pix = page.get_pixmap()
    out = fitz.open()
    dest = out.new_page(width=page.rect.width, height=page.rect.height)
    dest.insert_image(dest.rect, pixmap=pix)
    raw = out.tobytes()
    src.close()
    out.close()
    return raw


def _intake(raw: bytes, name: str, **kwargs):
    isolation.activate()
    db = isolation.SessionLocal()
    try:
        token = uuid.uuid4().hex[:8]
        result = intake_document(
            db,
            raw,
            name,
            title=kwargs.get("title") or f"Parse {token}",
            document_class=kwargs.get("document_class") or "Учебный материал",
            corpus_source=kwargs.get("corpus_source", "database_edu_lectcii"),
            owner="qa-parse",
            actor_id="qa-parse",
            content_type=kwargs.get("content_type"),
        )
        assert_isolated_path(result.file_path)
        return result
    finally:
        db.close()


def _elements(doc_id: str) -> list[DocumentElement]:
    db = SessionLocal()
    try:
        return (
            db.query(DocumentElement)
            .filter(DocumentElement.document_id == doc_id)
            .order_by(DocumentElement.order_index)
            .all()
        )
    finally:
        db.close()


def _doc(doc_id: str) -> KnowledgeDocument:
    db = SessionLocal()
    try:
        return db.query(KnowledgeDocument).filter(KnowledgeDocument.id == doc_id).one()
    finally:
        db.close()


def _assert_no_rag(doc_id: str) -> None:
    db = SessionLocal()
    try:
        assert db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id == doc_id).count() == 0
        row = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == doc_id).one()
        assert row.status == "draft"
        assert row.reviewed_at is None
    finally:
        db.close()


def test_docx_headings():
    created = _intake(_docx_bytes(headings=True), "head.docx")
    try:
        els = _elements(created.document_id)
        types = [e.element_type for e in els]
        assert "heading" in types or "article" in types
        texts = " ".join(e.text_original for e in els)
        assert "Глава 1" in texts
        assert "Статья 12.1" in texts
        assert all(e.page_number is None for e in els)
        _assert_no_rag(created.document_id)
    finally:
        _cleanup(created.document_id)


def test_docx_tables():
    created = _intake(_docx_bytes(table=True), "table.docx")
    try:
        els = _elements(created.document_id)
        assert any(e.element_type == "table" for e in els)
        assert any(e.element_type == "table_row" for e in els)
        cells = " ".join(e.text_original for e in els)
        assert "Статья 12.1" in cells
        assert "01.01.2020" in cells
    finally:
        _cleanup(created.document_id)


def test_docx_numbered_lists():
    created = _intake(_docx_bytes(lists=True), "list.docx")
    try:
        els = _elements(created.document_id)
        joined = "\n".join(e.text_original for e in els)
        assert "Первый пункт не отменяется" in joined
        assert any(e.element_type == "list_item" for e in els)
    finally:
        _cleanup(created.document_id)


def test_docx_hyperlinks():
    created = _intake(_docx_bytes(links=True), "link.docx")
    try:
        els = _elements(created.document_id)
        found = False
        for el in els:
            links = (el.source_location or {}).get("hyperlinks") or []
            if any((item or {}).get("url") == "https://example.test/law" for item in links):
                found = True
        assert found
    finally:
        _cleanup(created.document_id)


def test_pdf_text_layer_and_pages():
    raw = _pdf_bytes("Страница один. Статья 1.", "Страница два. не применяется.")
    created = _intake(raw, "multi.pdf")
    try:
        info = inspect_payload(raw, "multi.pdf")
        assert info.format == "pdf"
        els = _elements(created.document_id)
        pages = {e.page_number for e in els if e.page_number}
        assert 1 in pages and 2 in pages
        text = " ".join(e.text_normalized for e in els)
        assert "Страница один" in text
        assert "Страница два" in text
        assert "не применяется" in text
        assert any((e.source_location or {}).get("bbox") for e in els)
        _assert_no_rag(created.document_id)
    finally:
        _cleanup(created.document_id)


def test_scanned_pdf_does_not_invent_text():
    raw = _scanned_pdf_bytes("Секретный выдуманный текст XYZ123")
    created = _intake(raw, "scan.pdf")
    try:
        row = _doc(created.document_id)
        assert row.status == "draft"
        els = _elements(created.document_id)
        invented = any("XYZ123" in (e.text_original or "") for e in els)
        db = SessionLocal()
        try:
            run = (
                db.query(DocumentParseRun)
                .filter(
                    DocumentParseRun.document_id == created.document_id,
                    DocumentParseRun.is_current.is_(True),
                )
                .one()
            )
            joined_warn = " ".join(run.warnings or [])
            if invented:
                assert "ocr_used" in joined_warn
            else:
                assert run.needs_review is True
                assert "ocr_" in joined_warn or (run.errors or [])
        finally:
            db.close()
    finally:
        _cleanup(created.document_id)


def test_txt_cyrillic_and_legal_keep():
    raw = f"ГЛАВА 1\n\n{LEGAL}\n\n1. Первый абзац.\n2. Второй абзац.\n".encode("utf-8")
    created = _intake(raw, "norm.txt")
    try:
        els = _elements(created.document_id)
        text = "\n".join(e.text_normalized for e in els)
        assert "не применяется" in text
        assert "01.01.2020" in text
        assert "12.1" in text
        assert "1000" in text
        assert "не подлежит" in text
        assert "за исключением" in text
        types = {e.element_type for e in els}
        assert "heading" in types or "article" in types
        _assert_no_rag(created.document_id)
    finally:
        _cleanup(created.document_id)


def test_markdown_headings_and_tables():
    raw = (
        "# Раздел A\n\nПараграф не сокращается.\n\n"
        "| Колонка | Значение |\n| --- | --- |\n| Дата | 01.01.2020 |\n"
        "| Статья | 12.1 |\n"
    ).encode("utf-8")
    created = _intake(raw, "note.md")
    try:
        els = _elements(created.document_id)
        types = [e.element_type for e in els]
        assert "heading" in types
        assert "table" in types
        text = " ".join(e.text_original for e in els)
        assert "01.01.2020" in text
        assert "12.1" in text
    finally:
        _cleanup(created.document_id)


def test_reparse_is_idempotent():
    created = _intake(_docx_bytes(headings=True, legal=True), "again.docx")
    try:
        first = _elements(created.document_id)
        ids = [e.id for e in first]
        db = SessionLocal()
        try:
            second = parse_registered_document(db, created.document_id)
        finally:
            db.close()
        again = _elements(created.document_id)
        assert [e.id for e in again] == ids
        assert second.parser_version == PARSER_VERSION
        db = SessionLocal()
        try:
            current = (
                db.query(DocumentParseRun)
                .filter(
                    DocumentParseRun.document_id == created.document_id,
                    DocumentParseRun.is_current.is_(True),
                )
                .count()
            )
            assert current == 1
            assert db.query(DocumentElement).filter(DocumentElement.document_id == created.document_id).count() == len(ids)
        finally:
            db.close()
        assert _doc(created.document_id).status == "draft"
    finally:
        _cleanup(created.document_id)


def test_parser_error_keeps_original_file():
    created = _intake("Исходник кириллица".encode("utf-8"), "keep.txt")
    try:
        row = _doc(created.document_id)
        path = Path(row.file_path)
        if not path.is_absolute():
            from app.config import BASE_DIR

            path = BASE_DIR / row.file_path
        assert path.is_file()
        original = path.read_bytes()
        db = SessionLocal()
        try:
            with patch(
                "app.services.document_parser._parse_bytes",
                side_effect=RuntimeError("forced-parse"),
            ):
                result = parse_registered_document(db, created.document_id)
        finally:
            db.close()
        assert result.status == "error"
        assert path.is_file()
        assert path.read_bytes() == original
        after = _doc(created.document_id)
        assert after.status == "draft"
        assert after.file_path == row.file_path
        _assert_no_rag(created.document_id)
    finally:
        _cleanup(created.document_id)


def test_routes_still_use_intake():
    import inspect

    from app.routers import admin, most

    src = inspect.getsource(most.upload_document)
    assert "intake_document" in src
    assert "replace_chunks" not in src
    assert "intake_document" in inspect.getsource(admin._intake_from_bytes)


def test_isolated_from_prod_corpus_and_db():
    assert_test_database()
    from app.services import corpus as corpus_svc

    assert_isolated_path(corpus_svc.EDU_DIR)
    assert_isolated_path(corpus_svc.LAW_DIR)
    created = _intake("изолированный текст".encode("utf-8"), "iso.txt")
    try:
        row = _doc(created.document_id)
        assert_isolated_path(row.file_path)
        assert "data/corpus/database_edu_lectcii" not in (row.file_path or "").replace("\\", "/")
        assert_prod_corpus_untouched()
        _assert_no_rag(created.document_id)
    finally:
        _cleanup(created.document_id)
    assert_prod_corpus_untouched()
