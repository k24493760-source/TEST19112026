"""Единый intake: форматы, дубли, draft, чанки после разбора, rollback."""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

from docx import Document

from tests import isolation

isolation.activate()
SessionLocal = isolation.SessionLocal
from tests.isolation import assert_isolated_path, assert_test_database
from app.models import (
    ChunkEmbedding,
    CorpusReleaseItem,
    DocumentElement,
    DocumentHistory,
    DocumentParseRun,
    ExpertReview,
    KnowledgeChunk,
    KnowledgeDocument,
    UploadHistory,
)
from app.services import corpus as corpus_svc
from app.services.intake import IntakeError, inspect_payload, intake_document

assert_test_database()

def _blank_pdf() -> bytes:
    from PyPDF2 import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _docx_bytes(text: str) -> bytes:
    doc = Document()
    doc.add_paragraph(text)
    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _cleanup(doc_id: str) -> None:
    db = SessionLocal()
    try:
        doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == doc_id).first()
        path = None
        if doc and doc.file_path:
            path = Path(doc.file_path)
            if not path.is_absolute():
                from app.config import BASE_DIR

                path = BASE_DIR / doc.file_path
        db.query(DocumentElement).filter(DocumentElement.document_id == doc_id).delete()
        db.query(DocumentParseRun).filter(DocumentParseRun.document_id == doc_id).delete()
        chunk_ids = [
            row.id
            for row in db.query(KnowledgeChunk.id).filter(KnowledgeChunk.document_id == doc_id)
        ]
        if chunk_ids:
            db.query(ChunkEmbedding).filter(ChunkEmbedding.chunk_id.in_(chunk_ids)).delete(
                synchronize_session=False
            )
        db.query(CorpusReleaseItem).filter(CorpusReleaseItem.document_id == doc_id).delete(
            synchronize_session=False
        )
        db.query(DocumentHistory).filter(DocumentHistory.document_id == doc_id).delete(
            synchronize_session=False
        )
        db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id == doc_id).delete()
        db.query(UploadHistory).filter(UploadHistory.document_id == doc_id).delete()
        db.query(ExpertReview).filter(ExpertReview.object_id == doc_id).delete()
        if doc:
            db.delete(doc)
        db.commit()
        if path and path.is_file():
            assert_isolated_path(path)
            path.unlink()
        from app.services.document_parser import PARSED_DIR, PARSER_VERSION

        artifact = PARSED_DIR / f"{doc_id}_{PARSER_VERSION}.json"
        if artifact.is_file():
            assert_isolated_path(artifact)
            artifact.unlink()
    finally:
        db.close()


def _intake(raw: bytes, name: str, **kwargs):
    isolation.activate()
    db = isolation.SessionLocal()
    try:
        token = uuid.uuid4().hex[:8]
        return intake_document(
            db,
            raw,
            name,
            title=kwargs.get("title") or f"Intake {token}",
            document_class=kwargs.get("document_class") or "Учебный материал",
            corpus_source=kwargs.get("corpus_source", "database_edu_lectcii"),
            owner="qa-intake",
            version_action=kwargs.get("version_action") or "new",
            related_document_id=kwargs.get("related_document_id"),
            actor_id="qa",
            content_type=kwargs.get("content_type"),
        )
    finally:
        db.close()


def test_inspect_rejects_exe_and_bad_format():
    try:
        inspect_payload(b"MZ\x90\x00not-an-exe-really", "tool.exe")
        raise AssertionError("exe must fail")
    except IntakeError as exc:
        assert exc.code == "forbidden_type"
    try:
        inspect_payload(b"not a document", "file.bin")
        raise AssertionError("bin must fail")
    except IntakeError as exc:
        assert exc.code == "unsupported_format"


def test_inspect_rejects_corrupt_docx_and_pdf():
    try:
        inspect_payload(b"PK\x03\x04notzip", "a.docx")
        raise AssertionError("corrupt docx")
    except IntakeError as exc:
        assert exc.code == "corrupt_file"
    try:
        inspect_payload(b"%PDF-1.4\nthis is not a pdf", "a.pdf")
        raise AssertionError("corrupt pdf")
    except IntakeError as exc:
        assert exc.code in {"corrupt_file", "encrypted_pdf"}


def test_txt_md_docx_pdf_accepted_as_draft_and_text_is_chunked():
    created = []
    try:
        txt = _intake(f"plain {uuid.uuid4().hex}".encode("utf-8"), "note.txt")
        created.append(txt)
        md = _intake(f"# md {uuid.uuid4().hex}".encode("utf-8"), "note.md")
        created.append(md)
        docx = _intake(_docx_bytes(f"Word {uuid.uuid4().hex}"), "mod.docx")
        created.append(docx)
        pdf_bytes = _blank_pdf()
        pdf_info = inspect_payload(pdf_bytes, "scan.pdf")
        assert pdf_info.format == "pdf"
        pdf = _intake(pdf_bytes, "scan.pdf")
        created.append(pdf)
        assert pdf_info.needs_ocr is True
        db = SessionLocal()
        try:
            for item in created:
                assert item.status == "draft"
                assert item.duplicate is False
                row = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == item.document_id).one()
                assert row.status == "draft"
                assert row.reviewed_at is None
                assert row.corpus_source in {"database_edu_lectcii", "database_law"}
                chunks = db.query(KnowledgeChunk).filter(KnowledgeChunk.document_id == row.id).count()
                if item.document_id == pdf.document_id:
                    assert item.n_chunks == 0
                    assert chunks == 0
                else:
                    assert item.n_chunks >= 1
                    assert chunks >= 1
                assert row.file_path
                assert_isolated_path(row.file_path)
        finally:
            db.close()
        assert pdf.needs_ocr is True
    finally:
        for item in created:
            _cleanup(item.document_id)


def test_missing_corpus_source_not_guessed():
    try:
        _intake(b"hello source", "a.txt", corpus_source="")
        raise AssertionError("source required")
    except IntakeError as exc:
        assert exc.code == "corpus_source_required"


def test_duplicate_does_not_create_second_row():
    raw = f"dup-{uuid.uuid4().hex}".encode("utf-8")
    first = _intake(raw, "a.txt")
    try:
        second = _intake(raw, "a.txt")
        assert second.duplicate is True
        assert second.document_id == first.document_id
        db = SessionLocal()
        try:
            n = db.query(KnowledgeDocument).filter(KnowledgeDocument.file_hash == first.file_hash).count()
            assert n == 1
        finally:
            db.close()
    finally:
        _cleanup(first.document_id)


def test_parallel_same_bytes_single_document():
    raw = f"parallel-{uuid.uuid4().hex}".encode("utf-8")

    def run():
        return _intake(raw, "p.txt", title="Parallel same")

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            a, b = pool.submit(run).result(), pool.submit(run).result()
        ids = {a.document_id, b.document_id}
        assert len(ids) == 1
        winner = a if not a.duplicate else b
        db = SessionLocal()
        try:
            n = db.query(KnowledgeDocument).filter(KnowledgeDocument.file_hash == winner.file_hash).count()
            assert n == 1
        finally:
            db.close()
    finally:
        _cleanup(a.document_id)


def test_cleanup_removes_isolated_file_only():
    created = _intake(f"clean-{uuid.uuid4().hex}".encode("utf-8"), "gone.txt")
    db = isolation.SessionLocal()
    try:
        row = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == created.document_id).one()
        path = Path(row.file_path)
        if not path.is_absolute():
            from app.config import BASE_DIR

            path = BASE_DIR / row.file_path
        assert path.is_file()
        assert_isolated_path(path)
    finally:
        db.close()
    _cleanup(created.document_id)
    assert not path.exists()
    isolation.assert_prod_corpus_untouched()


def test_upload_route_uses_intake():
    import inspect

    from app.routers import admin, most

    src = inspect.getsource(most.upload_document)
    assert "intake_document" in src
    assert "replace_chunks" not in src
    admin_src = inspect.getsource(admin.admin_commit)
    assert "process_document_pipeline" not in admin_src
    assert "intake_document" in inspect.getsource(admin._intake_from_bytes)


def test_error_does_not_leave_partial_record_or_file():
    from hashlib import sha256
    from unittest.mock import patch

    from sqlalchemy.orm import Session

    raw = f"rollback-{uuid.uuid4().hex}".encode("utf-8")
    digest = sha256(raw).hexdigest()
    edu = corpus_svc.EDU_DIR
    assert_isolated_path(edu)
    files_before = {p.name for p in edu.glob("*") if p.is_file()}
    db = SessionLocal()
    try:
        docs_before = db.query(KnowledgeDocument).filter(KnowledgeDocument.file_hash == digest).count()
        hist_before = db.query(UploadHistory).count()
    finally:
        db.close()

    with patch.object(Session, "commit", side_effect=RuntimeError("forced")):
        try:
            _intake(raw, "rollback.txt")
            raise AssertionError("commit failure must abort intake")
        except IntakeError as exc:
            assert exc.code == "intake_failed"

    db = SessionLocal()
    try:
        docs_after = db.query(KnowledgeDocument).filter(KnowledgeDocument.file_hash == digest).count()
        hist_after = db.query(UploadHistory).count()
        assert docs_after == docs_before
        assert hist_after == hist_before
    finally:
        db.close()
    files_after = {p.name for p in edu.glob("*") if p.is_file()}
    assert files_after == files_before
