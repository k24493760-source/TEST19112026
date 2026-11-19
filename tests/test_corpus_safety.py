"""Безопасность корпуса: чужие чанки, approved, дубли, unknown source, ID/version."""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.routers import most

from app.services.corpus import (
    CORPUS_DIR,
    EDU_DIR,
    LAW_DIR,
    _process_file,
    audit_documents,
    corpus_source_from_path,
    find_live_duplicate,
)


def test_corpus_source_unknown_not_guessed_from_filename():
    path = CORPUS_DIR / "Текст_5_модуль_Медработник220725.docx"
    assert corpus_source_from_path(path) == "unknown"
    assert corpus_source_from_path(CORPUS_DIR / "Нормативно_правовая_база.docx") == "unknown"


def test_corpus_source_from_canonical_directory_only():
    assert corpus_source_from_path(LAW_DIR / "npa.docx") == "database_law"
    assert corpus_source_from_path(EDU_DIR / "lecture.docx") == "database_edu_lectcii"
    assert corpus_source_from_path(CORPUS_DIR / "quarantine" / "wait.pdf") == "quarantine"


def test_scan_does_not_mutate_existing_or_approved():
    approved = SimpleNamespace(
        id="keep-id",
        title="Модуль 6",
        document_class="Учебный материал",
        owner="загрузка через UI",
        version="1.0",
        status="approved",
        hash="abc",
        file_path="data/corpus/mod6.docx",
        file_format="docx",
    )
    db = MagicMock()
    with (
        patch("app.services.corpus.ensure_corpus_layout"),
        patch("app.services.corpus.write_file_registry") as write_reg,
        patch("app.services.corpus.document_audit_rows", return_value=[{"id": approved.id}]),
        patch("app.services.corpus.replace_chunks") as replace,
        patch("app.services.corpus._process_file") as process,
    ):
        rows = audit_documents(db)
    write_reg.assert_called_once_with(db)
    process.assert_not_called()
    replace.assert_not_called()
    assert rows[0]["id"] == "keep-id"


def test_process_file_keeps_id_version_and_skips_chunks():
    existing = SimpleNamespace(
        id="doc-stable",
        version="1.0",
        status="draft",
        document_class="Кейс",
        owner="owner-a",
        title="Same",
        hash="hh",
        file_path="data/corpus/a.txt",
        file_format="txt",
    )
    db = MagicMock()
    path = MagicMock()
    path.read_bytes.return_value = b"hello"
    path.suffix = ".txt"
    with (
        patch("app.services.corpus.sha256_bytes", return_value="hh"),
        patch("app.services.corpus.relative_corpus_path", return_value="data/corpus/a.txt"),
        patch("app.services.corpus.find_document_for_file", return_value=existing),
        patch("app.services.corpus.replace_chunks") as replace,
    ):
        result = _process_file(db, path, None)
    assert result is existing
    assert result.id == "doc-stable"
    assert result.version == "1.0"
    assert result.document_class == "Кейс"
    assert result.owner == "owner-a"
    assert result.status == "draft"
    replace.assert_not_called()


def test_upload_route_does_not_call_mass_audit():
    source = inspect.getsource(most.upload_document)
    assert "audit_documents(" not in source
    assert "document_audit_rows" in source
    assert "intake_document" in source


def test_reupload_helper_does_not_touch_foreign_chunks():
    other = SimpleNamespace(id="other-doc")
    db = MagicMock()
    with patch("app.services.corpus.replace_chunks") as replace:
        found = find_live_duplicate(db, "same-hash")
        replace.assert_not_called()
    db.query.assert_called()
    assert found is db.query.return_value.filter.return_value.filter.return_value.first.return_value
    assert other.id == "other-doc"


def test_duplicate_lookup_filters_hash_and_not_archived():
    db = MagicMock()
    sentinel = object()
    db.query.return_value.filter.return_value.filter.return_value.first.return_value = sentinel
    assert find_live_duplicate(db, "abc") is sentinel
