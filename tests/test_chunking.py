"""Специализированный чанкинг: шесть классов, draft, без embeddings."""

from __future__ import annotations

import uuid

from app.models import CorpusRelease, KnowledgeChunk, KnowledgeDocument
from app.services.chunking import chunk_parsed_document, count_tokens, load_chunking_config
from app.services.intake import intake_document
from tests import isolation
from tests.isolation import assert_isolated_path, assert_prod_corpus_untouched, assert_test_database
from tests.test_intake import _cleanup

isolation.activate()
assert_test_database()

LEGAL = (
    "Статья 12.1 не применяется до 01.01.2020. "
    "Сумма 1000 руб. не подлежит возврату, за исключением случая, указанного в пункте 2."
)


def _intake_md(body: str, cls: str, name: str = "doc.md"):
    db = isolation.SessionLocal()
    try:
        return intake_document(
            db,
            body.encode("utf-8"),
            name,
            title=f"Chunk {uuid.uuid4().hex[:8]}",
            document_class=cls,
            corpus_source="database_edu_lectcii",
            owner="qa-chunk",
            actor_id="qa-chunk",
            content_type="text/markdown",
        )
    finally:
        db.close()


def _chunks(doc_id: str) -> list[KnowledgeChunk]:
    db = isolation.SessionLocal()
    try:
        return (
            db.query(KnowledgeChunk)
            .filter(KnowledgeChunk.document_id == doc_id)
            .order_by(KnowledgeChunk.id)
            .all()
        )
    finally:
        db.close()


def _run(doc_id: str):
    db = isolation.SessionLocal()
    try:
        return chunk_parsed_document(db, doc_id)
    finally:
        db.close()


def _assert_draft_only(doc_id: str) -> None:
    rows = _chunks(doc_id)
    assert rows
    ids = [r.id for r in rows]
    assert len(ids) == len(set(ids))
    db = isolation.SessionLocal()
    try:
        doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == doc_id).one()
        assert doc.status == "draft"
        for row in rows:
            assert row.approval_status == "draft"
            assert row.embedding is None
            assert (row.embedding_status or "pending") == "pending"
            assert (row.evidence_text or "").strip()
            assert row.source_element_ids
            assert row.chunking_version == load_chunking_config()["chunking_version"]
            assert row.corpus_version == "working-draft"
            assert row.document_version == doc.version
            assert_isolated_path(doc.file_path)
    finally:
        db.close()


def test_six_classes_produce_draft_chunks():
    samples = {
        "Нормативный акт": f"# Статья 1\n\n{LEGAL}\n\n1. Первый пункт не отменяется.\n2. Второй пункт.\n",
        "Официальное разъяснение": f"# Раздел разъяснения\n\n{LEGAL}\n\nПояснение официального органа.\n",
        "Юридический комментарий": f"# Комментарий к статье 12.1\n\nКомментарий: {LEGAL}\n",
        "Учебный материал": f"# Лекция\n\n{LEGAL}\n\nУчебное пояснение темы.\n",
        "Кейс": (
            "# Контекст\n\nРаботник обратился за разъяснением.\n\n"
            f"# Действия\n\nНужно учесть, что {LEGAL}\n"
        ),
        "Методика": f"# Шаг 1\n\nВыполнить проверку.\n\n# Шаг 2\n\n{LEGAL}\n",
    }
    created = []
    try:
        for cls, body in samples.items():
            item = _intake_md(body, cls)
            created.append(item.document_id)
            result = _run(item.document_id)
            assert result.chunks
            _assert_draft_only(item.document_id)
            joined = "\n".join(c.evidence_text or "" for c in _chunks(item.document_id))
            assert "не применяется" in joined
            assert "12.1" in joined or "Статья 1" in joined or "не отменяется" in joined or "проверку" in joined
        db = isolation.SessionLocal()
        try:
            rel = db.query(CorpusRelease).filter(CorpusRelease.version == "working-draft").one()
            assert rel.status == "draft"
            assert rel.published_at is None
        finally:
            db.close()
        assert_prod_corpus_untouched()
    finally:
        for doc_id in created:
            _cleanup(doc_id)


def test_table_and_list_kept_with_condition():
    body = (
        "# Статья 2\n\n"
        "Норма не действует, если отсутствует согласие.\n\n"
        "1. Первый пункт списка не применяется отдельно.\n"
        "2. Второй пункт.\n\n"
        "| Колонка | Значение |\n| --- | --- |\n| Дата | 01.01.2020 |\n| Статья | 12.1 |\n"
    )
    item = _intake_md(body, "Нормативный акт")
    try:
        _run(item.document_id)
        texts = "\n".join((c.evidence_text or "") for c in _chunks(item.document_id))
        assert "01.01.2020" in texts
        assert "12.1" in texts
        assert "Первый пункт списка" in texts
        assert "если отсутствует согласие" in texts
        _assert_draft_only(item.document_id)
    finally:
        _cleanup(item.document_id)


def test_long_article_keeps_exception_and_may_parent():
    sentences = [
        "Статья 9 не применяется к отношениям, возникшим до вступления нормы в силу."
    ]
    for idx in range(40):
        sentences.append(
            f"Положение номер {idx} описывает порядок действий уполномоченного лица."
        )
    sentences.append(
        "Возврат суммы не производится, за исключением случая, указанного в пункте 2."
    )
    body = "# Статья 9\n\n" + " ".join(sentences)
    item = _intake_md(body, "Нормативный акт")
    try:
        result = _run(item.document_id)
        rows = _chunks(item.document_id)
        joined = " ".join(r.evidence_text or "" for r in rows)
        assert "не применяется" in joined
        assert "за исключением случая" in joined
        assert "пункте 2" in joined or "пункт 2" in joined
        for row in rows:
            if row.applicability != "oversized_unsplit":
                assert len(row.evidence_text or "") <= 1500
        assert count_tokens(joined) >= 150
        assert all((row.evidence_text or "").strip() for row in rows)
        assert result.chunking_version
        _assert_draft_only(item.document_id)
    finally:
        _cleanup(item.document_id)


def test_rechunk_no_duplicates_and_keeps_approved():
    item = _intake_md(f"# Статья 3\n\n{LEGAL}\n", "Нормативный акт")
    try:
        _run(item.document_id)
        first = _chunks(item.document_id)
        first_ids = sorted(r.id for r in first)
        db = isolation.SessionLocal()
        try:
            db.add(
                KnowledgeChunk(
                    id="keep-approved-chunk",
                    document_id=item.document_id,
                    text="утверждённый фрагмент",
                    approval_status="approved",
                    embedding_status="pending",
                    chunking_version="legacy",
                )
            )
            db.commit()
        finally:
            db.close()
        _run(item.document_id)
        second = _chunks(item.document_id)
        draft_ids = sorted(r.id for r in second if r.approval_status == "draft")
        assert draft_ids == first_ids
        assert len(draft_ids) == len(set(draft_ids))
        assert any(r.id == "keep-approved-chunk" for r in second)
        assert all(r.embedding is None for r in second)
    finally:
        _cleanup(item.document_id)


def test_no_text_loss_and_elements_match_parse_run():
    body = f"# Глава\n\nПервый абзац.\n\n{LEGAL}\n\nПоследний абзац.\n"
    item = _intake_md(body, "Учебный материал")
    try:
        _run(item.document_id)
        db = isolation.SessionLocal()
        try:
            from app.models import DocumentElement, DocumentParseRun

            run = (
                db.query(DocumentParseRun)
                .filter(
                    DocumentParseRun.document_id == item.document_id,
                    DocumentParseRun.is_current.is_(True),
                )
                .one()
            )
            el_ids = {
                e.id
                for e in db.query(DocumentElement).filter(
                    DocumentElement.parse_run_id == run.id
                )
            }
            chunks = (
                db.query(KnowledgeChunk)
                .filter(KnowledgeChunk.document_id == item.document_id)
                .all()
            )
            used = []
            for chunk in chunks:
                for eid in chunk.source_element_ids or []:
                    assert eid in el_ids
                    used.append(eid)
            evidence = "\n".join(c.evidence_text or "" for c in chunks)
            assert "Первый абзац" in evidence
            assert "Последний абзац" in evidence
            assert "не применяется" in evidence
        finally:
            db.close()
        assert_prod_corpus_untouched()
    finally:
        _cleanup(item.document_id)


def test_class_strategies_keep_structure():
    created = []
    try:
        law = _intake_md(
            "Настоящий закон регулирует отношения.\n\n"
            "# Статья 1\n\n1. Первый пункт.\n\n"
            "# Статья 2\n\n2. Второй пункт не смешивается.\n",
            "Нормативный акт",
        )
        created.append(law.document_id)
        assert law.n_chunks >= 2
        rows = _chunks(law.document_id)
        sections = [row.section or "" for row in rows]
        assert any("Преамбула" in item or "закон" in (row.evidence_text or "") for item, row in zip(sections, rows))
        article_chunks = [row for row in rows if "Статья" in (row.section or "") or "Статья" in (row.evidence_text or "")]
        assert article_chunks
        for row in rows:
            text = row.evidence_text or ""
            if "Первый пункт" in text:
                assert "Второй пункт" not in text
            if "Второй пункт" in text:
                assert "Первый пункт" not in text

        qa = _intake_md(
            "Вопрос: Когда направлять уведомление?\n\nОтвет: В течение четырнадцати дней.\n\n"
            "Вопрос: Кто подписывает?\n\nОтвет: Уполномоченное лицо.\n",
            "Официальное разъяснение",
        )
        created.append(qa.document_id)
        qa_rows = _chunks(qa.document_id)
        assert len(qa_rows) >= 2
        for row in qa_rows:
            text = row.evidence_text or ""
            if "уведомление" in text:
                assert "четырнадцати" in text
                assert "подписывает" not in text
    finally:
        for doc_id in created:
            _cleanup(doc_id)


def test_tokenizer_is_real():
    n = count_tokens("Статья 12.1 не применяется.")
    assert n >= 4
    assert n < 40
    assert_test_database()
    assert_prod_corpus_untouched()
