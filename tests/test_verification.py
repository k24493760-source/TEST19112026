from datetime import datetime, timedelta
from types import SimpleNamespace

from app.services.rag import AskResponse, Citation, LIMITATION_LEGAL, LIMITATION_MEDICAL
from app.services.verification import (
    LIMITATION_UNVERIFIED,
    VERIFICATION_HIDDEN,
    apply_citation_verification,
    quote_contained,
    verify_citations,
)


def test_quote_contained_uses_in():
    assert quote_contained("Статья 1", "Текст. Статья 1. Дальше.")
    assert quote_contained("  Статья 1  ", "Статья 1")
    assert not quote_contained("выдумка", "Статья 1. Норма.")
    assert not quote_contained("", "Статья 1")


class _Query:
    def __init__(self, row):
        self._row = row

    def join(self, *args, **kwargs):
        return self

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._row


class FakeDb:
    def __init__(self, row):
        self.row = row

    def query(self, *args, **kwargs):
        return _Query(self.row)


def _response(quote="Статья 1. Текст нормы.", chunk_id="chunk-1"):
    return AskResponse(
        answer="ok",
        claims=[],
        citations=[
            Citation(
                id="cit1",
                chunk_id=chunk_id,
                section="1",
                page="1",
                quote=quote,
            )
        ],
        limitations=[LIMITATION_LEGAL, LIMITATION_MEDICAL],
        review_required=False,
    )


def _live_row(text="Статья 1. Текст нормы.", status="approved", valid_to=None):
    chunk = SimpleNamespace(
        id="chunk-1",
        text=text,
        approval_status="approved",
        document_id="doc-1",
    )
    doc = SimpleNamespace(
        id="doc-1",
        status=status,
        valid_from=None,
        valid_to=valid_to,
    )
    return chunk, doc


def test_verify_ok_when_quote_in_approved_chunk():
    report = verify_citations(_response(), FakeDb(_live_row()))
    assert report["is_verified"] is True
    assert report["checked"][0]["ok"] is True


def test_verify_fails_missing_chunk():
    report = verify_citations(_response(), FakeDb(None))
    assert report["is_verified"] is False
    assert report["checked"][0]["reason"] == "chunk_not_found"


def test_verify_fails_draft_document():
    report = verify_citations(_response(), FakeDb(_live_row(status="draft")))
    assert report["is_verified"] is False
    assert report["checked"][0]["reason"] == "document_not_live"


def test_verify_fails_expired_valid_to():
    past = datetime.utcnow() - timedelta(days=1)
    report = verify_citations(_response(), FakeDb(_live_row(valid_to=past)))
    assert report["is_verified"] is False


def test_verify_fails_quote_mismatch():
    report = verify_citations(
        _response(quote="этого текста нет в чанке"),
        FakeDb(_live_row()),
    )
    assert report["is_verified"] is False
    assert report["checked"][0]["reason"] == "quote_mismatch"


def test_apply_hides_answer_on_failure():
    hidden = apply_citation_verification(_response(), FakeDb(None))
    assert hidden.answer == VERIFICATION_HIDDEN
    assert hidden.claims == []
    assert hidden.citations == []
    assert LIMITATION_UNVERIFIED in hidden.limitations
    assert hidden.review_required is True
