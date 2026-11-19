from app.services.rag import NoEvidenceResponse


def test_no_evidence_payload_shape():
    payload = NoEvidenceResponse(
        reviewed_sources=[{"id": "doc-1", "title": "Учебный материал"}]
    ).model_dump()
    assert payload["status"] == "no_evidence"
    assert payload["answer_status"] == "basis_not_found"
    assert payload["review_required"] is True
    assert payload["review_reason"] == "no_basis"
    assert payload["reviewed_sources"][0]["title"] == "Учебный материал"
    assert "reviewed_sources" in payload
