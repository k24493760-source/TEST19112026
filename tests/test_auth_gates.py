"""Шлюз авторизации без TestClient/httpx. most_db не используется."""
from __future__ import annotations

import os

os.environ["SECRET_KEY"] = "unit-test-secret-key-not-for-prod"
os.environ["MOST_SCHEMA_BOOTSTRAP"] = "0"

from tests import isolation

isolation.activate()

from app.models import User, generate_uuid
from app.services.approval import can
from app.services.auth import AuthError, effective_role, issue_session, parse_session
from app.services.passwords import hash_password, verify_password
from tests.isolation import assert_test_database

assert_test_database()


def test_password_and_session():
    digest = hash_password("unit-test-pass-ok")
    assert verify_password("unit-test-pass-ok", digest)
    assert not verify_password("unit-test-pass-ok", "111")
    user = User(
        id=generate_uuid(),
        email="a@local",
        login="admin1",
        password_hash=digest,
        role="admin",
        global_role="admin",
    )
    token = issue_session(user)
    payload = parse_session(token)
    assert payload["login"] == "admin1"
    try:
        parse_session(None)
        raise AssertionError("anon")
    except AuthError as exc:
        assert exc.http_status == 401


def test_roles_and_permissions():
    admin = User(id="1", email="a", login="a", password_hash="x", role="admin", global_role="admin")
    listener = User(id="2", email="b", login="b", password_hash="x", role="listener", global_role="listener")
    lawyer = User(id="3", email="c", login="c", password_hash="x", role="expert", global_role="lawyer")
    assert can(effective_role(admin), "upload_document")
    assert not can(effective_role(listener), "upload_document")
    assert can(effective_role(lawyer), "expert_approve")
    assert not can(effective_role(admin), "expert_approve")
    assert can(effective_role(admin), "view_original")


if __name__ == "__main__":
    test_password_and_session()
    print("OK test_password_and_session")
    test_roles_and_permissions()
    print("OK test_roles_and_permissions")
    print("2/2 PASS")
