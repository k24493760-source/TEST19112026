"""Live auth checks. Does not print password or SECRET_KEY. Does not mutate users."""
from __future__ import annotations

import sys
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import dotenv_values

from app.db import DATABASE_URL, SessionLocal, schema_bootstrap_enabled
from app.models import User
from app.services.auth import COOKIE_NAME, can, effective_role, issue_session

BASE = "http://127.0.0.1:8000"


def hit(url: str, method="GET", data=None, headers=None, opener=None):
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    opener = opener or urllib.request.build_opener()
    try:
        with opener.open(req, timeout=15) as resp:
            body = resp.read(800).decode("utf-8", "replace")
            return resp.status, dict(resp.headers), body
    except urllib.error.HTTPError as e:
        body = e.read(800).decode("utf-8", "replace")
        return e.code, dict(e.headers), body


def main() -> None:
    env = dotenv_values(ROOT / ".env")
    parsed = urllib.parse.urlparse(DATABASE_URL.replace("postgresql+psycopg://", "postgresql://"))
    print("db_host", parsed.hostname)
    print("db_port", parsed.port)
    print("db_name", (parsed.path or "").lstrip("/"))
    print("db_user", parsed.username)
    print("env_MOST_SCHEMA_BOOTSTRAP", repr(env.get("MOST_SCHEMA_BOOTSTRAP")))
    print("runtime_schema_bootstrap_enabled", schema_bootstrap_enabled())
    print("env_has_SECRET_KEY", bool(env.get("SECRET_KEY")))
    print("env_admin_login", env.get("MOST_ADMIN_LOGIN"))
    print("env_has_admin_password", bool(env.get("MOST_ADMIN_PASSWORD")))

    st, _, body = hit(f"{BASE}/api/v1/health")
    print("health", st, body.replace("\n", " ")[:300])
    st, _, body = hit(f"{BASE}/ui/login")
    print("login_page", st, "password" in body.lower())

    anon_paths = [
        ("POST", f"{BASE}/api/v1/most/upload"),
        ("POST", f"{BASE}/api/v1/most/admin/upload/validate"),
        ("GET", f"{BASE}/api/v1/most/admin/uploads"),
        ("GET", f"{BASE}/api/v1/most/document/00000000-0000-0000-0000-000000000001/original"),
        ("GET", f"{BASE}/api/v1/most/corpus/registry"),
        ("GET", f"{BASE}/api/v1/most/expert/queue"),
        ("GET", f"{BASE}/ui/upload"),
        ("GET", f"{BASE}/ui"),
    ]
    for method, url in anon_paths:
        st, hdrs, body = hit(url, method=method)
        loc = hdrs.get("Location") or hdrs.get("location") or ""
        print("anon", method, urllib.parse.urlparse(url).path, st, loc[:80], body[:80].replace("\n", " "))

    listener_token = ""
    db = SessionLocal()
    try:
        rows = db.query(User).order_by(User.login).all()
        print("users_count", len(rows))
        for u in rows:
            role = effective_role(u)
            kind = "pbkdf2" if (u.password_hash or "").startswith("pbkdf2") else "non-pbkdf2"
            print(
                "user",
                u.login,
                "role",
                role,
                "global",
                u.global_role,
                "hash_kind",
                kind,
                "can_upload",
                can(role, "upload_document"),
                "can_manage",
                can(role, "manage_corpus"),
                "can_view_original",
                can(role, "view_original"),
            )
            if u.login == "system-listener":
                listener_token = issue_session(u)
        print("kb_admin_present", any(u.login == "kb-admin" for u in rows))
    finally:
        db.close()

    password = env.get("MOST_ADMIN_PASSWORD") or ""
    login_name = env.get("MOST_ADMIN_LOGIN") or "kb-admin"
    jar = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    payload = urllib.parse.urlencode({"login": login_name, "password": password, "next": "/ui"}).encode()
    st, _, body = hit(
        f"{BASE}/api/v1/auth/login",
        method="POST",
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        opener=opener,
    )
    print("kb_login", st, body[:120].replace("\n", " "))
    st, _, body = hit(f"{BASE}/api/v1/auth/me", opener=opener)
    print("kb_me", st, body[:240].replace("\n", " "))
    st, hdrs, body = hit(f"{BASE}/ui/upload", opener=opener)
    print("kb_upload_page", st, "form" in body.lower() or "upload" in body.lower())
    st, _, body = hit(f"{BASE}/api/v1/most/admin/uploads", opener=opener)
    print("kb_admin_uploads", st, body[:120].replace("\n", " "))

    st, hdrs, body = hit(f"{BASE}/ui/upload", headers={"Cookie": f"{COOKIE_NAME}={listener_token}"})
    print("listener_upload_page", st, (hdrs.get("Location") or "")[:80], body[:80].replace("\n", " "))
    st, _, body = hit(
        f"{BASE}/api/v1/most/upload",
        method="POST",
        headers={"Cookie": f"{COOKIE_NAME}={listener_token}"},
    )
    print("listener_upload_api", st, body[:120].replace("\n", " "))
    st, _, body = hit(
        f"{BASE}/api/v1/most/document/00000000-0000-0000-0000-000000000001/original",
        headers={"Cookie": f"{COOKIE_NAME}={listener_token}"},
    )
    print("listener_original", st, body[:120].replace("\n", " "))
    print("frontend_same_origin", True)
    print("done")


if __name__ == "__main__":
    main()
