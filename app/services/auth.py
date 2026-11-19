"""Сессии и роли. Пароли и токены в логи не пишутся."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from base64 import urlsafe_b64decode, urlsafe_b64encode

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import User
from app.services.approval import can, is_granted_document_approver, load_policy

COOKIE_NAME = "most_session"
SESSION_HOURS = 8
EXPERT_ROLES = {"lawyer", "medical_methodologist", "methodist"}


class AuthError(Exception):
    def __init__(self, code: str, message: str, http_status: int = 401):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


def secret_key() -> bytes:
    raw = (os.getenv("SECRET_KEY") or "").strip()
    if not raw or raw in {"change-me", "changeme"}:
        raise AuthError("secret_unconfigured", "SECRET_KEY не задан.")
    return raw.encode("utf-8")


def effective_role(user: User) -> str:
    global_role = (user.global_role or "").strip()
    role = (user.role or "").strip()
    if global_role:
        return global_role
    if role == "expert":
        return ""
    return role


def issue_session(user: User) -> str:
    payload = {
        "uid": user.id,
        "login": user.login,
        "role": effective_role(user),
        "exp": int(time.time()) + SESSION_HOURS * 3600,
    }
    blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    body = urlsafe_b64encode(blob).rstrip(b"=")
    sig = hmac.new(secret_key(), body, hashlib.sha256).digest()
    return body.decode("ascii") + "." + urlsafe_b64encode(sig).rstrip(b"=").decode("ascii")


def parse_session(token: str | None) -> dict:
    if not token or "." not in token:
        raise AuthError("unauthenticated", "Нужна авторизация.")
    body, sig = token.split(".", 1)
    expected = hmac.new(secret_key(), body.encode("ascii"), hashlib.sha256).digest()
    try:
        got = urlsafe_b64decode(sig + "=" * (-len(sig) % 4))
    except Exception as exc:
        raise AuthError("unauthenticated", "Сессия недействительна.") from exc
    if not hmac.compare_digest(expected, got):
        raise AuthError("unauthenticated", "Сессия недействительна.")
    try:
        payload = json.loads(urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    except Exception as exc:
        raise AuthError("unauthenticated", "Сессия недействительна.") from exc
    if int(payload.get("exp") or 0) < int(time.time()):
        raise AuthError("unauthenticated", "Сессия истекла.")
    return payload


def load_user(db: Session, user_id: str) -> User:
    user = db.get(User, user_id)
    if user is None:
        raise AuthError("unauthenticated", "Пользователь не найден.")
    return user


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    try:
        payload = parse_session(request.cookies.get(COOKIE_NAME))
        return load_user(db, str(payload.get("uid") or ""))
    except AuthError as exc:
        raise HTTPException(status_code=exc.http_status, detail=exc.code) from exc


def require_roles(*roles: str):
    allowed = set(roles)

    def _inner(user: User = Depends(current_user)) -> User:
        role = effective_role(user)
        if allowed and role not in allowed:
            raise HTTPException(status_code=403, detail="forbidden")
        return user

    return _inner


def require_permission(action: str):
    def _inner(user: User = Depends(current_user)) -> User:
        if not can(effective_role(user), action):
            raise HTTPException(status_code=403, detail="forbidden")
        return user

    return _inner


def require_document_approval(user: User = Depends(current_user)) -> User:
    if can(effective_role(user), "manage_corpus") or is_granted_document_approver(user.login):
        return user
    raise HTTPException(status_code=403, detail="forbidden")


def public_paths() -> tuple[str, ...]:
    return (
        "/api/v1/health",
        "/ui/login",
        "/api/v1/auth/login",
        "/api/v1/auth/register",
        "/api/v1/auth/logout",
        "/static/",
        "/favicon.ico",
        "/docs",
        "/openapi.json",
        "/redoc",
    )


def is_public(path: str) -> bool:
    if path in {"/", "/ui"}:
        return False
    for prefix in public_paths():
        if prefix.endswith("/") and path.startswith(prefix):
            return True
        if path == prefix:
            return True
    return False


def session_user_payload(user: User) -> dict:
    role = effective_role(user)
    policy = load_policy()
    return {
        "id": user.id,
        "login": user.login,
        "name": user.name,
        "role": role,
        "can_upload": can(role, "upload_document"),
        "can_expert": role in EXPERT_ROLES or can(role, "view_queue"),
        "expert_roles": [role] if role in EXPERT_ROLES else [],
        "policy_roles": list((policy.get("expert_roles") or {}).keys()),
    }
