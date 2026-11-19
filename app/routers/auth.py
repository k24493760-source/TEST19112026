"""Вход, регистрация и текущий пользователь."""
from __future__ import annotations

import logging
import re

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.config import BASE_DIR
from app.db import get_db
from app.models import User, generate_uuid
from app.services.auth import (
    COOKIE_NAME,
    AuthError,
    current_user,
    issue_session,
    secret_key,
    session_user_payload,
)
from app.services.passwords import hash_password, verify_password

router = APIRouter(tags=["auth"])
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))
logger = logging.getLogger("most.auth")
_LOGIN_RE = re.compile(r"^[\w.-]{3,64}$", re.UNICODE)
_DB_DOWN = "База данных сейчас недоступна. Подождите минуту и повторите попытку."
_REGISTER_ROLE = "curator"


def _safe_next(value: str | None) -> str:
    raw = (value or "/ui").strip() or "/ui"
    if not raw.startswith("/") or raw.startswith("//") or ":" in raw:
        return "/ui"
    return raw


@router.get("/ui/login", response_class=HTMLResponse)
def login_page(request: Request, next: str = "/ui"):
    return _login_page(request, next, None)


@router.post("/api/v1/auth/login")
def login(
    request: Request,
    db: Session = Depends(get_db),
    login: str = Form(...),
    password: str = Form(...),
    next: str = Form("/ui"),
):
    try:
        secret_key()
    except AuthError as exc:
        raise HTTPException(status_code=500, detail=exc.code) from exc
    try:
        user = db.query(User).filter(User.login == login.strip()).one_or_none()
    except OperationalError:
        logger.exception("login_db_unavailable")
        db.rollback()
        return _login_page(request, next, _DB_DOWN, 503)
    if user is None or not verify_password(password, user.password_hash):
        logger.info("login_failed")
        accept = request.headers.get("accept") or ""
        if "text/html" in accept:
            return _login_page(request, next, "Неверный логин или пароль.", 401)
        raise HTTPException(status_code=401, detail="invalid_credentials")
    token = issue_session(user)
    logger.info("login_ok role_set=1")
    dest = _safe_next(next)
    if "application/json" in (request.headers.get("accept") or ""):
        response = JSONResponse({"ok": True, "role": session_user_payload(user)["role"]})
    else:
        response = RedirectResponse(dest, status_code=303)
    response.set_cookie(
        COOKIE_NAME,
        token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=8 * 3600,
        path="/",
    )
    return response


def _login_page(request: Request, next_url: str, error: str | None, status_code: int = 200):
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "next_url": _safe_next(next_url),
            "error": error,
            "register_error": None,
        },
        status_code=status_code,
    )


@router.post("/api/v1/auth/register")
def register(
    request: Request,
    db: Session = Depends(get_db),
    login: str = Form(...),
    password: str = Form(...),
    password_confirm: str = Form(...),
    next: str = Form("/ui/upload"),
):
    next_url = _safe_next(next)
    login_norm = (login or "").strip()
    if not _LOGIN_RE.fullmatch(login_norm):
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "next_url": next_url,
                "error": None,
                "register_error": "Логин: 3–64 символа, буквы, цифры, точка, _ или -.",
            },
            status_code=400,
        )
    if password != password_confirm:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "next_url": next_url,
                "error": None,
                "register_error": "Пароль и подтверждение не совпадают.",
            },
            status_code=400,
        )
    try:
        digest = hash_password(password)
        secret_key()
    except ValueError:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "next_url": next_url,
                "error": None,
                "register_error": "Пароль слишком короткий: минимум 10 символов.",
            },
            status_code=400,
        )
    except AuthError as exc:
        raise HTTPException(status_code=500, detail=exc.code) from exc
    try:
        existing = db.query(User).filter(User.login == login_norm).one_or_none()
    except OperationalError:
        logger.exception("register_db_unavailable")
        db.rollback()
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "next_url": next_url,
                "error": None,
                "register_error": _DB_DOWN,
            },
            status_code=503,
        )
    if existing is not None:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "next_url": next_url,
                "error": None,
                "register_error": "Такой логин уже занят.",
            },
            status_code=409,
        )
    user = User(
        id=generate_uuid(),
        email=f"{login_norm}@local",
        login=login_norm,
        password_hash=digest,
        role=_REGISTER_ROLE,
        global_role=_REGISTER_ROLE,
        name=login_norm,
    )
    db.add(user)
    try:
        db.commit()
    except OperationalError:
        logger.exception("register_db_unavailable")
        db.rollback()
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "next_url": next_url,
                "error": None,
                "register_error": _DB_DOWN,
            },
            status_code=503,
        )
    db.refresh(user)
    logger.info("register_ok role_set=1")
    token = issue_session(user)
    response = RedirectResponse(next_url, status_code=303)
    response.set_cookie(
        COOKIE_NAME,
        token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=8 * 3600,
        path="/",
    )
    return response


@router.post("/api/v1/auth/logout")
def logout():
    response = RedirectResponse("/ui/login", status_code=303)
    response.delete_cookie(COOKIE_NAME, path="/")
    return response


@router.get("/api/v1/auth/me")
def me(user: User = Depends(current_user)):
    return session_user_payload(user)
