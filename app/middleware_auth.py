"""Шлюз авторизации. Не логирует cookie и пароли."""
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse

from app.db import SessionLocal
from app.services.auth import AuthError, COOKIE_NAME, is_public, load_user, parse_session, secret_key


class AuthGateMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if is_public(path) or request.method == "OPTIONS":
            return await call_next(request)
        try:
            secret_key()
            payload = parse_session(request.cookies.get(COOKIE_NAME))
            db = SessionLocal()
            try:
                user = load_user(db, str(payload.get("uid") or ""))
            finally:
                db.close()
            request.state.user_id = user.id
            request.state.user_role = payload.get("role")
        except AuthError:
            if path.startswith("/api/"):
                return JSONResponse({"detail": "unauthenticated"}, status_code=401)
            nxt = path if path.startswith("/") else "/ui"
            return RedirectResponse(f"/ui/login?next={nxt}", status_code=303)
        return await call_next(request)
