"""Создание или смена пароля администратора. Пароль только из окружения."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from app.db import SessionLocal
from app.models import User, generate_uuid
from app.services.passwords import hash_password


def main() -> int:
    login = (os.getenv("MOST_ADMIN_LOGIN") or "admin").strip()
    password = os.getenv("MOST_ADMIN_PASSWORD") or ""
    role = (os.getenv("MOST_ADMIN_ROLE") or "admin").strip() or "admin"
    if not password:
        print("Задайте MOST_ADMIN_PASSWORD в окружении.", file=sys.stderr)
        return 2
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.login == login).one_or_none()
        digest = hash_password(password)
        if user is None:
            db.add(
                User(
                    id=generate_uuid(),
                    email=f"{login}@local",
                    login=login,
                    password_hash=digest,
                    role=role,
                    global_role=role,
                    name="Администратор базы знаний",
                )
            )
            db.commit()
            print("admin_ready created=1 reset=0")
            return 0
        reset = (os.getenv("MOST_ADMIN_RESET") or "").strip() == "1"
        if not reset:
            print("admin_exists reset_required=1", file=sys.stderr)
            return 3
        user.password_hash = digest
        if not (user.role or "").strip():
            user.role = role
        if not (user.global_role or "").strip():
            user.global_role = role
        db.commit()
        print("admin_ready created=0 reset=1")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
