"""Дописывает недостающие ключи в .env. Значения в stdout не печатает."""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from dotenv import dotenv_values

from app.config import BASE_DIR

ENV_PATH = BASE_DIR / ".env"
ADMIN_LOGIN = "kb-admin"


def _upsert(text: str, key: str, value: str) -> str:
    lines = text.splitlines()
    prefix = key + "="
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            if line[len(prefix) :].strip():
                return text
            lines[i] = prefix + value
            return "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    sep = "" if not text or text.endswith("\n") else "\n"
    return text + sep + prefix + value + "\n"


def main() -> int:
    current = ENV_PATH.read_text(encoding="utf-8") if ENV_PATH.is_file() else ""
    parsed = dotenv_values(ENV_PATH) if ENV_PATH.is_file() else {}
    updated = current
    if not (parsed.get("SECRET_KEY") or "").strip() or parsed.get("SECRET_KEY") in {"change-me", "changeme"}:
        updated = _upsert(updated, "SECRET_KEY", secrets.token_urlsafe(48))
    updated = _upsert(updated, "MOST_SCHEMA_BOOTSTRAP", "0")
    updated = _upsert(updated, "MOST_EMBEDDING_OFFLINE", "1")
    updated = _upsert(updated, "MOST_QA_LLM_ENABLED", "0")
    updated = _upsert(updated, "MOST_ADMIN_LOGIN", ADMIN_LOGIN)
    if not (parsed.get("MOST_ADMIN_PASSWORD") or "").strip():
        updated = _upsert(updated, "MOST_ADMIN_PASSWORD", secrets.token_urlsafe(24))
    if updated != current:
        ENV_PATH.write_text(updated, encoding="utf-8")
        print("env_updated=1")
    else:
        print("env_updated=0")
    print("admin_login", ADMIN_LOGIN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
