"""Хеширование паролей. Секреты в код не записываются."""
from __future__ import annotations

import hashlib
import hmac
import os

SCHEME = "pbkdf2_sha256"
ITERATIONS = 200_000


def hash_password(password: str) -> str:
    if not password or len(password) < 10:
        raise ValueError("Пароль слишком короткий.")
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    return f"{SCHEME}${ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str | None) -> bool:
    if not password or not stored:
        return False
    parts = stored.split("$")
    if len(parts) != 4 or parts[0] != SCHEME:
        return False
    try:
        iterations = int(parts[1])
        salt = bytes.fromhex(parts[2])
        expected = bytes.fromhex(parts[3])
    except ValueError:
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(dk, expected)
