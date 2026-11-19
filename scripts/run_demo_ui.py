"""Запуск того же приложения на most_db_demo. Строку подключения не печатает."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
values = dotenv_values(ROOT / ".env")
raw = values.get("DATABASE_URL") or ""
parsed = urlparse(raw)
if parsed.path.rsplit("/", 1)[-1] != "most_db":
    raise SystemExit("Ожидался DATABASE_URL с базой most_db, демо-сервер не запущен")
demo_url = urlunparse(parsed._replace(path="/most_db_demo"))
env = os.environ.copy()
for key, value in values.items():
    if key and value is not None:
        env[key] = value
env["DATABASE_URL"] = demo_url
env["MOST_EMBEDDING_OFFLINE"] = "0"
env["MOST_ALLOW_HF_DOWNLOAD"] = "1"
env["HF_HUB_OFFLINE"] = "0"
env["TRANSFORMERS_OFFLINE"] = "0"
env["MOST_SCHEMA_BOOTSTRAP"] = "0"
env["MOST_QA_LLM_ENABLED"] = "0"
os.execve(
    sys.executable,
    [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8001"],
    env,
)
