"""Резервное копирование most_db. Пароли в вывод не попадают."""
from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.engine import make_url

from app.config import BASE_DIR

load_dotenv(BASE_DIR / ".env")
PROD = "most_db"
RESTORE_DB = "most_db_restore_check"


def _url():
    raw = (os.getenv("DATABASE_URL") or "").replace("postgresql+psycopg://", "postgresql://")
    return make_url(raw)


def _bin(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for root in (
        Path(r"C:\Program Files\PostgreSQL"),
        Path(r"C:\Program Files (x86)\PostgreSQL"),
    ):
        if not root.is_dir():
            continue
        matches = list(root.glob(f"**/bin/{name}.exe"))
        if matches:
            return str(sorted(matches)[-1])
    return None


def _env(url) -> dict:
    env = os.environ.copy()
    if url.password:
        env["PGPASSWORD"] = url.password
    return env


def dump_most_db(dest: Path | None = None) -> Path:
    url = _url()
    if url.database != PROD:
        raise SystemExit("Отказ: резервная копия только для most_db из DATABASE_URL.")
    pg_dump = _bin("pg_dump")
    if not pg_dump:
        raise SystemExit("pg_dump не найден. Установите клиент PostgreSQL или выполните дамп из контейнера.")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")
    dest = dest or (BASE_DIR / "backups" / f"{stamp}_most_db.sql")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        pg_dump,
        "-h",
        url.host or "127.0.0.1",
        "-p",
        str(url.port or 5432),
        "-U",
        url.username or "user",
        "-d",
        PROD,
        "--no-owner",
        "--no-acl",
        "-f",
        str(dest),
    ]
    proc = subprocess.run(cmd, env=_env(url), capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit("pg_dump завершился с ошибкой (детали не печатаются).")
    if dest.stat().st_size < 1024:
        raise SystemExit("Дамп подозрительно мал.")
    return dest


def verify_toc(dump: Path) -> int:
    text = dump.read_text(encoding="utf-8", errors="replace")
    if "CREATE TABLE" not in text and "CREATE TABLE" not in text.upper():
        if "CREATE TABLE" not in text:
            raise SystemExit("Дамп не содержит CREATE TABLE.")
    return text.count("CREATE TABLE")


def restore_to_temp(dump: Path) -> str:
    url = _url()
    psql = _bin("psql")
    createdb = _bin("createdb")
    dropdb = _bin("dropdb")
    if not all([psql, createdb, dropdb]):
        raise SystemExit("Нет createdb/dropdb/psql.")
    env = _env(url)
    common = ["-h", url.host or "127.0.0.1", "-p", str(url.port or 5432), "-U", url.username or "user"]
    subprocess.run([dropdb, *common, "--if-exists", RESTORE_DB], env=env, check=False, capture_output=True)
    created = subprocess.run([createdb, *common, RESTORE_DB], env=env, capture_output=True, text=True)
    if created.returncode != 0:
        raise SystemExit("Не удалось создать временную БД восстановления.")
    sql = dump.read_text(encoding="utf-8", errors="replace")
    sql = "\n".join(ln for ln in sql.splitlines() if "transaction_timeout" not in ln)
    filtered = dump.with_suffix(".pg16.sql")
    filtered.write_text(sql, encoding="utf-8")
    restored = subprocess.run(
        [psql, *common, "-d", RESTORE_DB, "-v", "ON_ERROR_STOP=1", "-f", str(filtered)],
        env=env,
        capture_output=True,
        text=True,
    )
    if restored.returncode != 0:
        err = restored.stderr or restored.stdout or ""
        hint = next((ln.strip() for ln in err.splitlines() if ln.strip()), "unknown")
        if "password" in hint.lower():
            hint = "ошибка подключения (секреты скрыты)"
        raise SystemExit(f"Пробное восстановление не удалось: {hint[:180]}")
    return RESTORE_DB


def rollback_plan() -> str:
    return (
        "Откат: 1) остановить uvicorn; "
        "2) dropdb most_db только после подтверждения; "
        "3) createdb most_db; "
        "4) psql -d most_db -f backups/<stamp>_most_db.sql (убрать SET transaction_timeout, если клиент новее сервера); "
        "5) не применять ALLOW_ALEMBIC_MOST_DB. "
        "Временная БД most_db_restore_check — только проверка, не рабочий контур."
    )


if __name__ == "__main__":
    path = dump_most_db()
    n = verify_toc(path)
    dbname = restore_to_temp(path)
    print(f"dump_ok bytes={path.stat().st_size} toc_entries={n} restore_db={dbname}")
    print(rollback_plan())
