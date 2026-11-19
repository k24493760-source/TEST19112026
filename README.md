# УКЗ-15 «Эксперт МОСТ»

RAG + детерминированная оценка. Ответы только из утверждённых источников.

## Окружение

Python 3.11+ (проверено также 3.14). PostgreSQL 15+ с расширением `vector`.

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

В `.env` задайте `DATABASE_URL` рабочей БД. Пароли не хранятся в коде.

Совместимые версии (зафиксированы в `requirements.txt`): FastAPI 0.115.x, SQLAlchemy 2.0.35, Alembic 1.14–1.20, psycopg 3.x, pgvector 0.3.x, sentence-transformers 3.1.1, PyMuPDF ≥1.24.

## Миграции

Скрипты лежат в `migrations/` (не в каталоге `alembic/`, чтобы не конфликтовать с пакетом Alembic).

```text
set ALEMBIC_DATABASE_URL=postgresql+psycopg://USER:PASS@localhost:5433/most_db_test
python -m app.migrate upgrade head
```

- Без `ALEMBIC_DATABASE_URL` команда завершается с ошибкой.
- Миграция `most_db` запрещена, пока не задано `ALLOW_ALEMBIC_MOST_DB=1`.
- Повторный `upgrade head` идемпотентен.

## Эмбеддинги MiniLM

По умолчанию веса берутся из локального кэша Hugging Face (`~/.cache/huggingface/hub`). Сеть не используется, пока не задано `MOST_ALLOW_HF_DOWNLOAD=1`. Тесты включают `MOST_EMBEDDING_OFFLINE=1` и не скачивают модель.

## Тесты

Только `most_db_test` / `most_db_schema` и `tests/_tmp`. Рабочие `most_db` и `data/corpus` не меняются.

```text
set MOST_EMBEDDING_OFFLINE=1
python -m tests.test_e2e_rag
```
