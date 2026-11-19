# УКЗ-15 — шаг 2.4Б. Безопасная векторизация

Дата: 2026-11-17.

- PostgreSQL `most_db` не мигрировалась и не наполнялась.
- `retrieve_chunks` по-прежнему читает только `knowledge_chunks.embedding` + FTS.
- Alembic `20261117_02` создан, **не применялся** к рабочей БД.

Прогон: `python tests/test_embedding_pipeline.py` → **10/10 OK** на `most_db_test`.

## Пример

Длинный `retrieval_text` (>256 токенов MiniLM со спецтокенами) → несколько строк `chunk_embeddings` с `segment_index` 0..n, смежные `char_start`/`char_end`, склейка равна исходному тексту. Короткий чанк → один вектор.

## Память

Модель MiniLM CPU, RSS процесса тестов порядка **400+ МБ**. Пакет 1–2 при свободной RAM < 0.5 ГБ.

## Ограничения

- Только draft, отказ на `most_db`.
- Модель MVP: `all-MiniLM-L6-v2` / 384 / 256.
- Скрытая обрезка ST не используется как решение: сегмент длиннее 256 отклоняется.
- Утверждённые векторы в retrieval не подключены.

## Следующий этап

Можно проектировать 2.4В/публикацию: JOIN к `chunk_embeddings` только после expert approval и `corpus_releases.status=published`, с фильтром одной `(embedding_model, model_revision)`. Нельзя включать draft-векторы в прод-поиск.
