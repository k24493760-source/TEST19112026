# УКЗ-15 — шаг 2.5А. Экспертное утверждение и публикация корпуса

Дата: 2026-11-17. `most_db` не мигрировалась. `retrieve_chunks` не читает `chunk_embeddings`.

Прогон: `python tests/test_approval_publication.py` → **8/8 OK**.

Статус документа: существующий `ready_for_review` = pending_review; добавлен `rejected`. Автоутверждения нет: документ становится `approved` только после полного комплекта ролей.
