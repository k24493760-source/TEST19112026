import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from app.db import SessionLocal
from app.models import KnowledgeChunk, KnowledgeDocument
from app import models  # noqa: F401


def clear_knowledge_corpus() -> None:
    db = SessionLocal()
    try:
        n_docs_before = db.query(KnowledgeDocument).count()
        n_chunks_before = db.query(KnowledgeChunk).count()
        print(f"До очистки: документов={n_docs_before}, фрагментов={n_chunks_before}")

        deleted_chunks = db.query(KnowledgeChunk).delete()
        deleted_docs = db.query(KnowledgeDocument).delete()
        db.commit()

        n_docs_after = db.query(KnowledgeDocument).count()
        n_chunks_after = db.query(KnowledgeChunk).count()
        print(f"Удалено: документов={deleted_docs}, фрагментов={deleted_chunks}")
        print(f"После очистки: документов={n_docs_after}, фрагментов={n_chunks_after}")
        if n_docs_after == 0 and n_chunks_after == 0:
            print("Корпус пуст. Можно загружать реальные источники через /ui/upload.")
        else:
            print("Внимание: после удаления записи ещё остались.")
    except Exception as exc:
        db.rollback()
        print(f"Ошибка очистки: {exc}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    clear_knowledge_corpus()
