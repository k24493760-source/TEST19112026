import sys
import pathlib

# Добавляем корневую папку в путь, чтобы импорты работали
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from app.db import SessionLocal, engine, Base, ensure_knowledge_schema
from app.models import User
from app import models  # noqa: F401 — регистрация таблиц в metadata


def seed_data():
    Base.metadata.create_all(bind=engine)
    ensure_knowledge_schema(engine)

    db = SessionLocal()
    try:
        if db.query(User).first():
            print("Пользователи уже есть. Документы не создаём — корпус заполняется через загрузку.")
            return

        print("Создаем пользователей...")
        users = [
            User(name="Иван Смирнов", email="ivan@portal-ramn.ru", login="ivan", password_hash="111", role="admin"),
            User(name="Анна Петрова", email="anna@portal-ramn.ru", login="anna", password_hash="111", role="curator"),
            User(name="Сергей Иванов", email="sergey@portal-ramn.ru", login="sergey", password_hash="111", role="listener"),
        ]
        db.add_all(users)
        db.commit()
        print("Пользователи созданы. Тестовые документы больше не добавляются.")
    except Exception as e:
        db.rollback()
        print(f"Ошибка: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    seed_data()
