"""Превращает текст чанка в вектор эмбеддинга (all-MiniLM-L6-v2, 384 измерения)."""

from sqlalchemy.orm import Session

from app.config import EMBEDDING_DIM, EMBEDDING_MODEL
from app.models import KnowledgeChunk


def _load_model():
    from app.services.embedding_model import load_sentence_transformer

    return load_sentence_transformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _load_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=len(texts) > 16,
    )
    result = []
    for vector in vectors:
        values = vector.tolist()
        if len(values) != EMBEDDING_DIM:
            raise ValueError(
                f"Размерность модели {EMBEDDING_MODEL} равна {len(values)}, "
                f"в конфиге EMBEDDING_DIM={EMBEDDING_DIM}."
            )
        result.append(values)
    return result


def embed_text(text: str) -> list[float]:
    """Возвращает нормализованный вектор длины EMBEDDING_DIM."""
    return embed_texts([text or ""])[0]


def embedding_as_list(vector) -> list[float] | None:
    """pgvector отдаёт numpy-массив: его нельзя проверять через `if vector`."""
    if vector is None:
        return None
    if hasattr(vector, "tolist"):
        values = vector.tolist()
    else:
        try:
            values = list(vector)
        except TypeError:
            return None
    if not values:
        return None
    if isinstance(values[0], (list, tuple)):
        values = list(values[0])
    return [float(item) for item in values]


def embedding_is_ready(vector, dim: int = EMBEDDING_DIM) -> bool:
    values = embedding_as_list(vector)
    return values is not None and len(values) == dim


def cosine_similarity(left, right) -> float:
    left_values = embedding_as_list(left)
    right_values = embedding_as_list(right)
    if not left_values or not right_values or len(left_values) != len(right_values):
        return -1.0
    return float(sum(a * b for a, b in zip(left_values, right_values)))


def apply_embeddings(
    db: Session,
    document_id: str | None = None,
    force: bool = False,
) -> int:
    """Считает и сохраняет embedding для чанков без вектора (или всех, если force)."""
    query = db.query(KnowledgeChunk)
    if document_id:
        query = query.filter(KnowledgeChunk.document_id == document_id)
    chunks = query.all()
    pending = [
        chunk
        for chunk in chunks
        if force or not embedding_is_ready(chunk.embedding)
    ]
    if not pending:
        return 0
    vectors = embed_texts([chunk.text or "" for chunk in pending])
    for chunk, vector in zip(pending, vectors):
        chunk.embedding = vector
    return len(pending)
