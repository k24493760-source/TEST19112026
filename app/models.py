import uuid
from datetime import datetime
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from app.db import Base

def generate_uuid():
    return str(uuid.uuid4())

class KnowledgeDocument(Base):
    """Источник базы знаний. Поле class в ТЗ = document_class (закон / учебник)."""
    __tablename__ = "knowledge_documents"

    id = Column(String, primary_key=True, default=generate_uuid)
    title = Column(String, nullable=False)
    document_class = Column(String, nullable=False)  # закон / учебник
    owner = Column(String, nullable=False)
    version = Column(String, default="1.0")
    status = Column(String, default="draft")  # draft / approved / archived
    valid_from = Column(DateTime, nullable=True)
    valid_to = Column(DateTime, nullable=True)
    official_url = Column(String, nullable=True)
    hash = Column(String, nullable=True)
    file_hash = Column(String, unique=True, nullable=True)
    version_action = Column(String, nullable=True)
    related_document_id = Column(String, nullable=True)
    file_path = Column(String, nullable=True)
    file_format = Column(String, nullable=True)
    original_filename = Column(String, nullable=True)
    corpus_source = Column(String, nullable=True)
    parse_status = Column(String, nullable=True)
    parser_version = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    reviewed_at = Column(DateTime, nullable=True)
    last_updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    chunks = relationship("KnowledgeChunk", back_populates="document")
    history = relationship(
        "DocumentHistory",
        back_populates="document",
        cascade="all, delete-orphan",
    )
    parse_runs = relationship("DocumentParseRun", back_populates="document")
    parse_elements = relationship("DocumentElement", back_populates="document")


class DocumentParseRun(Base):
    """Отчёт структурного парсинга. Не заменяет оригинал и не публикует RAG."""

    __tablename__ = "document_parse_runs"

    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(String, ForeignKey("knowledge_documents.id"), nullable=False)
    parser_version = Column(String, nullable=False)
    document_version = Column(String, nullable=True)
    file_format = Column(String, nullable=True)
    status = Column(String, nullable=False, default="parsed")
    needs_review = Column(Boolean, default=False)
    is_current = Column(Boolean, default=True)
    warnings = Column(JSONB, nullable=True)
    errors = Column(JSONB, nullable=True)
    element_count = Column(Integer, default=0)
    artifact_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("KnowledgeDocument", back_populates="parse_runs")
    elements = relationship(
        "DocumentElement",
        back_populates="parse_run",
        cascade="all, delete-orphan",
    )


class DocumentElement(Base):
    """Структурный элемент документа для последующего цитирования."""

    __tablename__ = "document_elements"

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("knowledge_documents.id"), nullable=False)
    parse_run_id = Column(String, ForeignKey("document_parse_runs.id"), nullable=False)
    element_type = Column(String, nullable=False)
    order_index = Column(Integer, nullable=False)
    text_original = Column(Text, nullable=False, default="")
    text_normalized = Column(Text, nullable=False, default="")
    section_path = Column(String, nullable=True)
    page_number = Column(Integer, nullable=True)
    source_location = Column(JSONB, nullable=True)
    normalization_actions = Column(JSONB, nullable=True)

    document = relationship("KnowledgeDocument", back_populates="parse_elements")
    parse_run = relationship("DocumentParseRun", back_populates="elements")


class DocumentHistory(Base):
    __tablename__ = "document_history"

    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(String, ForeignKey("knowledge_documents.id"), nullable=False)
    action = Column(String, nullable=False)
    old_status = Column(String, nullable=True)
    new_status = Column(String, nullable=True)
    actor_id = Column(String, nullable=True)
    actor_role = Column(String, nullable=True)
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("KnowledgeDocument", back_populates="history")


class UploadHistory(Base):
    __tablename__ = "upload_history"

    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(String, ForeignKey("knowledge_documents.id"), nullable=True)
    action = Column(String, nullable=False)
    actor_id = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    details = Column(JSONB, nullable=True)


class CorpusRelease(Base):
    """Версия корпуса. Retrieval использует только status=published."""

    __tablename__ = "corpus_releases"

    id = Column(String, primary_key=True, default=generate_uuid)
    version = Column(String, unique=True, nullable=False)
    status = Column(String, nullable=False, default="draft")
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    published_at = Column(DateTime, nullable=True)
    published_by = Column(String, nullable=True)
    embedding_model = Column(String, nullable=True)
    model_revision = Column(String, nullable=True)
    embedding_config_version = Column(String, nullable=True)
    previous_version = Column(String, nullable=True)
    manifest_hash = Column(String, nullable=True)
    snapshot = Column(JSONB, nullable=True)

    items = relationship("CorpusReleaseItem", back_populates="release")


class CorpusReleaseItem(Base):
    """Состав опубликованного релиза. Старые строки не удаляются при новом выпуске."""

    __tablename__ = "corpus_release_items"

    id = Column(String, primary_key=True, default=generate_uuid)
    release_id = Column(String, ForeignKey("corpus_releases.id", ondelete="CASCADE"), nullable=False, index=True)
    document_id = Column(String, nullable=False, index=True)
    document_version = Column(String, nullable=True)
    chunk_id = Column(String, nullable=False)
    embedding_id = Column(String, nullable=True)
    chunking_version = Column(String, nullable=True)

    release = relationship("CorpusRelease", back_populates="items")


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(String, ForeignKey("knowledge_documents.id"), nullable=False)
    text = Column(Text, nullable=False)
    section = Column(String, nullable=True)
    page = Column(Integer, nullable=True)
    approval_status = Column(String, default="draft")
    embedding_status = Column(String, default="pending")
    embedding = Column(Vector(384), nullable=True)
    search_vector = Column(
        TSVECTOR,
        Computed("to_tsvector('russian', coalesce(text, ''))", persisted=True),
    )
    document_version = Column(String, nullable=True)
    corpus_version = Column(String, nullable=True)
    chunking_version = Column(String, nullable=True)
    source_element_ids = Column(JSONB, nullable=True)
    token_count = Column(Integer, nullable=True)
    parent_chunk_id = Column(String, nullable=True)
    applicability = Column(String, nullable=True)
    retrieval_text = Column(Text, nullable=True)
    evidence_text = Column(Text, nullable=True)

    document = relationship("KnowledgeDocument", back_populates="chunks")
    embeddings = relationship(
        "ChunkEmbedding",
        back_populates="chunk",
        cascade="all, delete-orphan",
    )


class ChunkEmbedding(Base):
    """Поисковый сегмент и вектор. Не заменяет evidence_text исходного чанка."""

    __tablename__ = "chunk_embeddings"
    __table_args__ = (
        UniqueConstraint(
            "chunk_id",
            "segment_index",
            "embedding_model",
            "model_revision",
            "embedding_config_version",
            "content_hash",
            name="uq_chunk_embedding_identity",
        ),
    )

    embedding_id = Column(String, primary_key=True)
    chunk_id = Column(
        String,
        ForeignKey("knowledge_chunks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    segment_index = Column(Integer, nullable=False)
    segment_text = Column(Text, nullable=False)
    char_start = Column(Integer, nullable=False, default=0)
    char_end = Column(Integer, nullable=False, default=0)
    embedding = Column(Vector(384), nullable=True)
    embedding_model = Column(String, nullable=False)
    model_revision = Column(String, nullable=False)
    embedding_config_version = Column(String, nullable=False)
    content_hash = Column(String, nullable=False)
    token_count = Column(Integer, nullable=True)
    embedding_status = Column(String, nullable=False, default="ready")
    created_at = Column(DateTime, default=datetime.utcnow)

    chunk = relationship("KnowledgeChunk", back_populates="embeddings")

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, nullable=False)
    login = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="listener")  # listener, curator, expert, admin
    global_role = Column(String, default="listener")
    learner_subrole = Column(String, nullable=True)
    name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=False, default="Диалог")
    learner_role = Column(String, nullable=True)
    allowed_corpus_version = Column(String, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    last_active_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)  # user или assistant
    content = Column(JSONB, nullable=False)
    review_required = Column(Boolean, default=False)
    is_appealed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    session = relationship("ChatSession", back_populates="messages")


class RagAuditLog(Base):
    __tablename__ = "rag_audit_logs"

    id = Column(String, primary_key=True, default=generate_uuid)
    request_id = Column(String, nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    question_text = Column(Text, nullable=False)
    retrieved_chunk_ids = Column(JSONB, nullable=True)
    search_score = Column(Float, nullable=True)
    error_code = Column(String, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)


class AnalyticsEvent(Base):
    __tablename__ = "analytics_events"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=True)
    event_type = Column(String, nullable=False)
    payload = Column(JSONB, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

class ExpertReview(Base):
    """Задача эксперту: конфликт, нет основания, апелляция."""
    __tablename__ = "expert_reviews"

    id = Column(String, primary_key=True, default=generate_uuid)
    object_type = Column(String, nullable=False)
    object_id = Column(String, nullable=True)
    reason = Column(String, nullable=False)
    reviewer = Column(String, nullable=True)
    expert_id = Column(String, nullable=True)
    expert_role = Column(String, nullable=True)
    priority = Column(String, nullable=True, default="normal")
    sla_deadline = Column(DateTime, nullable=True)
    resolution_action = Column(JSON, nullable=True)
    decision = Column(String, nullable=True)
    expert_decision = Column(String, nullable=True)
    comment = Column(Text, nullable=True)
    expert_comment = Column(Text, nullable=True)
    evidence = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)
    time_spent_seconds = Column(Integer, nullable=True)
    status = Column(String, default="pending")

    comments = relationship(
        "ExpertComment",
        back_populates="review",
        cascade="all, delete-orphan",
    )


class ExpertComment(Base):
    __tablename__ = "expert_comments"

    id = Column(String, primary_key=True, default=generate_uuid)
    review_id = Column(String, ForeignKey("expert_reviews.id"), nullable=False)
    expert_id = Column(String, nullable=True)
    comment = Column(Text, nullable=False)
    internal_only = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    review = relationship("ExpertReview", back_populates="comments")


class AuditEvent(Base):
    __tablename__ = "audit_events"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    actor = Column(String, nullable=False)
    action = Column(String, nullable=False)
    object = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    payload = Column(JSON, nullable=True)

class Attempt(Base):
    """Попытка тестирования (ТЗ, раздел 10)."""
    __tablename__ = "attempts"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    module_id = Column(String, nullable=False)
    question_version = Column(String, nullable=False)
    answer = Column(Text, nullable=True)
    score = Column(Float, nullable=True)
    rationale = Column(Text, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    finished_at = Column(DateTime, nullable=True)


class LearningModule(Base):
    """Учебный модуль (ТЗ F15-01). Не является карточкой Q&A."""

    __tablename__ = "learning_modules"

    id = Column(String, primary_key=True, default=generate_uuid)
    title = Column(String, nullable=False)
    competencies = Column(JSONB, nullable=True)
    prerequisites = Column(JSONB, nullable=True)
    corpus_version = Column(String, nullable=True)
    owner = Column(String, nullable=True)
    status = Column(String, nullable=False, default="draft")
    created_at = Column(DateTime, default=datetime.utcnow)


class Rubric(Base):
    """Версионированная рубрика свободного ответа. Не утверждается LLM."""

    __tablename__ = "rubrics"

    id = Column(String, primary_key=True, default=generate_uuid)
    version = Column(String, nullable=False, default="1.0")
    competency = Column(String, nullable=True)
    title = Column(String, nullable=True)
    required_elements = Column(JSONB, nullable=True)
    allowed_formulations = Column(JSONB, nullable=True)
    critical_errors = Column(JSONB, nullable=True)
    owner = Column(String, nullable=True)
    status = Column(String, nullable=False, default="draft")
    created_at = Column(DateTime, default=datetime.utcnow)


class Question(Base):
    """Банк вопросов ТЗ. Может ссылаться на карточку QaCard."""

    __tablename__ = "questions"

    id = Column(String, primary_key=True, default=generate_uuid)
    type = Column(String, nullable=False, default="free")
    competency = Column(String, nullable=True)
    difficulty = Column(String, nullable=True)
    prompt = Column(Text, nullable=False)
    key = Column(JSONB, nullable=True)
    rubric_version = Column(String, nullable=True)
    evidence_ids = Column(JSONB, nullable=True)
    status = Column(String, nullable=False, default="draft")
    qa_id = Column(String, ForeignKey("qa_cards.qa_id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class QaCard(Base):
    """Кандидат / версия Q&A. Golden и обучение утверждаются независимо."""

    __tablename__ = "qa_cards"

    qa_id = Column(String, primary_key=True, default=generate_uuid)
    lineage_id = Column(String, nullable=False, index=True)
    parent_qa_id = Column(String, ForeignKey("qa_cards.qa_id"), nullable=True)
    qa_version = Column(String, nullable=False, default="1.0")
    question_text = Column(Text, nullable=False)
    reference_answer = Column(Text, nullable=False, default="")
    question_type = Column(String, nullable=False, default="factual")
    difficulty = Column(String, nullable=True)
    document_version = Column(String, nullable=True)
    corpus_version = Column(String, nullable=True)
    generation_model = Column(String, nullable=True)
    generation_prompt_version = Column(String, nullable=True)
    content_hash = Column(String, nullable=False, index=True)
    generation_status = Column(String, nullable=False, default="draft")
    golden_status = Column(String, nullable=False, default="none")
    learning_status = Column(String, nullable=False, default="none")
    golden_test_kind = Column(String, nullable=True)
    expected_answer_status = Column(String, nullable=True)
    expected_chunk_ids = Column(JSONB, nullable=True)
    eval_split = Column(String, nullable=True)
    competency = Column(String, nullable=True)
    module_id = Column(String, ForeignKey("learning_modules.id"), nullable=True)
    rubric_id = Column(String, ForeignKey("rubrics.id"), nullable=True)
    rubric_version = Column(String, nullable=True)
    answer_type = Column(String, nullable=True)
    answer_key = Column(JSONB, nullable=True)
    required_elements = Column(JSONB, nullable=True)
    allowed_formulations = Column(JSONB, nullable=True)
    critical_errors = Column(JSONB, nullable=True)
    warnings = Column(JSONB, nullable=True)
    ambiguity_flags = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    sources = relationship("QaSourceChunk", back_populates="card", cascade="all, delete-orphan")
    quotes = relationship("QaEvidenceQuote", back_populates="card", cascade="all, delete-orphan")


class QaSourceChunk(Base):
    """Связь карточки с утверждённым чанком. Не embedding_id."""

    __tablename__ = "qa_source_chunks"

    id = Column(String, primary_key=True, default=generate_uuid)
    qa_id = Column(String, ForeignKey("qa_cards.qa_id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_id = Column(String, ForeignKey("knowledge_chunks.id"), nullable=False, index=True)
    document_id = Column(String, nullable=True)
    document_version = Column(String, nullable=True)
    section = Column(String, nullable=True)
    page = Column(Integer, nullable=True)
    role = Column(String, nullable=False, default="primary")

    card = relationship("QaCard", back_populates="sources")


class QaEvidenceQuote(Base):
    __tablename__ = "qa_evidence_quotes"

    id = Column(String, primary_key=True, default=generate_uuid)
    qa_id = Column(String, ForeignKey("qa_cards.qa_id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_id = Column(String, nullable=True)
    quote = Column(Text, nullable=False)
    source_element_ids = Column(JSONB, nullable=True)

    card = relationship("QaCard", back_populates="quotes")


class QaEvalSnapshot(Base):
    """Зафиксированный результат RAG-теста. Не переписывается при новой версии карточки."""

    __tablename__ = "qa_eval_snapshots"

    id = Column(String, primary_key=True, default=generate_uuid)
    qa_id = Column(String, nullable=False, index=True)
    qa_version = Column(String, nullable=True)
    corpus_version = Column(String, nullable=True)
    eval_split = Column(String, nullable=True)
    payload = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class QaGenerationJob(Base):
    """Фоновая генерация кандидатов Q&A. Не публикует и не утверждает."""

    __tablename__ = "qa_generation_jobs"
    __table_args__ = (UniqueConstraint("idempotency_key", name="uq_qa_generation_jobs_idempotency"),)

    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(String, nullable=True, index=True)
    chunk_id = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False, default="queued")
    attempt = Column(Integer, nullable=False, default=0)
    error_code = Column(String, nullable=True)
    skipped_reason = Column(String, nullable=True)
    n_candidates = Column(Integer, nullable=False, default=0)
    n_persisted = Column(Integer, nullable=False, default=0)
    idempotency_key = Column(String, nullable=False)
    model = Column(String, nullable=True)
    revision = Column(String, nullable=True)
    prompt_version = Column(String, nullable=True)
    adapter = Column(String, nullable=True)
    document_version = Column(String, nullable=True)
    chunking_version = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)