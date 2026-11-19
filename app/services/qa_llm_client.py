"""Заменяемые клиенты LLM. По умолчанию сеть выключена."""
from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from app.services.qa_generator import GenerateRequest, QaGenerator, StubQaGenerator
from app.services.qa_llm_config import LlmConfigError, assert_url_allowed, resolved_llm_config
from app.services.qa_llm_prompt import SYSTEM_PROMPT, user_prompt

logger = logging.getLogger("most.qa_llm")


class LlmTransportError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class LlmRawResponse:
    text: str
    model: str | None = None
    revision: str | None = None


class QaLlmClient(Protocol):
    def complete(self, request: GenerateRequest) -> LlmRawResponse: ...


class DisabledQaLlmClient:
    def complete(self, request: GenerateRequest) -> LlmRawResponse:
        raise LlmTransportError("llm_disabled", "Генеративная LLM отключена.")


class MockQaLlmClient:
    """Тестовый клиент. Не открывает сеть."""

    def __init__(self, payloads: list[str] | None = None, errors: list[Exception] | None = None):
        self.payloads = list(payloads or [])
        self.errors = list(errors or [])
        self.calls = 0

    def complete(self, request: GenerateRequest) -> LlmRawResponse:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        if self.payloads:
            return LlmRawResponse(text=self.payloads.pop(0), model="mock", revision="test")
        chunk = request.primary
        quote = (chunk.evidence_text or "")[:80]
        payload = {
            "candidates": [
                {
                    "question_text": "Что сказано в исходном фрагменте?",
                    "reference_answer": quote,
                    "question_type": "factual",
                    "difficulty": "easy",
                    "source_chunk_ids": [chunk.chunk_id],
                    "evidence_quotes": [{"chunk_id": chunk.chunk_id, "quote": quote}],
                    "golden_test_kind": "single",
                    "expected_answer_status": "answered",
                    "warnings": [],
                    "ambiguity_flags": [],
                }
            ],
            "skipped_reason": None,
        }
        return LlmRawResponse(text=json.dumps(payload, ensure_ascii=False), model="mock", revision="test")


class HttpQaLlmClient:
    """OpenAI-совместимый /v1/chat/completions на доверенном сервере."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or resolved_llm_config()
        if not self.cfg.get("enabled"):
            raise LlmConfigError("llm_disabled", "Сетевой адаптер выключен.")
        assert_url_allowed(self.cfg["base_url"], self.cfg["allowed_hosts"])
        self._sem = threading.BoundedSemaphore(int(self.cfg["max_parallel"]))

    def complete(self, request: GenerateRequest) -> LlmRawResponse:
        body = {
            "model": self.cfg.get("model") or "local",
            "temperature": self.cfg.get("temperature"),
            "max_tokens": self.cfg.get("max_tokens"),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": user_prompt(
                        {
                            "chunk_id": request.primary.chunk_id,
                            "document_class": request.primary.document_class,
                            "document_version": request.primary.document_version,
                            "section_path": request.primary.section_path,
                            "page": request.primary.page,
                            "evidence_text": request.primary.evidence_text,
                        }
                    ),
                },
            ],
        }
        url = self.cfg["base_url"].rstrip("/") + "/v1/chat/completions"
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        import os

        key = (os.getenv("LLM_API_KEY") or "").strip()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        timeout = float(self.cfg.get("timeout_seconds") or 30)
        logger.info(
            "qa_llm_request host_ok=1 model=%s prompt_version=%s timeout=%s",
            self.cfg.get("model"),
            self.cfg.get("prompt_version"),
            timeout,
        )
        try:
            with self._sem:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8")
        except TimeoutError as exc:
            raise LlmTransportError("timeout", "Таймаут LLM-сервера.") from exc
        except urllib.error.URLError as exc:
            reason = str(getattr(exc, "reason", exc))
            if "timed out" in reason.lower():
                raise LlmTransportError("timeout", "Таймаут LLM-сервера.") from exc
            raise LlmTransportError("server_unavailable", "LLM-сервер недоступен.") from exc
        except Exception as exc:
            raise LlmTransportError("server_unavailable", "LLM-сервер недоступен.") from exc
        try:
            parsed = json.loads(raw)
            text = parsed["choices"][0]["message"]["content"]
        except Exception as exc:
            raise LlmTransportError("invalid_json", "Ответ сервера не JSON chat-completions.") from exc
        return LlmRawResponse(
            text=text,
            model=self.cfg.get("model"),
            revision=self.cfg.get("revision"),
        )


class LlmQaGenerator:
    def __init__(self, client: QaLlmClient, model: str | None = None, revision: str | None = None):
        self.client = client
        self.model = model
        self.revision = revision

    def generate(self, request: GenerateRequest) -> "GenerateResult":
        from app.services.qa_generator import GenerateResult, is_contentful, load_qa_generation_config
        from app.services.qa_llm_parse import parse_and_filter

        cfg = load_qa_generation_config()
        if not is_contentful(request.primary.evidence_text, int(cfg.get("min_evidence_chars") or 80)):
            return GenerateResult(skipped_reason="empty_or_technical_chunk")
        raw = self.client.complete(request)
        return parse_and_filter(raw.text, request)


def build_generator_from_config(client: QaLlmClient | None = None) -> QaGenerator:
    cfg = resolved_llm_config()
    if client is not None:
        return LlmQaGenerator(client, model=cfg.get("model"), revision=cfg.get("revision"))
    adapter = cfg["adapter"]
    if not cfg["enabled"] or adapter in {"stub", "disabled", ""}:
        return StubQaGenerator()
    if adapter == "mock":
        return LlmQaGenerator(MockQaLlmClient(), model="mock", revision=cfg.get("revision"))
    if adapter == "http":
        return LlmQaGenerator(HttpQaLlmClient(cfg), model=cfg.get("model"), revision=cfg.get("revision"))
    return StubQaGenerator()
