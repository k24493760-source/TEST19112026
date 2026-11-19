"""Конфиг LLM для Golden Q&A. URL не принимается из пользовательского запроса."""
from __future__ import annotations

import os
from urllib.parse import urlparse

from app.services.qa_generator import load_qa_generation_config
from app.services.qa_llm_prompt import PROMPT_VERSION


class LlmConfigError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def llm_section() -> dict:
    cfg = load_qa_generation_config()
    return dict(cfg.get("llm") or {})


def prompt_version() -> str:
    cfg = load_qa_generation_config()
    return str(cfg.get("prompt_version") or PROMPT_VERSION)


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def resolved_llm_config() -> dict:
    section = llm_section()
    enabled = env_flag("MOST_QA_LLM_ENABLED", bool(section.get("enabled")))
    adapter = (os.getenv("MOST_QA_LLM_ADAPTER") or section.get("adapter") or "stub").strip().lower()
    base_url = (os.getenv("MOST_QA_LLM_URL") or section.get("base_url") or "").strip()
    model = (os.getenv("MOST_QA_LLM_MODEL") or section.get("model") or "").strip()
    revision = (os.getenv("MOST_QA_LLM_REVISION") or section.get("revision") or "").strip()
    timeout = int(os.getenv("MOST_QA_LLM_TIMEOUT") or section.get("timeout_seconds") or 30)
    max_tokens = int(os.getenv("MOST_QA_LLM_MAX_TOKENS") or section.get("max_tokens") or 1024)
    temperature = float(os.getenv("MOST_QA_LLM_TEMPERATURE") or section.get("temperature") or 0.0)
    max_parallel = int(section.get("max_parallel") or 1)
    enqueue = env_flag("MOST_QA_ENQUEUE_AFTER_CHUNK", bool(section.get("enqueue_after_chunking", True)))
    hosts = [str(h).strip().lower() for h in (section.get("allowed_hosts") or []) if str(h).strip()]
    return {
        "enabled": enabled,
        "adapter": adapter,
        "base_url": base_url,
        "model": model,
        "revision": revision,
        "timeout_seconds": timeout,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "max_parallel": max(1, max_parallel),
        "enqueue_after_chunking": enqueue,
        "allowed_hosts": hosts,
        "prompt_version": prompt_version(),
    }


def assert_url_allowed(url: str, allowed_hosts: list[str]) -> None:
    if not url:
        raise LlmConfigError("llm_url_missing", "URL LLM-сервера не задан.")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise LlmConfigError("llm_url_scheme", "Допустимы только http/https.")
    host = (parsed.hostname or "").lower()
    if host not in {h.lower() for h in allowed_hosts}:
        raise LlmConfigError("llm_host_forbidden", "Хост LLM не входит в список доверенных.")
