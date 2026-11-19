"""Загрузка MiniLM из локального кэша. Без отправки документов во внешние сервисы."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.config import BASE_DIR, EMBEDDING_DIM, EMBEDDING_MODEL

HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
MIN_FREE_MB = 400


class EmbeddingModelError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _allow_download() -> bool:
    if (os.getenv("MOST_EMBEDDING_OFFLINE") or "").strip() == "1":
        return False
    if (os.getenv("HF_HUB_OFFLINE") or "").strip() == "1":
        return False
    if (os.getenv("TRANSFORMERS_OFFLINE") or "").strip() == "1":
        return False
    return (os.getenv("MOST_ALLOW_HF_DOWNLOAD") or "").strip() == "1"


def _hf_model_dir(model_name: str) -> Path:
    slug = "models--" + model_name.replace("/", "--")
    if not model_name.startswith("sentence-transformers/") and "/" not in model_name:
        slug = "models--sentence-transformers--" + model_name
    return HF_HUB / slug


def _snapshot_revision(model_dir: Path) -> str | None:
    refs = model_dir / "refs" / "main"
    if refs.is_file():
        return refs.read_text(encoding="utf-8").strip() or None
    snaps = model_dir / "snapshots"
    if snaps.is_dir():
        names = sorted(p.name for p in snaps.iterdir() if p.is_dir())
        return names[-1] if names else None
    return None


def _available_mb() -> float | None:
    try:
        import ctypes
        from ctypes import wintypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullAvailPhys / 1024**2
    except Exception:
        try:
            import os as _os

            page = _os.sysconf("SC_PAGE_SIZE")
            avail = _os.sysconf("SC_AVPHYS_PAGES")
            return page * avail / 1024**2
        except Exception:
            return None


@dataclass
class EmbeddingDiagnosis:
    model: str
    embedding_dim: int
    cache_dir: str
    local_files: bool
    disk_revision: str | None
    config_model: str
    allow_download: bool
    available_mb: float | None
    status: str
    detail: str


def diagnose(model_name: str | None = None) -> EmbeddingDiagnosis:
    import yaml

    name = model_name or EMBEDDING_MODEL
    cfg_path = BASE_DIR / "contracts" / "embedding_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    config_model = str(cfg.get("embedding_model") or name)
    cache_dir = _hf_model_dir(name)
    local = cache_dir.is_dir() and any(cache_dir.rglob("config.json"))
    revision = _snapshot_revision(cache_dir) if local else None
    avail = _available_mb()
    if config_model.split("/")[-1] != name.split("/")[-1]:
        status, detail = "config_mismatch", f"конфиг {config_model}, запрос {name}"
    elif not local:
        status, detail = "cache_missing", f"нет весов в {cache_dir}"
    elif avail is not None and avail < MIN_FREE_MB:
        status, detail = "low_memory", (
            f"веса локально есть; свободно {avail:.0f} МБ "
            f"(рекомендуется ≥{MIN_FREE_MB}, загрузка всё равно будет попыткой)"
        )
    else:
        status, detail = "local_ok", "веса доступны локально"
    return EmbeddingDiagnosis(
        model=name,
        embedding_dim=int(cfg.get("embedding_dim") or EMBEDDING_DIM),
        cache_dir=str(cache_dir),
        local_files=local,
        disk_revision=revision,
        config_model=config_model,
        allow_download=_allow_download(),
        available_mb=avail,
        status=status,
        detail=detail,
    )


@lru_cache(maxsize=1)
def load_sentence_transformer(model_name: str | None = None):
    from sentence_transformers import SentenceTransformer

    name = model_name or EMBEDDING_MODEL
    info = diagnose(name)
    if info.status == "config_mismatch":
        raise EmbeddingModelError("config_mismatch", info.detail)
    if info.status == "low_memory":
        print(f"Предупреждение MiniLM: {info.detail}")
    local_only = not _allow_download()
    if local_only and not info.local_files:
        raise EmbeddingModelError(
            "cache_missing",
            f"{info.detail}. Для разовой загрузки задайте MOST_ALLOW_HF_DOWNLOAD=1. "
            "Тесты не скачивают модель сами.",
        )
    print(
        f"Загрузка модели эмбеддингов {name} "
        f"({'локальный кэш' if info.local_files else 'сеть разрешена'})."
    )
    try:
        model = SentenceTransformer(name, local_files_only=local_only)
    except OSError as exc:
        raise EmbeddingModelError("cache_missing", str(exc)) from exc
    except MemoryError as exc:
        raise EmbeddingModelError("oom", "Недостаточно памяти для MiniLM.") from exc
    dim = int(model.get_sentence_embedding_dimension() or 0)
    if dim and dim != EMBEDDING_DIM:
        raise EmbeddingModelError(
            "config_mismatch",
            f"Размерность модели {dim}, в конфиге EMBEDDING_DIM={EMBEDDING_DIM}.",
        )
    loaded_rev = info.disk_revision
    print(f"Модель {name} готова (revision={loaded_rev or 'unknown'}).")
    return model
