"""Локальный зонд MiniLM: лимит токенов, обрезка, RAM. Без PostgreSQL."""
from __future__ import annotations

import time
from pathlib import Path

import psutil
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL

out = Path(__file__).resolve().parent.parent / "_tmp" / "embed_eval"
out.mkdir(parents=True, exist_ok=True)

mem0 = psutil.virtual_memory()
proc0 = psutil.Process().memory_info().rss
t0 = time.perf_counter()
model = SentenceTransformer(EMBEDDING_MODEL)
load_s = time.perf_counter() - t0
tok = model.tokenizer
tok.model_max_length = getattr(model, "max_seq_length", None) or tok.model_max_length

text = "Статья 12.1 не применяется. " * 80
ids_full = tok.encode(text, add_special_tokens=False, truncation=False)
ids_trunc = tok.encode(text, add_special_tokens=True, truncation=True, max_length=model.max_seq_length)
vec_short = model.encode(["Статья 12.1 не применяется."], normalize_embeddings=True)
vec_long = model.encode([text], normalize_embeddings=True)
t1 = time.perf_counter()
for _ in range(20):
    model.encode(["Каков срок действия статьи 12.1?"], normalize_embeddings=True)
embed_s = (time.perf_counter() - t1) / 20

mem1 = psutil.virtual_memory()
proc1 = psutil.Process().memory_info().rss
lines = [
    f"model={EMBEDDING_MODEL}",
    f"max_seq_length={model.max_seq_length}",
    f"tokenizer_model_max_length={tok.model_max_length}",
    f"dim={len(vec_short[0])}",
    f"full_tokens={len(ids_full)} trunc_tokens={len(ids_trunc)}",
    f"long_equals_short={bool((vec_short[0] == vec_long[0]).all())}",
    f"load_s={load_s:.3f} embed_query_s={embed_s:.4f}",
    f"ram_total_gb={mem1.total/1024**3:.2f} ram_avail_gb={mem1.available/1024**3:.2f}",
    f"proc_rss_mb={proc1/1024**2:.1f} delta_rss_mb={(proc1-proc0)/1024**2:.1f}",
    f"cuda=False",
]
(out / "minilm_probe.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
