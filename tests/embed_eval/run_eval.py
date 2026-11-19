"""Изолированная оценка embedding-моделей. Без PostgreSQL, без most_db, без рабочего корпуса."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent
GOLD_PATH = ROOT / "gold_set.json"
OUT_DIR = ROOT.parent / "_tmp" / "embed_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = [
    {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "aliases": ["all-MiniLM-L6-v2"],
        "doc_dim": 384,
        "doc_ctx": 256,
        "license": "Apache-2.0",
        "prefixes": None,
        "hf_dir": "models--sentence-transformers--all-MiniLM-L6-v2",
        "min_ram_gb": 1.0,
        "weight_gb": 0.09,
    },
    {
        "name": "BAAI/bge-m3",
        "aliases": [],
        "doc_dim": 1024,
        "doc_ctx": 8192,
        "license": "MIT",
        "prefixes": "без instruction-префикса (dense)",
        "hf_dir": "models--BAAI--bge-m3",
        "min_ram_gb": 6.0,
        "weight_gb": 2.3,
    },
    {
        "name": "intfloat/multilingual-e5-base",
        "aliases": [],
        "doc_dim": 768,
        "doc_ctx": 512,
        "license": "MIT",
        "prefixes": "query: / passage:",
        "hf_dir": "models--intfloat--multilingual-e5-base",
        "min_ram_gb": 3.5,
        "weight_gb": 1.1,
    },
    {
        "name": "intfloat/multilingual-e5-large",
        "aliases": [],
        "doc_dim": 1024,
        "doc_ctx": 512,
        "license": "MIT",
        "prefixes": "query: / passage:",
        "hf_dir": "models--intfloat--multilingual-e5-large",
        "min_ram_gb": 6.0,
        "weight_gb": 2.2,
    },
]


def hf_cache_root() -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub"


def model_cached(spec: dict) -> bool:
    root = hf_cache_root()
    names = [spec["hf_dir"]] + [f"models--{a.replace('/', '--')}" for a in spec["aliases"]]
    return any((root / n).is_dir() for n in names)


def skip_reason(spec: dict) -> str | None:
    mem = psutil.virtual_memory()
    avail = mem.available / 1024**3
    if not model_cached(spec) and spec["weight_gb"] + 0.8 > avail:
        return (
            f"веса не в кэше (~{spec['weight_gb']} ГБ) и доступно RAM {avail:.2f} ГБ; "
            "загрузка не запускалась"
        )
    if avail < spec["min_ram_gb"] and spec["weight_gb"] > 0.5:
        return f"доступно RAM {avail:.2f} ГБ, для безопасной загрузки нужно ≥ {spec['min_ram_gb']} ГБ"
    return None


def prepare_texts(spec: dict, queries: list[str], passages: list[str]) -> tuple[list[str], list[str]]:
    pref = spec.get("prefixes")
    if pref and "query:" in str(pref):
        return [f"query: {q}" for q in queries], [f"passage: {p}" for p in passages]
    return list(queries), list(passages)


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    top = ranked[:k]
    if not gold:
        return 0.0
    return len(gold.intersection(top)) / len(gold)


def mrr_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    for i, cid in enumerate(ranked[:k], start=1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = 0.0
    for i, cid in enumerate(ranked[:k], start=1):
        if cid in gold:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = min(len(arr) - 1, max(0, int(round((p / 100) * (len(arr) - 1)))))
    return arr[idx]


def max_tokens_without_trunc(model) -> dict:
    tok = model.tokenizer
    max_len = int(getattr(model, "max_seq_length", None) or tok.model_max_length or 256)
    unit = " срок "
    n = 1
    last_ok = 0
    last_ids = 0
    while n < 20000:
        text = ("Статья 8." + unit * n).strip()
        ids = tok.encode(text, add_special_tokens=True, truncation=False)
        if len(ids) > max_len:
            break
        last_ok = n
        last_ids = len(ids)
        n = int(n * 1.35) + 1
    truncated_text = ("Статья 8." + unit * (last_ok * 4 + 80)).strip()
    full = tok.encode(truncated_text, add_special_tokens=True, truncation=False)
    trunc = tok.encode(
        truncated_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_len,
    )
    return {
        "max_seq_length_runtime": max_len,
        "tokenizer_model_max_length": int(tok.model_max_length or 0),
        "chars_fit": len("Статья 8." + unit * last_ok),
        "tokens_fit": last_ids,
        "overlong_tokens_no_trunc": len(full),
        "overlong_tokens_with_trunc": len(trunc),
        "truncation_applied": len(full) > len(trunc),
    }


def evaluate_model(spec: dict, gold: dict) -> dict:
    from sentence_transformers import SentenceTransformer

    chunks = gold["chunks"]
    queries = gold["queries"]
    passages = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    qtexts = [q["text"] for q in queries]
    q_in, p_in = prepare_texts(spec, qtexts, passages)

    mem0 = psutil.virtual_memory()
    proc0 = psutil.Process().memory_info().rss
    t0 = time.perf_counter()
    model = SentenceTransformer(spec["name"] if model_cached(spec) else spec["aliases"][0])
    load_s = time.perf_counter() - t0
    dim = int(model.get_sentence_embedding_dimension())
    trunc_info = max_tokens_without_trunc(model)

    t_pass = time.perf_counter()
    p_vec = model.encode(p_in, normalize_embeddings=True, convert_to_numpy=True, batch_size=16)
    pass_s = time.perf_counter() - t_pass
    pass_each = [pass_s / max(len(p_in), 1)]

    q_times = []
    q_vec = []
    for text in q_in:
        tq = time.perf_counter()
        vec = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
        q_times.append(time.perf_counter() - tq)
        q_vec.append(vec[0])
    q_mat = np.stack(q_vec)
    scores = q_mat @ p_vec.T

    search_times = []
    metrics = []
    hard_ok = 0
    hard_total = 0
    for i, q in enumerate(queries):
        ts = time.perf_counter()
        order = np.argsort(-scores[i])
        ranked = [ids[j] for j in order]
        search_times.append(time.perf_counter() - ts)
        gold_ids = set(q["relevant"])
        rec5 = recall_at_k(ranked, gold_ids, 5)
        mrr = mrr_at_k(ranked, gold_ids, 10)
        ndcg = ndcg_at_k(ranked, gold_ids, 10)
        metrics.append(
            {
                "id": q["id"],
                "recall@5": rec5,
                "mrr@10": mrr,
                "ndcg@10": ndcg,
                "top5": ranked[:5],
            }
        )
        if q["id"] in {"q05", "q27", "q32", "q04", "q34", "q19", "q21"}:
            hard_total += 1
            if rec5 > 0:
                hard_ok += 1

    proc1 = psutil.Process().memory_info().rss
    mem1 = psutil.virtual_memory()
    mean_embed = float(np.mean(q_times + pass_each))
    p95_embed = percentile(q_times, 95)
    return {
        "name": spec["name"],
        "ran": True,
        "skip_reason": None,
        "license": spec["license"],
        "doc_dim": spec["doc_dim"],
        "measured_dim": dim,
        "doc_ctx": spec["doc_ctx"],
        "prefixes": spec["prefixes"],
        "load_s": round(load_s, 3),
        "embed_query_mean_s": round(float(np.mean(q_times)), 4),
        "embed_query_p95_s": round(p95_embed, 4),
        "embed_passages_total_s": round(pass_s, 4),
        "search_mean_s": round(float(np.mean(search_times)), 6),
        "search_p95_s": round(percentile(search_times, 95), 6),
        "recall@5": round(float(np.mean([m["recall@5"] for m in metrics])), 4),
        "mrr@10": round(float(np.mean([m["mrr@10"] for m in metrics])), 4),
        "ndcg@10": round(float(np.mean([m["ndcg@10"] for m in metrics])), 4),
        "hard_subset_hit_rate": round(hard_ok / hard_total, 4) if hard_total else None,
        "n_queries": len(queries),
        "n_chunks": len(chunks),
        "ram_avail_before_gb": round(mem0.available / 1024**3, 2),
        "ram_avail_after_gb": round(mem1.available / 1024**3, 2),
        "proc_rss_mb": round(proc1 / 1024**2, 1),
        "proc_delta_mb": round((proc1 - proc0) / 1024**2, 1),
        "cuda": False,
        "truncation": trunc_info,
        "per_query": metrics,
    }


def main() -> None:
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    host = {
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "ram_avail_gb": round(psutil.virtual_memory().available / 1024**3, 2),
        "cuda": False,
    }
    results = []
    for spec in CANDIDATES:
        reason = skip_reason(spec)
        if reason:
            results.append(
                {
                    "name": spec["name"],
                    "ran": False,
                    "skip_reason": reason,
                    "license": spec["license"],
                    "doc_dim": spec["doc_dim"],
                    "measured_dim": None,
                    "doc_ctx": spec["doc_ctx"],
                    "prefixes": spec["prefixes"],
                    "recall@5": None,
                    "mrr@10": None,
                    "ndcg@10": None,
                    "embed_query_mean_s": None,
                    "embed_query_p95_s": None,
                    "search_mean_s": None,
                    "proc_rss_mb": None,
                    "cuda": False,
                }
            )
            print(f"SKIP {spec['name']}: {reason}")
            continue
        print(f"RUN {spec['name']}")
        results.append(evaluate_model(spec, gold))

    payload = {
        "host": host,
        "disclaimer": gold["disclaimer"],
        "results": results,
        "postgres_untouched": True,
        "productive_rag_untouched": True,
    }
    (OUT_DIR / "eval_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({r["name"]: {k: r.get(k) for k in ("ran", "recall@5", "mrr@10", "ndcg@10", "skip_reason")} for r in results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
