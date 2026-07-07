"""
retrieval/ranking.py — BM25, RRF fusion, MMR filter, source diversification.

Four independent ranking algorithms used in the Stage 3 pipeline.
Each function does ONE thing — easy to test and swap individually.
"""

import math
import re
from collections import defaultdict
from config import RRF_K


# ── BM25 ─────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25:
    """
    BM25 keyword scoring — implemented from scratch (no rank-bm25 dependency).

    Complements FAISS semantic search:
    - FAISS: great at conceptual similarity ("car" ≈ "automobile")
    - BM25:  great at exact matches (names, model numbers, serial numbers)
    Combined via RRF fusion for best of both worlds.
    """
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1          = k1
        self.b           = b
        self.corpus_size = len(corpus)
        self.tokenized   = [tokenize(doc) for doc in corpus]
        self.avgdl       = sum(len(d) for d in self.tokenized) / max(self.corpus_size, 1)
        self.idf: dict[str, float]     = {}
        self.tf:  list[dict[str, int]] = []
        df: dict[str, int] = defaultdict(int)
        for doc_tokens in self.tokenized:
            term_counts: dict[str, int] = defaultdict(int)
            for token in doc_tokens:
                term_counts[token] += 1
            self.tf.append(dict(term_counts))
            for term in set(doc_tokens):
                df[term] += 1
        for term, freq in df.items():
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)

    def score(self, query: str, top_k: int) -> list[tuple[int, float]]:
        query_tokens = tokenize(query)
        scores: dict[int, float] = defaultdict(float)
        for token in query_tokens:
            if token not in self.idf:
                continue
            idf = self.idf[token]
            for doc_id, term_freq in enumerate(self.tf):
                tf = term_freq.get(token, 0)
                if tf == 0:
                    continue
                dl          = len(self.tokenized[doc_id])
                numerator   = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_id] += idf * numerator / denominator
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ── RRF fusion ────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """
    Merge multiple ranked lists with Reciprocal Rank Fusion.
    Score = sum(1 / (k + rank)) across all lists.
    """
    rrf_scores: dict[int, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            rrf_scores[doc_id] += 1.0 / (k + rank)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# ── MMR filter ────────────────────────────────────────────────────────────────

def mmr_filter(
    candidates: list[dict],
    top_k: int,
    lambda_param: float = 0.7,
    similarity_threshold: float = 0.85,
) -> list[dict]:
    """
    Maximal Marginal Relevance — remove near-duplicate chunks.

    At each step picks the candidate that best balances:
      relevance (rrf_score) vs diversity from already-selected chunks.

    MMR score = λ × relevance − (1−λ) × max_similarity(chunk, selected)
    Runs before cross-encoder so it only scores distinct, useful chunks.
    """
    if len(candidates) <= top_k:
        return candidates

    def text_to_vec(text: str) -> dict:
        tokens = re.findall(r"\b\w+\b", text.lower())
        counts: dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        return counts

    def dict_cosine(a: dict, b: dict) -> float:
        keys = set(a) | set(b)
        if not keys:
            return 0.0
        dot    = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    vecs      = [text_to_vec(c.get("child_text", c["text"])[:400]) for c in candidates]
    max_rrf   = max((c.get("rrf_score", 0) for c in candidates), default=1)
    min_rrf   = min((c.get("rrf_score", 0) for c in candidates), default=0)
    rrf_range = max_rrf - min_rrf or 1.0

    selected_indices: list[int] = []
    selected_vecs:    list[dict] = []
    remaining = list(range(len(candidates)))

    while len(selected_indices) < top_k and remaining:
        best_idx, best_score = None, -float("inf")
        for i in remaining:
            rel = (candidates[i].get("rrf_score", 0) - min_rrf) / rrf_range
            if not selected_vecs:
                mmr_score = rel
            else:
                max_sim = max(dict_cosine(vecs[i], sv) for sv in selected_vecs)
                if max_sim > similarity_threshold:
                    continue
                mmr_score = lambda_param * rel - (1 - lambda_param) * max_sim
            if mmr_score > best_score:
                best_score, best_idx = mmr_score, i
        if best_idx is None:
            break
        selected_indices.append(best_idx)
        selected_vecs.append(vecs[best_idx])
        remaining.remove(best_idx)

    result  = [candidates[i] for i in selected_indices]
    removed = len(candidates) - len(result)
    if removed > 0:
        print(f"  [MMR] Filtered {removed} near-duplicate(s), kept {len(result)}")
    else:
        print(f"  [MMR] No near-duplicates found, all {len(result)} distinct")
    return result


# ── Source diversification ────────────────────────────────────────────────────

def diversify(candidates: list[dict], top_k: int) -> list[dict]:
    """
    Cap chunks per source so one large document can't fill all slots.
    """
    selected:        list[dict]     = []
    per_source:      dict[str, int] = {}
    max_per_source   = max(2, top_k // 2)

    for c in candidates:
        if len(selected) >= top_k:
            break
        src = c["source"]
        if per_source.get(src, 0) >= max_per_source:
            continue
        selected.append(c)
        per_source[src] = per_source.get(src, 0) + 1

    if len(selected) < top_k:
        remaining = [c for c in candidates if c not in selected]
        selected.extend(remaining[:top_k - len(selected)])
    return selected


def force_one_chunk_per_source(
    candidates: list[dict],
    metadata: list[dict],
    top_k: int,
) -> list[dict]:
    """
    For cross-doc questions: guarantee at least one chunk from EVERY document.

    Solves "Durga missing from name list" — even if Durga's chunks all
    rank below top_k by similarity score, they still get a guaranteed slot.
    """
    all_sources = list(dict.fromkeys(m["source"] for m in metadata))
    if len(all_sources) <= 1:
        return candidates[:top_k]

    best_per_source: dict[str, dict] = {}
    for c in candidates:
        src = c["source"]
        if src not in best_per_source:
            best_per_source[src] = c

    guaranteed      = [best_per_source[s] for s in all_sources if s in best_per_source]
    guaranteed_ids  = {id(c) for c in guaranteed}
    remaining_slots = max(0, top_k - len(guaranteed))
    fillers         = [c for c in candidates if id(c) not in guaranteed_ids][:remaining_slots]

    print(f"  [Adaptive] Guaranteed 1 chunk from each of {len(guaranteed)}/{len(all_sources)} source(s)")
    return guaranteed + fillers
