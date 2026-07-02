"""
Retrieval helpers — session-scoped hybrid search with full Stage 3 pipeline.

Pipeline:
1. Query rewriting      — rephrase question to be more search-friendly
2. HyDE                 — generate hypothetical answer, embed that instead
3. Multi-query          — generate 3 variants, retrieve for each
4. BM25 keyword search  — exact match, names, dates, numbers
5. FAISS semantic search — conceptual similarity
6. RRF fusion           — merge all ranked lists into one
7. Cross-encoder        — re-rank top candidates by reading question+chunk together
8. Source diversity     — cap chunks per source
9. Deduplication        — remove duplicate chunks across multi-query results

Why HyDE works:
  A vague question like "tell me about leadership" has a weak embedding
  because it lacks specific vocabulary. A hypothetical answer like
  "Leadership involves setting vision, motivating teams, making decisions..."
  has a MUCH richer embedding that matches actual document content better.
  We embed both the question AND the hypothetical answer and retrieve for both.
"""

import json
import math
import re
import os
from collections import defaultdict

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"
DEFAULT_TOP_K = 6
RRF_K = 60

_cross_encoder = None


def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        if os.environ.get("DISABLE_CROSS_ENCODER") == "true":
            print("[CrossEncoder] Disabled via env var")
            _cross_encoder = "unavailable"
            return None
        try:
            from sentence_transformers import CrossEncoder
            print("[CrossEncoder] Loading model...")
            _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("[CrossEncoder] Ready")
        except ImportError:
            print("[CrossEncoder] sentence-transformers not installed — skipping")
            _cross_encoder = "unavailable"
        except Exception as e:
            print(f"[CrossEncoder] Failed to load: {e} — skipping")
            _cross_encoder = "unavailable"
    return _cross_encoder if _cross_encoder != "unavailable" else None


def load_index(session_id: str):
    store_dir = f"vector_store/{session_id}"
    index = faiss.read_index(f"{store_dir}/index.faiss")
    with open(f"{store_dir}/metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def embed_query(question: str) -> np.ndarray:
    response = client.embeddings.create(model=EMBED_MODEL, input=[question])
    return np.array([response.data[0].embedding], dtype="float32")


def rewrite_query(question: str) -> str:
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a search query optimizer. Rewrite the user's question as a keyword-rich search query. Return ONLY the rewritten query."},
                {"role": "user", "content": f"Rewrite this for document search: {question}"},
            ],
            temperature=0.3, max_tokens=80,
        )
        rewritten = response.choices[0].message.content.strip()
        print(f"  [Rewrite] '{question}' → '{rewritten}'")
        return rewritten if rewritten else question
    except Exception as e:
        print(f"  [Rewrite] Failed: {e}")
        return question


def generate_hypothetical_answer(question: str) -> str:
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "Write a short hypothetical passage (2-3 sentences) that would appear in a real document and directly answer this question. Return ONLY the passage."},
                {"role": "user", "content": f"Write a hypothetical document passage that answers: {question}"},
            ],
            temperature=0.5, max_tokens=150,
        )
        hypothesis = response.choices[0].message.content.strip()
        print(f"  [HyDE] '{hypothesis[:80]}...'")
        return hypothesis if hypothesis else question
    except Exception as e:
        print(f"  [HyDE] Failed: {e}")
        return question


def generate_query_variants(question: str) -> list[str]:
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": 'Generate 3 different search query variants. Return ONLY a JSON array of 3 strings. No markdown.\nExample: ["variant 1", "variant 2", "variant 3"]'},
                {"role": "user", "content": f"Generate 3 search variants for: {question}"},
            ],
            temperature=0.7, max_tokens=200,
        )
        raw = re.sub(r"```json|```", "", response.choices[0].message.content.strip()).strip()
        variants = json.loads(raw)
        if isinstance(variants, list):
            result = [str(v) for v in variants[:3]]
            print(f"  [Multi-query] {len(result)} variants")
            return result
    except Exception as e:
        print(f"  [Multi-query] Failed: {e}")
    return [question]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.tokenized = [tokenize(doc) for doc in corpus]
        self.avgdl = sum(len(d) for d in self.tokenized) / max(self.corpus_size, 1)
        self.idf: dict[str, float] = {}
        self.tf: list[dict[str, int]] = []
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
                dl = len(self.tokenized[doc_id])
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_id] += idf * numerator / denominator
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


def reciprocal_rank_fusion(ranked_lists: list[list[tuple[int, float]]], k: int = RRF_K) -> list[tuple[int, float]]:
    rrf_scores: dict[int, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            rrf_scores[doc_id] += 1.0 / (k + rank)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def rerank_with_cross_encoder(question: str, candidates: list[dict], top_k: int) -> list[dict]:
    cross_encoder = get_cross_encoder()
    if cross_encoder is None:
        print("  [CrossEncoder] Unavailable — using RRF order")
        return candidates[:top_k]
    rerank_pool = candidates[:top_k * 2]
    if not rerank_pool:
        return candidates[:top_k]
    try:
        pairs = [(question, c["text"]) for c in rerank_pool]
        scores = cross_encoder.predict(pairs)
        for i, score in enumerate(scores):
            rerank_pool[i]["cross_encoder_score"] = float(score)
        reranked = sorted(rerank_pool, key=lambda x: x.get("cross_encoder_score", 0.0), reverse=True)
        print(f"  [CrossEncoder] Re-ranked {len(rerank_pool)} → top: {reranked[0].get('cross_encoder_score', 0):.4f}")
        return reranked[:top_k]
    except Exception as e:
        print(f"  [CrossEncoder] Failed: {e}")
        return candidates[:top_k]


def _diversify(candidates: list[dict], top_k: int) -> list[dict]:
    selected, per_source_count = [], {}
    max_per_source = max(2, top_k // 2)
    for c in candidates:
        if len(selected) >= top_k:
            break
        src = c["source"]
        if per_source_count.get(src, 0) >= max_per_source:
            continue
        selected.append(c)
        per_source_count[src] = per_source_count.get(src, 0) + 1
    if len(selected) < top_k:
        remaining = [c for c in candidates if c not in selected]
        selected.extend(remaining[:top_k - len(selected)])
    return selected


def retrieve(question: str, session_id: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    index, metadata = load_index(session_id)
    corpus = [m["text"] for m in metadata]
    if not corpus:
        return []

    candidate_k = min(top_k * 4, len(metadata))
    bm25 = BM25(corpus)

    print(f"\n[Retrieval] Question: '{question}'")

    rewritten  = rewrite_query(question)
    hypothesis = generate_hypothetical_answer(question)
    variants   = generate_query_variants(rewritten)

    all_queries = list(dict.fromkeys([question, rewritten, hypothesis] + variants))
    print(f"  [Retrieval] {len(all_queries)} total queries")

    all_ranked_lists = []
    semantic_dist_map: dict[int, float] = {}

    for q in all_queries:
        query_vec = embed_query(q)
        distances, indices = index.search(query_vec, candidate_k)
        semantic = [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0]) if idx != -1]
        for doc_id, dist in semantic:
            if doc_id not in semantic_dist_map or dist < semantic_dist_map[doc_id]:
                semantic_dist_map[doc_id] = dist
        bm25_results = bm25.score(q, top_k=candidate_k)
        fused = reciprocal_rank_fusion([semantic, bm25_results])
        all_ranked_lists.append(fused)

    final_ranked = reciprocal_rank_fusion(all_ranked_lists)

    seen_ids: set[int] = set()
    candidates = []
    for doc_id, rrf_score in final_ranked:
        if doc_id in seen_ids or doc_id >= len(metadata):
            continue
        seen_ids.add(doc_id)
        candidates.append({**metadata[doc_id], "distance": semantic_dist_map.get(doc_id, 1.0), "rrf_score": rrf_score})

    reranked = rerank_with_cross_encoder(question=question, candidates=candidates, top_k=top_k * 2)
    result = _diversify(reranked, top_k)

    print(f"  [Retrieval] Final: {len(result)} chunks from {len(set(c['source'] for c in result))} source(s)")
    return result