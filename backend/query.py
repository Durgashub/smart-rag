"""
Retrieval helpers — session-scoped hybrid search with full Stage 3 pipeline.

Pipeline:
1. Query rewriting      — rephrase question to be more search-friendly
                          (skipped for identity questions to preserve specificity)
2. HyDE                 — generate hypothetical answer, embed that instead
                          (skipped for identity questions to prevent hallucination)
3. Multi-query          — generate 3 variants, retrieve for each
4. BM25 keyword search  — exact match, names, dates, numbers
5. FAISS semantic search — conceptual similarity
6. RRF fusion           — merge all ranked lists into one
7. Cross-encoder        — re-rank using child chunks (400 chars, model's sweet spot)
8. Source diversity     — cap chunks per source
9. Deduplication        — remove duplicate chunks across multi-query results

Why HyDE is skipped for identity questions:
  "What is my name?" → HyDE hallucinates "Your name is Alex J..." → wrong
  embedding → retrieves irrelevant chunks → cross-encoder scores -10.
  For identity questions, the original literal question + BM25 exact match
  is far more reliable.

Why child chunks are used for cross-encoder:
  ms-marco-MiniLM-L-6-v2 was trained on short passages (~100-300 chars).
  Sending 1200-char parent chunks produces extreme negative scores even
  for genuinely relevant content. Child chunks (400 chars) stay within
  the model's trained range and produce reliable positive scores.
"""

import json
import math
import re
from collections import defaultdict

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4.1-mini"
DEFAULT_TOP_K = 6
RRF_K = 60

# Patterns that indicate the question is about personal identity.
# For these, skip HyDE and query rewriting — literal matching works better.
IDENTITY_PATTERNS = [
    r"\bmy name\b",
    r"\bwho am i\b",
    r"\bmy email\b",
    r"\bmy phone\b",
    r"\bmy address\b",
    r"\bmy contact\b",
    r"\bmy age\b",
    r"\bmy dob\b",
    r"\bmy birthday\b",
    r"\bmy number\b",
    r"\bmy linkedin\b",
    r"\bmy github\b",
]

def _is_identity_question(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in IDENTITY_PATTERNS)


# ── Cross-encoder — loaded once and cached ────────────────────────────────────

_cross_encoder = None

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
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


# ── Index loading ─────────────────────────────────────────────────────────────

def load_index(session_id: str):
    store_dir = f"vector_store/{session_id}"
    index = faiss.read_index(f"{store_dir}/index.faiss")
    with open(f"{store_dir}/metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_query(question: str) -> np.ndarray:
    response = client.embeddings.create(model=EMBED_MODEL, input=[question])
    return np.array([response.data[0].embedding], dtype="float32")


# ── Query rewriting ───────────────────────────────────────────────────────────

def rewrite_query(question: str) -> str:
    """
    Rewrite the question to be keyword-rich for document search.

    Skipped for identity questions — rewriting "what is my name?" into
    "methods to identify personal name, techniques for name recognition..."
    is generic and loses the specificity needed for BM25 exact matching.
    """
    if _is_identity_question(question):
        print(f"  [Rewrite] Skipped — identity question, keeping literal")
        return question

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a search query optimizer. "
                        "Rewrite the user's question as a keyword-rich search query "
                        "that will find the most relevant document chunks. "
                        "Remove filler words, expand abbreviations, add synonyms. "
                        "Return ONLY the rewritten query — no explanation, no quotes."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Rewrite this for document search: {question}",
                },
            ],
            temperature=0.3,
            max_tokens=80,
        )
        rewritten = response.choices[0].message.content.strip()
        print(f"  [Rewrite] '{question}' → '{rewritten}'")
        return rewritten if rewritten else question
    except Exception as e:
        print(f"  [Rewrite] Failed: {e}")
        return question


# ── HyDE ─────────────────────────────────────────────────────────────────────

def generate_hypothetical_answer(question: str) -> str:
    """
    HyDE: Hypothetical Document Embeddings.

    Instead of embedding the question directly (which is short and vague),
    ask GPT to write a hypothetical answer as if it exists in the document.
    Then embed THAT — it has much richer vocabulary matching actual content.

    Example:
      Question:   "tell me about leadership"
      Hypothesis: "Leadership is the ability to guide and motivate teams toward
                   a shared goal. Effective leaders demonstrate vision, emotional
                   intelligence, strategic thinking, and communication skills..."

    Skipped for identity questions — GPT hallucinates wrong names/details
    which poisons retrieval:
      Question:   "what is my name?"
      Hypothesis: "Your registered name is Alex Johnson..." (WRONG)
      → wrong embedding → irrelevant chunks → cross-encoder scores -10
    """
    if _is_identity_question(question):
        print(f"  [HyDE] Skipped — identity question (hallucination risk)")
        return question

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document content simulator. "
                        "Given a question, write a short hypothetical passage (2-3 sentences) "
                        "that would appear in a real document and directly answer this question. "
                        "Write it as factual document content, not as a direct answer to the user. "
                        "Use specific vocabulary, technical terms, and concrete details. "
                        "Return ONLY the passage — no preamble, no explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Write a hypothetical document passage that answers: {question}",
                },
            ],
            temperature=0.5,
            max_tokens=150,
        )
        hypothesis = response.choices[0].message.content.strip()
        print(f"  [HyDE] Generated hypothesis: '{hypothesis[:80]}...'")
        return hypothesis if hypothesis else question
    except Exception as e:
        print(f"  [HyDE] Failed: {e} — using original question")
        return question


# ── Multi-query generation ────────────────────────────────────────────────────

def generate_query_variants(question: str) -> list[str]:
    """Generate 3 different phrasings of the question."""
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate 3 different search query variants for document retrieval. "
                        "Each variant should approach the topic differently. "
                        "Return ONLY a JSON array of 3 strings. No markdown."
                        '\nExample: ["variant 1", "variant 2", "variant 3"]'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Generate 3 search variants for: {question}",
                },
            ],
            temperature=0.7,
            max_tokens=200,
        )
        raw = re.sub(r"```json|```", "", response.choices[0].message.content.strip()).strip()
        variants = json.loads(raw)
        if isinstance(variants, list):
            result = [str(v) for v in variants[:3]]
            print(f"  [Multi-query] {len(result)} variants generated")
            return result
    except Exception as e:
        print(f"  [Multi-query] Failed: {e}")
    return [question]


# ── BM25 ─────────────────────────────────────────────────────────────────────

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
            self.idf[term] = math.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1
            )

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
                numerator   = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_id] += idf * numerator / denominator
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ── RRF ──────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    rrf_scores: dict[int, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            rrf_scores[doc_id] += 1.0 / (k + rank)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# ── Cross-encoder re-ranking ──────────────────────────────────────────────────

def rerank_with_cross_encoder(
    question: str,
    candidates: list[dict],
    top_k: int,
) -> list[dict]:
    """
    Re-rank candidates by reading question + chunk together.

    Uses child_text (400 chars) instead of parent text (1200 chars).
    ms-marco-MiniLM-L-6-v2 was trained on short passages — sending long
    parent chunks produces extreme negative scores (-10) even for relevant
    content, making the model useless. Child chunks stay within the model's
    trained range and produce reliable positive relevance scores.
    """
    # Skip cross-encoder for identity questions — ms-marco scores name
    # lookup questions near -10 regardless of chunk quality, producing
    # misleading low accuracy. BM25 + RRF already rank the name chunk #1.
    if _is_identity_question(question):
        print("  [CrossEncoder] Skipped — identity question, using RRF order")
        return candidates[:top_k]

    cross_encoder = get_cross_encoder()
    if cross_encoder is None:
        print("  [CrossEncoder] Unavailable — using RRF order")
        return candidates[:top_k]

    rerank_pool = candidates[:top_k * 2]
    if not rerank_pool:
        return candidates[:top_k]

    try:
        # Use child_text for scoring (short, precise, within model's range)
        # Fall back to truncated parent text if child_text not stored
        pairs = [
            (question, c.get("child_text", c["text"])[:400])
            for c in rerank_pool
        ]
        scores = cross_encoder.predict(pairs)
        for i, score in enumerate(scores):
            rerank_pool[i]["cross_encoder_score"] = float(score)
        reranked = sorted(
            rerank_pool,
            key=lambda x: x.get("cross_encoder_score", 0.0),
            reverse=True,
        )
        print(
            f"  [CrossEncoder] Re-ranked {len(rerank_pool)} → "
            f"top score: {reranked[0].get('cross_encoder_score', 0):.4f}"
        )
        return reranked[:top_k]
    except Exception as e:
        print(f"  [CrossEncoder] Failed: {e} — using RRF order")
        return candidates[:top_k]


# ── MMR (Maximal Marginal Relevance) ─────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D numpy vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def mmr_filter(
    candidates: list[dict],
    top_k: int,
    lambda_param: float = 0.7,
    similarity_threshold: float = 0.85,
) -> list[dict]:
    """
    Maximal Marginal Relevance — remove near-duplicate chunks from the
    candidate list before cross-encoder re-ranking.

    At each step, MMR keeps the candidate that best balances:
      - Relevance to the query  (represented by rrf_score)
      - Diversity from already-selected chunks (cosine similarity on text)

    Formula:
      MMR score = λ × rrf_score − (1−λ) × max_sim(chunk, selected)

    Parameters:
      lambda_param        0.0 = pure diversity, 1.0 = pure relevance (no MMR)
                          0.7 = 70% relevance, 30% diversity penalty (default)
      similarity_threshold  chunks with cosine sim > this to ANY already-
                            selected chunk are blocked immediately, regardless
                            of MMR score — faster and catches near-duplicates
                            that only differ by a few words (0.85 = very strict)

    Uses child_text for similarity so the same 400-char window the
    cross-encoder uses is also what MMR compares — keeps behaviour consistent.

    Why here (before cross-encoder, not after):
      The cross-encoder is expensive. Filtering duplicates first means it only
      scores distinct, useful chunks — no wasted computation on rephrases.
    """
    if len(candidates) <= top_k:
        return candidates

    # Build simple TF-IDF-style bag-of-words vectors for fast cosine sim.
    # Using the child_text (short, precise) keeps vectors tight and comparable.
    # We avoid calling the OpenAI embedding API here to keep latency low.
    def text_to_vec(text: str) -> np.ndarray:
        tokens = re.findall(r"\b\w+\b", text.lower())
        vocab: dict[str, int] = {}
        for tok in tokens:
            vocab[tok] = vocab.get(tok, 0) + 1
        return vocab

    def dict_cosine(a: dict, b: dict) -> float:
        keys = set(a) | set(b)
        if not keys:
            return 0.0
        dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # Vectorise all candidates once
    vecs = [text_to_vec(c.get("child_text", c["text"])[:400]) for c in candidates]

    selected_indices: list[int] = []
    selected_vecs:    list[dict] = []

    # Normalise rrf_scores to [0,1] for the MMR formula
    max_rrf = max((c.get("rrf_score", 0) for c in candidates), default=1)
    min_rrf = min((c.get("rrf_score", 0) for c in candidates), default=0)
    rrf_range = max_rrf - min_rrf or 1.0

    remaining = list(range(len(candidates)))

    while len(selected_indices) < top_k and remaining:
        best_idx = None
        best_score = -float("inf")

        for i in remaining:
            rel = (candidates[i].get("rrf_score", 0) - min_rrf) / rrf_range

            if not selected_vecs:
                mmr_score = rel
            else:
                max_sim = max(dict_cosine(vecs[i], sv) for sv in selected_vecs)

                # Hard block: identical or near-identical chunk
                if max_sim > similarity_threshold:
                    continue

                mmr_score = lambda_param * rel - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        if best_idx is None:
            break

        selected_indices.append(best_idx)
        selected_vecs.append(vecs[best_idx])
        remaining.remove(best_idx)

    result = [candidates[i] for i in selected_indices]
    removed = len(candidates) - len(result)
    if removed > 0:
        print(f"  [MMR] Filtered {removed} near-duplicate chunk(s), kept {len(result)}")
    else:
        print(f"  [MMR] No near-duplicates found, all {len(result)} chunks distinct")
    return result


# ── Source diversification ────────────────────────────────────────────────────

def _diversify(candidates: list[dict], top_k: int) -> list[dict]:
    selected: list[dict] = []
    per_source_count: dict[str, int] = {}
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


# ── Main retrieval ────────────────────────────────────────────────────────────

def retrieve(
    question: str,
    session_id: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    Full Stage 3 retrieval pipeline:

    1. Rewrite query          → keyword-rich version (skipped for identity Qs)
    2. HyDE                   → hypothetical answer embedding (skipped for identity Qs)
    3. Multi-query variants   → 3 different angles
    4. For ALL queries (original + rewritten + HyDE + 3 variants):
       a. FAISS semantic search
       b. BM25 keyword search
       c. RRF fusion per query
    5. Merge all per-query ranked lists with RRF
    6. Deduplicate
    7. Cross-encoder re-ranking on child chunks (400 chars)
    8. Source diversification
    9. Return top_k chunks (GPT receives parent text for full context)
    """
    index, metadata = load_index(session_id)
    corpus = [m["text"] for m in metadata]

    if not corpus:
        return []

    candidate_k = min(top_k * 4, len(metadata))
    bm25 = BM25(corpus)

    print(f"\n[Retrieval] Question: '{question}'")

    # ── Step 1: Rewrite ──
    rewritten = rewrite_query(question)

    # ── Step 2: HyDE ──
    hypothesis = generate_hypothetical_answer(question)

    # ── Step 3: Multi-query variants ──
    variants = generate_query_variants(rewritten)

    # Deduplicate queries — hypothesis == question when HyDE is skipped,
    # so dict.fromkeys naturally deduplicates it
    all_queries = list(dict.fromkeys(
        [question, rewritten, hypothesis] + variants
    ))
    print(f"  [Retrieval] {len(all_queries)} total queries")

    # ── Step 4: Retrieve for each query ──
    all_ranked_lists: list[list[tuple[int, float]]] = []
    semantic_dist_map: dict[int, float] = {}

    for q in all_queries:
        # FAISS semantic search
        query_vec = embed_query(q)
        distances, indices = index.search(query_vec, candidate_k)
        semantic = [
            (int(idx), float(dist))
            for idx, dist in zip(indices[0], distances[0])
            if idx != -1
        ]
        for doc_id, dist in semantic:
            if doc_id not in semantic_dist_map or dist < semantic_dist_map[doc_id]:
                semantic_dist_map[doc_id] = dist

        # BM25 keyword search
        bm25_results = bm25.score(q, top_k=candidate_k)

        # RRF per query
        fused = reciprocal_rank_fusion([semantic, bm25_results])
        all_ranked_lists.append(fused)

    # ── Step 5: Merge all ranked lists ──
    final_ranked = reciprocal_rank_fusion(all_ranked_lists)

    # ── Step 6: Deduplicate and build candidate dicts ──
    seen_ids: set[int] = set()
    candidates: list[dict] = []
    for doc_id, rrf_score in final_ranked:
        if doc_id in seen_ids or doc_id >= len(metadata):
            continue
        seen_ids.add(doc_id)
        candidates.append({
            **metadata[doc_id],
            "distance": semantic_dist_map.get(doc_id, 1.0),
            "rrf_score": rrf_score,
        })

    # ── Step 7: MMR — remove near-duplicate chunks before cross-encoder ──
    candidates = mmr_filter(candidates, top_k=top_k * 3)

    # ── Step 8: Cross-encoder re-ranking (on child chunks) ──
    reranked = rerank_with_cross_encoder(
        question=question,
        candidates=candidates,
        top_k=top_k * 2,
    )

    # ── Step 9: Source diversification ──
    result = _diversify(reranked, top_k)

    print(
        f"  [Retrieval] Final: {len(result)} chunks from "
        f"{len(set(c['source'] for c in result))} source(s)"
    )
    return result