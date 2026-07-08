"""
retrieval/pipeline.py — retrieve() orchestrating the full Stage 3 pipeline.

Steps:
  1. rewrite_query()                → retrieval/expansion.py
  2. generate_hypothetical_answer() → retrieval/expansion.py
  3. generate_query_variants()      → retrieval/expansion.py
  4. pgvector search per query      → retrieval/store.py (search_chunks)
  5. BM25 search per query          → retrieval/ranking.py
  6. RRF fusion per query           → retrieval/ranking.py
  7. Merge all RRF lists            → retrieval/ranking.py
  8. MMR deduplication              → retrieval/ranking.py
  9. Cross-encoder re-ranking       → retrieval/rerank.py
  10. Source diversification        → retrieval/ranking.py
"""

from retrieval.store     import load_index, embed_query, search_chunks, has_chunks
from retrieval.expansion import rewrite_query, generate_hypothetical_answer, generate_query_variants
from retrieval.ranking   import BM25, reciprocal_rank_fusion, mmr_filter, diversify, force_one_chunk_per_source
from retrieval.rerank    import rerank_with_cross_encoder
from retrieval.patterns  import is_identity_question, is_cross_document_question
from config import DEFAULT_TOP_K


def retrieve(
    question:   str,
    session_id: str,
    top_k:      int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    Full Stage 3 retrieval — runs all 10 steps.

    Returns top_k chunks. Each chunk dict contains:
      source, text (parent → sent to GPT), child_text,
      distance, rrf_score, cross_encoder_score (if ran).
    """
    # Load metadata from PostgreSQL (replaces faiss.read_index)
    _, metadata = load_index(session_id)
    corpus      = [m["text"] for m in metadata]
    if not corpus:
        return []

    is_id    = is_identity_question(question)
    is_cross = is_cross_document_question(question) and not is_id

    if is_cross:
        print("  [Intent] Cross-document question detected")
    elif is_id:
        print("  [Intent] Identity question detected")
    else:
        print("  [Intent] Single-document question")

    candidate_k = min(top_k * 6 if is_cross else top_k * 4, len(metadata))
    bm25        = BM25(corpus)

    print(f"\n[Retrieval] Question: '{question}'")

    # Steps 1-3: query expansion
    rewritten  = rewrite_query(question)
    hypothesis = generate_hypothetical_answer(question)
    variants   = generate_query_variants(rewritten)
    all_queries = list(dict.fromkeys([question, rewritten, hypothesis] + variants))
    print(f"  [Retrieval] {len(all_queries)} total queries")

    # Steps 4-7: pgvector search + BM25 + RRF per query, then merge
    all_ranked_lists:  list[list[tuple[int, float]]] = []
    semantic_dist_map: dict[int, float]              = {}

    for q in all_queries:
        query_vec = embed_query(q)

        # pgvector search — returns [{source, text, child_text, distance}, ...]
        pg_results = search_chunks(session_id, query_vec, top_k=candidate_k)

        # Map pgvector results to metadata indices for RRF compatibility
        semantic: list[tuple[int, float]] = []
        for pg_chunk in pg_results:
            # Find this chunk's index in metadata list
            for idx, m in enumerate(metadata):
                if (m["source"] == pg_chunk["source"] and
                        m["text"][:100] == pg_chunk["text"][:100]):
                    dist = pg_chunk["distance"]
                    semantic.append((idx, dist))
                    if idx not in semantic_dist_map or dist < semantic_dist_map[idx]:
                        semantic_dist_map[idx] = dist
                    break

        bm25_results = bm25.score(q, top_k=candidate_k)
        fused        = reciprocal_rank_fusion([semantic, bm25_results])
        all_ranked_lists.append(fused)

    final_ranked = reciprocal_rank_fusion(all_ranked_lists)

    # Build candidate dicts
    seen_ids:   set[int]   = set()
    candidates: list[dict] = []
    for doc_id, rrf_score in final_ranked:
        if doc_id in seen_ids or doc_id >= len(metadata):
            continue
        seen_ids.add(doc_id)
        candidates.append({
            **metadata[doc_id],
            "distance":  semantic_dist_map.get(doc_id, 1.0),
            "rrf_score": rrf_score,
        })

    # Step 8: MMR
    candidates = mmr_filter(candidates, top_k=top_k * 3)

    # Step 9: cross-encoder
    reranked = rerank_with_cross_encoder(question, candidates, top_k=top_k * 2)

    # Step 10: diversification
    if is_cross:
        result = force_one_chunk_per_source(reranked, metadata, top_k)
    else:
        result = diversify(reranked, top_k)

    print(f"  [Retrieval] Final: {len(result)} chunks from {len(set(c['source'] for c in result))} source(s)")
    return result
