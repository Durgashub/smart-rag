"""
retrieval/rerank.py — cross-encoder re-ranking.

Uses ms-marco-MiniLM-L-6-v2 on child_text[:400] (not parent text).

Why child_text[:400]:
  The model was trained on short passages (~100-300 chars).
  Sending 1200-char parent chunks produces scores around -10 even for
  relevant content, making the model useless. Child chunks produce
  reliable positive scores (+2 to +6 for relevant content).

Why skipped for identity questions:
  "what is my name?" scores -10 against a resume name chunk because the
  cross-encoder can't judge lookup questions reliably. BM25 + RRF already
  rank the name chunk #1, so skipping cross-encoder preserves that order.
"""

from retrieval.patterns import is_identity_question

_cross_encoder = None


def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            from config import CROSS_ENCODER_MODEL
            print("[CrossEncoder] Loading model...")
            _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
            print("[CrossEncoder] Ready")
        except ImportError:
            print("[CrossEncoder] sentence-transformers not installed — skipping")
            _cross_encoder = "unavailable"
        except Exception as e:
            print(f"[CrossEncoder] Failed to load: {e} — skipping")
            _cross_encoder = "unavailable"
    return _cross_encoder if _cross_encoder != "unavailable" else None


def rerank_with_cross_encoder(
    question: str,
    candidates: list[dict],
    top_k: int,
) -> list[dict]:
    """Re-rank candidates by relevance. Uses child_text[:400] for scoring."""
    if is_identity_question(question):
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
        pairs  = [(question, c.get("child_text", c["text"])[:400]) for c in rerank_pool]
        scores = cross_encoder.predict(pairs)
        for i, score in enumerate(scores):
            rerank_pool[i]["cross_encoder_score"] = float(score)
        reranked = sorted(rerank_pool, key=lambda x: x.get("cross_encoder_score", 0.0), reverse=True)
        print(
            f"  [CrossEncoder] Re-ranked {len(rerank_pool)} → "
            f"top score: {reranked[0].get('cross_encoder_score', 0):.4f}"
        )
        return reranked[:top_k]
    except Exception as e:
        print(f"  [CrossEncoder] Failed: {e} — using RRF order")
        return candidates[:top_k]
