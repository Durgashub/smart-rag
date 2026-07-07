"""
services/accuracy.py — confidence % formula.

Converts raw retrieval signals to a 0-100 integer shown in the UI.

Design decisions:
- Cross-encoder score preferred over FAISS distance (direct relevance)
- Noise floor at -5.0: cross-encoder scores below this are unreliable
- max(scores) not avg: best chunk answers the question; averaging misleads
"""

from config import CROSS_ENCODER_NOISE_FLOOR, FAISS_MIN_DIST, FAISS_MAX_DIST


def calculate_accuracy(distance: float, cross_encoder_score: float = None) -> int:
    """
    Cross-encoder score mapping: -5.0 → 50%, +10.0 → 100%
    FAISS distance fallback:      0.3  → 92%, 1.5   → 46%
    """
    if cross_encoder_score is not None and cross_encoder_score > CROSS_ENCODER_NOISE_FLOOR:
        normalized = (cross_encoder_score - CROSS_ENCODER_NOISE_FLOOR) / (
            10.0 - CROSS_ENCODER_NOISE_FLOOR
        )
        return round(min(100, max(50, normalized * 50 + 50)))
    clamped = max(FAISS_MIN_DIST, min(FAISS_MAX_DIST, distance))
    return round(92 - ((clamped - FAISS_MIN_DIST) / (FAISS_MAX_DIST - FAISS_MIN_DIST)) * 46)


def avg_accuracy(chunks: list[dict]) -> int:
    if not chunks:
        return 0
    return max(
        calculate_accuracy(c.get("distance", 1.0), c.get("cross_encoder_score"))
        for c in chunks
    )
