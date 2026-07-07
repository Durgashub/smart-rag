"""
services/generation.py — GPT answer generation helpers.

  build_messages_with_history()  — injects conversation turns into messages
  verify_answer()                — checks answer against retrieved chunks
  get_top_k()                    — returns right chunk count per intent
"""

from config import client, CHAT_MODEL, MAX_HISTORY_TURNS
from prompts import build_context_prompt, get_system_prompt


def build_messages_with_history(
    system_prompt: str,
    context: str,
    question: str,
    history: list[dict],
) -> list[dict]:
    """
    Build the full GPT messages array with conversation history injected.

    Structure GPT receives:
      [system]
      [user: Q1]  [assistant: A1]   ← prior turns (no context, just the text)
      [user: Q2]  [assistant: A2]
      ...
      [user: current question + fresh retrieved context]

    History makes follow-ups work: "make it shorter" → GPT knows what "it" is.
    """
    messages = [{"role": "system", "content": system_prompt}]

    recent = history[-(MAX_HISTORY_TURNS * 2):]
    for turn in recent:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({
        "role":    "user",
        "content": f"Context from your documents:\n{context}\n\nQuestion: {question}",
    })
    return messages


def verify_answer(question: str, answer: str, chunks: list[dict]) -> tuple[str, bool]:
    """
    Second GPT call — checks if the answer is supported by retrieved chunks.

    Removes hallucinated claims (facts from GPT training, not the documents).
    Only runs for single_doc / identity — cross_doc is already isolated by map-reduce.
    Returns (verified_answer, was_modified).
    """
    context = "\n\n".join(c["text"][:400] for c in chunks[:5])
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an answer verifier for a RAG system. "
                        "Check if the answer is fully supported by the context.\n\n"
                        "If the answer contains claims NOT in the context, remove or correct them.\n"
                        "If fully supported, return it unchanged.\n"
                        "Return ONLY the (possibly corrected) answer — no preamble."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer to verify:\n{answer}",
                },
            ],
            temperature=0.1,
            max_tokens=1000,
        )
        verified     = response.choices[0].message.content.strip()
        was_modified = verified != answer
        print(f"  [Verify] {'Answer modified' if was_modified else 'Fully supported'}")
        return verified, was_modified
    except Exception as e:
        print(f"  [Verify] Failed: {e}")
        return answer, False


def get_top_k(intent_type: str, question: str) -> int:
    if intent_type in ("analyzer", "skill_gap"):
        return 12
    if intent_type == "cross_doc":
        return 8
    if len(question.split()) <= 5:
        return 12
    return 8
