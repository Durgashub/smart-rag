"""
retrieval/expansion.py — query expansion: rewrite, HyDE, multi-query.

These run BEFORE search and generate additional queries that improve recall.

Why HyDE is skipped for identity questions:
  "What is my name?" → HyDE generates "Your name is Alex Johnson" →
  wrong embedding → wrong chunks retrieved.

Why rewriting is skipped for identity questions:
  "what is my name?" → rewritten to "personal name identification methods" →
  generic, loses BM25 exact matching advantage.
"""

import json
import re
from config import client, CHAT_MODEL
from retrieval.patterns import is_identity_question


def rewrite_query(question: str) -> str:
    """Keyword-rich rewrite of the question for better document search."""
    if is_identity_question(question):
        print("  [Rewrite] Skipped — identity question, keeping literal")
        return question
    try:
        resp = client.chat.completions.create(
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
                {"role": "user", "content": f"Rewrite this for document search: {question}"},
            ],
            temperature=0.3,
            max_tokens=80,
        )
        rewritten = resp.choices[0].message.content.strip()
        print(f"  [Rewrite] '{question}' → '{rewritten}'")
        return rewritten if rewritten else question
    except Exception as e:
        print(f"  [Rewrite] Failed: {e}")
        return question


def generate_hypothetical_answer(question: str) -> str:
    """
    HyDE: generate a hypothetical document passage, embed that instead of the question.

    The hypothesis has richer vocabulary matching real document content.
    Skipped for identity questions — GPT hallucinates wrong names.
    """
    if is_identity_question(question):
        print("  [HyDE] Skipped — identity question (hallucination risk)")
        return question
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document content simulator. "
                        "Write a short hypothetical passage (2-3 sentences) "
                        "that would appear in a real document and directly answer this question. "
                        "Write it as factual document content, not as a direct answer to the user. "
                        "Use specific vocabulary, technical terms, and concrete details. "
                        "Return ONLY the passage — no preamble, no explanation."
                    ),
                },
                {"role": "user", "content": f"Write a hypothetical document passage that answers: {question}"},
            ],
            temperature=0.5,
            max_tokens=150,
        )
        hypothesis = resp.choices[0].message.content.strip()
        print(f"  [HyDE] Generated hypothesis: '{hypothesis[:80]}...'")
        return hypothesis if hypothesis else question
    except Exception as e:
        print(f"  [HyDE] Failed: {e} — using original question")
        return question


def generate_query_variants(question: str) -> list[str]:
    """Generate 3 different phrasings of the question for multi-query retrieval."""
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate 3 different search query variants for document retrieval. "
                        "Each variant should approach the topic differently. "
                        "Return ONLY a JSON array of 3 strings. No markdown.\n"
                        'Example: ["variant 1", "variant 2", "variant 3"]'
                    ),
                },
                {"role": "user", "content": f"Generate 3 search variants for: {question}"},
            ],
            temperature=0.7,
            max_tokens=200,
        )
        raw      = re.sub(r"```json|```", "", resp.choices[0].message.content.strip()).strip()
        variants = json.loads(raw)
        if isinstance(variants, list):
            result = [str(v) for v in variants[:3]]
            print(f"  [Multi-query] {len(result)} variants generated")
            return result
    except Exception as e:
        print(f"  [Multi-query] Failed: {e}")
    return [question]
