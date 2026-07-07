"""
services/intent.py — LLM-based intent classification.

Replaces 40+ regex patterns with one GPT call.
To add a new intent: add it to the classifier prompt + handle it in routers/ask.py.
"""

import json
import re
from config import client, CHAT_MODEL
from schemas import Intent


def classify_intent(question: str, history: list[dict]) -> Intent:
    """
    One GPT call → returns Intent(type, reasoning, is_followup).

    Fallback: single_doc if the call fails — always produces an answer.
    """
    recent = history[-4:] if history else []
    history_summary = ""
    if recent:
        lines = [
            f"{'User' if t.get('role') == 'user' else 'Assistant'}: {t.get('content', '')[:200]}"
            for t in recent
        ]
        history_summary = "\nRecent conversation:\n" + "\n".join(lines)

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an intent classifier for a document Q&A system. "
                        "Classify the user's question into exactly one type:\n\n"
                        "- single_doc: about one specific document or person\n"
                        "- cross_doc: needs ALL documents "
                        "(list names, compare all, who are the candidates, summarize all, "
                        "which candidate, compare resumes, list all)\n"
                        "- identity: personal info about ONE person "
                        "(my name, my email, my phone, my contact)\n"
                        "- resume: rewrite, optimize, improve, tailor a resume\n"
                        "- analyzer: score, rate, evaluate, grade a resume\n"
                        "- cover_letter: write a cover letter or application letter\n"
                        "- skill_gap: missing skills, qualification match, am I qualified\n"
                        "- out_of_scope: ONLY if completely unrelated to documents "
                        "(weather, sports, jokes). "
                        "If the question COULD be answered from any uploaded document, "
                        "classify as single_doc. When in doubt, default to single_doc.\n\n"
                        "IMPORTANT: 'who are the candidates', 'list all names', "
                        "'compare the resumes', 'which candidate should I hire', "
                        "'what are the hazards' are NEVER out_of_scope.\n\n"
                        "is_followup: true ONLY if question uses pronouns like "
                        "'it', 'that', 'them' referring to a specific prior answer. "
                        "Questions about ALL documents are NOT follow-ups.\n\n"
                        "Return ONLY valid JSON:\n"
                        '{"type": "<intent>", "reasoning": "<one sentence>", "is_followup": <true|false>}'
                    ),
                },
                {"role": "user", "content": f"Question: {question}{history_summary}"},
            ],
            temperature=0.1,
            max_tokens=120,
        )
        raw  = re.sub(r"```json|```", "", response.choices[0].message.content.strip()).strip()
        data = json.loads(raw)
        intent = Intent(
            type=data.get("type", "single_doc"),
            reasoning=data.get("reasoning", ""),
            is_followup=data.get("is_followup", False),
        )
        print(f"  [Intent] {intent.type} | followup={intent.is_followup} | {intent.reasoning}")
        return intent
    except Exception as e:
        print(f"  [Intent] Classification failed: {e} — defaulting to single_doc")
        return Intent(type="single_doc", reasoning="fallback", is_followup=False)
