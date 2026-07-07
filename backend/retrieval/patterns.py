"""
retrieval/patterns.py — regex pattern lists for identity/cross-doc detection.

These are used by retrieval/pipeline.py to skip HyDE/rewriting for identity
questions and force adaptive retrieval for cross-doc questions.

Note: The LLM intent classifier in services/intent.py is the primary router.
These patterns act as a fast local pre-check inside the retrieval pipeline
to adjust retrieval behaviour (not routing behaviour).
"""

import re

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
    r"\bwhat name\b",
    r"\bwhat.s the name\b",
    r"\bwhat is the name\b",
    r"\bthe name of\b",
    r"\bwho is this\b",
    r"\bwho does this belong\b",
    r"\bwhose resume\b",
    r"\bwhose cv\b",
    r"\bthe email\b",
    r"\bthe phone\b",
    r"\bcontact details\b",
    r"\bcontact info\b",
    r"\bwho.*uploaded\b",
    r"\bpeople.*in.*doc",
]

CROSS_DOC_PATTERNS = [
    r"\ball\b.*\b(resume|document|file|pdf|candidate|person|people)\b",
    r"\b(resume|document|file|pdf|candidate|person|people)\b.*\ball\b",
    r"\bcompare\b",
    r"\blist.*all\b",
    r"\ball.*names?\b",
    r"\bevery\b.*\b(resume|document|candidate)\b",
    r"\beach\b.*\b(resume|document|candidate)\b",
    r"\bsummariz.*all\b",
    r"\bacross.*document",
    r"\bwhich.*document",
    r"\bwhich.*file",
    r"\blist.*names?\b",
    r"\blist.*the.*names?\b",
    r"\bwhat.*names?\b",
    r"\bnames?.*resume\b",
    r"\bresume.*names?\b",
    r"\bnames.*from.*all\b",
    r"\bnames?.*and.*skills?\b",
    r"\blist.*names?.*skills?",
    r"\bskills.*of.*all\b",
    r"\blist.*people\b",
]


def is_identity_question(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in IDENTITY_PATTERNS)


def is_cross_document_question(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in CROSS_DOC_PATTERNS)
