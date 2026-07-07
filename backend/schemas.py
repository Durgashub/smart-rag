"""
schemas.py — ALL Pydantic request/response models.

Add new models here. Import from here everywhere else.
"""

from pydantic import BaseModel
from typing import Optional


class HistoryTurn(BaseModel):
    role: str       # "user" or "assistant"
    content: str


class QuestionRequest(BaseModel):
    question: str
    mode: Optional[str] = None
    history: list[HistoryTurn] = []


class Intent(BaseModel):
    type: str         # single_doc | cross_doc | identity | resume |
                      # analyzer | cover_letter | skill_gap | out_of_scope
    reasoning: str    # one sentence why (for logging)
    is_followup: bool # true if question references a prior answer
