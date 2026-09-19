"""Pydantic data models shared across the LLM analyzer and fuzzy engine."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from src.config import ISSUE_TYPES


class TicketAnalysis(BaseModel):
    """Structured information extracted from a customer complaint by the LLM.

    The LLM is only responsible for producing these values. It must never
    decide the final priority itself.
    """

    issue_type: str = Field(description="Category of the complaint")
    urgency: float = Field(ge=0, le=100, description="How time-sensitive the issue is")
    financial_impact: float = Field(ge=0, le=100, description="Monetary severity of the issue")
    sentiment: float = Field(ge=0, le=100, description="0=neutral/positive, 100=extremely negative")
    delay: float = Field(ge=0, le=100, description="0=no delay, 100=extremely long delay")
    summary: str = Field(description="Short summary of the complaint")

    @field_validator("issue_type")
    @classmethod
    def normalize_issue_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        return normalized if normalized in ISSUE_TYPES else "other"


class PriorityResult(BaseModel):
    """Output of the fuzzy inference engine."""

    score: float = Field(ge=0, le=100)
    level: str
    input_values: dict[str, float]
    membership_values: dict[str, dict[str, float]]
    activated_rules: list[str]
    explanation: str
