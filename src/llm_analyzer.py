"""LangChain-based ticket analyzer.

Responsible ONLY for turning a natural-language complaint into a structured
TicketAnalysis. It must never decide the final priority - that is the fuzzy
engine's job.
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.config import config
from src.models import TicketAnalysis
from src.prompts import ANALYSIS_PROMPT


class LLMAnalyzerError(Exception):
    """Raised when the LLM pipeline fails to produce a valid analysis."""


class LLMAnalyzer:
    """Wraps a LangChain pipeline that extracts structured ticket data."""

    def __init__(self) -> None:
        if not config.has_api_key:
            raise LLMAnalyzerError(
                "AI service is not configured. Please configure the required API key."
            )
        llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.llm_temperature,
            api_key=config.openai_api_key,
        )
        self._structured_llm = llm.with_structured_output(TicketAnalysis)
        self._chain = ANALYSIS_PROMPT | self._structured_llm

    def analyze(self, complaint: str) -> TicketAnalysis:
        try:
            result = self._chain.invoke({"complaint": complaint})
        except Exception as exc:  # network/API errors, malformed responses, etc.
            raise LLMAnalyzerError(
                "Unable to analyze the complaint right now. Please try again."
            ) from exc

        if not isinstance(result, TicketAnalysis):
            raise LLMAnalyzerError(
                "Unable to analyze the complaint right now. Please try again."
            )
        return result


_analyzer: LLMAnalyzer | None = None


def get_analyzer() -> LLMAnalyzer:
    """Return a lazily-created singleton LLMAnalyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = LLMAnalyzer()
    return _analyzer
