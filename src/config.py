"""Centralized configuration for SupportAI.

Reads secrets from environment variables (.env locally) or Streamlit
secrets when deployed on Streamlit Community Cloud.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str, default: str | None = None) -> str | None:
    """Fetch a secret from Streamlit secrets first, then environment variables."""
    try:
        import streamlit as st

        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


@dataclass(frozen=True)
class PriorityThresholds:
    """Configurable priority classification boundaries (0-100 scale)."""

    low_max: float = 30.0
    medium_max: float = 55.0
    high_max: float = 80.0

    def classify(self, score: float) -> str:
        if score < self.low_max:
            return "LOW"
        if score < self.medium_max:
            return "MEDIUM"
        if score < self.high_max:
            return "HIGH"
        return "CRITICAL"


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str | None = field(default_factory=lambda: _get_secret("OPENAI_API_KEY"))
    llm_model: str = field(default_factory=lambda: _get_secret("LLM_MODEL", "gpt-4o-mini") or "gpt-4o-mini")
    llm_temperature: float = 0.0
    thresholds: PriorityThresholds = field(default_factory=PriorityThresholds)

    @property
    def has_api_key(self) -> bool:
        return bool(self.openai_api_key)


config = AppConfig()

ISSUE_TYPES = [
    "payment",
    "refund",
    "delivery",
    "order",
    "account",
    "technical",
    "product",
    "complaint",
    "other",
]
