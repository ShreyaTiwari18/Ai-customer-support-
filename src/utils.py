"""Small shared utility helpers (input validation, formatting)."""
from __future__ import annotations

MIN_COMPLAINT_LENGTH = 10


class ValidationError(Exception):
    """Raised when user-provided input fails validation."""


def validate_complaint(text: str) -> str:
    """Validate and normalize a raw complaint string.

    Raises ValidationError with a user-friendly message on failure.
    """
    if text is None:
        raise ValidationError("Please enter a customer complaint.")

    cleaned = text.strip()

    if not cleaned:
        raise ValidationError("Please enter a customer complaint.")

    if len(cleaned) < MIN_COMPLAINT_LENGTH:
        raise ValidationError("Please enter a meaningful customer complaint.")

    return cleaned


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Clamp a numeric value into [low, high]."""
    return max(low, min(high, value))
