"""Tests for input-validation utility functions."""
import pytest

from src.utils import ValidationError, clamp, validate_complaint


def test_validate_complaint_accepts_valid_text() -> None:
    assert validate_complaint("My payment was deducted but order cancelled.") == (
        "My payment was deducted but order cancelled."
    )


def test_validate_complaint_rejects_empty_string() -> None:
    with pytest.raises(ValidationError):
        validate_complaint("")


def test_validate_complaint_rejects_whitespace_only() -> None:
    with pytest.raises(ValidationError):
        validate_complaint("    ")


def test_validate_complaint_rejects_too_short() -> None:
    with pytest.raises(ValidationError):
        validate_complaint("hi")


def test_validate_complaint_rejects_none() -> None:
    with pytest.raises(ValidationError):
        validate_complaint(None)


def test_validate_complaint_strips_whitespace() -> None:
    assert validate_complaint("  a meaningful complaint here  ") == "a meaningful complaint here"


def test_clamp_within_range() -> None:
    assert clamp(50) == 50


def test_clamp_above_range() -> None:
    assert clamp(150) == 100


def test_clamp_below_range() -> None:
    assert clamp(-10) == 0
