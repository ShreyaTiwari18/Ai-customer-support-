"""Tests for Pydantic models."""
import pytest
from pydantic import ValidationError

from src.models import PriorityResult, TicketAnalysis


def test_ticket_analysis_valid() -> None:
    analysis = TicketAnalysis(
        issue_type="payment",
        urgency=90,
        financial_impact=85,
        sentiment=80,
        delay=40,
        summary="Customer needs an urgent refund.",
    )
    assert analysis.issue_type == "payment"
    assert analysis.urgency == 90


def test_ticket_analysis_rejects_out_of_range_values() -> None:
    with pytest.raises(ValidationError):
        TicketAnalysis(
            issue_type="payment",
            urgency=150,
            financial_impact=85,
            sentiment=80,
            delay=40,
            summary="Invalid urgency.",
        )


def test_ticket_analysis_normalizes_unknown_issue_type() -> None:
    analysis = TicketAnalysis(
        issue_type="something_unexpected",
        urgency=50,
        financial_impact=50,
        sentiment=50,
        delay=50,
        summary="Unknown category.",
    )
    assert analysis.issue_type == "other"


def test_priority_result_valid() -> None:
    result = PriorityResult(
        score=87.3,
        level="CRITICAL",
        input_values={"urgency": 90},
        membership_values={"urgency": {"high": 1.0}},
        activated_rules=["R1"],
        explanation="High urgency and impact.",
    )
    assert result.level == "CRITICAL"


def test_priority_result_rejects_out_of_range_score() -> None:
    with pytest.raises(ValidationError):
        PriorityResult(
            score=150,
            level="CRITICAL",
            input_values={},
            membership_values={},
            activated_rules=[],
            explanation="",
        )
