"""Tests for the fuzzy inference engine using fixed numerical inputs.

These tests do not require an LLM API key - they exercise the fuzzy engine
directly, as required by the project spec.
"""
import pytest

from src.fuzzy_engine import FuzzyPriorityEngine
from src.models import PriorityResult


@pytest.fixture(scope="module")
def engine() -> FuzzyPriorityEngine:
    return FuzzyPriorityEngine()


def test_engine_returns_priority_result(engine: FuzzyPriorityEngine) -> None:
    result = engine.analyze(urgency=50, financial_impact=50, sentiment=50, delay=50)
    assert isinstance(result, PriorityResult)


def test_low_inputs_produce_low_priority(engine: FuzzyPriorityEngine) -> None:
    result = engine.analyze(urgency=5, financial_impact=5, sentiment=5, delay=5)
    assert result.level in ("LOW", "MEDIUM")
    assert result.score < 55


def test_high_inputs_produce_high_priority(engine: FuzzyPriorityEngine) -> None:
    result = engine.analyze(urgency=95, financial_impact=95, sentiment=95, delay=95)
    assert result.level in ("HIGH", "CRITICAL")
    assert result.score > 55


def test_critical_scenario(engine: FuzzyPriorityEngine) -> None:
    result = engine.analyze(urgency=90, financial_impact=85, sentiment=80, delay=40)
    assert result.level in ("HIGH", "CRITICAL")


def test_score_within_bounds(engine: FuzzyPriorityEngine) -> None:
    for urgency in (0, 25, 50, 75, 100):
        result = engine.analyze(urgency=urgency, financial_impact=50, sentiment=50, delay=50)
        assert 0 <= result.score <= 100


def test_classification_thresholds(engine: FuzzyPriorityEngine) -> None:
    from src.config import config

    assert config.thresholds.classify(0) == "LOW"
    assert config.thresholds.classify(29) == "LOW"
    assert config.thresholds.classify(30) == "MEDIUM"
    assert config.thresholds.classify(55) == "HIGH"
    assert config.thresholds.classify(80) == "CRITICAL"
    assert config.thresholds.classify(100) == "CRITICAL"


def test_clamping_out_of_range_inputs(engine: FuzzyPriorityEngine) -> None:
    result = engine.analyze(urgency=150, financial_impact=-20, sentiment=50, delay=50)
    assert 0 <= result.score <= 100


def test_activated_rules_present_for_extreme_inputs(engine: FuzzyPriorityEngine) -> None:
    result = engine.analyze(urgency=95, financial_impact=95, sentiment=95, delay=95)
    assert len(result.activated_rules) > 0


def test_membership_values_structure(engine: FuzzyPriorityEngine) -> None:
    result = engine.analyze(urgency=65, financial_impact=50, sentiment=50, delay=50)
    assert set(result.membership_values.keys()) == {
        "urgency",
        "financial_impact",
        "sentiment",
        "delay",
    }
    for degrees in result.membership_values.values():
        for degree in degrees.values():
            assert 0 <= degree <= 1
