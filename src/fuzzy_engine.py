"""Fuzzy inference engine for support-ticket prioritization.

Implements a genuine Mamdani fuzzy inference system using scikit-fuzzy:
fuzzification -> rule evaluation -> aggregation -> centroid defuzzification.

The LLM never decides priority directly; this module is the sole authority
on the final priority score, given four crisp numeric inputs.
"""
from __future__ import annotations

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from src.config import config
from src.models import PriorityResult
from src.utils import clamp

UNIVERSE = np.arange(0, 101, 1)


class FuzzyPriorityEngine:
    """Mamdani fuzzy inference system for ticket priority scoring."""

    def __init__(self) -> None:
        self._build_variables()
        self._build_rules()
        self._system = ctrl.ControlSystem(self.rules)

    def _build_variables(self) -> None:
        self.urgency = ctrl.Antecedent(UNIVERSE, "urgency")
        self.financial_impact = ctrl.Antecedent(UNIVERSE, "financial_impact")
        self.sentiment = ctrl.Antecedent(UNIVERSE, "sentiment")
        self.delay = ctrl.Antecedent(UNIVERSE, "delay")
        self.priority = ctrl.Consequent(UNIVERSE, "priority")

        self.urgency["low"] = fuzz.trimf(UNIVERSE, [0, 0, 40])
        self.urgency["medium"] = fuzz.trimf(UNIVERSE, [30, 50, 70])
        self.urgency["high"] = fuzz.trimf(UNIVERSE, [60, 100, 100])

        self.financial_impact["low"] = fuzz.trimf(UNIVERSE, [0, 0, 40])
        self.financial_impact["medium"] = fuzz.trimf(UNIVERSE, [30, 50, 70])
        self.financial_impact["high"] = fuzz.trimf(UNIVERSE, [60, 100, 100])

        self.sentiment["neutral"] = fuzz.trimf(UNIVERSE, [0, 0, 40])
        self.sentiment["negative"] = fuzz.trimf(UNIVERSE, [30, 50, 70])
        self.sentiment["very_negative"] = fuzz.trimf(UNIVERSE, [60, 100, 100])

        self.delay["short"] = fuzz.trimf(UNIVERSE, [0, 0, 40])
        self.delay["moderate"] = fuzz.trimf(UNIVERSE, [30, 50, 70])
        self.delay["long"] = fuzz.trimf(UNIVERSE, [60, 100, 100])

        self.priority["low"] = fuzz.trimf(UNIVERSE, [0, 0, 30])
        self.priority["medium"] = fuzz.trimf(UNIVERSE, [20, 42, 65])
        self.priority["high"] = fuzz.trimf(UNIVERSE, [55, 70, 85])
        self.priority["critical"] = fuzz.trimf(UNIVERSE, [75, 100, 100])

    def _build_rules(self) -> None:
        u, f, s, d, p = (
            self.urgency,
            self.financial_impact,
            self.sentiment,
            self.delay,
            self.priority,
        )

        self.rule_definitions = [
            ("R1: urgency HIGH and financial_impact HIGH -> CRITICAL", u["high"] & f["high"], p["critical"]),
            ("R2: urgency HIGH and sentiment VERY_NEGATIVE -> HIGH", u["high"] & s["very_negative"], p["high"]),
            ("R3: urgency HIGH and delay LONG -> CRITICAL", u["high"] & d["long"], p["critical"]),
            ("R4: financial_impact HIGH and sentiment VERY_NEGATIVE -> CRITICAL", f["high"] & s["very_negative"], p["critical"]),
            ("R5: urgency MEDIUM and financial_impact HIGH -> HIGH", u["medium"] & f["high"], p["high"]),
            ("R6: delay LONG and urgency MEDIUM -> HIGH", d["long"] & u["medium"], p["high"]),
            ("R7: sentiment VERY_NEGATIVE and urgency MEDIUM -> HIGH", s["very_negative"] & u["medium"], p["high"]),
            ("R8: urgency LOW and financial_impact LOW -> LOW", u["low"] & f["low"], p["low"]),
            ("R9: urgency MEDIUM and financial_impact MEDIUM and delay MODERATE -> MEDIUM", u["medium"] & f["medium"] & d["moderate"], p["medium"]),
            ("R10: urgency LOW and sentiment NEUTRAL and delay SHORT -> LOW", u["low"] & s["neutral"] & d["short"], p["low"]),
            ("R11: financial_impact HIGH and delay LONG -> CRITICAL", f["high"] & d["long"], p["critical"]),
            ("R12: urgency HIGH and financial_impact MEDIUM -> HIGH", u["high"] & f["medium"], p["high"]),
            ("R13: urgency LOW and financial_impact MEDIUM -> MEDIUM", u["low"] & f["medium"], p["medium"]),
            ("R14: sentiment NEUTRAL and delay SHORT and financial_impact LOW -> LOW", s["neutral"] & d["short"] & f["low"], p["low"]),
            ("R15: urgency MEDIUM and sentiment NEGATIVE -> MEDIUM", u["medium"] & s["negative"], p["medium"]),
            ("R16: urgency HIGH and financial_impact LOW -> MEDIUM", u["high"] & f["low"], p["medium"]),
            ("R17: urgency LOW and financial_impact HIGH -> MEDIUM", u["low"] & f["high"], p["medium"]),
        ]
        self.rules = [ctrl.Rule(antecedent, consequent) for _, antecedent, consequent in self.rule_definitions]

    def _membership_degrees(self, variable: ctrl.Antecedent, value: float) -> dict[str, float]:
        degrees = {}
        for term_name, term in variable.terms.items():
            degrees[term_name] = float(fuzz.interp_membership(UNIVERSE, term.mf, value))
        return degrees

    def _activated_rules(self, memberships: dict[str, dict[str, float]]) -> list[str]:
        activated = []
        var_map = {
            "urgency": self.urgency,
            "financial_impact": self.financial_impact,
            "sentiment": self.sentiment,
            "delay": self.delay,
        }
        for label, antecedent, _ in self.rule_definitions:
            firing = self._evaluate_antecedent(antecedent, memberships, var_map)
            if firing > 0:
                activated.append(f"{label} (firing strength {firing:.2f})")
        return activated

    def _evaluate_antecedent(self, node, memberships, var_map) -> float:
        """Recursively evaluate a scikit-fuzzy antecedent tree against membership degrees."""
        if hasattr(node, "term1") and hasattr(node, "term2"):
            left = self._evaluate_antecedent(node.term1, memberships, var_map)
            right = self._evaluate_antecedent(node.term2, memberships, var_map)
            if type(node).__name__ == "OperatorAND":
                return min(left, right)
            return max(left, right)
        var_name = node.parent.label
        term_name = node.label
        return memberships[var_name][term_name]

    def analyze(
        self,
        urgency: float,
        financial_impact: float,
        sentiment: float,
        delay: float,
    ) -> PriorityResult:
        """Run the full fuzzy pipeline and return a structured PriorityResult."""
        urgency = clamp(urgency)
        financial_impact = clamp(financial_impact)
        sentiment = clamp(sentiment)
        delay = clamp(delay)

        simulation = ctrl.ControlSystemSimulation(self._system)
        simulation.input["urgency"] = urgency
        simulation.input["financial_impact"] = financial_impact
        simulation.input["sentiment"] = sentiment
        simulation.input["delay"] = delay
        simulation.compute()

        if "priority" not in simulation.output:
            # No rule fired above zero membership for this input combination;
            # fall back to a simple weighted average of the crisp inputs.
            score = 0.35 * urgency + 0.35 * financial_impact + 0.2 * sentiment + 0.1 * delay
        else:
            score = float(simulation.output["priority"])
        level = config.thresholds.classify(score)

        memberships = {
            "urgency": self._membership_degrees(self.urgency, urgency),
            "financial_impact": self._membership_degrees(self.financial_impact, financial_impact),
            "sentiment": self._membership_degrees(self.sentiment, sentiment),
            "delay": self._membership_degrees(self.delay, delay),
        }
        activated_rules = self._activated_rules(memberships)

        explanation = self._build_explanation(
            urgency, financial_impact, sentiment, delay, score, level
        )

        return PriorityResult(
            score=round(score, 1),
            level=level,
            input_values={
                "urgency": urgency,
                "financial_impact": financial_impact,
                "sentiment": sentiment,
                "delay": delay,
            },
            membership_values=memberships,
            activated_rules=activated_rules,
            explanation=explanation,
        )

    @staticmethod
    def _build_explanation(
        urgency: float,
        financial_impact: float,
        sentiment: float,
        delay: float,
        score: float,
        level: str,
    ) -> str:
        drivers = []
        if urgency >= 60:
            drivers.append("high urgency")
        elif urgency >= 30:
            drivers.append("moderate urgency")

        if financial_impact >= 60:
            drivers.append("significant financial impact")
        elif financial_impact >= 30:
            drivers.append("some financial impact")

        if sentiment >= 60:
            drivers.append("strongly negative customer sentiment")
        elif sentiment >= 30:
            drivers.append("negative customer sentiment")

        if delay >= 60:
            drivers.append("a long delay")
        elif delay >= 30:
            drivers.append("a moderate delay")

        if not drivers:
            drivers.append("generally low severity across all factors")

        driver_text = ", ".join(drivers[:-1]) + (" and " + drivers[-1] if len(drivers) > 1 else drivers[0])

        return (
            f"The ticket received a {level.lower()} priority (score {score:.1f}/100) "
            f"because it shows {driver_text}."
        )


_engine: FuzzyPriorityEngine | None = None


def get_engine() -> FuzzyPriorityEngine:
    """Return a lazily-created singleton FuzzyPriorityEngine instance."""
    global _engine
    if _engine is None:
        _engine = FuzzyPriorityEngine()
    return _engine
