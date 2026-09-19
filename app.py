"""SupportAI - AI-Based Customer Support Ticket Priority Analyzer.

Streamlit entry point. Wires together the LangChain LLM analyzer and the
fuzzy inference engine, and renders results with explanations and charts.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.config import config
from src.fuzzy_engine import UNIVERSE, get_engine
from src.llm_analyzer import LLMAnalyzerError, get_analyzer
from src.models import PriorityResult, TicketAnalysis
from src.utils import ValidationError, validate_complaint

st.set_page_config(page_title="SupportAI - Ticket Priority Analyzer", page_icon="🤖", layout="centered")


def render_header() -> None:
    st.title("🤖 SupportAI")
    st.caption("AI-Based Customer Support Ticket Priority Analyzer")
    st.write("AI-powered ticket understanding using LangChain and fuzzy logic")


def render_extracted_values(analysis: TicketAnalysis) -> None:
    st.subheader("AI Analysis")
    st.write(f"**Issue Type:** {analysis.issue_type.title()}")
    st.write(f"**Summary:** {analysis.summary}")

    col1, col2 = st.columns(2)
    col1.metric("Urgency", f"{analysis.urgency:.0f}")
    col2.metric("Financial Impact", f"{analysis.financial_impact:.0f}")

    col3, col4 = st.columns(2)
    col3.metric("Sentiment", f"{analysis.sentiment:.0f}")
    col4.metric("Delay", f"{analysis.delay:.0f}")


def render_priority(result: PriorityResult) -> None:
    st.subheader("Fuzzy Result")
    st.metric("Priority Score", f"{result.score:.1f} / 100")

    level_colors = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}
    icon = level_colors.get(result.level, "")
    st.markdown(f"### {icon} Priority Level: **{result.level}**")

    st.subheader("Explanation")
    st.write(result.explanation)


def render_fuzzy_details(result: PriorityResult) -> None:
    with st.expander("View Fuzzy Logic Details"):
        st.markdown("**Membership Degrees**")
        for variable, degrees in result.membership_values.items():
            formatted = ", ".join(f"{term}={value:.2f}" for term, value in degrees.items() if value > 0)
            st.write(f"- {variable}: {formatted or 'none'}")

        st.markdown("**Activated Rules**")
        if result.activated_rules:
            for rule in result.activated_rules:
                st.write(f"- {rule}")
        else:
            st.write("No rules fired above zero membership.")

        st.markdown(
            "**Methodology:** Mamdani fuzzy inference with triangular membership "
            "functions, max-min composition, and centroid defuzzification."
        )


def render_membership_chart(result: PriorityResult) -> None:
    engine = get_engine()
    fig, ax = plt.subplots(figsize=(6, 3))
    for term_name, term in engine.priority.terms.items():
        ax.plot(UNIVERSE, term.mf, label=term_name)
    ax.axvline(result.score, color="black", linestyle="--", label=f"Score = {result.score:.1f}")
    ax.set_xlabel("Priority Score")
    ax.set_ylabel("Membership Degree")
    ax.set_title("Priority Membership Functions")
    ax.legend(loc="upper left", fontsize="small")
    st.pyplot(fig)


def main() -> None:
    render_header()

    complaint = st.text_area("Enter customer complaint", height=150)
    analyze_clicked = st.button("Analyze Ticket", type="primary")

    if not analyze_clicked:
        return

    try:
        cleaned_complaint = validate_complaint(complaint)
    except ValidationError as exc:
        st.error(str(exc))
        return

    if not config.has_api_key:
        st.error("AI service is not configured. Please configure the required API key.")
        return

    try:
        analyzer = get_analyzer()
        analysis = analyzer.analyze(cleaned_complaint)
    except LLMAnalyzerError as exc:
        st.error(str(exc))
        return

    try:
        engine = get_engine()
        result = engine.analyze(
            urgency=analysis.urgency,
            financial_impact=analysis.financial_impact,
            sentiment=analysis.sentiment,
            delay=analysis.delay,
        )
    except Exception:
        st.error("Something went wrong while calculating priority. Please try again.")
        return

    render_extracted_values(analysis)
    render_priority(result)
    render_fuzzy_details(result)
    render_membership_chart(result)


if __name__ == "__main__":
    main()
