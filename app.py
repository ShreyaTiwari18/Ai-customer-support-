"""SupportAI - AI-Based Customer Support Ticket Priority Analyzer.

Streamlit entry point. Wires together the LangChain LLM analyzer and the
fuzzy inference engine, and renders a clean, card-based result view.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from src.config import config
from src.fuzzy_engine import UNIVERSE, get_engine
from src.llm_analyzer import LLMAnalyzerError, get_analyzer
from src.models import PriorityResult, TicketAnalysis
from src.utils import ValidationError, validate_complaint

st.set_page_config(page_title="SupportAI - Ticket Priority Analyzer", page_icon="🤖", layout="centered")


def inject_css() -> None:
    css_path = Path(__file__).parent / "assets" / "style.css"
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        """
        <div class="sa-hero">
            <h1>🤖 SupportAI</h1>
            <p>Understand customer complaints and prioritize tickets instantly.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_extracted_values(analysis: TicketAnalysis) -> None:
    st.markdown(
        f"""
        <div class="sa-card">
            <h3>Ticket Summary</h3>
            <p><b>Issue Type:</b> {analysis.issue_type.replace('_', ' ').title()}</p>
            <p>{analysis.summary}</p>
            <div class="sa-stat-grid">
                <div class="sa-stat">
                    <div class="sa-stat-label">Urgency</div>
                    <div class="sa-stat-value">{analysis.urgency:.0f}</div>
                </div>
                <div class="sa-stat">
                    <div class="sa-stat-label">Financial Impact</div>
                    <div class="sa-stat-value">{analysis.financial_impact:.0f}</div>
                </div>
                <div class="sa-stat">
                    <div class="sa-stat-label">Sentiment</div>
                    <div class="sa-stat-value">{analysis.sentiment:.0f}</div>
                </div>
                <div class="sa-stat">
                    <div class="sa-stat-label">Delay</div>
                    <div class="sa-stat-value">{analysis.delay:.0f}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_priority(result: PriorityResult) -> None:
    st.markdown(
        f"""
        <div class="sa-priority {result.level.lower()}">
            <div class="sa-score">{result.score:.1f} / 100</div>
            <div class="sa-level">{result.level} Priority</div>
        </div>
        <div class="sa-explanation">{result.explanation}</div>
        """,
        unsafe_allow_html=True,
    )


def render_membership_chart(result: PriorityResult) -> None:
    engine = get_engine()
    colors = {"low": "#22c55e", "medium": "#eab308", "high": "#f97316", "critical": "#ef4444"}
    fig, ax = plt.subplots(figsize=(6, 2.6))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    for term_name, term in engine.priority.terms.items():
        ax.plot(UNIVERSE, term.mf, label=term_name, color=colors.get(term_name), linewidth=2)
    ax.axvline(result.score, color="#374151", linestyle="--", linewidth=1.5, label=f"Score = {result.score:.1f}")
    ax.set_xlabel("Priority Score")
    ax.set_ylabel("Membership Degree")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize="small", frameon=False)
    st.pyplot(fig, transparent=True)


def render_fuzzy_details(result: PriorityResult) -> None:
    with st.expander("View Fuzzy Logic Details"):
        st.markdown("**Priority Membership Chart**")
        render_membership_chart(result)

        st.markdown("**Membership Degrees**")
        for variable, degrees in result.membership_values.items():
            formatted = ", ".join(f"{term}={value:.2f}" for term, value in degrees.items() if value > 0)
            st.markdown(
                f"<div class='sa-membership-line'><b>{variable}:</b> {formatted or 'none'}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("**Activated Rules**")
        if result.activated_rules:
            for rule in result.activated_rules:
                st.markdown(f"<div class='sa-rule-item'>{rule}</div>", unsafe_allow_html=True)
        else:
            st.write("No rules fired above zero membership.")


def main() -> None:
    inject_css()
    render_header()

    complaint = st.text_area("Enter customer complaint", height=140, label_visibility="visible")
    analyze_clicked = st.button("Analyze Ticket", type="primary", use_container_width=True)

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


if __name__ == "__main__":
    main()
