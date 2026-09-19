# AI Customer Support Ticket Priority Analyzer

## Overview

**SupportAI** is a hybrid AI system that analyzes natural-language customer
complaints and calculates an explainable support-ticket priority score. It
combines LangChain + an LLM (for language understanding) with a genuine
fuzzy inference system (for the actual priority decision).

## Problem Statement

Customer support teams receive complaints with varying levels of urgency
and impact. A simple keyword-based system cannot reliably distinguish "Do
you deliver on Sundays?" from "Rs 10,000 was deducted and my order was
cancelled, I need it back today." This project uses an LLM to interpret
complaints and fuzzy logic to handle the graded, uncertain nature of
urgency, financial impact, sentiment, and delay.

## Objectives

- Accept natural-language complaints through a web UI.
- Use LangChain + an LLM to extract structured ticket attributes.
- Use a real fuzzy inference system (not if/else rules) to calculate a
  priority score.
- Keep the LLM and fuzzy-logic responsibilities strictly separated.
- Provide an explainable, visual, testable, deployable application.

## Features

- Natural language complaint input with validation.
- LangChain pipeline extracting: issue type, urgency, financial impact,
  sentiment, delay, and a summary - validated via Pydantic.
- Mamdani fuzzy inference engine (scikit-fuzzy): fuzzification, rule
  evaluation, aggregation, centroid defuzzification.
- 0-100 priority score classified into Low / Medium / High / Critical
  using centrally configured thresholds.
- Human-readable explanation generated from the actual extracted values.
- Expandable "Fuzzy Logic Details" section showing membership degrees and
  activated rules.
- Priority membership function chart with the calculated score marked.
- Graceful error handling (missing API key, API failures, invalid input).
- Automated tests for the fuzzy engine, models, and utilities.

## Architecture

```
Customer Complaint -> Streamlit UI -> LangChain + LLM -> TicketAnalysis (Pydantic)
    -> Fuzzy Inference Engine -> Priority Score -> Priority Level -> Explanation -> UI
```

See [docs/architecture.md](docs/architecture.md) for the full diagram and
module breakdown.

## Technology Stack

Python 3.12+, Streamlit, LangChain, LangChain-OpenAI, Pydantic,
scikit-fuzzy, NumPy, SciPy, Pandas, Matplotlib, python-dotenv, pytest.

## LangChain Pipeline

`src/prompts.py` defines a `ChatPromptTemplate` instructing the LLM to
extract only structured facts (never a priority judgement). `src/llm_analyzer.py`
chains this prompt into `ChatOpenAI(...).with_structured_output(TicketAnalysis)`,
so the LLM's output is always a validated `TicketAnalysis` Pydantic object.

## Fuzzy Logic Methodology

Four fuzzy input variables (Urgency, Financial Impact, Sentiment, Delay),
17 fuzzy rules, and centroid defuzzification produce the final 0-100
priority score. See [docs/fuzzy_logic.md](docs/fuzzy_logic.md) for full
details on membership functions, rules, and the defuzzification method.

## Installation

```bash
git clone <this-repository-url>
cd Ai-customer-support-
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your key:

```
OPENAI_API_KEY=your_api_key_here
LLM_MODEL=gpt-4o-mini
```

Never commit `.env` or real secrets - both are excluded via `.gitignore`.
On Streamlit Community Cloud, set secrets via the app's deployment
settings instead.

## Running Locally

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically http://localhost:8501).

## Testing

```bash
pytest
```

Tests cover the fuzzy engine (fixed numeric inputs, no API key required),
Pydantic model validation, and input-validation utilities. Sample
end-to-end scenarios are listed in `tests/test_cases.json`.

## Deployment

1. Push this repository to GitHub.
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud), connect
   your GitHub account.
3. Create a new app, select this repo/branch, set the main file to `app.py`.
4. In "Advanced settings", add `OPENAI_API_KEY` as a secret.
5. Deploy. The app auto-redeploys on every push to the connected branch.

## Live Demo

[Open Live Application](YOUR_STREAMLIT_URL)

## Project Structure

```
app.py
src/
  __init__.py
  config.py
  models.py
  prompts.py
  llm_analyzer.py
  fuzzy_engine.py
  utils.py
tests/
  __init__.py
  test_fuzzy_engine.py
  test_models.py
  test_utils.py
  test_cases.json
docs/
  architecture.md
  fuzzy_logic.md
.streamlit/
  config.toml
.env.example
.gitignore
requirements.txt
README.md
```

## Limitations

- Priority accuracy depends on the LLM correctly extracting urgency,
  financial impact, sentiment, and delay from the complaint text.
- No persistent storage - each analysis is stateless and not saved.
- No authentication, ticket database, or real support-system integration
  (out of scope for this project).

## Future Scope

- Database storage and ticket history.
- Login/authentication and an admin dashboard.
- Multiple LLM provider support.
- Email/WhatsApp integration and automatic ticket routing.
- SLA prediction and customer segmentation analytics.
- Feedback-based fuzzy rule tuning and human-in-the-loop review.

## Conclusion

SupportAI demonstrates a clean separation of concerns between language
understanding (LangChain + LLM) and decision-making (fuzzy logic),
producing an explainable, testable, and deployable ticket-prioritization
tool suitable for a college-level AI/software engineering project.

## Author

Rajalingam Muthiah
