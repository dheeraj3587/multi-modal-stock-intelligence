# Multi‑Modal Stock Intelligence Platform — Project Charter

Project: Multi‑Modal Stock Intelligence Platform: Price Forecasting, Growth Scoring, and Market Sentiment
Author: Project Team
Date: 18 Nov 2025

Introduction
The stock market is complex and driven by price history, fundamentals, macro indicators, and unstructured text (news, social media). This project builds a multi‑modal platform that integrates these data sources to produce short‑ and medium‑term price forecasts, objective growth scores, and continuous sentiment signals to support data‑driven investment decisions.

Objective
- Develop and validate multi‑modal ML models and a lightweight dashboard that together: (1) lower short‑term price forecast error versus classical baselines; (2) produce actionable growth scores; and (3) surface timely sentiment that meaningfully correlates with price movements.

Scope (In scope)
- Data: historical prices, volumes, company fundamentals, macro indicators, news and social sentiment feeds (India focus initially).
- Models: time‑series models (LSTM/GRU/Transformer), BERT‑style sentiment models (FinBERT or fine‑tuned variants), and ensemble growth scorers (RF/GBT + linear baselines).
- Deliverables: reproducible data pipeline, model training code, evaluation notebooks, and a minimal interactive dashboard showing forecasts, confidence intervals, growth scores, and sentiment trends.
- Validation: backtests, cross‑validation, and deployment‑ready inference tests (latency and throughput checks).

Exclusions (Out of scope)
- Live trading execution, custody/integration with brokers, or automated order placement (the platform provides signals, not trade execution).
- Proprietary paid data integrations beyond agreed APIs (Bloomberg, Quandl) unless budgeted separately.
- Full regulatory compliance certifications — we will provide basic controls and audit logs but not compliance/legal sign‑off.

Primary Research Questions
- Can multi‑modal models that combine price history, company fundamentals, and text‑based sentiment achieve lower 7‑day Mean Absolute Error (MAE) than a baseline ARIMA model and lead‑lag ensembles? (primary)
- Does adding domain‑adapted sentiment (FinBERT) to price + fundamentals materially improve short‑term (1–7 day) forecast accuracy and directional accuracy versus price‑only models? (secondary)
- Can a growth scoring algorithm rank stocks such that the top decile (by score) outperforms the market/top‑sector benchmark over a 60–90 day horizon? (secondary)
- How do accuracy, latency, and robustness trade off under different market regimes (volatile vs. stable) and for India‑specific macro events? (tertiary)

Success Metrics
- Forecast accuracy: 7‑day MAE reduction >= X% vs ARIMA baseline (define X during setup; suggested target 10–20%).
- Directional accuracy: >55% for 1–3 day directional moves on test sets (statistical significance evaluated).
- Sentiment quality: F1 score >= 0.80 on held‑out financial sentiment labels (or improvement over generic BERT).
- Growth score utility: top decile cumulative return > benchmark by a statistically significant margin over 60–90 days.
- Production readiness: inference latency <= 300ms per symbol (single‑node) and data pipeline uptime >= 99% during evaluation windows.

Stakeholders
- Supervisor / Sponsor: [Supervisor Name] — strategic approval, prioritization, and business acceptance.
- Project Lead: responsible for delivery cadence, scope, and stakeholder communication.
- Data Engineers: data ingestion, ETL, and pipeline reliability.
- ML Engineers / Researchers: model design, training, validation, and backtesting.
- NLP Engineer: text pipelines, named‑entity extraction, and sentiment modelling.
- Frontend / UX: dashboard and visualization implementation.
- QA / Risk: validation, backtests, and basic controls review.
- Compliance / Legal (advisory): guidance on data licensing and communication of signal use.

Timeline & Next Steps (high level)
- Weeks 1–2: finalize success targets, data access, and baseline ARIMA/RF experiments.
- Weeks 3–6: develop multi‑modal models and sentiment pipeline; initial backtests.
- Weeks 7–10: ensemble scoring, dashboard MVP, and evaluation against success metrics.
- Deliverable: `docs/charter.md`, evaluation report, model notebooks, and dashboard MVP for supervisor review.

Acceptance
This charter is intended as the concise project agreement. When the supervisor reviews and endorses it (“this is our project”), the team will proceed with the detailed project plan and sprint backlog.
