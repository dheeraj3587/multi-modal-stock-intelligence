# Baseline Implementation Plan

**Project:** Multi-Modal Stock Intelligence Platform
**Version:** 1.0
**Date:** 18 Nov 2025
**Purpose:** Define 2-3 defensible baseline models for each module to implement first, before advanced multi-modal models.

---

## Overview

This document specifies the **initial baseline models** we will implement to establish performance benchmarks. These baselines must be:
1. **Simple and well-understood** (reproducible, debuggable)
2. **Industry-standard** (defensible to reviewers)
3. **Fast to implement** (1-2 weeks per module)

All advanced models (multi-modal LSTM+fundamentals+sentiment, PatchTST, event-driven) will be evaluated **against these baselines**.

---

## 1. Forecasting Module Baselines

### Baseline 1: ARIMA (Primary Classical Baseline)

**Model:** Auto-ARIMA with grid search over (p, d, q)

**Rationale:**
- Industry standard for time-series forecasting
- Simple, interpretable, no deep learning required
- Establishes minimum acceptable performance
- Used in Fischer & Krauss (2018) and most financial forecasting papers as baseline

**Implementation Details:**
- **Library:** `statsmodels.tsa.arima.model.ARIMA` or `pmdarima.auto_arima`
- **Parameter Search:** Grid search over:
  - p (AR order): [0, 1, 2, 3]
  - d (differencing): [0, 1, 2]
  - q (MA order): [0, 1, 2, 3]
- **Selection Criterion:** AIC (Akaike Information Criterion)
- **Forecast Horizon:** 1-day, 3-day, 7-day ahead
- **Training Window:** Rolling 252 trading days (~1 year)

**Expected Performance (from literature):**
- 7-day MAE: 2-5% of stock price (baseline)
- Directional accuracy: ~50-52% (near random)
- Sharpe ratio: 0.3-0.5 (buy-and-hold level)

**Success Criteria:**
- Our multi-modal models must beat ARIMA by ≥15% MAE reduction (per metrics doc)

---

### Baseline 2: Simple LSTM (Primary Deep Learning Baseline)

**Model:** 2-layer LSTM with price-only features

**Rationale:**
- Widely used in financial forecasting (Fischer & Krauss 2018 achieved 55.9% directional accuracy)
- Captures temporal dependencies better than ARIMA
- Establishes neural network baseline before adding multi-modal features
- Directly comparable to our advanced LSTM+fundamentals+sentiment model

**Implementation Details:**
- **Framework:** PyTorch or TensorFlow/Keras
- **Architecture:**
  ```
  Input: [lookback_window, n_features]
  ├─ LSTM Layer 1 (64 units, return_sequences=True)
  ├─ Dropout (0.2)
  ├─ LSTM Layer 2 (32 units)
  ├─ Dropout (0.2)
  └─ Dense (1 unit, linear activation) → predicted price
  ```
- **Features:**
  - Past 60 days of close prices (normalized)
  - Past 60 days of volume (normalized)
  - Past 60 days of returns (log returns)
- **Hyperparameters:**
  - Lookback window: 60 trading days
  - Batch size: 32
  - Learning rate: 0.001 (Adam optimizer)
  - Epochs: 50 with early stopping (patience=10)
- **Training:** Rolling window (same as ARIMA: 252-day train, predict next 1/3/7 days)

**Expected Performance (from Fischer & Krauss 2018):**
- Directional accuracy: 53-56% (1-day ahead)
- MAE: 10-15% better than ARIMA
- Sharpe ratio: 0.8-1.2

**Success Criteria:**
- Must beat ARIMA
- Our multi-modal LSTM must beat this price-only LSTM by ≥5% MAE

---

### Baseline 3: Prophet (Secondary Classical Baseline)

**Model:** Facebook Prophet with default settings

**Rationale:**
- Robust to missing data and outliers
- Handles seasonality and trend automatically
- Widely used in industry (Meta, Uber, Airbnb)
- Good for longer-term forecasts (30-day, 90-day)
- Provides uncertainty intervals out-of-the-box

**Implementation Details:**
- **Library:** `prophet` (Facebook/Meta)
- **Configuration:**
  - Daily seasonality: enabled
  - Weekly seasonality: enabled
  - Yearly seasonality: enabled (for stocks with seasonal patterns)
  - Changepoint detection: automatic
- **Features:**
  - Univariate (close price only)
  - Optional: add volume as regressor
- **Forecast Horizon:** 7-day, 30-day

**Expected Performance:**
- 7-day MAPE: 5-10%
- Better than ARIMA for stocks with strong seasonal patterns (e.g., retail, agriculture)
- Likely underperforms LSTM on volatile tech stocks

**Success Criteria:**
- Useful for longer horizons (30+ days) where LSTM may struggle
- Ensemble potential: combine Prophet (trend) + LSTM (short-term)

---

### Summary: Forecasting Baselines

| Baseline | Type | Primary Use | Expected DA (1-day) | Implementation Time |
|----------|------|-------------|---------------------|---------------------|
| **ARIMA** | Classical | Primary benchmark | 50-52% | 2-3 days |
| **Simple LSTM** | Deep learning | NN baseline | 53-56% | 5-7 days |
| **Prophet** | Classical | Long-term + seasonal | N/A (regression) | 2-3 days |

**Implementation Order:** ARIMA → Simple LSTM → Prophet (if time permits)

---

## 2. Sentiment Analysis Module Baselines

### Baseline 1: FinBERT (Primary Baseline)

**Model:** Pre-trained FinBERT fine-tuned on financial sentiment

**Rationale:**
- State-of-the-art for financial sentiment (Araci 2019: F1=0.86)
- Domain-adapted BERT specifically for finance
- Outperforms generic BERT by 10-15% on financial text
- Publicly available pre-trained weights

**Implementation Details:**
- **Model:** `ProsusAI/finbert` (Hugging Face Transformers)
- **Fine-tuning Dataset:**
  - Option 1: Use pre-trained FinBERT as-is (zero-shot on Indian news)
  - Option 2: Fine-tune on labeled Indian financial news (if we label 500-1000 samples)
- **Classification:** 3-class (Positive, Neutral, Negative)
- **Input:** News headlines + first 128 tokens of article body
- **Hyperparameters:**
  - Max length: 128 tokens
  - Batch size: 16
  - Learning rate: 2e-5
  - Epochs: 3-5
- **Output:** Sentiment probabilities [P(pos), P(neu), P(neg)]

**Expected Performance:**
- F1 score: 0.80-0.86 (meets our pass threshold of 0.80)
- Precision/Recall: ≥0.78 per class

**Success Criteria:**
- Must achieve F1 ≥ 0.80 on held-out test set
- Our advanced models (if any) must beat this by ≥3% F1

---

### Baseline 2: Generic BERT (Secondary Baseline)

**Model:** BERT-base-uncased fine-tuned on financial data

**Rationale:**
- Shows benefit of domain adaptation (FinBERT vs. generic BERT)
- Simpler fallback if FinBERT underperforms on Indian market
- Establishes lower bound (expected F1 ~0.72-0.75)

**Implementation Details:**
- **Model:** `bert-base-uncased` (Hugging Face)
- **Fine-tuning:** Same setup as FinBERT
- **Dataset:** Same labeled financial news

**Expected Performance:**
- F1 score: 0.72-0.75 (10% worse than FinBERT)

**Success Criteria:**
- Demonstrates value of domain adaptation (FinBERT should beat by ≥5% F1)

---

### Baseline 3: VADER (Tertiary Baseline - Optional)

**Model:** VADER (Valence Aware Dictionary and sEntiment Reasoner)

**Rationale:**
- Extremely fast (rule-based, no training required)
- Useful for real-time applications
- Shows improvement of ML over lexicon-based methods

**Implementation Details:**
- **Library:** `vaderSentiment` (Python)
- **Input:** News text (headlines + body)
- **Output:** Compound score [-1, 1]
- **Threshold:**
  - Positive: compound ≥ 0.05
  - Neutral: -0.05 < compound < 0.05
  - Negative: compound ≤ -0.05

**Expected Performance:**
- F1 score: 0.55-0.65 (much worse than BERT)
- Fast: 1000 texts/second

**Success Criteria:**
- Establishes lower bound
- Useful for ablation studies (ML vs. rules)

---

### Summary: Sentiment Baselines

| Baseline | Type | Expected F1 | Training Time | Inference Speed |
|----------|------|-------------|---------------|-----------------|
| **FinBERT** | Transformer | 0.80-0.86 | 1-2 hours | 50 texts/sec |
| **Generic BERT** | Transformer | 0.72-0.75 | 1-2 hours | 50 texts/sec |
| **VADER** | Lexicon | 0.55-0.65 | 0 (rule-based) | 1000 texts/sec |

**Implementation Order:** FinBERT → Generic BERT (for comparison) → VADER (if time permits)

---

## 3. Growth Score Module Baselines

### Baseline 1: Random Forest Regressor (Primary Baseline)

**Model:** Random Forest trained on technical + fundamental features

**Rationale:**
- Widely used in Krauss et al. (2017) for stock prediction
- Handles non-linear relationships and feature interactions
- Interpretable (feature importance)
- Robust to outliers and missing data

**Implementation Details:**
- **Library:** `scikit-learn.ensemble.RandomForestRegressor`
- **Features (30-40 total):**
  - **Technical (15):**
    - SMA (5, 20, 50, 200 days)
    - EMA (12, 26 days)
    - RSI (14 days)
    - MACD (12, 26, 9)
    - Bollinger Bands (20 days)
    - ATR (14 days)
    - Volume ratios
  - **Fundamental (15):**
    - P/E ratio
    - P/B ratio
    - ROE, ROA
    - Debt-to-equity
    - Revenue growth (YoY, QoQ)
    - Earnings growth (YoY, QoQ)
    - Free cash flow
    - Dividend yield
  - **Macro (5):**
    - Nifty 50 returns (1M, 3M)
    - Sector index returns
    - VIX (volatility)
    - Interest rates (10Y G-Sec)
- **Target:** 60-day or 90-day forward returns
- **Hyperparameters:**
  - n_estimators: 100-500 (tune via grid search)
  - max_depth: 10-30
  - min_samples_split: 5-20
  - max_features: 'sqrt'
- **Training:** Rolling window (2 years train → predict next 60-90 days)

**Expected Performance:**
- Spearman ρ: 0.25-0.35 (between predicted score and realized returns)
- Top-10 precision: 60-70%

**Success Criteria:**
- Spearman ρ ≥ 0.30 (meets our pass threshold)
- Top-10 precision ≥ 70%

---

### Baseline 2: Gradient Boosting Trees (GBT) (Secondary Baseline)

**Model:** XGBoost or LightGBM

**Rationale:**
- Krauss et al. (2017) found GBT achieved best Sharpe ratio (5.2)
- Often outperforms Random Forest on tabular data
- Handles feature interactions and non-linearities

**Implementation Details:**
- **Library:** `xgboost.XGBRegressor` or `lightgbm.LGBMRegressor`
- **Features:** Same as Random Forest (30-40 features)
- **Hyperparameters:**
  - learning_rate: 0.01-0.1
  - n_estimators: 100-1000
  - max_depth: 5-10
  - subsample: 0.8
  - colsample_bytree: 0.8
- **Regularization:** L1/L2 to prevent overfitting

**Expected Performance:**
- Spearman ρ: 0.30-0.40 (slightly better than RF)
- Top-10 precision: 65-75%

**Success Criteria:**
- Should match or beat Random Forest
- Provides ensemble diversity (can combine RF + GBT)

---

### Baseline 3: Equal-Weight Portfolio (Tertiary Baseline)

**Model:** Assign equal scores to all stocks (no prediction)

**Rationale:**
- Simplest possible baseline
- Shows value of any scoring system
- Equal-weight portfolios often outperform market-cap weighted (low-cap effect)

**Implementation Details:**
- **Score:** All stocks get score = 1.0
- **Portfolio:** Equal allocation to all N stocks
- **Rebalance:** Monthly

**Expected Performance:**
- Spearman ρ: 0.0 (by design, no ranking)
- Top-10 precision: ~50% (random)
- Returns: Similar to Nifty 50 equal-weight index

**Success Criteria:**
- Any ML-based growth score must beat this significantly
- Provides sanity check

---

### Baseline 4: Momentum Strategy (Tertiary Baseline - Optional)

**Model:** Rank stocks by past 12-month returns (classic momentum factor)

**Rationale:**
- Well-known anomaly in finance (Jegadeesh & Titman 1993)
- Simple, interpretable
- Shows if ML adds value beyond simple momentum

**Implementation Details:**
- **Score:** 12-month cumulative return (skip most recent month to avoid reversal)
- **Ranking:** Sort stocks by score, long top decile
- **Rebalance:** Monthly

**Expected Performance:**
- Spearman ρ: 0.15-0.25 (momentum works but noisy)
- Top-10 precision: 55-65%
- Annualized excess return: 2-4% vs. market

**Success Criteria:**
- ML growth score should beat momentum by ≥5% precision

---

### Summary: Growth Score Baselines

| Baseline | Type | Expected Spearman ρ | Expected Top-10 Precision | Implementation Time |
|----------|------|---------------------|---------------------------|---------------------|
| **Random Forest** | ML | 0.25-0.35 | 60-70% | 3-5 days |
| **Gradient Boosting** | ML | 0.30-0.40 | 65-75% | 3-5 days |
| **Equal-Weight** | Naive | 0.0 | ~50% | 1 day |
| **Momentum** | Factor | 0.15-0.25 | 55-65% | 1 day |

**Implementation Order:** Random Forest → Equal-Weight (sanity check) → GBT → Momentum (if time permits)

---

## 4. Implementation Timeline (Weeks 1-2)

### Week 1: Classical Baselines
**Days 1-2:** ARIMA forecasting baseline
- Implement auto-ARIMA with grid search
- Run on 10-20 Nifty 50 stocks
- Compute MAE, RMSE, DA (1/3/7-day)
- Document results in `experiments/01_arima_baseline.md`

**Days 3-4:** Equal-weight growth score baseline
- Compute equal-weight portfolio returns
- Establish random baseline (Top-10 precision ~50%)
- Document results

**Day 5:** FinBERT sentiment baseline
- Load pre-trained FinBERT from Hugging Face
- Test zero-shot on sample Indian financial news (50-100 articles)
- Compute F1 score (if labeled data available)
- Document results

### Week 2: Machine Learning Baselines
**Days 1-3:** Simple LSTM forecasting
- Implement 2-layer LSTM with price-only features
- Train on same stocks as ARIMA
- Compare MAE, RMSE, DA vs. ARIMA
- Document results in `experiments/02_lstm_baseline.md`

**Days 4-5:** Random Forest growth score
- Implement RF with technical + fundamental features
- Compute Spearman ρ, Top-K precision
- Compare vs. equal-weight baseline
- Document results in `experiments/03_rf_growth_score.md`

**Buffer:** Fine-tune FinBERT if needed (optional, depends on zero-shot performance)

---

## 5. Success Criteria for Baseline Phase

By end of Week 2, we must have:

1. **Forecasting:**
   - ✅ ARIMA results on ≥10 stocks (7-day MAE documented)
   - ✅ Simple LSTM results on same stocks
   - ✅ LSTM beats ARIMA by ≥10% MAE (if not, debug before proceeding)

2. **Sentiment:**
   - ✅ FinBERT F1 ≥ 0.75 on test set (even if below target 0.80, acceptable for baseline)
   - ✅ Labeled test set of 200-500 Indian financial news articles

3. **Growth Score:**
   - ✅ Random Forest Spearman ρ ≥ 0.25
   - ✅ Top-10 precision ≥ 60%
   - ✅ Results statistically significant (p < 0.05)

4. **Documentation:**
   - ✅ All experiments reproducible (code + configs in Git)
   - ✅ Results tables in experiment logs
   - ✅ Comparison vs. acceptance thresholds from `metrics_and_evaluation.md`

**If any baseline fails to meet minimum criteria:** Spend Week 3 debugging before advancing to multi-modal models.

---

## 6. Code Structure

```
baselines/
├── forecasting/
│   ├── arima_baseline.py          # ARIMA implementation
│   ├── lstm_baseline.py           # Simple LSTM (price-only)
│   ├── prophet_baseline.py        # Prophet (optional)
│   └── configs/
│       ├── arima_config.yaml
│       └── lstm_config.yaml
├── sentiment/
│   ├── finbert_baseline.py        # FinBERT zero-shot / fine-tuned
│   ├── bert_baseline.py           # Generic BERT (for comparison)
│   ├── vader_baseline.py          # VADER (optional)
│   └── configs/
│       └── finbert_config.yaml
├── growth_score/
│   ├── random_forest_baseline.py  # RF with technical + fundamentals
│   ├── gradient_boosting_baseline.py  # XGBoost/LightGBM
│   ├── equal_weight_baseline.py   # Naive baseline
│   ├── momentum_baseline.py       # Momentum factor (optional)
│   └── configs/
│       ├── rf_config.yaml
│       └── xgb_config.yaml
└── utils/
    ├── data_loader.py             # Load stock prices, fundamentals, news
    ├── feature_engineering.py     # Technical indicators, normalization
    ├── evaluation.py              # MAE, RMSE, DA, F1, Spearman, etc.
    └── backtesting.py             # Walk-forward validation
```

---

## 7. Data Requirements

### For Forecasting Baselines:
- **Stock prices:** Daily OHLCV for Nifty 50/100 (2019-2024, ~5 years)
- **Source:** `yfinance`, Alpha Vantage, or NSE API
- **Format:** CSV with columns: [Date, Symbol, Open, High, Low, Close, Volume]

### For Sentiment Baselines:
- **News articles:** 500-1000 labeled (Positive/Neutral/Negative)
- **Source:** Economic Times, Moneycontrol, Financial Express (web scraping)
- **Format:** JSON with fields: [date, symbol, headline, body, sentiment_label]
- **Labeling:** Manual (team labels) or semi-supervised (FinBERT pseudo-labels + human review)

### For Growth Score Baselines:
- **Fundamentals:** Quarterly financial statements (2019-2024)
- **Source:** Screener.in, Tijori Finance API, or Yahoo Finance
- **Features:** P/E, P/B, ROE, ROA, debt-to-equity, revenue/earnings growth, FCF, dividend yield
- **Format:** CSV with columns: [Date, Symbol, PE, PB, ROE, ROA, ...]

---

## 8. Risk Mitigation

### Risk 1: ARIMA underperforms (MAE very high)
**Mitigation:**
- Try SARIMA (seasonal ARIMA) for stocks with seasonality
- Use ensemble ARIMA (different parameter sets, average predictions)
- Fallback to simple moving average if ARIMA fails

### Risk 2: FinBERT F1 < 0.75 on Indian financial news
**Mitigation:**
- Fine-tune on 500-1000 labeled Indian samples
- Try RoBERTa-based models or domain-adapted BERT (e.g., `bert-base-cased` fine-tuned on Indian corpus)
- Augment training data with translated Financial PhraseBank

### Risk 3: Random Forest Spearman ρ < 0.25 (growth score fails)
**Mitigation:**
- Check for data leakage (fundamentals available at prediction time?)
- Add more features (momentum, volatility, sector dummies)
- Try different target (rank stocks by % gain rather than absolute return)
- Increase training window (3 years instead of 2)

---

## 9. Next Steps After Baseline Phase

Once baselines are validated (Week 3):

1. **Multi-modal LSTM:** Add fundamentals + sentiment to LSTM (Week 3-4)
2. **Advanced Transformers:** Implement PatchTST or Transformer encoder-decoder (Week 5-6)
3. **Ensemble Growth Score:** Combine RF + GBT + momentum (Week 5)
4. **Event-driven Sentiment:** Extract events from news (optional, Week 7+)
5. **Dashboard MVP:** Visualize baseline predictions (Week 6-7)

---

## 10. Approval and Sign-off

**Baseline choices approved by:** [Supervisor Name]

**Acceptance Criteria:**
- ✅ All baselines are defensible (cited in bibliography)
- ✅ Implementation complexity is reasonable (2 weeks total)
- ✅ Performance expectations are realistic (based on literature)
- ✅ Code structure is modular and reusable

**Ready to proceed:** YES / NO

---

**Document Owner:** ML Engineering Team
**Last Updated:** 18 Nov 2025
