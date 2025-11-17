# Metrics and Evaluation Plan

**Project:** Multi-Modal Stock Intelligence Platform
**Version:** 1.0
**Date:** 18 Nov 2025
**Purpose:** Define quantitative success criteria, evaluation protocols, and acceptance thresholds for all model components.

---

## 1. Forecasting Module Metrics

### 1.1 Primary Metrics

#### Mean Absolute Error (MAE)
**Formula:**
```
MAE = (1/n) Σ|yᵢ - ŷᵢ|
```
where:
- `yᵢ` = actual price at time i
- `ŷᵢ` = predicted price at time i
- `n` = number of predictions

**Acceptance Threshold:**
- **Pass:** 7-day MAE reduction ≥ 15% vs. baseline ARIMA
- **Target:** 20% reduction for "strong pass"
- **Measured on:** Out-of-sample test set (last 20% of time-series data)

#### Root Mean Squared Error (RMSE)
**Formula:**
```
RMSE = √[(1/n) Σ(yᵢ - ŷᵢ)²]
```

**Acceptance Threshold:**
- **Pass:** 7-day RMSE reduction ≥ 12% vs. baseline ARIMA
- **Note:** RMSE penalizes large errors more heavily than MAE

#### Directional Accuracy (DA)
**Formula:**
```
DA = (1/n) Σ I(sign(Δyᵢ) = sign(Δŷᵢ))
```
where:
- `Δyᵢ = yᵢ - yᵢ₋₁` (actual price change)
- `Δŷᵢ = ŷᵢ - yᵢ₋₁` (predicted price change)
- `I(·)` = indicator function (1 if true, 0 if false)

**Acceptance Thresholds:**
- **1-day ahead:** DA ≥ 55% (p < 0.05 via binomial test)
- **3-day ahead:** DA ≥ 53%
- **7-day ahead:** DA ≥ 52%
- **Baseline:** Random guess = 50%

### 1.2 Trading Performance Metric

#### Sharpe Ratio (Backtested Strategy)
**Formula:**
```
Sharpe = (Rₚ - Rբ) / σₚ
```
where:
- `Rₚ` = mean portfolio return
- `Rբ` = risk-free rate (use 10-year Indian G-Sec yield, ~7.2%)
- `σₚ` = standard deviation of portfolio returns

**Acceptance Threshold:**
- **Pass:** Annualized Sharpe ratio ≥ 1.0 on backtest
- **Target:** Sharpe ratio ≥ 1.5 for "strong pass"
- **Benchmark:** Compare against buy-and-hold Nifty 50

### 1.3 Additional Metrics

#### Mean Absolute Percentage Error (MAPE)
**Formula:**
```
MAPE = (100/n) Σ|(yᵢ - ŷᵢ) / yᵢ|
```

**Threshold:** MAPE ≤ 5% for 7-day forecasts

#### Maximum Drawdown (MDD)
**Formula:**
```
MDD = max[1 - (Vₜ / max(V₀, ..., Vₜ₋₁))]
```
where `Vₜ` = portfolio value at time t

**Threshold:** MDD ≤ 20% during backtest period

---

## 2. Sentiment Analysis Module Metrics

### 2.1 Classification Metrics

#### Precision
**Formula:**
```
Precision = TP / (TP + FP)
```
where:
- `TP` = True Positives
- `FP` = False Positives

**Acceptance Threshold:**
- **Per-class precision ≥ 0.75** (for positive, negative, neutral)
- **Macro-averaged precision ≥ 0.78**

#### Recall (Sensitivity)
**Formula:**
```
Recall = TP / (TP + FN)
```
where `FN` = False Negatives

**Acceptance Threshold:**
- **Per-class recall ≥ 0.75**
- **Macro-averaged recall ≥ 0.78**

#### F1 Score
**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Acceptance Threshold:**
- **Pass:** Macro F1 ≥ 0.80 on financial sentiment test set
- **Target:** F1 ≥ 0.85 for "strong pass"
- **Baseline:** Generic BERT on financial text (expected F1 ~0.72)

### 2.2 Multi-class Evaluation

#### Confusion Matrix
Required for 3-class sentiment (Positive, Neutral, Negative):

```
                  Predicted
              Pos   Neu   Neg
    Actual Pos [ ]   [ ]   [ ]
           Neu [ ]   [ ]   [ ]
           Neg [ ]   [ ]   [ ]
```

**Threshold:** No single off-diagonal cell > 15% of class total

#### Weighted F1 Score
**Formula:**
```
F1_weighted = Σ(wᵢ × F1ᵢ)
```
where `wᵢ = nᵢ / n` (class proportion)

**Threshold:** Weighted F1 ≥ 0.82 (to handle class imbalance)

### 2.3 Sentiment-Price Correlation

#### Lagged Correlation
**Formula:**
```
ρ_lag_k = corr(sentiment_t, return_{t+k})
```
for lag k ∈ {1, 2, 3, 5, 7} days

**Acceptance Threshold:**
- **Pass:** |ρ| ≥ 0.15 for at least one lag k (p < 0.05)
- **Target:** |ρ| ≥ 0.25 for strong predictive power

---

## 3. Growth Score Module Metrics

### 3.1 Ranking Quality

#### Spearman Rank Correlation
**Formula:**
```
ρ_s = 1 - [6 Σdᵢ²] / [n(n² - 1)]
```
where:
- `dᵢ` = difference between rank(score) and rank(realized return)
- `n` = number of stocks

**Acceptance Threshold:**
- **Pass:** ρ_s ≥ 0.30 (p < 0.01) between growth scores and 60-90 day realized returns
- **Target:** ρ_s ≥ 0.45 for "strong pass"

#### Top-K Precision
**Formula:**
```
Precision@K = (# of top-K stocks that outperformed benchmark) / K
```

**Acceptance Thresholds:**
- **Top-10 precision ≥ 70%** (7 out of 10 stocks beat benchmark)
- **Top-20 precision ≥ 65%**
- **Top-50 precision ≥ 60%**

### 3.2 Portfolio Performance

#### Top Decile Excess Return
**Formula:**
```
Excess Return = R_top_decile - R_benchmark
```
where:
- `R_top_decile` = cumulative return of top 10% stocks by growth score
- `R_benchmark` = Nifty 50 or sector-specific index return

**Acceptance Threshold:**
- **Pass:** Excess return ≥ 3% over 60-90 day horizon (statistically significant via t-test, p < 0.05)
- **Target:** Excess return ≥ 5%

#### Information Ratio
**Formula:**
```
IR = (R_portfolio - R_benchmark) / σ_tracking_error
```

**Threshold:** IR ≥ 0.5

---

## 4. Backtesting Protocol

### 4.1 Walk-Forward Validation

**Protocol:**
1. **Training Window:** Rolling 2-year (504 trading days) window
2. **Validation Window:** 3 months (63 trading days)
3. **Test Window:** 1 month (21 trading days)
4. **Step Size:** Advance by 1 month each iteration

**Timeline:**
```
Train: [T-730, T-63] → Validate: [T-62, T-1] → Test: [T, T+20]
Then advance T by 21 days and repeat
```

**Minimum Iterations:** 12 test periods (covering 1 year of out-of-sample data)

### 4.2 Lookahead Prevention Rules

**Strict Rules:**
1. **No future data:** Features at time `t` can only use data available up to `t-1`
2. **News timestamp:** Use publication time, not ingestion time
3. **Fundamentals lag:** Quarterly reports available only 45 days after quarter-end
4. **Technical indicators:** Use close prices from previous day
5. **Sentiment:** News from day `t` can only influence predictions for `t+1` onwards

**Automated Checks:**
- All feature timestamps < prediction timestamp
- No survivorship bias (use point-in-time index constituents)
- No backfilled data (use raw data as received)

### 4.3 Data Leakage Prevention

**Prohibited Practices:**
1. ❌ Normalizing/scaling with test set statistics
2. ❌ Feature selection using full dataset
3. ❌ Hyperparameter tuning on test set
4. ❌ Using forward-filled prices beyond last available price
5. ❌ Training on stocks delisted during test period

**Required Practices:**
1. ✓ Separate preprocessing pipelines for train/val/test
2. ✓ Fit scalers only on training data
3. ✓ Preserve temporal order in all splits
4. ✓ Use nested cross-validation for hyperparameter search
5. ✓ Document all data sources and their availability lags

### 4.4 Train/Validation/Test Split

**Temporal Split (Out-of-Sample):**
- **Training:** 60% of data (oldest)
- **Validation:** 20% (middle)
- **Test:** 20% (most recent)

**Example for 5 years of data (2020-2024):**
- Train: Jan 2020 - Dec 2022
- Validation: Jan 2023 - Jun 2023
- Test: Jul 2023 - Dec 2024

**Cross-Stock Validation:**
- For growth score: train on 80% of stocks, test on held-out 20%
- Stratify by sector to ensure representation

---

## 5. Statistical Significance Testing

### 5.1 Hypothesis Tests

#### Model Comparison (MAE reduction)
- **Null Hypothesis (H₀):** MAE_multimodal ≥ MAE_baseline
- **Alternative (H₁):** MAE_multimodal < MAE_baseline
- **Test:** Paired t-test on per-stock MAE differences
- **Threshold:** p < 0.05 (reject H₀)

#### Directional Accuracy
- **Null Hypothesis:** DA = 0.50 (random)
- **Test:** Binomial test
- **Threshold:** p < 0.05

#### Spearman Correlation
- **Null Hypothesis:** ρ_s = 0
- **Test:** Spearman correlation test
- **Threshold:** p < 0.01

### 5.2 Multiple Comparison Correction

When testing multiple stocks/timeframes:
- **Apply Bonferroni correction:** α_adjusted = α / m
- Where m = number of comparisons
- Or use False Discovery Rate (FDR) control via Benjamini-Hochberg

---

## 6. Production Readiness Metrics

### 6.1 Latency

**Inference Time (per symbol):**
- **Pass:** ≤ 300ms at p95 (95th percentile)
- **Target:** ≤ 200ms at p95

**Batch Processing (50 symbols):**
- **Pass:** ≤ 10 seconds

### 6.2 Throughput

**Minimum Requirements:**
- Process 500 stocks per minute
- Handle 100 concurrent sentiment analyses

### 6.3 Pipeline Reliability

**Data Pipeline Uptime:**
- **Pass:** ≥ 99.0% during evaluation windows
- **Target:** ≥ 99.5%

**Model Availability:**
- **Pass:** ≥ 99.5% uptime for inference API

---

## 7. Acceptance Criteria Summary

### 7.1 Pass/Fail Thresholds

| Module | Metric | Pass | Strong Pass |
|--------|--------|------|-------------|
| **Forecasting** | 7-day MAE reduction | ≥ 15% | ≥ 20% |
| | Directional Accuracy (1-day) | ≥ 55% | ≥ 58% |
| | Sharpe Ratio (backtest) | ≥ 1.0 | ≥ 1.5 |
| **Sentiment** | Macro F1 Score | ≥ 0.80 | ≥ 0.85 |
| | Sentiment-Price Correlation | \|ρ\| ≥ 0.15 | \|ρ\| ≥ 0.25 |
| **Growth Score** | Spearman ρ_s | ≥ 0.30 | ≥ 0.45 |
| | Top-10 Precision | ≥ 70% | ≥ 80% |
| | Excess Return (60-90d) | ≥ 3% | ≥ 5% |
| **Production** | Inference Latency (p95) | ≤ 300ms | ≤ 200ms |
| | Pipeline Uptime | ≥ 99.0% | ≥ 99.5% |

### 7.2 Experiment Acceptance Process

**For each experiment:**
1. Document hypothesis and expected outcome
2. Run with predefined train/val/test splits
3. Report all metrics in table above
4. Include statistical significance tests
5. Compare against baselines (ARIMA, Buy-and-Hold, Generic BERT)
6. **Accept if:** All "Pass" thresholds met + statistical significance achieved
7. **Reject if:** Any critical metric below pass threshold or p-value > 0.05

### 7.3 Baseline Models

**Required Comparisons:**
- **Forecasting:** ARIMA(1,1,1), Simple Moving Average, Exponential Smoothing
- **Sentiment:** Generic BERT, VADER, TextBlob
- **Growth Score:** Equal-weight portfolio, Market-cap weighted, Momentum (12-month)

---

## 8. Reporting Requirements

### 8.1 Experiment Reports Must Include

1. **Metrics Table:** All metrics from Section 7.1
2. **Confusion Matrix:** For sentiment module
3. **Error Distribution Plot:** MAE/RMSE by stock and time
4. **Backtest Equity Curve:** Cumulative returns over time
5. **Statistical Tests:** p-values for all comparisons
6. **Failure Analysis:** Cases where model performed poorly
7. **Computational Cost:** Training time, inference time, memory usage

### 8.2 Version Control

- Tag each experiment with Git commit hash
- Store configs, hyperparameters, and random seeds
- Enable full reproducibility

---

## 9. Data Leakage Checklist

Before running any experiment, verify:

- [ ] All features use only past data (t-1 and earlier)
- [ ] Scalers/normalizers fitted only on training set
- [ ] No test set data used in feature selection
- [ ] No hyperparameter tuning on test set
- [ ] Temporal ordering preserved in all splits
- [ ] Survivorship bias eliminated (point-in-time constituents)
- [ ] News timestamps checked (publication time, not scrape time)
- [ ] Fundamentals released with appropriate lag (45 days post-quarter)
- [ ] No forward-filling beyond last known value
- [ ] Cross-validation folds respect temporal order (no shuffling)

**Sign-off required from:** ML Lead and Data Engineer before each major experiment.

---

## 10. Review and Updates

**This document will be reviewed:**
- After initial baseline experiments (Week 2)
- Mid-project (Week 6)
- Before final evaluation (Week 10)

**Change Log:**
- v1.0 (18 Nov 2025): Initial metrics specification

---

**Document Owner:** ML Engineering Team
**Approved By:** [Supervisor Name]
**Next Review:** Week 2 (after baseline experiments)
