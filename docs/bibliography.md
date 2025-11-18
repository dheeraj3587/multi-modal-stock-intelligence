# Annotated Bibliography

**Project:** Multi-Modal Stock Intelligence Platform
**Version:** 1.0
**Date:** 18 Nov 2025
**Purpose:** Survey of relevant prior work for time-series forecasting, sentiment analysis, and growth scoring in financial markets.

---

## 1. Time-Series Forecasting: LSTM and Deep Learning

### 1.1 Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research*, 270(2), 654-669.

**Summary:**
Applied LSTM networks to predict daily directional movements (up/down) for constituents of the S&P-500 from 1992–2015. Compared LSTM performance against memory-free classifiers (random forest, deep neural networks, logistic regression).

**Dataset:**
- S&P-500 constituent stocks (1992–2015, ~23 years)
- Daily returns and technical indicators
- ~500 stocks, cross-sectional prediction setup

**Key Performance Claims:**
- LSTM achieved **0.559 directional accuracy** (vs. 0.50 random baseline)
- Economically significant returns in backtesting after transaction costs
- Outperformed random forest (0.529), deep NN (0.531), and logistic regression (0.524)

**What We'll Reuse/Compare:**
- **Baseline LSTM architecture** for our forecasting module
- Use as comparison for our multi-modal LSTM+fundamentals+sentiment model
- Replicate directional accuracy metric (target: >55% for 1-day ahead)
- Similar cross-sectional setup for Indian stocks (Nifty 50/100)

**Limitations:**
- Price-only features; no fundamentals or sentiment
- Simpler LSTM architecture (pre-Transformer era)

---

### 1.2 Krauss, C., Do, X. A., & Huck, N. (2017). Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500. *European Journal of Operational Research*, 259(2), 689-702.

**Summary:**
Systematic comparison of machine-learning models (DNN, gradient-boosted trees, random forest) for one-day-ahead trading signals across S&P-500 stocks. Evaluated ensemble combinations and economic backtests.

**Dataset:**
- S&P-500 stocks (1992–2015)
- Lagged returns and technical features
- Rolling window training

**Key Performance Claims:**
- **Gradient Boosting Trees (GBT)** achieved best Sharpe ratio: **5.2 annualized**
- Ensemble (RF + DNN + GBT) improved robustness
- All ML methods outperformed linear baselines significantly

**What We'll Reuse/Compare:**
- **Random Forest and GBT baselines** for growth score module
- Ensemble strategy (combine multiple weak learners)
- Sharpe ratio as primary trading performance metric
- Economic backtest methodology

**Limitations:**
- No external data (news, fundamentals, macroeconomic indicators)
- Focus on 1-day prediction only

---

## 2. Time-Series Forecasting: Modern Transformers

### 2.1 Nie, Y., et al. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. *ICLR 2023*.

**Paper:** PatchTST (Patching Time Series Transformer)

**Summary:**
State-of-the-art Transformer architecture for time-series forecasting. Uses patching (segmenting time series into subseries) and channel-independence to improve efficiency and accuracy. Pre-training on large time-series datasets enables transfer learning.

**Dataset:**
- Multiple benchmark datasets: Electricity, Traffic, Weather, ETT (Electricity Transformer Temperature)
- Multivariate time-series (up to 862 channels)
- Forecasting horizons: 96, 192, 336, 720 steps

**Key Performance Claims:**
- **Reduced MSE by 13-21%** vs. prior SOTA (Autoformer, FEDformer)
- 2-5× faster training than full-attention Transformers
- Transfer learning improves performance on small datasets

**What We'll Reuse/Compare:**
- **Advanced baseline for multi-step forecasting** (7-day, 30-day horizons)
- Patching technique to handle long sequences efficiently
- Pre-training strategy if we have multi-stock data
- Compare against our LSTM/GRU models

**Limitations:**
- Designed for continuous time-series; may need adaptation for irregular financial data (gaps, halts)
- No built-in support for exogenous features (fundamentals, sentiment)

---

### 2.2 Google Research (2024). TimesFM: Time Series Foundation Model. *ICML 2024*.

**Summary:**
Pre-trained foundation model for zero-shot time-series forecasting. Trained on 100 billion real-world time points from Google's proprietary datasets. Supports variable-length forecasting without fine-tuning.

**Dataset:**
- 100 billion time points from diverse domains (web traffic, retail, finance)
- Zero-shot evaluation on public benchmarks (M4, M5 competitions)

**Key Performance Claims:**
- **Competitive with task-specific models** on zero-shot forecasting
- Outperforms ARIMA, Prophet, and simple neural baselines
- No fine-tuning required for new datasets

**What We'll Reuse/Compare:**
- **Strong zero-shot baseline** for stock price forecasting
- Benchmark our models against a foundation model (if weights are accessible)
- Evaluate if domain-specific training (finance) beats general pre-training

**Limitations:**
- Proprietary training data (not fully reproducible)
- Open-source weights available but limited documentation
- Unclear performance on high-frequency financial data

---

### 2.3 Li, Y., et al. (2022). Stock market index prediction using a deep Transformer model. *Expert Systems with Applications*, 208, 118128.

**Summary:**
Applied Transformer encoder-decoder architecture to predict stock index prices (Shanghai Composite, S&P 500). Used multi-head attention to capture temporal dependencies in price data.

**Dataset:**
- Shanghai Composite Index (2000–2020)
- S&P 500 Index (2000–2020)
- Daily OHLCV (Open, High, Low, Close, Volume)

**Key Performance Claims:**
- **RMSE reduction of 12-18%** vs. LSTM on 10-day forecasts
- Better long-term dependencies than RNN/LSTM
- Attention weights provide interpretability

**What We'll Reuse/Compare:**
- Transformer architecture for price forecasting
- Compare against LSTM baseline
- Attention mechanism for interpretability (which past days matter most)

**Limitations:**
- Index-level prediction (single asset), not multi-stock
- No fundamentals or sentiment integration

---

## 3. Sentiment Analysis and Event-Driven Models

### 3.1 Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. *arXiv:1908.10063*.

**Summary:**
BERT-based model fine-tuned on financial communication (earnings calls, analyst reports, financial news). Achieves state-of-the-art performance on financial sentiment classification tasks.

**Dataset:**
- **Financial PhraseBank** (4,840 sentences labeled positive/negative/neutral)
- **TRC2-financial** (5,000 sentences from financial news)
- Fine-tuned from BERT-base-uncased

**Key Performance Claims:**
- **F1 score: 0.86** on Financial PhraseBank (vs. 0.75 for generic BERT)
- **Accuracy: 0.87** on financial sentiment tasks
- Domain adaptation significantly outperforms general-purpose models

**What We'll Reuse/Compare:**
- **Primary sentiment baseline** for our sentiment module
- Pre-trained FinBERT as starting point (fine-tune on Indian financial news if needed)
- Target F1 ≥ 0.80 (our threshold); FinBERT sets high bar at 0.86

**Limitations:**
- Trained on English financial text (may need adaptation for Indian market idioms)
- Sentence-level classification (not document-level)

---

### 3.2 Ding, X., et al. (2015). Deep Learning for Event-Driven Stock Prediction. *IJCAI 2015*, 2327-2333.

**Summary:**
Introduced event-driven deep learning approach: extract structured events from news text, represent events as dense vectors (neural tensor networks), model short- and long-term influence on stock prices.

**Dataset:**
- **Reuters news** + **S&P 500 stock prices** (2006–2013)
- Event extraction: subject-predicate-object tuples (e.g., "Apple-release-iPhone")
- 10 million events extracted

**Key Performance Claims:**
- **6% accuracy improvement** over bag-of-words baseline
- **3-4% improvement** over generic sentiment models
- Event embeddings capture financial relationships (e.g., "acquisition" → positive impact)

**What We'll Reuse/Compare:**
- Event extraction pipeline (NER + relation extraction)
- Compare event-based features vs. pure sentiment scores
- Neural tensor network for event embeddings (if time permits)

**Limitations:**
- Complex pipeline (requires NLP expertise and labeled event data)
- Computational cost for real-time event extraction
- 2015 architecture (pre-Transformer)

---

## 4. Multi-Modal and Ensemble Approaches

### 4.1 Zhang, Y., et al. (2024). Deep Convolutional Transformer Network for Stock Movement Prediction (DCT). *Electronics*, 13(21), 4225.

**Summary:**
Combines CNNs (to extract local patterns) and Transformers (for long-range dependencies) for stock movement prediction. Uses multi-modal inputs: price, volume, technical indicators.

**Dataset:**
- Chinese stock market (CSI 300 constituents, 2010–2022)
- Daily OHLCV + 20 technical indicators (RSI, MACD, Bollinger Bands)

**Key Performance Claims:**
- **Accuracy: 58.3%** for next-day movement prediction
- **MCC (Matthews Correlation Coefficient): 0.167** (vs. 0.12 for LSTM)
- Hybrid CNN-Transformer outperforms pure CNN or pure Transformer

**What We'll Reuse/Compare:**
- Hybrid architecture inspiration (CNN for local patterns + Transformer for global)
- Multi-modal input strategy (price + technical indicators)
- Compare against our LSTM and Transformer baselines

**Limitations:**
- Still price-only features (no fundamentals or sentiment)
- Chinese market may have different dynamics than Indian market

---

### 4.2 Boyle, P., & Kalita, J. (2023). Spatiotemporal Transformer for stock movement prediction. *arXiv:2305.03835*.

**Summary:**
Uses spatiotemporal attention to model both temporal dependencies (time) and cross-stock relationships (space). Predicts stock movements by learning interactions between stocks in the same sector.

**Dataset:**
- S&P 500 stocks (2010–2020)
- Daily returns + sector embeddings
- Graph structure: stocks as nodes, sector relationships as edges

**Key Performance Claims:**
- **Accuracy: 56.2%** for 1-day movement prediction
- **Improves sector-level predictions** by 8% over stock-only models
- Captures contagion effects (e.g., tech stock correlations)

**What We'll Reuse/Compare:**
- Cross-stock attention mechanism (learn from similar stocks)
- Sector embeddings to enrich single-stock models
- Compare against our independent per-stock models

**Limitations:**
- Requires high-quality sector/graph data
- Computational complexity for large stock universes

---

## 5. Classical Baselines (ARIMA, Prophet)

### 5.1 Box, G. E. P., & Jenkins, G. M. (1976). Time Series Analysis: Forecasting and Control. *Holden-Day*.

**Model:** ARIMA (AutoRegressive Integrated Moving Average)

**Summary:**
Classical statistical method for time-series forecasting. Models time series as linear combination of past values, past errors, and differencing for stationarity.

**Performance:**
- Widely used industry baseline for financial forecasting
- Typical MAPE: 5-10% for short-term stock predictions (domain-dependent)
- Struggles with non-linear patterns and volatility

**What We'll Reuse/Compare:**
- **Primary baseline for forecasting module**
- ARIMA(1,1,1) or auto-tuned ARIMA via grid search
- Target: beat ARIMA by ≥15% MAE reduction

---

### 5.2 Taylor, S. J., & Letham, B. (2018). Forecasting at Scale. *The American Statistician*, 72(1), 37-45.

**Model:** Prophet (Facebook/Meta)

**Summary:**
Open-source forecasting tool designed for business time series. Decomposes series into trend, seasonality, and holidays using additive regression model.

**Performance:**
- Handles missing data and outliers robustly
- Works well for data with strong seasonal patterns
- Median MAPE: 5-15% on Facebook internal datasets

**What We'll Reuse/Compare:**
- **Secondary baseline** for forecasting (especially for longer horizons)
- Compare against ARIMA and LSTM
- Useful for stocks with strong seasonal patterns (e.g., retail, agriculture)

---

## 6. Growth Scoring and Factor Models

### 6.1 Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1-22.

**Summary:**
Five-factor model for expected stock returns: market risk, size, value, profitability, investment. Widely used in quantitative finance for portfolio construction and risk assessment.

**Performance:**
- Explains ~90% of cross-sectional return variation
- Outperforms CAPM and three-factor models

**What We'll Reuse/Compare:**
- **Feature engineering for growth score**: use Fama-French factors as inputs
- Benchmark growth score rankings against factor-based portfolios
- Compare ML-based growth score vs. linear factor model

---

## Summary Table

| **Paper** | **Domain** | **Dataset** | **Key Metric** | **Our Use** |
|-----------|-----------|-------------|----------------|-------------|
| Fischer & Krauss (2018) | LSTM forecasting | S&P 500, 1992-2015 | DA: 55.9% | LSTM baseline |
| Krauss et al. (2017) | Ensemble ML | S&P 500, 1992-2015 | Sharpe: 5.2 | GBT/RF growth score baseline |
| PatchTST (2023) | Transformer forecasting | Electricity, Traffic, ETT | MSE -13-21% | Advanced forecasting baseline |
| TimesFM (2024) | Foundation model | 100B time points | Zero-shot SOTA | Zero-shot baseline (if accessible) |
| Li et al. (2022) | Transformer | Shanghai/S&P 500, 2000-2020 | RMSE -12-18% | Transformer architecture |
| FinBERT (2019) | Sentiment | Financial PhraseBank | F1: 0.86 | Sentiment baseline |
| Ding et al. (2015) | Event-driven | Reuters + S&P 500, 2006-2013 | Acc +6% | Event extraction (optional) |
| DCT (2024) | Hybrid CNN-Transformer | CSI 300, 2010-2022 | Acc: 58.3% | Hybrid architecture idea |
| Boyle & Kalita (2023) | Spatiotemporal | S&P 500, 2010-2020 | Acc: 56.2% | Cross-stock attention |
| ARIMA (1976) | Classical | N/A | Industry standard | Primary baseline |
| Prophet (2018) | Additive model | Business time series | MAPE: 5-15% | Secondary baseline |
| Fama-French (2015) | Factor model | US equities | R²: ~90% | Growth score feature engineering |

---

## References

1. Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research*, 270(2), 654-669.

2. Krauss, C., Do, X. A., & Huck, N. (2017). Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500. *European Journal of Operational Research*, 259(2), 689-702.

3. Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. *ICLR 2023*.

4. Das, A., Kong, W., Leach, A., Mathur, S., Sen, R., & Yu, R. (2024). A decoder-only foundation model for time-series forecasting. *ICML 2024*. [TimesFM]

5. Li, Y., Luo, Y., & Cai, Y. (2022). Stock market index prediction using a deep Transformer model. *Expert Systems with Applications*, 208, 118128.

6. Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. *arXiv:1908.10063*.

7. Ding, X., Zhang, Y., Liu, T., & Duan, J. (2015). Deep Learning for Event-Driven Stock Prediction. *IJCAI 2015*, 2327-2333.

8. Zhang, Y., Li, J., & Chen, H. (2024). Deep Convolutional Transformer Network for Stock Movement Prediction. *Electronics*, 13(21), 4225.

9. Boyle, P., & Kalita, J. (2023). Spatiotemporal Transformer for stock movement prediction. *arXiv:2305.03835*.

10. Box, G. E. P., & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control*. Holden-Day.

11. Taylor, S. J., & Letham, B. (2018). Forecasting at Scale. *The American Statistician*, 72(1), 37-45.

12. Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1-22.

---

**Document Owner:** ML Research Team
**Next Steps:** Implement baselines in `baseline_plan.md`
**Last Updated:** 18 Nov 2025
