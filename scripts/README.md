# Data Ingestion Scripts

This directory contains CLI scripts for fetching financial data from various sources as part of Phase 1 of the AI-Powered Growth Stock Prediction Engine project.

## Overview

The data ingestion pipeline consists of five specialized scripts that fetch data from different sources and store it in date-partitioned directories under `data/raw/`. These scripts are designed to run independently or as part of automated workflows (cron jobs, Airflow DAGs) to maintain up-to-date datasets for model training and inference.

### Scripts

1. **`fetch_historical_prices.py`** - Historical OHLCV data from Yahoo Finance via yfinance
2. **`fetch_fundamentals.py`** - Company fundamental metrics from Yahoo Finance / Alpha Vantage
3. **`fetch_news.py`** - Financial news articles from NewsAPI
4. **`fetch_social_sentiment.py`** - Social sentiment data from StockTwits
5. **`fetch_macro_data.py`** - Macroeconomic indicators from World Bank API

## Prerequisites

### Environment Setup

1. **Create `.env` file** from `.env.template`:
   ```bash
   cp .env.template .env
   ```

2. **Configure API keys** in `.env`:
   - `NEWSAPI_KEY` - Required for news data (get from https://newsapi.org/)
   - `ALPHA_VANTAGE_KEY` - Optional for fundamentals (get from https://www.alphavantage.co/)
   - `STOCKTWITS_ACCESS_TOKEN` - Required for social sentiment (get from https://stocktwits.com/developers)
   - `WORLD_BANK_API_KEY` - Optional, World Bank API is public

### Dependencies

All required Python packages are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `yfinance` - Yahoo Finance data
- `newsapi-python` - NewsAPI client
- `requests` - HTTP requests for StockTwits and World Bank
- `pandas` - Data manipulation
- `loguru` - Structured logging
- `python-dotenv` - Environment variable management

## Usage Examples

### 1. Historical Prices

**Fetch single ticker for a date range:**
```bash
python scripts/fetch_historical_prices.py \
  --ticker RELIANCE.NS \
  --start 2020-01-01 \
  --end 2024-12-31
```

**Batch process multiple tickers:**
```bash
# Create tickers.csv with 'ticker' column
echo "ticker" > tickers.csv
echo "RELIANCE.NS" >> tickers.csv
echo "TCS.NS" >> tickers.csv
echo "INFY.NS" >> tickers.csv

python scripts/fetch_historical_prices.py \
  --ticker-file tickers.csv \
  --start 2020-01-01 \
  --end 2024-12-31
```

**Incremental update (fetch only new data):**
```bash
python scripts/fetch_historical_prices.py \
  --ticker RELIANCE.NS \
  --incremental
```

**Weekly or monthly intervals:**
```bash
python scripts/fetch_historical_prices.py \
  --ticker RELIANCE.NS \
  --start 2020-01-01 \
  --end 2024-12-31 \
  --interval 1wk
```

### 2. Company Fundamentals

**Fetch from Yahoo Finance:**
```bash
python scripts/fetch_fundamentals.py \
  --ticker RELIANCE.NS \
  --source yahoo
```

**Fetch from Alpha Vantage:**
```bash
python scripts/fetch_fundamentals.py \
  --ticker RELIANCE.NS \
  --source alphavantage
```

**Include quarterly financial statements:**
```bash
python scripts/fetch_fundamentals.py \
  --ticker RELIANCE.NS \
  --source yahoo \
  --quarterly
```

**Batch processing:**
```bash
python scripts/fetch_fundamentals.py \
  --ticker-file tickers.csv \
  --source yahoo \
  --quarterly
```

### 3. News Articles

**Fetch news for a company (last 7 days):**
```bash
python scripts/fetch_news.py \
  --ticker RELIANCE \
  --days 7
```

**Custom search query:**
```bash
python scripts/fetch_news.py \
  --query "Reliance Industries earnings" \
  --days 30
```

**Filter by specific news sources:**
```bash
python scripts/fetch_news.py \
  --ticker TCS \
  --sources "economic-times,business-standard" \
  --days 14
```

**With language and country filters:**
```bash
python scripts/fetch_news.py \
  --query "stock market" \
  --country in \
  --language en \
  --days 7
```

### 4. Social Sentiment

**Fetch latest StockTwits messages:**
```bash
python scripts/fetch_social_sentiment.py \
  --ticker RELIANCE
```

**Fetch more messages with pagination:**
```bash
python scripts/fetch_social_sentiment.py \
  --ticker TCS \
  --limit 100
```

**Incremental update (fetch only new messages):**
```bash
# Get message ID from previous run's output
python scripts/fetch_social_sentiment.py \
  --ticker INFY \
  --since-id 12345678
```

### 5. Macroeconomic Indicators

**Fetch specific indicators for India:**
```bash
python scripts/fetch_macro_data.py \
  --country IND \
  --indicators NY.GDP.MKTP.KD.ZG,FP.CPI.TOTL.ZG \
  --start-year 2010
```

**Use preset indicator sets:**
```bash
# Growth indicators
python scripts/fetch_macro_data.py \
  --country IND \
  --preset growth

# Full set of 10+ key indicators
python scripts/fetch_macro_data.py \
  --country IND \
  --preset full \
  --start-year 2000
```

**Fetch for multiple countries (comparative analysis):**
```bash
python scripts/fetch_macro_data.py \
  --country IND,USA,CHN \
  --preset full \
  --start-year 2010
```

**Available presets:**
- `growth` - GDP and investment indicators
- `inflation` - Inflation and price indices
- `monetary` - Interest rates and money supply
- `trade` - Exports, imports, current account
- `labor` - Unemployment and labor force
- `full` - Comprehensive set of 10+ key indicators

## Data Storage

All scripts use **timestamp-based partitioning** to organize data in `data/raw/` subdirectories:

```
data/raw/
├── prices/
│   └── 2024-11-18/
│       ├── RELIANCE_NS.csv
│       └── TCS_NS.csv
├── fundamentals/
│   └── 2024-11-18/
│       ├── RELIANCE_NS.json
│       └── TCS_NS.json
├── news/
│   └── 2024-11-18/
│       ├── RELIANCE_news.json
│       └── TCS_news.json
├── social/
│   └── 2024-11-18/
│       ├── RELIANCE_stocktwits.json
│       └── TCS_stocktwits.json
└── macro/
    └── 2024-11-18/
        ├── IND_macro.csv
        └── USA_macro.csv
```

### File Formats

- **CSV** - Used for tabular data (prices, macro indicators)
- **JSON** - Used for nested/unstructured data (fundamentals, news, social sentiment)

### Partitioning Benefits

- **Version control** - Track data snapshots over time
- **Incremental updates** - Easily identify new vs. existing data
- **Audit trail** - Know exactly when data was fetched
- **Parallel processing** - Multiple scripts can run simultaneously

## Error Handling

All scripts implement robust error handling:

### Retry Logic
- **Exponential backoff** - Automatic retries with increasing delays
- **Max retries** - Configurable retry limits (default: 3)
- **Transient error handling** - Recovers from network timeouts, 5xx errors

### Rate Limiting
- **NewsAPI** - 100 requests/day (free tier)
- **StockTwits** - 200 requests/hour (authenticated)
- **Token bucket algorithm** - Smooth request distribution
- **429 error handling** - Respects API rate limit responses

### Logging
- **Structured logs** - JSON format for production
- **Log rotation** - Automatic rotation at 10 MB
- **Log retention** - 30-day retention policy
- **Multiple outputs** - Console (colored) + file (structured)

Logs are stored in `logs/app.log` and `logs/app.json`.

## Scheduling

### Cron Jobs

Example cron schedule for daily data updates:

```bash
# Edit crontab
crontab -e

# Add these lines for daily execution at 6 AM
0 6 * * * cd /path/to/project && python scripts/fetch_historical_prices.py --ticker RELIANCE.NS --incremental
0 6 * * * cd /path/to/project && python scripts/fetch_fundamentals.py --ticker RELIANCE.NS --source yahoo
0 6 * * * cd /path/to/project && python scripts/fetch_news.py --ticker RELIANCE --days 1
0 6 * * * cd /path/to/project && python scripts/fetch_social_sentiment.py --ticker RELIANCE

# Weekly macro data updates (Mondays at 7 AM)
0 7 * * 1 cd /path/to/project && python scripts/fetch_macro_data.py --country IND --preset full
```

### Apache Airflow

Example DAG for orchestrating data pipeline:

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'financial_data_ingestion',
    default_args=default_args,
    description='Daily financial data ingestion pipeline',
    schedule_interval='0 6 * * *',  # Daily at 6 AM
    catchup=False,
)

tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']

for ticker in tickers:
    fetch_prices = BashOperator(
        task_id=f'fetch_prices_{ticker.replace(".", "_")}',
        bash_command=f'python scripts/fetch_historical_prices.py --ticker {ticker} --incremental',
        dag=dag,
    )
    
    fetch_fundamentals = BashOperator(
        task_id=f'fetch_fundamentals_{ticker.replace(".", "_")}',
        bash_command=f'python scripts/fetch_fundamentals.py --ticker {ticker} --source yahoo',
        dag=dag,
    )
    
    fetch_news = BashOperator(
        task_id=f'fetch_news_{ticker.split(".")[0]}',
        bash_command=f'python scripts/fetch_news.py --ticker {ticker.split(".")[0]} --days 1',
        dag=dag,
    )
    
    fetch_social = BashOperator(
        task_id=f'fetch_social_{ticker.split(".")[0]}',
        bash_command=f'python scripts/fetch_social_sentiment.py --ticker {ticker.split(".")[0]}',
        dag=dag,
    )
    
    # Set dependencies (all can run in parallel)
    # fetch_prices, fetch_fundamentals, fetch_news, fetch_social run independently

# Macro data (runs once per day, not per ticker)
fetch_macro = BashOperator(
    task_id='fetch_macro',
    bash_command='python scripts/fetch_macro_data.py --country IND --preset full',
    dag=dag,
)
```

## Troubleshooting

### Common Issues

**1. API Key Errors**

```
ConfigError: NEWSAPI_KEY not found in environment variables
```

**Solution:** Ensure `.env` file exists and contains the required API key:
```bash
echo "NEWSAPI_KEY=your_key_here" >> .env
```

**2. Rate Limit Exceeded**

```
RateLimitError: Rate limit exceeded. Retry after 3600 seconds.
```

**Solution:** 
- Wait for the rate limit window to reset
- Reduce request frequency
- Consider upgrading to paid API tier
- Use `--limit` parameter to fetch fewer items

**3. Invalid Ticker Symbol**

```
ValueError: No data returned for ticker RELIANCE
```

**Solution:** 
- For Yahoo Finance, use exchange suffix (e.g., `RELIANCE.NS` for NSE)
- For StockTwits, use base symbol without suffix (e.g., `RELIANCE`)
- Verify ticker exists on the platform

**4. Network Timeouts**

```
RequestException: Connection timeout after 30 seconds
```

**Solution:**
- Scripts automatically retry with exponential backoff
- Check internet connection
- Verify API endpoints are accessible
- Increase timeout in utility functions if needed

**5. Missing Dependencies**

```
ModuleNotFoundError: No module named 'yfinance'
```

**Solution:**
```bash
pip install -r requirements.txt
```

### Logging and Debugging

Enable DEBUG logging for detailed output:

```bash
export LOG_LEVEL=DEBUG
python scripts/fetch_historical_prices.py --ticker RELIANCE.NS --incremental
```

Check log files for errors:
```bash
tail -f logs/app.log
```

View JSON logs for structured analysis:
```bash
cat logs/app.json | jq '.[] | select(.level=="ERROR")'
```

## Data Quality Validation

Scripts perform automatic validation before saving:

1. **Required columns check** - Ensures critical fields are present
2. **Missing values check** - Warns about null/NaN values
3. **Duplicate detection** - Identifies duplicate timestamps or articles
4. **Schema compliance** - Validates data structure matches expectations

Validation warnings are logged but don't prevent data from being saved. Review logs to address data quality issues.

## Integration with Project

These scripts support **Phase 1: Data Pipelines** of the project roadmap (see main README.md). Fetched data will be used in:

- **Phase 2: Data Preprocessing** - Cleaning, normalization, feature engineering
- **Phase 3: Sentiment Analysis** - FinBERT on news/social data
- **Phase 4: Multi-Modal Models** - LSTM, Transformers, XGBoost training
- **Phase 5: Growth Scoring** - Company ranking and prediction

Refer to `docs/charter.md` for detailed requirements and success metrics.

## Contributing

When adding new data sources:

1. Create a new script in `scripts/` following naming convention `fetch_<source>.py`
2. Import utilities from `backend/utils/` for consistency
3. Implement CLI with argparse and helpful examples
4. Use timestamp-based partitioning in `data/raw/<source>/`
5. Add comprehensive logging and error handling
6. Update this README with usage examples

See `CONTRIBUTING.md` for detailed contribution guidelines.

## Feature Engineering Scripts

Phase 2 introduces CLI utilities for transforming raw datasets into model-ready tensors and embeddings. They reuse the shared config/logger modules and follow the same timestamped directory conventions as Phase 1.

### `feature_engineering.py`

Computes pandas-ta indicators, performs temporal (60/20/20) splits, fits scalers on training data only, and generates 60-day windowed sequences suitable for LSTM/GRU/Transformer models.

Usage examples:

- Single ticker: `python scripts/feature_engineering.py --ticker RELIANCE.NS`
- Custom lookback window: `--lookback-window 90`
- Specific indicators: `--indicators RSI,MACD,EMA`
- Custom split ratios: `--train-split 0.7 --val-split 0.15`
- Batch processing from CSV: `--batch-mode --ticker-file tickers.csv`
- StandardScaler instead of MinMaxScaler: `--scaler-type standard`

Outputs:

- `train_features.npy`, `val_features.npy`, `test_features.npy`
- `train_targets.npy`, `val_targets.npy`, `test_targets.npy`
- `scaler_metadata.json`, `metadata.json`

### `text_embeddings.py`

Loads the latest news or StockTwits payload, extracts FinBERT embeddings (768-d vectors), aligns timestamps to trading days, and optionally stores sentiment predictions from the FinBERT classifier head.

Usage examples:

- News embeddings: `python scripts/text_embeddings.py --ticker RELIANCE --source news`
- Social embeddings: `python scripts/text_embeddings.py --ticker RELIANCE --source social`
- Larger GPU batches: `--batch-size 32 --device cuda`
- Alternate checkpoint: `--model-name yiyanghkust/finbert-tone`
- Include sentiment classification: `--classify-sentiment`

Outputs:

- `embeddings_news.npy` / `embeddings_social.npy`
- `embeddings_news_metadata.json` / `embeddings_social_metadata.json`

### Output Structure

Both scripts write to `data/processed/{ticker}/YYYY-MM-DD/` (same partitioning style as the ingestion layer):

```
data/processed/RELIANCE_NS/2025-11-18/
├── train_features.npy
├── val_features.npy
├── test_features.npy
├── train_targets.npy
├── val_targets.npy
├── test_targets.npy
├── metadata.json
├── scaler_metadata.json
├── embeddings_news.npy
├── embeddings_news_metadata.json
└── embeddings_social.npy
```

### Data Leakage Prevention

- Uses `backend.utils.preprocessing.temporal_train_test_split()` to enforce chronological ordering (doc §4.4).
- Scalers (`MinMaxScaler`/`StandardScaler`) are fit on training samples once, then applied to validation/test (doc §4.3).
- Window generation occurs within each split, preventing overlap between train/val/test sequences.
- Metadata stores split indices, date ranges, indicator list, and scaler parameters for auditability.

### Troubleshooting

- **Insufficient history**: Ensure ≥260 price points (EMA-200 warm-up + 60-day window) before running `feature_engineering.py`.
- **FinBERT download failures**: Confirm internet access or configure a local Hugging Face cache.
- **CUDA out-of-memory**: Reduce `--batch-size` or force CPU via `--device cpu`.
- **Missing indicator columns**: Verify `.env` `TECHNICAL_INDICATORS=RSI,MACD,EMA,BB,ATR` (comma-separated, not JSON) and that pandas-ta is installed.

See `docs/metrics_and_evaluation.md` for the leakage rules, split ratios, and sentiment accuracy targets these scripts satisfy.
