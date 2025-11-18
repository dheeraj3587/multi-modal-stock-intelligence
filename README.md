# Multi-Modal Stock Intelligence

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-teal?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

**Advanced financial analytics platform integrating deep learning, sentiment analysis, and real-time market data.**

[Features](#key-features) • [Architecture](#architecture) • [Quick Start](#quick-start) • [Documentation](docs/)

---

## Overview

Multi-Modal Stock Intelligence is a production-ready forecasting engine that fuses quantitative market data with qualitative sentiment signals. Unlike traditional technical analysis tools, this platform leverages Transformer-based architectures (PatchTST) and FinBERT embeddings to capture both market momentum and investor sentiment.

### Key Capabilities

| Feature | Description | Tech Stack |
|---------|-------------|------------|
| **Deep Forecasting** | Multi-horizon price prediction (7-day) using LSTM, GRU, and Transformers. | PyTorch, Optuna |
| **Sentiment Engine** | Real-time news and social media sentiment analysis with FinBERT. | HuggingFace, NewsAPI |
| **Real-Time Stream** | Live market data ingestion via WebSocket for sub-second updates. | Upstox, Redis |
| **Growth Scoring** | Proprietary fundamental analysis algorithm for long-term value investing. | Scikit-learn, Pandas |
| **MLOps Pipeline** | End-to-end experiment tracking, model registry, and hyperparameter tuning. | MLflow, Docker |

---

## Architecture

The system follows a modular microservices architecture designed for scalability and maintainability.

```mermaid
flowchart TB
    subgraph DataSources["Data Sources Layer"]
        direction LR
        YF["Yahoo Finance"]
        NEWS["NewsAPI"]
        ST["StockTwits"]
        UP["Upstox WebSocket"]
        MACRO["World Bank"]
    end
    
    subgraph Storage["Data Storage Layer"]
        direction TB
        RAW[("Raw Data")]
        PROC[("Processed Data")]
    end
    
    subgraph FeatureEng["Feature Engineering Pipeline"]
        direction TB
        TA["Technical Indicators"]
        EMB["FinBERT Embeddings"]
        NORM["Scaling & Windowing"]
        SPLIT["Train/Val/Test Split"]
    end
    
    subgraph Models["Machine Learning Models"]
        direction TB
        subgraph Forecasting["Time-Series Forecasting"]
            LSTM["LSTM"]
            GRU["GRU"]
            TRANS["Transformer (PatchTST)"]
        end
        subgraph TextModels["NLP Models"]
            FINBERT["FinBERT Sentiment"]
        end
        subgraph Analytics["Analytics"]
            GROWTH["Growth Scorer"]
        end
    end
    
    subgraph MLOps["MLOps & Experiment Tracking"]
        direction LR
        MLFLOW["MLflow"]
        OPTUNA["Optuna"]
        CKPT["Checkpoints"]
    end
    
    subgraph Backend["Backend Services"]
        direction TB
        API["FastAPI"]
        REDIS["Redis"]
        PG["PostgreSQL"]
        MONGO["MongoDB"]
    end
    
    subgraph Frontend["User Interface"]
        NEXT["Next.js Dashboard"]
    end
    
    YF --> RAW
    NEWS --> RAW
    ST --> RAW
    MACRO --> RAW
    
    RAW --> TA
    RAW --> EMB
    
    TA --> NORM
    EMB --> NORM
    
    NORM --> SPLIT
    SPLIT --> PROC
    
    PROC --> LSTM
    PROC --> GRU
    PROC --> TRANS
    PROC --> FINBERT
    PROC --> GROWTH
    
    LSTM -.-> MLFLOW
    GRU -.-> MLFLOW
    TRANS -.-> MLFLOW
    
    OPTUNA -.-> LSTM
    OPTUNA -.-> GRU
    OPTUNA -.-> TRANS
    
    LSTM --> API
    GRU --> API
    TRANS --> API
    FINBERT --> API
    GROWTH --> API
    UP --> API
    
    API <--> REDIS
    API <--> PG
    API <--> MONGO
    
    API <--> NEXT
```

### Architecture Highlights

- **Multi-Modal Data Integration**: Combines price data, news, social sentiment, and macroeconomic indicators.
- **Comprehensive Feature Engineering**: 60-day windowing with technical indicators and FinBERT embeddings.
- **State-of-the-Art Models**: LSTM, GRU, and Transformer (PatchTST) for time-series forecasting.
- **MLOps Pipeline**: Automated experiment tracking (MLflow) and hyperparameter optimization (Optuna).
- **High-Performance Backend**: Async FastAPI with Redis caching and dual database architecture.

---

## Technology Stack

- **Core Infrastructure**: Python 3.10+, FastAPI, Docker, Redis, PostgreSQL, MongoDB
- **Machine Learning**: PyTorch, Hugging Face Transformers, Scikit-Learn, MLflow, Optuna
- **Frontend**: Next.js, TailwindCSS, Plotly

---

## Project Structure

```bash
test/
├── backend/                 # FastAPI application
├── frontend/                # Next.js dashboard
├── data/                    # Datasets (raw, processed, external)
├── models/                  # ML model implementations (LSTM, GRU, Transformer)
├── scripts/                 # Data ingestion and training scripts
├── notebooks/               # Jupyter notebooks for EDA and evaluation
├── tests/                   # Unit and integration tests
├── docker/                  # Docker configurations
├── docs/                    # Project documentation
└── mlruns/                  # MLflow tracking artifacts
```

---

## Prerequisites

- **Python**: 3.10 or higher
- **Docker**: Latest version
- **API Keys**: NewsAPI, Upstox Developer, StockTwits (optional)

---

## Installation & Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd test
```

### 2. Environment Configuration

Copy the template and configure your API keys:

```bash
cp .env.template .env
# Edit .env with your credentials
```

### 3. Docker Setup (Recommended)

```bash
docker-compose up -d
```

Services will be available at:
- Backend API: http://localhost:8000
- Frontend Dashboard: http://localhost:3000
- MLflow UI: http://localhost:5000

### 4. Local Development Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

---

## Usage

### Dashboard & API

- **Dashboard**: Real-time analytics and visualizations.
- **API Documentation**: Interactive Swagger UI at `/docs`.
- **MLflow**: Experiment tracking and model registry.

### Core Workflows

#### Data Ingestion

Fetch data from various sources using the scripts in `scripts/`:

```bash
# Fetch historical prices
python scripts/fetch_historical_prices.py --ticker RELIANCE.NS --start 2020-01-01 --end 2024-12-31

# Fetch news articles
python scripts/fetch_news.py --ticker RELIANCE --days 30
```

#### Feature Engineering

Transform raw data into model-ready features:

```bash
python scripts/feature_engineering.py --ticker RELIANCE.NS
python scripts/text_embeddings.py --ticker RELIANCE --source news
```

#### Model Training

Train forecasting and sentiment models:

```bash
# Train LSTM model
python scripts/train_forecasting_models.py --model-type lstm --ticker RELIANCE.NS --max-epochs 100

# Train Sentiment model
python scripts/train_sentiment_model.py --data-dir data/raw/news --price-dir data/raw/prices

# Train Growth Scorer
python scripts/train_growth_scorer.py --fundamentals-dir data/raw/fundamentals --technical-dir data/processed --price-dir data/raw/prices
```

---

## Evaluation Metrics

We target specific performance metrics as defined in our documentation:

- **Forecasting**: MAE reduction >= 15%, Directional Accuracy >= 55%, Sharpe Ratio >= 1.0
- **Sentiment**: F1 Score >= 0.80
- **Growth Scoring**: Spearman Correlation >= 0.30, Top-10 Precision >= 70%

Refer to `docs/metrics_and_evaluation.md` for detailed criteria.

---

## Testing

Run the test suite to ensure system stability:

```bash
pytest tests/
```

---

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on code style, testing, and the pull request process.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
