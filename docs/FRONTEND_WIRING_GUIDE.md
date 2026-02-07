# Frontend Wiring Guide

> Everything you need to rebuild the frontend and connect it to the existing FastAPI backend.

---

## Table of Contents

1. [Backend Base URL](#backend-base-url)
2. [Authentication Flow](#authentication-flow)
3. [API Endpoints Reference](#api-endpoints-reference)
4. [WebSocket (Real-Time Data)](#websocket-real-time-data)
5. [TypeScript Data Models](#typescript-data-models)
6. [Environment Variables](#environment-variables)
7. [Pages to Design](#pages-to-design)

---

## Backend Base URL

All REST calls go to `VITE_API_URL` (default `http://localhost:8000`).

---

## Authentication Flow

The app uses **Upstox OAuth 2.0** — no custom user DB.

```
┌──────────┐  1. click "Login"   ┌──────────┐  2. redirect   ┌──────────┐
│ Frontend  │ ──────────────────► │ Backend  │ ─────────────► │ Upstox   │
│           │                     │ /auth/   │                │ OAuth    │
│           │  5. store token     │ upstox/  │  3. user       │ Dialog   │
│           │ ◄────────────────── │ login    │ ◄───────────── │          │
└──────────┘  in localStorage     └──────────┘  4. callback   └──────────┘
                                                   with code
```

| Step | Detail |
|------|--------|
| **1** | Navigate user to `GET {API_URL}/auth/upstox/login` |
| **2** | Backend redirects to Upstox consent page |
| **3** | User approves → Upstox redirects to backend callback |
| **4** | Backend redirects to `{FRONTEND_URL}/auth/callback?code={code}` |
| **5** | Frontend `AuthCallback` page calls `GET {API_URL}/auth/upstox/callback?code={code}` |
| **6** | Backend returns `{ access_token: string }` |
| **7** | Store `access_token` in `localStorage` as `upstox_access_token` |

**Sending auth:** Add header `Authorization: Bearer {token}` on every API call.

**Logout:** Remove `upstox_access_token` from localStorage, redirect to `/`.

**Note:**  Most endpoints work **without** auth too — they fall back to Yahoo Finance.  Auth is only needed for live Upstox quotes and order data.

---

## API Endpoints Reference

### Health & Info

| Method | Path | Response |
|--------|------|----------|
| `GET` | `/` | `{ message, version, docs, health }` |
| `GET` | `/health` | `{ status: "healthy", service, version, timestamp, environment }` |

### Auth

| Method | Path | Params | Response |
|--------|------|--------|----------|
| `GET` | `/auth/upstox/login` | — | `302` redirect to Upstox |
| `GET` | `/auth/upstox/callback` | `code` (query) | `{ access_token, extended_token? }` |

### Stocks & Market Data

| Method | Path | Params | Response |
|--------|------|--------|----------|
| `GET` | `/stocks` | — | `Stock[]` — all tracked stocks |
| `GET` | `/stocks/quotes` | `symbols` (comma-separated, optional) | `StockQuote[]` |
| `GET` | `/stocks/with-sentiment` | — | `StockWithSentiment[]` — price + AI sentiment for all stocks |
| `GET` | `/stocks/{symbol}/historical` | `interval` (default `"1minute"`), `days` (default `7`) | `CandleData[]` |
| `GET` | `/stocks/{symbol}/sentiment` | — | `RealTimeSentiment` — single-stock sentiment |
| `GET` | `/stocks/{symbol}/analysis` | — | `StockAnalysis` — comprehensive analysis object |
| `GET` | `/stocks/{symbol}/fundamental-analysis` | — | Scraped fundamentals (Screener.in / Yahoo fallback) |
| `GET` | `/indices` | — | `IndexQuote[]` — NIFTY 50, SENSEX, BANK NIFTY |
| `GET` | `/leaderboard` | — | `LeaderboardStock[]` — top stocks ranked by growth score |

### News & Sentiment

| Method | Path | Params | Response |
|--------|------|--------|----------|
| `GET` | `/news/{symbol}` | `company_name` (optional) | `SentimentResponse` — overall sentiment + news articles |

### Scorecard

All paths are prefixed with `/scorecard`.

| Method | Path | Params | Response |
|--------|------|--------|----------|
| `GET` | `/scorecard/{symbol}` | `include_ai_summary` (bool), `refresh` (bool) | `ScorecardData` — full scorecard |
| `GET` | `/scorecard/` | — | `ScorecardListItem[]` — summary list for all stocks |
| `POST` | `/scorecard/refresh/{symbol}` | — | `{ status, symbol, source, company_name, message }` |
| `POST` | `/scorecard/batch-refresh` | `symbols` (query, list) | `{ total, success_count, failed_count, success[], failed[] }` |
| `GET` | `/scorecard/search/{query}` | — | `SearchResult[]` — search stocks by name/symbol |

---

## WebSocket (Real-Time Data)

The Upstox live feed connects **directly from the browser** (not through the backend):

```
URL:    wss://api.upstox.com/v2/feed/market-data-feed
Auth:   Authorization: Bearer {upstox_access_token}
Format: binary (Protocol Buffers)
```

You'll need `protobufjs` to decode. The proto schema is fetched from:
`https://assets.upstox.com/feed/market-data-feed/v1/MarketDataFeed.proto`

Message types: `ltpc` (last traded price), `ff` (full feed with OHLC, OI, depth).

---

## TypeScript Data Models

```typescript
// ─── Stocks ─────────────────────────────────────────────

interface Stock {
  symbol: string;
  name: string;
  sector: string;
  instrument_key: string;
  isin: string;
  yf_ticker: string;
}

interface StockQuote {
  symbol: string;
  name: string;
  sector: string;
  price: number;
  change: number;
  changePercent: number;
  high: number;
  low: number;
  open: number;
  volume: number;
  instrument_key: string;
}

interface StockWithSentiment {
  symbol: string;
  name: string;
  sector: string;
  price: number;
  change: number;
  changePercent: number;
  forecastConfidence: number;
  sentiment: "bullish" | "neutral" | "bearish";
  riskLevel: "low" | "medium" | "high";
  lastUpdated?: string;
}

interface IndexQuote {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
}

interface CandleData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface LeaderboardStock {
  rank: number;
  symbol: string;
  name: string;
  sector: string;
  growthScore: number;
  sentiment: string;
  forecastPercent: number;
}

// ─── Analysis ───────────────────────────────────────────

interface StockAnalysis {
  symbol: string;
  name: string;
  sector: string;
  currentPrice: number;
  change: number;
  changePercent: number;
  sentiment: string;
  sentimentScore: number;
  sentimentConfidence: number;
  sentimentReasoning: string;
  riskLevel: string;
  riskFactors: string[];
  growthPotential: string;
  debtLevel: string;
  predictedPrice: number;
  forecastConfidence: number;
  shortTermOutlook: string;
  longTermOutlook: string;
  recommendation: string;
  recentNews: NewsArticle[];
  newsCount: number;
}

interface NewsArticle {
  title: string;
  source: string;
  url: string;
  published_at: string;
  sentiment_score?: number;
  description?: string;
}

// ─── Sentiment ──────────────────────────────────────────

interface RealTimeSentiment {
  symbol: string;
  name: string;
  sentiment_score: number;
  sentiment_label: string;
  confidence: number;
  risk_level: string;
  reasoning: string;
  last_updated?: string;
  cached: boolean;
}

interface SentimentResponse {
  symbol: string;
  overall_sentiment: string;
  confidence?: number;
  reasoning?: string;
  key_themes?: string[];
  risk_level?: string;
  article_count: number;
  news: NewsArticle[];
  model?: string;
}

// ─── Scorecard ──────────────────────────────────────────

interface ScorecardData {
  symbol: string;
  company_name: string;
  sector?: string;
  industry?: string;
  source?: string;
  generated_at: string;
  overall_score: number;
  overall_max: number;
  overall_verdict: string;
  overall_badge: string;   // "Strong Buy" | "Buy" | "Hold" | "Sell" etc.
  categories: Record<string, CategoryScore>;
  key_stats: KeyStats;
  pros: string[];
  cons: string[];
  strengths: StrengthWeakness[];
  weaknesses: StrengthWeakness[];
  peers: string[];
  news_summary: {
    sentiment?: string;
    key_themes?: string[];
    article_count: number;
  };
  ai_summary?: string;
}

interface CategoryScore {
  score: number;
  max: number;
  label: string;
  details: Record<string, string | number>;
}

interface KeyStats {
  market_cap_cr?: number;
  pe_ratio?: number;
  pb_ratio?: number;
  dividend_yield?: number;
  roce?: number;
  roe?: number;
  debt_to_equity?: number;
  current_ratio?: number;
  promoter_holding?: number;
  eps?: number;
  book_value?: number;
  face_value?: number;
  industry_pe?: number;
}

interface StrengthWeakness {
  text: string;
  category: string;
}

interface ScorecardListItem {
  symbol: string;
  company_name: string;
  sector?: string;
  overall_score: number;
  overall_verdict: string;
  overall_badge: string;
  pe_ratio?: number;
  roce?: number;
  market_cap_cr?: number;
}

// ─── Live Market (Upstox WebSocket) ─────────────────────

interface PriceTick {
  symbol: string;
  timestamp: number;
  price: number;
  volume: number;
  side: "buy" | "sell";
}

interface Candle {
  symbol: string;
  start: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  predictedClose?: number;
  confidenceLower?: number;
  confidenceUpper?: number;
}

type Timeframe = "1m" | "5m" | "1h" | "1d" | "1w";
```

---

## Environment Variables

### Frontend `.env`

```env
VITE_API_URL=http://localhost:8000
```

### Backend `.env` (for reference)

```env
# Upstox OAuth
UPSTOX_API_KEY=
UPSTOX_API_SECRET=
UPSTOX_REDIRECT_URI=http://localhost:8000/auth/upstox/callback
FRONTEND_BASE_URL=http://localhost:5173

# AI & LLM
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3-flash

# Embeddings  (local | openai | huggingface | gemini)
EMBEDDING_MODEL_TYPE=local
OPENAI_API_KEY=
HUGGINGFACE_TOKEN=

# Cache
REDIS_HOST=redis
REDIS_PORT=6379

# Database
DATABASE_URL=sqlite:///data/processed/stock_data.db
```

---

## Pages to Design

Below are the pages your new frontend needs. Each entry shows the route, purpose, and which APIs it calls.

### 1. Login Page
- **Route:** `/login`  
- **Purpose:** Unauthenticated landing. Explain the product, provide "Login with Upstox" button.  
- **API:** Navigates to `GET /auth/upstox/login`

### 2. Auth Callback
- **Route:** `/auth/callback`  
- **Purpose:** Invisible/loading page. Receives OAuth `code`, exchanges for token, stores in localStorage, redirects to dashboard.  
- **API:** `GET /auth/upstox/callback?code={code}`

### 3. Dashboard (Home)
- **Route:** `/`  
- **Purpose:** Overview of the market. Show market indices, top movers, sentiment overview, quick leaderboard.  
- **APIs:**  
  - `GET /stocks/with-sentiment` — all stocks with price + sentiment  
  - `GET /indices` — NIFTY 50, SENSEX, BANK NIFTY  
  - `GET /leaderboard` — top growth stocks  

### 4. Stock Screener / Explorer
- **Route:** `/screener` or `/explore`  
- **Purpose:** Browse all 120 tracked stocks. Filter by sector, sentiment, risk. Search by name/symbol.  
- **APIs:**  
  - `GET /stocks/with-sentiment`  
  - `GET /scorecard/search/{query}` — search  

### 5. Stock Detail
- **Route:** `/stock/:symbol`  
- **Purpose:** Deep-dive into a single stock. Price chart, AI analysis, sentiment, news, fundamentals.  
- **APIs:**  
  - `GET /stocks/{symbol}/analysis` — full AI analysis  
  - `GET /stocks/{symbol}/historical?interval=1minute&days=7` — chart data  
  - `GET /news/{symbol}` — news articles + sentiment  
  - `GET /stocks/{symbol}/fundamental-analysis` — financials from Screener.in  

### 6. Scorecard
- **Route:** `/scorecard/:symbol`  
- **Purpose:** Tickertape-style investment scorecard. Overall score ring, category breakdowns, pros/cons, peer comparison, AI summary.  
- **APIs:**  
  - `GET /scorecard/{symbol}?include_ai_summary=true`  

### 7. All Scorecards
- **Route:** `/scorecards`  
- **Purpose:** Table/grid of all stocks with their overall scorecard ratings. Quick comparison view.  
- **APIs:**  
  - `GET /scorecard/` — list of all scorecard summaries  

### 8. Forecasts
- **Route:** `/forecasts`  
- **Purpose:** Show ML price predictions for all stocks. Predicted price, confidence, direction.  
- **APIs:**  
  - `GET /stocks/with-sentiment` — includes `forecastConfidence`  
  - `GET /stocks/{symbol}/analysis` — includes `predictedPrice`, `forecastConfidence`, `shortTermOutlook`, `longTermOutlook`  

### 9. Sentiment Dashboard
- **Route:** `/sentiment`  
- **Purpose:** Market-wide sentiment view. Heatmap or cards showing bullish/neutral/bearish distribution across stocks.  
- **APIs:**  
  - `GET /stocks/with-sentiment` — sentiment for all stocks  
  - `GET /stocks/{symbol}/sentiment` — drill-down for individual stock  

### 10. Growth Leaderboard
- **Route:** `/growth`  
- **Purpose:** Ranked list of stocks by growth score. Show score breakdown, badges (Strong Buy / Buy / Hold / Sell).  
- **APIs:**  
  - `GET /leaderboard`  
  - `GET /scorecard/{symbol}` — for detailed breakdown  

### 11. Portfolio (Mock / Upstox)
- **Route:** `/portfolio`  
- **Purpose:** User's portfolio view. If Upstox token is available, could show real holdings. Otherwise, mock or manual portfolio.  
- **APIs:**  
  - `GET /stocks/quotes?symbols=RELIANCE,TCS,...` — prices for held stocks  

### 12. Live Trading View (Optional)
- **Route:** `/trading`  
- **Purpose:** Real-time candlestick chart with order book & live tick stream. Requires Upstox WebSocket.  
- **APIs:**  
  - WebSocket: `wss://api.upstox.com/v2/feed/market-data-feed`  
  - `GET /stocks/{symbol}/historical` — initial candle data  

---

## Architecture Recommendations

```
src/
├── app/                  # App entry, router, providers
├── components/           # Shared UI components (Button, Card, Modal, Chart, etc.)
├── features/             # Feature-based modules
│   ├── auth/             # Login, Callback, AuthContext, ProtectedRoute
│   ├── dashboard/        # Home dashboard page
│   ├── screener/         # Stock explorer/screener
│   ├── stock-detail/     # Single stock deep-dive
│   ├── scorecard/        # Scorecard page + list
│   ├── forecasts/        # ML forecasts view
│   ├── sentiment/        # Sentiment dashboard
│   ├── growth/           # Growth leaderboard
│   ├── portfolio/        # Holdings view
│   └── trading/          # Live trading (optional)
├── hooks/                # Custom hooks (useApi, useAuth, useWebSocket)
├── services/             # API client, WebSocket service
├── types/                # Shared TypeScript interfaces
├── utils/                # Helpers, formatters, constants
└── styles/               # Global styles, theme
```

---

## Quick Reference: API Client Skeleton

```typescript
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function getHeaders(): HeadersInit {
  const headers: HeadersInit = { "Content-Type": "application/json" };
  const token = localStorage.getItem("upstox_access_token");
  if (token) headers["Authorization"] = `Bearer ${token}`;
  return headers;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { headers: getHeaders() });
  if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
  return res.json();
}

// Example usage:
// const stocks = await get<StockWithSentiment[]>("/stocks/with-sentiment");
// const analysis = await get<StockAnalysis>(`/stocks/${symbol}/analysis`);
// const scorecard = await get<ScorecardData>(`/scorecard/${symbol}?include_ai_summary=true`);
```
