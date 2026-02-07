# Data Fetching Robustness Guide

This document describes the robustness improvements implemented across all data-fetching components to ensure reliable operation in production.

## Overview

All external API calls and data fetching operations now include:
- ✅ **Retry logic** with exponential backoff
- ✅ **Timeout handling** with appropriate limits
- ✅ **Rate limiting** to prevent API overload
- ✅ **Cache fallback** for graceful degradation
- ✅ **Enhanced logging** for debugging
- ✅ **Error recovery** with multiple fallback layers

---

## Configuration Parameters

### Screener Parser

**File:** `backend/services/screener_parser.py`

```python
# HTTP retry configuration
RETRY_TOTAL = 3                      # Max retry attempts
RETRY_BACKOFF_FACTOR = 0.8           # Exponential backoff multiplier
RETRY_STATUS_FORCELIST = [429, 500, 502, 503, 504]  # HTTP codes to retry
SCREENER_TIMEOUT = 10                # seconds
```

**Retry behavior:**
- 1st attempt: immediate
- 2nd attempt: after 0.8s
- 3rd attempt: after 1.6s
- Then falls back to yfinance

### RAG Sentiment Analyzer

**File:** `backend/services/rag_sentiment_analyzer.py`

```python
LLM_TIMEOUT = 30                     # seconds for LLM API calls
LLM_MAX_RETRIES = 2                  # Retry attempts for LLM
LLM_RETRY_DELAY = 2.0                # Base delay between retries
```

**Retry behavior:**
- 1st attempt: 30s timeout
- 2nd attempt: 30s timeout, after 2s delay
- 3rd attempt: 30s timeout, after 4s delay (exponential)

### RSS News Fetcher

**File:** `backend/services/rss_news_fetcher.py`

```python
RSS_TIMEOUT = 15                     # seconds per request
RSS_MAX_RETRIES = 3                  # Max retry attempts
RSS_BACKOFF_FACTOR = 0.5             # Exponential backoff
RSS_STATUS_FORCELIST = [429, 500, 502, 503, 504]
```

**Throttling:**
- 500ms delay between different query batches
- 300ms delay between different news sources
- 300ms delay between ET RSS feeds

### Scorecard API

**File:** `backend/api/scorecard.py`

```python
MAX_CONCURRENT_SCORECARDS = 5        # Max parallel scorecard generations
CACHE_DEBOUNCE_SECONDS = 2           # Prevent rapid duplicate requests
```

**Timeouts:**
- Fundamentals fetch: 30 seconds
- Sentiment analysis: 45 seconds
- AI summary generation: 30 seconds

**Batch processing:**
- Process 5 stocks at a time
- 0.5s delay between batches

### Fundamentals Database

**File:** `backend/services/fundamentals_db.py`

```python
CACHE_DEFAULT_HOURS = 12             # Default cache lifetime
```

**Cache fallback layers:**
1. Fresh cache (< 12h)
2. Stale cache (12h - 48h) with `allow_stale=True`
3. Very stale cache (> 48h) as last resort

---

## Robustness Features by Component

### 1. Screener Parser

**Improvements:**
- ✅ HTTP session with automatic retry adapter
- ✅ Timeout on all requests (10s)
- ✅ Specific exception handling (Timeout, HTTPError, ConnectionError)
- ✅ Yfinance fallback when Screener.in fails
- ✅ Detailed logging of errors with URL/status codes

**Usage:**
```python
from backend.services.screener_parser import parser

# Automatically retries on failure
fundamentals = parser.fetch_fundamentals('RELIANCE')

# Falls back to yfinance if Screener.in unavailable
if fundamentals and fundamentals.get('source') == 'yfinance':
    print("Using fallback data")
```

### 2. RAG Sentiment Analyzer

**Improvements:**
- ✅ LLM timeout increased to 30s
- ✅ Retry loop with exponential backoff (2 retries)
- ✅ Special Gemini timeout handling with ThreadPoolExecutor
- ✅ Structured error responses with details
- ✅ Graceful degradation on persistent failures

**Usage:**
```python
from backend.services.rag_sentiment_analyzer import create_sentiment_analyzer

analyzer = create_sentiment_analyzer()

# Automatically retries LLM calls on failure
sentiment = analyzer.analyze_sentiment(
    company_name="Reliance Industries",
    symbol="RELIANCE",
    articles=articles,
    context_articles=context
)

# Check for errors
if "error" in sentiment:
    print(f"Sentiment analysis failed: {sentiment['error']}")
```

### 3. RSS News Fetcher

**Improvements:**
- ✅ HTTP session with retry adapter
- ✅ Timeout on all requests (15s)
- ✅ Throttling between sources to avoid rate limits
- ✅ Specific error handling per news source
- ✅ Continues with partial results on failures

**Usage:**
```python
from backend.services.rss_news_fetcher import rss_fetcher

# Fetches from multiple sources with automatic retries
articles = rss_fetcher.fetch_company_news(
    symbol="RELIANCE",
    company_name="Reliance Industries",
    max_articles=20
)

# Even if some sources fail, returns available articles
print(f"Fetched {len(articles)} articles")
```

### 4. Scorecard API

**Improvements:**
- ✅ Semaphore limiting (max 5 concurrent)
- ✅ Request debouncing (2s window)
- ✅ Timeout on all async operations
- ✅ Triple-layer cache fallback (fresh → stale → very stale)
- ✅ Batch processing with throttling
- ✅ Detailed error responses with context

**Usage:**
```python
# GET /api/scorecard/RELIANCE
# Automatically:
# - Limits concurrent requests
# - Deduplicates rapid requests
# - Tries fresh → stale → very stale cache
# - Times out long operations
# - Returns partial results on failures

# POST /api/scorecard/batch-refresh
# Body: {"symbols": ["RELIANCE", "TCS", "INFY", ...]}
# Processes 5 at a time with delays
```

### 5. Fundamentals Database

**Improvements:**
- ✅ Multi-layer cache with age tracking
- ✅ `allow_stale` parameter for graceful degradation
- ✅ Detailed logging of cache age
- ✅ Safe datetime parsing with error handling

**Usage:**
```python
from backend.services.fundamentals_db import db

# Try fresh cache first
data = db.get_latest('RELIANCE', allow_stale=False)

if not data:
    # Fetch new data from source
    new_data = fetch_from_screener('RELIANCE')
    if new_data:
        db.upsert(new_data)
    else:
        # Fall back to stale cache
        data = db.get_latest('RELIANCE', allow_stale=True)
```

---

## Debugging

### Enable Detailed HTTP Logging

Set environment variable:
```bash
export DEBUG_HTTP=true
```

This enables full request/response logging via `backend/utils/http_debug.py`.

**Output includes:**
- Request method, URL, headers (secrets masked)
- Query parameters and body (truncated if large)
- Response status, headers, body snippet
- Request duration in milliseconds
- Full exception details on errors

### Test Provider Endpoints

Use the test utility to manually verify endpoints:

```python
from backend.utils.http_debug import test_provider_endpoint

# Test Screener.in directly
response = test_provider_endpoint(
    'https://www.screener.in/company/RELIANCE/consolidated/',
    method='GET',
    timeout=10
)
```

### Check Cache Status

```python
from backend.services.fundamentals_db import db

# Check if data is stale
is_stale = db.is_stale('RELIANCE', max_age_hours=12)

# Get data with age information (check logs)
data = db.get_latest('RELIANCE', allow_stale=False)
```

### Monitor Retry Counts

All retry operations log attempts:
```
INFO: Attempting fundamentals fetch for RELIANCE (attempt 1/3)
WARNING: Request failed, retrying after 0.8s...
INFO: Attempting fundamentals fetch for RELIANCE (attempt 2/3)
INFO: Successfully fetched fundamentals for RELIANCE
```

---

## Testing

### Run Robustness Tests

```bash
cd /Users/dheerajjoshi/multi-modal-stock-intelligence
pytest backend/tests/test_robustness.py -v
```

**Test coverage:**
- ✅ Retry logic on timeouts
- ✅ Retry on HTTP 5xx errors
- ✅ Fallback mechanisms
- ✅ Cache staleness handling
- ✅ Throttling behavior
- ✅ Timeout handling
- ✅ Semaphore limiting
- ✅ Exponential backoff

### Manual Integration Test

```bash
# Start services
docker-compose up

# Test single scorecard (with all robustness features)
curl http://localhost:8000/api/scorecard/RELIANCE

# Test batch processing (with throttling)
curl -X POST http://localhost:8000/api/scorecard/batch-refresh \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]}'
```

---

## Error Handling Patterns

### 1. Graceful Degradation

When a service fails, the system falls back to increasingly stale data:

```
1. Try fresh API call (with retries)
   ↓ fails
2. Try fresh cache (< 12h)
   ↓ not found
3. Try stale cache (12h - 48h)
   ↓ not found
4. Try very stale cache (> 48h)
   ↓ not found
5. Return error response (not a crash!)
```

### 2. Partial Success

If some data sources fail, return what's available:

```json
{
  "symbol": "RELIANCE",
  "overall_score": 75,
  "fundamentals": {...},     // ✅ Success
  "sentiment": {             // ⚠️ Partial failure
    "sentiment_score": 0.0,
    "confidence": 0.0,
    "error": "Timeout after 45s"
  },
  "ai_summary": "..."        // ✅ Success
}
```

### 3. Structured Error Responses

All errors include context for debugging:

```json
{
  "error": "Timeout fetching Screener.in data",
  "details": {
    "url": "https://www.screener.in/company/RELIANCE/",
    "timeout": "10s",
    "retry_count": 3,
    "fallback_used": "yfinance"
  }
}
```

---

## Performance Considerations

### Rate Limits

**Scorecard generation:**
- Max 5 concurrent requests (semaphore)
- Additional requests queue automatically
- Average scorecard: 10-15 seconds

**Batch operations:**
- Process 5 symbols at a time
- 0.5s delay between batches
- 100 symbols: ~3-5 minutes

### Timeout Guidelines

| Operation | Timeout | Retries | Total Max Time |
|-----------|---------|---------|----------------|
| Screener.in HTTP | 10s | 3 | ~30s |
| yfinance fallback | 8s | 3 | ~24s |
| RSS news fetch | 15s | 3 | ~45s |
| LLM API call | 30s | 2 | ~90s |
| Full sentiment analysis | 45s | 1 | 45s |
| Scorecard generation | N/A | 0 | ~2 minutes |

### Caching Strategy

**Cache TTLs:**
- Fundamentals: 12 hours (default)
- News articles: 2 hours
- Vector store: persistent (saved every 50 docs)

**Cache warming:**
- Run batch refresh for top stocks during off-peak hours
- Pre-populate cache before market opens

---

## Troubleshooting

### Issue: Constant Timeouts

**Symptoms:** All requests timing out after 10-30s

**Possible causes:**
1. Network connectivity issues
2. Provider blocking/rate limiting
3. Provider API changes

**Solutions:**
```bash
# 1. Test connectivity
curl -I https://www.screener.in

# 2. Check if rate limited
curl -v https://www.screener.in/company/RELIANCE/ 2>&1 | grep -i "429\|rate"

# 3. Enable debug logging
export DEBUG_HTTP=true
python -c "from backend.services.screener_parser import parser; parser.fetch_fundamentals('RELIANCE')"

# 4. Test with different provider
# (add logic to switch providers based on failures)
```

### Issue: Stale Data Being Returned

**Symptoms:** Data is old despite fresh fetch attempts

**Check cache staleness:**
```python
from backend.services.fundamentals_db import db
import logging

logging.basicConfig(level=logging.DEBUG)

# This will log cache age in hours
data = db.get_latest('RELIANCE')
is_stale = db.is_stale('RELIANCE', max_age_hours=12)
print(f"Stale: {is_stale}")
```

**Force refresh:**
```bash
# Force refresh via API (bypasses cache)
curl -X POST http://localhost:8000/api/scorecard/refresh/RELIANCE
```

### Issue: Sentiment Analysis Failing

**Symptoms:** Sentiment returns 0.0 with error

**Debugging steps:**
```python
# 1. Check LLM provider availability
from backend.services.rag_sentiment_analyzer import create_sentiment_analyzer
analyzer = create_sentiment_analyzer()

# 2. Test with simple query
result = analyzer.analyze_sentiment(
    company_name="Test",
    symbol="TEST",
    articles=[{"title": "Test", "content": "Positive news"}],
    context_articles=[]
)
print(result)

# 3. Check API keys
import os
print("OpenAI API key:", "✅" if os.getenv("OPENAI_API_KEY") else "❌")
print("Anthropic API key:", "✅" if os.getenv("ANTHROPIC_API_KEY") else "❌")
print("Gemini API key:", "✅" if os.getenv("GEMINI_API_KEY") else "❌")
```

### Issue: High Memory Usage

**Symptoms:** Memory grows over time, OOM errors

**Possible causes:**
1. Vector store growing unbounded
2. HTTP sessions not being reused
3. Cache not being cleaned up

**Solutions:**
```python
# 1. Limit vector store size
from backend.services.vector_store import create_vector_store
vector_store = create_vector_store(...)
vector_store.clear_old_documents(days=7)  # Keep only recent

# 2. Sessions are already singleton (check _get_session patterns)

# 3. Clear old cache entries
from backend.services.fundamentals_db import db
# Add method to clean cache older than 7 days
```

---

## Future Improvements

### Planned Enhancements

1. **Circuit Breaker Pattern**
   - Temporarily disable failing providers
   - Auto-recovery after cool-down period

2. **Metrics & Monitoring**
   - Track retry counts, timeout frequencies
   - Alert on high error rates
   - Dashboard for cache hit rates

3. **Dynamic Timeout Adjustment**
   - Adjust timeouts based on provider latency
   - Use P95 latency metrics

4. **Provider Fallback Chain**
   - Multiple data sources per metric
   - Automatic provider switching on failures

5. **Distributed Caching**
   - Redis instead of SQLite
   - Shared cache across instances

6. **Request Deduplication**
   - Coalesce identical concurrent requests
   - Return same result to all callers

---

## References

- **Retry Logic:** urllib3.util.retry.Retry
- **Async Timeouts:** asyncio.wait_for
- **Rate Limiting:** asyncio.Semaphore
- **HTTP Sessions:** requests.Session with adapters
- **Threading:** concurrent.futures.ThreadPoolExecutor

**Documentation:**
- [requests retry](https://urllib3.readthedocs.io/en/stable/reference/urllib3.util.html#urllib3.util.Retry)
- [asyncio timeouts](https://docs.python.org/3/library/asyncio-task.html#asyncio.wait_for)
- [HTTP status codes](https://httpstatuses.com/)

---

**Last Updated:** December 2024
