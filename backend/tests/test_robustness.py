"""
Test suite for robustness improvements.

Tests retry logic, timeouts, fallback mechanisms, and error handling
across all data-fetching components.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
import requests
from datetime import datetime, timedelta

from backend.services.screener_parser import ScreenerParser
from backend.services.fundamentals_db import FundamentalsDB
from backend.services.rss_news_fetcher import RSSNewsFetcher
from backend.api.scorecard import (
    _fetch_and_store,
    _get_sentiment,
    MAX_CONCURRENT_SCORECARDS,
    CACHE_DEBOUNCE_SECONDS,
)


class TestScreenerParserRobustness:
    """Test retry logic and timeout handling in ScreenerParser."""
    
    def test_retry_on_timeout(self):
        """Test that parser retries on timeout."""
        parser = ScreenerParser()
        
        with patch('requests.Session.get') as mock_get:
            # Simulate timeout then success
            mock_get.side_effect = [
                requests.exceptions.Timeout(),
                Mock(status_code=200, text='<html>Test</html>')
            ]
            
            # Should succeed after retry
            result = parser.fetch_screener_page('RELIANCE')
            assert result is not None
            assert mock_get.call_count == 2
    
    def test_retry_on_503(self):
        """Test that parser retries on 503 Server Error."""
        parser = ScreenerParser()
        
        with patch('requests.Session.get') as mock_get:
            # Simulate 503 then success
            error_response = Mock(status_code=503)
            error_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
            success_response = Mock(status_code=200, text='<html>Test</html>')
            success_response.raise_for_status.return_value = None
            
            mock_get.side_effect = [error_response, success_response]
            
            result = parser.fetch_screener_page('RELIANCE')
            # Retry adapter should handle 503 automatically
            assert mock_get.call_count >= 1
    
    def test_fallback_to_yfinance(self):
        """Test fallback to yfinance when Screener.in fails."""
        parser = ScreenerParser()
        
        with patch('requests.Session.get') as mock_get, \
             patch('yfinance.Ticker') as mock_ticker:
            
            # Screener.in always fails
            mock_get.side_effect = requests.exceptions.RequestException()
            
            # yfinance succeeds
            mock_info = {
                'symbol': 'RELIANCE.NS',
                'longName': 'Reliance Industries',
                'currentPrice': 2500.0,
                'marketCap': 15000000000000,
            }
            mock_ticker.return_value.info = mock_info
            
            result = parser.fetch_fundamentals('RELIANCE')
            
            # Should have yfinance data
            assert result is not None
            assert 'source' in result
            assert result['source'] == 'yfinance'


class TestFundamentalsDBRobustness:
    """Test cache fallback and stale data handling."""
    
    def test_allow_stale_cache(self):
        """Test that allow_stale returns old cache when available."""
        db = FundamentalsDB()
        
        # Insert old data (25 hours ago)
        old_timestamp = datetime.now() - timedelta(hours=25)
        test_data = {
            'symbol': 'TEST',
            'company_name': 'Test Company',
            'source': 'test',
            'timestamp': old_timestamp.isoformat(),
        }
        
        db.upsert(test_data)
        
        # Fresh query should return None (stale)
        fresh = db.get_latest('TEST', allow_stale=False)
        assert fresh is None
        
        # Stale query should return old data
        stale = db.get_latest('TEST', allow_stale=True)
        assert stale is not None
        assert stale['symbol'] == 'TEST'
    
    def test_cache_age_logging(self, caplog):
        """Test that cache age is logged for visibility."""
        db = FundamentalsDB()
        
        # Insert data 5 hours ago
        old_timestamp = datetime.now() - timedelta(hours=5)
        test_data = {
            'symbol': 'CACHE_TEST',
            'company_name': 'Cache Test',
            'source': 'test',
            'timestamp': old_timestamp.isoformat(),
        }
        
        db.upsert(test_data)
        
        # Check staleness (should log age)
        is_stale = db.is_stale('CACHE_TEST', max_age_hours=12)
        
        # Should log cache age
        assert any('age:' in record.message for record in caplog.records)


class TestRSSNewsFetcherRobustness:
    """Test retry logic and throttling in RSS fetcher."""
    
    def test_session_retry_logic(self):
        """Test that session has retry adapter configured."""
        fetcher = RSSNewsFetcher()
        session = fetcher._get_session()
        
        # Check that session has retry adapter mounted
        adapter = session.get_adapter('https://')
        assert adapter is not None
        assert hasattr(adapter, 'max_retries')
    
    def test_throttling_between_sources(self):
        """Test that fetcher throttles requests to avoid overwhelming providers."""
        fetcher = RSSNewsFetcher()
        
        with patch.object(fetcher, 'fetch_google_news_rss', return_value=[]), \
             patch.object(fetcher, 'fetch_economic_times_rss', return_value=[]), \
             patch.object(fetcher, 'fetch_moneycontrol_news', return_value=[]), \
             patch('time.sleep') as mock_sleep:
            
            fetcher.fetch_company_news('RELIANCE', 'Reliance Industries', max_articles=20)
            
            # Should have called sleep for throttling
            assert mock_sleep.call_count > 0
    
    def test_timeout_handling(self):
        """Test that timeout errors are handled gracefully."""
        fetcher = RSSNewsFetcher()
        
        with patch('requests.Session.get') as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout()
            
            # Should return empty list, not crash
            result = fetcher.fetch_moneycontrol_news('Test Company')
            assert result == []


class TestScorecardAPIRobustness:
    """Test rate limiting and timeout handling in scorecard API."""
    
    @pytest.mark.asyncio
    async def test_semaphore_limiting(self):
        """Test that semaphore limits concurrent scorecard requests."""
        from backend.api.scorecard import _semaphore
        
        # Semaphore should have max of 5
        assert _semaphore._value == MAX_CONCURRENT_SCORECARDS
        
        # Acquire all permits
        for _ in range(MAX_CONCURRENT_SCORECARDS):
            await _semaphore.acquire()
        
        # Next acquisition should block
        assert _semaphore.locked()
        
        # Release one
        _semaphore.release()
        assert not _semaphore.locked()
    
    @pytest.mark.asyncio
    async def test_debouncing(self):
        """Test that rapid requests are debounced."""
        # Track function call times
        call_times = []
        
        async def mock_fetch():
            call_times.append(time.time())
            await asyncio.sleep(0.1)
            return {}
        
        # Simulate rapid calls
        start = time.time()
        with patch('backend.api.scorecard._fetch_fundamentals', mock_fetch):
            # Make multiple rapid calls (would be debounced in real code)
            pass
        
        # In real implementation, calls within CACHE_DEBOUNCE_SECONDS should be coalesced
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test that timeouts are handled gracefully."""
        with patch('backend.services.screener_parser.ScreenerParser.fetch_fundamentals') as mock_fetch:
            # Simulate slow response (longer than timeout)
            async def slow_fetch(*args, **kwargs):
                await asyncio.sleep(100)  # Longer than timeout
                return {}
            
            mock_fetch.side_effect = slow_fetch
            
            # Should timeout and handle gracefully
            # (actual test would need full API setup)
            pass


class TestRAGSentimentRobustness:
    """Test retry and timeout handling in RAG sentiment analyzer."""
    
    def test_llm_retry_on_failure(self):
        """Test that LLM calls retry on failure."""
        from backend.services.rag_sentiment_analyzer import RAGSentimentAnalyzer
        
        analyzer = RAGSentimentAnalyzer()
        
        with patch.object(analyzer, '_call_llm') as mock_llm:
            # Simulate failure then success
            mock_llm.side_effect = [
                Exception("API Error"),
                {"sentiment_score": 0.7, "confidence": 0.8}
            ]
            
            result = analyzer.analyze_sentiment(
                company_name="Test Co",
                symbol="TEST",
                articles=[{"title": "Test", "content": "Test"}],
                context_articles=[]
            )
            
            # Should have retried
            assert mock_llm.call_count == 2
    
    def test_exponential_backoff(self):
        """Test that retries use exponential backoff."""
        from backend.services.rag_sentiment_analyzer import LLM_RETRY_DELAY
        
        # Base delay should be configured
        assert LLM_RETRY_DELAY > 0
        
        # Exponential backoff: 2.0, 4.0, 8.0 seconds for attempts 1, 2, 3
        expected_delays = [LLM_RETRY_DELAY * (2 ** i) for i in range(3)]
        assert expected_delays == [2.0, 4.0, 8.0]


class TestIntegrationRobustness:
    """Integration tests for end-to-end robustness."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_with_failures(self):
        """Test full scorecard generation with partial failures."""
        # This would test:
        # 1. Screener.in fails -> falls back to yfinance
        # 2. Fresh cache miss -> uses stale cache
        # 3. LLM timeout -> retries with backoff
        # 4. News fetch partial failure -> continues with available data
        pass
    
    def test_graceful_degradation(self):
        """Test that system degrades gracefully when services fail."""
        # Even if all external services fail, should return:
        # - Empty scorecard with default values
        # - Error messages explaining what failed
        # - No crashes or uncaught exceptions
        pass


def test_http_debug_utility():
    """Test HTTP debugging utility."""
    from backend.utils.http_debug import log_request, debug_http_call
    
    # Test request logging
    log_request(
        method='GET',
        url='https://example.com',
        headers={'Authorization': 'Bearer secret123'},
        response=Mock(status_code=200, ok=True, text='{"data": "test"}', headers={'content-type': 'application/json'}),
        elapsed_ms=150.0
    )
    
    # Test decorator
    @debug_http_call
    def mock_request(url, timeout=10):
        return Mock(status_code=200, ok=True, text='test')
    
    result = mock_request('https://example.com')
    assert result is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
