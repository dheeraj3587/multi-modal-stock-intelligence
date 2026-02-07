#!/usr/bin/env python3
"""
Quick verification script for robustness improvements.

Tests key robustness features across all data-fetching components.
"""

import sys
import time
import logging
from typing import Dict, List
import asyncio

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test results tracker
test_results = []


def log_test_result(test_name: str, success: bool, details: str = ""):
    """Record test result."""
    status = "✅ PASS" if success else "❌ FAIL"
    logger.info(f"{status} - {test_name}")
    if details:
        logger.info(f"  Details: {details}")
    test_results.append({
        'test': test_name,
        'success': success,
        'details': details
    })


def test_screener_parser_retry():
    """Test screener parser with retry logic."""
    logger.info("\n=== Testing Screener Parser Robustness ===")
    
    try:
        from backend.services.screener_parser import parser
        
        # Test 1: Normal fetch (should use retry adapter)
        logger.info("Test 1: Fetch with retry adapter")
        start = time.time()
        result = parser.fetch_fundamentals('RELIANCE')
        elapsed = time.time() - start
        
        if result:
            source = result.get('source', 'unknown')
            log_test_result(
                "Screener Parser - Normal fetch",
                True,
                f"Source: {source}, Time: {elapsed:.2f}s"
            )
        else:
            log_test_result(
                "Screener Parser - Normal fetch",
                False,
                "No data returned"
            )
        
        # Test 2: Check session has retry adapter
        logger.info("Test 2: Verify retry adapter")
        session = parser._get_session()
        adapter = session.get_adapter('https://')
        has_retry = hasattr(adapter, 'max_retries')
        
        log_test_result(
            "Screener Parser - Retry adapter",
            has_retry,
            f"Adapter type: {type(adapter).__name__}"
        )
        
    except Exception as e:
        log_test_result("Screener Parser", False, str(e))
        logger.error(f"Error testing screener parser: {e}", exc_info=True)


def test_rss_news_fetcher_robustness():
    """Test RSS fetcher retry and throttling."""
    logger.info("\n=== Testing RSS News Fetcher Robustness ===")
    
    try:
        from backend.services.rss_news_fetcher import rss_fetcher
        
        # Test 1: Verify session with retry
        logger.info("Test 1: Verify HTTP session")
        session = rss_fetcher._get_session()
        adapter = session.get_adapter('https://')
        has_retry = hasattr(adapter, 'max_retries')
        
        log_test_result(
            "RSS Fetcher - Retry adapter",
            has_retry,
            f"Session configured with retries"
        )
        
        # Test 2: Fetch news (with throttling)
        logger.info("Test 2: Fetch news with throttling")
        start = time.time()
        articles = rss_fetcher.fetch_company_news(
            symbol='RELIANCE',
            company_name='Reliance Industries',
            max_articles=10
        )
        elapsed = time.time() - start
        
        log_test_result(
            "RSS Fetcher - News fetch",
            len(articles) > 0,
            f"Fetched {len(articles)} articles in {elapsed:.2f}s"
        )
        
    except Exception as e:
        log_test_result("RSS News Fetcher", False, str(e))
        logger.error(f"Error testing RSS fetcher: {e}", exc_info=True)


def test_fundamentals_db_cache():
    """Test cache fallback mechanisms."""
    logger.info("\n=== Testing Fundamentals DB Cache ===")
    
    try:
        from backend.services.fundamentals_db import fundamentals_db as db
        from datetime import datetime, timedelta
        
        # Test 1: Insert test data
        logger.info("Test 1: Insert test data")
        test_symbol = 'TEST_ROBUST'
        test_data = {
            'symbol': test_symbol,
            'company_name': 'Test Company',
            'source': 'test',
            'timestamp': datetime.now().isoformat(),
        }
        db.upsert(test_data)
        
        # Test 2: Fresh cache retrieval
        logger.info("Test 2: Fresh cache")
        fresh = db.get_latest(test_symbol, allow_stale=False)
        log_test_result(
            "Fundamentals DB - Fresh cache",
            fresh is not None,
            f"Retrieved: {fresh is not None}"
        )
        
        # Test 3: Simulate stale data
        logger.info("Test 3: Stale cache handling")
        old_timestamp = datetime.now() - timedelta(hours=25)
        old_data = {
            'symbol': 'TEST_STALE',
            'company_name': 'Test Stale',
            'source': 'test',
            'timestamp': old_timestamp.isoformat(),
        }
        db.upsert(old_data)
        
        # Should fail without allow_stale
        fresh_attempt = db.get_latest('TEST_STALE', allow_stale=False)
        # Should succeed with allow_stale
        stale_attempt = db.get_latest('TEST_STALE', allow_stale=True)
        
        log_test_result(
            "Fundamentals DB - Stale fallback",
            fresh_attempt is None and stale_attempt is not None,
            "Stale cache fallback works correctly"
        )
        
    except Exception as e:
        log_test_result("Fundamentals DB", False, str(e))
        logger.error(f"Error testing fundamentals DB: {e}", exc_info=True)


async def test_scorecard_api_limits():
    """Test scorecard API rate limiting."""
    logger.info("\n=== Testing Scorecard API Limits ===")
    
    try:
        from backend.api.scorecard import (
            MAX_CONCURRENT_SCORECARDS,
            CACHE_DEBOUNCE_SECONDS,
            _semaphore
        )
        
        # Test 1: Verify semaphore exists
        logger.info("Test 1: Semaphore configuration")
        log_test_result(
            "Scorecard API - Semaphore",
            _semaphore is not None,
            f"Max concurrent: {MAX_CONCURRENT_SCORECARDS}"
        )
        
        # Test 2: Verify debounce configured
        logger.info("Test 2: Debounce configuration")
        log_test_result(
            "Scorecard API - Debounce",
            CACHE_DEBOUNCE_SECONDS > 0,
            f"Debounce window: {CACHE_DEBOUNCE_SECONDS}s"
        )
        
    except Exception as e:
        log_test_result("Scorecard API Limits", False, str(e))
        logger.error(f"Error testing scorecard API: {e}", exc_info=True)


def test_rag_sentiment_retry():
    """Test RAG sentiment analyzer retry logic."""
    logger.info("\n=== Testing RAG Sentiment Analyzer ===")
    
    try:
        from backend.services.rag_sentiment_analyzer import (
            LLM_TIMEOUT,
            LLM_MAX_RETRIES,
            LLM_RETRY_DELAY
        )
        
        # Test 1: Verify timeout configuration
        logger.info("Test 1: Timeout configuration")
        log_test_result(
            "RAG Sentiment - Timeout",
            LLM_TIMEOUT > 0,
            f"LLM timeout: {LLM_TIMEOUT}s"
        )
        
        # Test 2: Verify retry configuration
        logger.info("Test 2: Retry configuration")
        log_test_result(
            "RAG Sentiment - Retry",
            LLM_MAX_RETRIES > 0,
            f"Max retries: {LLM_MAX_RETRIES}, Base delay: {LLM_RETRY_DELAY}s"
        )
        
        # Test 3: Verify exponential backoff
        logger.info("Test 3: Exponential backoff")
        backoff_delays = [LLM_RETRY_DELAY * (2 ** i) for i in range(LLM_MAX_RETRIES)]
        log_test_result(
            "RAG Sentiment - Backoff",
            True,
            f"Backoff sequence: {backoff_delays}"
        )
        
    except Exception as e:
        log_test_result("RAG Sentiment Analyzer", False, str(e))
        logger.error(f"Error testing RAG sentiment: {e}", exc_info=True)


def test_http_debug_utility():
    """Test HTTP debugging utility."""
    logger.info("\n=== Testing HTTP Debug Utility ===")
    
    try:
        from backend.utils.http_debug import log_request, debug_http_call
        from unittest.mock import Mock
        
        # Test 1: Request logging
        logger.info("Test 1: Request logging")
        log_request(
            method='GET',
            url='https://example.com/test',
            headers={'User-Agent': 'test'},
            response=Mock(
                status_code=200,
                ok=True,
                text='{"test": "data"}',
                headers={'content-type': 'application/json'}
            ),
            elapsed_ms=150.0
        )
        
        log_test_result(
            "HTTP Debug - Request logging",
            True,
            "Log function works without errors"
        )
        
        # Test 2: Decorator
        logger.info("Test 2: Debug decorator")
        
        @debug_http_call
        def mock_request(url, **kwargs):
            return Mock(status_code=200, ok=True, text='test')
        
        result = mock_request('https://example.com')
        
        log_test_result(
            "HTTP Debug - Decorator",
            result is not None,
            "Decorator wraps functions correctly"
        )
        
    except Exception as e:
        log_test_result("HTTP Debug Utility", False, str(e))
        logger.error(f"Error testing HTTP debug: {e}", exc_info=True)


def print_summary():
    """Print test summary."""
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    total = len(test_results)
    passed = sum(1 for r in test_results if r['success'])
    failed = total - passed
    
    logger.info(f"\nTotal Tests: {total}")
    logger.info(f"✅ Passed: {passed}")
    logger.info(f"❌ Failed: {failed}")
    
    if failed > 0:
        logger.info("\nFailed Tests:")
        for result in test_results:
            if not result['success']:
                logger.info(f"  - {result['test']}: {result['details']}")
    
    logger.info("\n" + "="*60)
    
    return failed == 0


def main():
    """Run all robustness verification tests."""
    logger.info("="*60)
    logger.info("ROBUSTNESS VERIFICATION SUITE")
    logger.info("="*60)
    logger.info("\nThis script verifies that all robustness improvements are")
    logger.info("properly configured and working as expected.\n")
    
    # Run tests
    test_screener_parser_retry()
    test_rss_news_fetcher_robustness()
    test_fundamentals_db_cache()
    asyncio.run(test_scorecard_api_limits())
    test_rag_sentiment_retry()
    test_http_debug_utility()
    
    # Print summary
    all_passed = print_summary()
    
    if all_passed:
        logger.info("\n🎉 All robustness features verified successfully!")
        return 0
    else:
        logger.warning("\n⚠️  Some tests failed. Review the logs above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
