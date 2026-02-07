"""
HTTP request debugging utilities for troubleshooting API issues.

Provides request/response logging with configurable detail levels.
"""

import logging
import time
from typing import Optional, Dict, Any
import requests
from functools import wraps

logger = logging.getLogger(__name__)

# Enable detailed HTTP logging for debugging (set via env)
import os
DEBUG_HTTP = os.getenv("DEBUG_HTTP", "false").lower() == "true"


def log_request(
    method: str,
    url: str,
    headers: Optional[Dict] = None,
    params: Optional[Dict] = None,
    data: Any = None,
    response: Optional[requests.Response] = None,
    elapsed_ms: Optional[float] = None,
    error: Optional[Exception] = None,
):
    """
    Log HTTP request details for debugging.
    
    Args:
        method: HTTP method
        url: Full URL
        headers: Request headers (secrets will be masked)
        params: Query parameters
        data: Request body (truncated if large)
        response: Response object
        elapsed_ms: Request duration in milliseconds
        error: Exception if request failed
    """
    if not DEBUG_HTTP:
        return
    
    # Mask sensitive headers
    safe_headers = {}
    if headers:
        for k, v in headers.items():
            if k.lower() in ('authorization', 'api-key', 'x-api-key', 'token'):
                safe_headers[k] = f"{v[:8]}***" if len(v) > 8 else "***"
            else:
                safe_headers[k] = v
    
    log_parts = [
        f"HTTP {method} {url}",
        f"params={params}" if params else None,
        f"headers={safe_headers}" if safe_headers else None,
    ]
    
    if data:
        data_str = str(data)
        if len(data_str) > 200:
            data_str = data_str[:200] + "..."
        log_parts.append(f"data={data_str}")
    
    if elapsed_ms is not None:
        log_parts.append(f"elapsed={elapsed_ms:.0f}ms")
    
    if error:
        log_parts.append(f"ERROR: {type(error).__name__}: {str(error)}")
        logger.error(" | ".join(filter(None, log_parts)))
    elif response:
        log_parts.append(f"status={response.status_code}")
        
        # Log response snippet
        try:
            if response.headers.get('content-type', '').startswith('application/json'):
                resp_text = response.text[:300]
                if len(response.text) > 300:
                    resp_text += "..."
                log_parts.append(f"response={resp_text}")
        except:
            pass
        
        if response.ok:
            logger.info(" | ".join(filter(None, log_parts)))
        else:
            logger.warning(" | ".join(filter(None, log_parts)))
    else:
        logger.debug(" | ".join(filter(None, log_parts)))


def debug_http_call(func):
    """
    Decorator to log HTTP request/response details for debugging.
    
    Usage:
        @debug_http_call
        def fetch_data(url):
            return requests.get(url)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not DEBUG_HTTP:
            return func(*args, **kwargs)
        
        start = time.time()
        error = None
        response = None
        
        try:
            response = func(*args, **kwargs)
            return response
        except Exception as e:
            error = e
            raise
        finally:
            elapsed_ms = (time.time() - start) * 1000
            
            # Try to extract request details from args/kwargs
            method = kwargs.get('method', 'GET')
            url = args[0] if args else kwargs.get('url', 'unknown')
            headers = kwargs.get('headers')
            params = kwargs.get('params')
            data = kwargs.get('data') or kwargs.get('json')
            
            log_request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                data=data,
                response=response,
                elapsed_ms=elapsed_ms,
                error=error,
            )
    
    return wrapper


def test_provider_endpoint(url: str, method: str = 'GET', **kwargs):
    """
    Test an API endpoint and log full request/response for debugging.
    
    Args:
        url: Full endpoint URL
        method: HTTP method
        **kwargs: Additional arguments for requests (params, headers, etc.)
        
    Example:
        test_provider_endpoint('https://api.example.com/quote', params={'symbol': 'RELIANCE'})
    """
    logger.info(f"Testing endpoint: {method} {url}")
    logger.info(f"Request params: {kwargs}")
    
    start = time.time()
    try:
        response = requests.request(method, url, **kwargs)
        elapsed = (time.time() - start) * 1000
        
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response headers: {dict(response.headers)}")
        logger.info(f"Response time: {elapsed:.0f}ms")
        
        # Log response body
        try:
            if response.headers.get('content-type', '').startswith('application/json'):
                import json
                logger.info(f"Response JSON: {json.dumps(response.json(), indent=2)}")
            else:
                logger.info(f"Response text (first 500 chars): {response.text[:500]}")
        except:
            logger.info(f"Response text (first 500 chars): {response.text[:500]}")
        
        return response
    
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        logger.error(f"Request failed after {elapsed:.0f}ms: {type(e).__name__}: {str(e)}")
        raise
