"""
API interaction utilities with retry logic, rate limiting, and error handling.

Provides decorators and helper functions for robust API calls with exponential
backoff, token bucket rate limiting, and standardized error handling.
"""

import time
import functools
import json
from typing import Callable, Dict, Optional, Any
from datetime import datetime, timedelta
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError


class APIError(Exception):
    """Base exception for API errors."""
    pass


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    pass


class APIResponseError(APIError):
    """Raised when API returns an error response."""
    pass


class RateLimiter:
    """
    Token bucket rate limiter for API requests.
    
    Implements token bucket algorithm to enforce rate limits.
    """
    
    def __init__(self, max_requests: int, time_window: int):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed in time window.
            time_window: Time window in seconds.
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.tokens = max_requests
        self.last_update = datetime.now()
    
    def _refill_tokens(self):
        """Refill tokens based on elapsed time."""
        now = datetime.now()
        elapsed = (now - self.last_update).total_seconds()
        
        # Calculate tokens to add based on elapsed time
        tokens_to_add = (elapsed / self.time_window) * self.max_requests
        self.tokens = min(self.max_requests, self.tokens + tokens_to_add)
        self.last_update = now
    
    def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens for a request.
        
        Args:
            tokens: Number of tokens to acquire (default 1).
            
        Returns:
            True if tokens acquired, False otherwise.
        """
        self._refill_tokens()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def wait_for_token(self, tokens: int = 1):
        """
        Wait until tokens are available.
        
        Args:
            tokens: Number of tokens to acquire (default 1).
        """
        while not self.acquire(tokens):
            # Calculate wait time
            wait_time = (tokens - self.tokens) / self.max_requests * self.time_window
            time.sleep(max(0.1, wait_time))


def retry_with_backoff(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    exceptions: tuple = (RequestException, Timeout, ConnectionError)
):
    """
    Decorator to retry function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts.
        backoff_factor: Multiplier for delay between retries.
        initial_delay: Initial delay in seconds before first retry.
        exceptions: Tuple of exceptions to catch and retry.
        
    Returns:
        Decorated function with retry logic.
        
    Example:
        @retry_with_backoff(max_retries=3, backoff_factor=2)
        def fetch_data():
            return requests.get('https://api.example.com/data')
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        # Log retry attempt (import logger to avoid circular dependency)
                        from backend.utils.logger import get_logger
                        logger = get_logger(__name__)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {str(e)}. "
                            f"Retrying in {delay:.1f} seconds..."
                        )
                        
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        # Max retries exceeded
                        from backend.utils.logger import get_logger
                        logger = get_logger(__name__)
                        logger.error(f"Max retries ({max_retries}) exceeded. Last error: {str(e)}")
                        raise last_exception
            
            # Should never reach here, but raise last exception if it does
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


def make_api_request(
    url: str,
    method: str = 'GET',
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    json_data: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
    rate_limiter: Optional[RateLimiter] = None
) -> requests.Response:
    """
    Make an API request with retry logic and rate limiting.
    
    Args:
        url: API endpoint URL.
        method: HTTP method (GET, POST, etc.).
        headers: Request headers.
        params: Query parameters.
        json_data: JSON request body.
        timeout: Request timeout in seconds.
        rate_limiter: Optional RateLimiter instance.
        
    Returns:
        Response object.
        
    Raises:
        RateLimitError: If rate limit is exceeded.
        APIResponseError: If API returns error status code.
        RequestException: If request fails after retries.
    """
    # Wait for rate limiter token if provided
    if rate_limiter:
        rate_limiter.wait_for_token()
    
    # Make request with retry logic
    @retry_with_backoff(max_retries=3, backoff_factor=2.0)
    def _do_request():
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=json_data,
            timeout=timeout
        )
        
        # Check for rate limit error
        if response.status_code == 429:
            retry_after = response.headers.get('Retry-After')
            error_msg = f"Rate limit exceeded. "
            if retry_after:
                error_msg += f"Retry after {retry_after} seconds."
            raise RateLimitError(error_msg)
        
        # Check for other HTTP errors
        if not response.ok:
            raise APIResponseError(
                f"API request failed with status {response.status_code}: {response.text}"
            )
        
        return response
    
    return _do_request()


def validate_api_response(
    response: requests.Response,
    expected_keys: Optional[list] = None,
    response_format: str = 'json'
) -> Any:
    """
    Validate and parse API response.
    
    Args:
        response: Response object from requests.
        expected_keys: List of required keys in JSON response.
        response_format: Expected response format ('json' or 'text').
        
    Returns:
        Parsed response data (dict for JSON, string for text).
        
    Raises:
        APIResponseError: If response validation fails.
    """
    try:
        # Parse response based on format
        if response_format == 'json':
            data = response.json()
            
            # Validate expected keys if provided
            if expected_keys:
                if not isinstance(data, dict):
                    raise APIResponseError("Expected JSON object in response")
                
                missing_keys = [key for key in expected_keys if key not in data]
                if missing_keys:
                    raise APIResponseError(
                        f"Missing required keys in response: {', '.join(missing_keys)}"
                    )
            
            return data
        
        elif response_format == 'text':
            return response.text
        
        else:
            raise ValueError(f"Unsupported response format: {response_format}")
            
    except (ValueError, json.JSONDecodeError) as e:
        raise APIResponseError(f"Failed to parse JSON response: {str(e)}")


def handle_api_error(error: Exception, context: str = "") -> None:
    """
    Handle and log API errors with context.
    
    Args:
        error: Exception that occurred.
        context: Additional context about the operation.
    """
    from backend.utils.logger import get_logger
    logger = get_logger(__name__)
    
    error_msg = f"API error"
    if context:
        error_msg += f" ({context})"
    error_msg += f": {str(error)}"
    
    if isinstance(error, RateLimitError):
        logger.warning(error_msg)
    elif isinstance(error, (APIResponseError, RequestException)):
        logger.error(error_msg)
    else:
        logger.exception(error_msg)
