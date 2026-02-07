"""
Screener.in HTML parser - Fetches and parses company fundamentals from screener.in.

Flow: Screener.in (HTML) → Parser → JSON → DB (company_fundamentals)
"""

import re
import json
import logging
from typing import Dict, Optional, List
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Screener.in base URL
SCREENER_BASE_URL = "https://www.screener.in/company"

# Configure robust requests session with retries
_session = None

def _get_session() -> requests.Session:
    """Get or create a requests session with retry configuration."""
    global _session
    if _session is None:
        _session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.8,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        _session.mount('https://', adapter)
        _session.mount('http://', adapter)
        logger.info("Initialized requests session with retry logic")
    return _session

# Common NSE ticker → Screener.in slug mapping
TICKER_SLUG_MAP = {
    "RELIANCE": "RELIANCE",
    "TCS": "TCS",
    "HDFCBANK": "HDFCBANK",
    "INFY": "INFY",
    "ICICIBANK": "ICICIBANK",
    "SBIN": "SBIN",
    "BHARTIARTL": "BHARTIARTL",
    "ITC": "ITC",
    "KOTAKBANK": "KOTAKBANK",
    "LT": "LT",
    "HINDUNILVR": "HINDUNILVR",
    "AXISBANK": "AXISBANK",
    "WIPRO": "WIPRO",
    "BAJFINANCE": "BAJFINANCE",
    "MARUTI": "MARUTI",
    "ASIANPAINT": "ASIANPAINT",
    "TATAMOTORS": "TATAMOTORS",
    "SUNPHARMA": "SUNPHARMA",
    "ULTRACEMCO": "ULTRACEMCO",
    "TITAN": "TITAN",
    "NESTLEIND": "NESTLEIND",
    "POWERGRID": "POWERGRID",
    "NTPC": "NTPC",
    "TATASTEEL": "TATASTEEL",
    "ADANIENT": "ADANIENT",
    "JSWSTEEL": "JSWSTEEL",
    "ONGC": "ONGC",
    "COALINDIA": "COALINDIA",
    "TECHM": "TECHM",
    "HCLTECH": "HCLTECH",
}

# User-Agent to mimic a browser
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def _safe_float(text: str) -> Optional[float]:
    """Safely parse a float from text, stripping commas, %, ₹, Cr., etc."""
    if not text:
        return None
    cleaned = text.strip().replace(",", "").replace("₹", "").replace("%", "").replace("Cr.", "").strip()
    if cleaned in ("", "-", "—", "N/A"):
        return None
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _safe_int(text: str) -> Optional[int]:
    """Safely parse an int from text."""
    val = _safe_float(text)
    if val is not None:
        return int(val)
    return None


def _extract_ratios_section(soup: BeautifulSoup) -> Dict:
    """Extract the top-level ratios section (Market Cap, P/E, ROCE, etc.)."""
    ratios = {}
    
    # The top ratios are in a list with id="top-ratios"
    top_ratios = soup.find("ul", {"id": "top-ratios"})
    if not top_ratios:
        # Fallback: look for the ratios container
        top_ratios = soup.find("div", {"id": "top-ratios"})
    
    if top_ratios:
        items = top_ratios.find_all("li")
        for item in items:
            name_el = item.find("span", {"class": "name"})
            value_el = item.find("span", {"class": "number"})
            if name_el and value_el:
                key = name_el.get_text(strip=True).lower()
                val = value_el.get_text(strip=True)
                
                key_map = {
                    "market cap": "market_cap_cr",
                    "current price": "current_price",
                    "high / low": "high_low",
                    "stock p/e": "pe_ratio",
                    "book value": "book_value",
                    "dividend yield": "dividend_yield",
                    "roce": "roce",
                    "roe": "roe",
                    "face value": "face_value",
                    "industry pe": "industry_pe",
                    "debt": "debt_cr",
                    "eps": "eps",
                    "promoter holding": "promoter_holding",
                    "pledged percentage": "pledged_pct",
                    "price to book value": "price_to_book",
                    "price to earning": "pe_ratio",
                }
                
                mapped_key = key_map.get(key, key.replace(" ", "_").replace("/", "_"))
                ratios[mapped_key] = val
    
    return ratios


def _extract_table_data(soup: BeautifulSoup, section_id: str) -> List[Dict]:
    """Extract tabular data from a screener section (quarterly results, P&L, etc.)."""
    rows = []
    section = soup.find("section", {"id": section_id})
    if not section:
        return rows
    
    table = section.find("table")
    if not table:
        return rows
    
    # Extract headers
    thead = table.find("thead")
    headers = []
    if thead:
        for th in thead.find_all("th"):
            headers.append(th.get_text(strip=True))
    
    # Extract rows
    tbody = table.find("tbody")
    if tbody:
        for tr in tbody.find_all("tr"):
            cells = tr.find_all("td")
            if cells and headers:
                row = {}
                for i, cell in enumerate(cells):
                    if i < len(headers):
                        row[headers[i]] = cell.get_text(strip=True)
                if row:
                    rows.append(row)
    
    return rows


def _extract_peers(soup: BeautifulSoup) -> List[Dict]:
    """Extract peer comparison data."""
    peers = []
    peer_section = soup.find("section", {"id": "peers"})
    if not peer_section:
        return peers
    
    table = peer_section.find("table")
    if not table:
        return peers
    
    headers = []
    thead = table.find("thead")
    if thead:
        for th in thead.find_all("th"):
            headers.append(th.get_text(strip=True))
    
    tbody = table.find("tbody")
    if tbody:
        for tr in tbody.find_all("tr"):
            cells = tr.find_all("td")
            row = {}
            for i, cell in enumerate(cells):
                if i < len(headers):
                    # Check if the cell contains a link (company name)
                    link = cell.find("a")
                    if link:
                        row[headers[i]] = link.get_text(strip=True)
                    else:
                        row[headers[i]] = cell.get_text(strip=True)
            if row:
                peers.append(row)
    
    return peers


def _extract_pros_cons(soup: BeautifulSoup) -> Dict[str, List[str]]:
    """Extract pros and cons sections."""
    result = {"pros": [], "cons": []}
    
    for section_class, key in [("pros", "pros"), ("cons", "cons")]:
        section = soup.find("div", {"class": section_class})
        if section:
            items = section.find_all("li")
            for item in items:
                text = item.get_text(strip=True)
                if text:
                    result[key].append(text)
    
    return result


def _extract_shareholding(soup: BeautifulSoup) -> List[Dict]:
    """Extract shareholding pattern data."""
    return _extract_table_data(soup, "shareholding")


def _extract_company_info(soup: BeautifulSoup) -> Dict:
    """Extract basic company information from the page header."""
    info = {}
    
    # Company name
    h1 = soup.find("h1")
    if h1:
        info["company_name"] = h1.get_text(strip=True)
    
    # Sector / Industry info
    company_info = soup.find("div", {"class": "company-info"})
    if company_info:
        links = company_info.find_all("a")
        sectors = [a.get_text(strip=True) for a in links]
        if len(sectors) >= 1:
            info["sector"] = sectors[0]
        if len(sectors) >= 2:
            info["industry"] = sectors[1]
    
    return info


def fetch_screener_page(symbol: str, timeout: int = 10) -> Optional[str]:
    """
    Fetch HTML from screener.in for a given stock symbol with robust retry logic.
    
    Args:
        symbol: NSE ticker symbol (e.g., "RELIANCE")
        timeout: Request timeout in seconds (default 10s)
        
    Returns:
        HTML content as string, or None if fetch failed
    """
    slug = TICKER_SLUG_MAP.get(symbol.upper(), symbol.upper())
    url = f"{SCREENER_BASE_URL}/{slug}/consolidated/"
    
    logger.info(f"Fetching screener.in page: {url}")
    session = _get_session()
    
    try:
        # Try consolidated URL first (with automatic retries via session)
        response = session.get(url, headers=HEADERS, timeout=timeout)
        
        if response.status_code == 404:
            # Try standalone (non-consolidated) URL
            url = f"{SCREENER_BASE_URL}/{slug}/"
            logger.info(f"Consolidated URL returned 404, trying standalone: {url}")
            response = session.get(url, headers=HEADERS, timeout=timeout)
        
        response.raise_for_status()
        logger.info(f"Successfully fetched screener page for {symbol} (status: {response.status_code})")
        return response.text
            
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout fetching screener.in for {symbol} after {timeout}s: {e}")
        return None
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error {e.response.status_code} for {symbol}: {e}")
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error fetching screener.in for {symbol}: {e}")
        return None
    except requests.RequestException as e:
        logger.error(f"Request failed for screener.in {symbol}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error fetching screener.in for {symbol}: {e}")
        return None


def parse_screener_html(html: str, symbol: str) -> Dict:
    """
    Parse Screener.in HTML into structured JSON.
    
    Args:
        html: Raw HTML string from screener.in
        symbol: Stock ticker symbol
        
    Returns:
        Structured dictionary with all parsed fundamentals
    """
    soup = BeautifulSoup(html, "lxml")
    
    # Extract all sections
    company_info = _extract_company_info(soup)
    ratios = _extract_ratios_section(soup)
    pros_cons = _extract_pros_cons(soup)
    quarterly_results = _extract_table_data(soup, "quarters")
    profit_loss = _extract_table_data(soup, "profit-loss")
    balance_sheet = _extract_table_data(soup, "balance-sheet")
    cash_flow = _extract_table_data(soup, "cash-flow")
    shareholding = _extract_shareholding(soup)
    peers = _extract_peers(soup)
    
    # Build structured output
    parsed = {
        "symbol": symbol.upper(),
        "company_name": company_info.get("company_name", symbol),
        "sector": company_info.get("sector"),
        "industry": company_info.get("industry"),
        "source": "screener.in",
        "fetched_at": datetime.now().isoformat(),
        
        # Key ratios
        "ratios": {
            "market_cap_cr": _safe_float(ratios.get("market_cap_cr", "")),
            "current_price": _safe_float(ratios.get("current_price", "")),
            "pe_ratio": _safe_float(ratios.get("pe_ratio", "")),
            "book_value": _safe_float(ratios.get("book_value", "")),
            "dividend_yield": _safe_float(ratios.get("dividend_yield", "")),
            "roce": _safe_float(ratios.get("roce", "")),
            "roe": _safe_float(ratios.get("roe", "")),
            "face_value": _safe_float(ratios.get("face_value", "")),
            "industry_pe": _safe_float(ratios.get("industry_pe", "")),
            "debt_cr": _safe_float(ratios.get("debt_cr", "")),
            "eps": _safe_float(ratios.get("eps", "")),
            "promoter_holding": _safe_float(ratios.get("promoter_holding", "")),
            "high_low": ratios.get("high_low"),
        },
        
        # Qualitative
        "pros": pros_cons["pros"],
        "cons": pros_cons["cons"],
        
        # Financial tables
        "quarterly_results": quarterly_results[:8],  # Last 8 quarters
        "profit_loss": profit_loss,
        "balance_sheet": balance_sheet,
        "cash_flow": cash_flow,
        
        # Shareholding
        "shareholding": shareholding[:4],  # Last 4 quarters
        
        # Peers
        "peers": peers[:10],  # Top 10 peers
    }
    
    return parsed


def fetch_and_parse(symbol: str) -> Optional[Dict]:
    """
    Complete pipeline: Fetch from screener.in and parse to JSON.
    
    Args:
        symbol: NSE ticker symbol
        
    Returns:
        Parsed fundamentals dict, or None if failed
    """
    html = fetch_screener_page(symbol)
    if not html:
        logger.warning(f"Could not fetch screener data for {symbol}, falling back to yfinance")
        return _fallback_yfinance(symbol)
    
    parsed = parse_screener_html(html, symbol)
    
    # Validate we got meaningful data
    if not parsed["ratios"]["market_cap_cr"] and not parsed["ratios"]["pe_ratio"]:
        logger.warning(f"Screener parse yielded empty ratios for {symbol}, falling back to yfinance")
        return _fallback_yfinance(symbol)
    
    return parsed


def _fallback_yfinance(symbol: str, timeout: int = 8) -> Optional[Dict]:
    """
    Fallback to yfinance when screener.in is unavailable.
    Maps yfinance data to the same schema as screener parser output.
    Uses robust session with retries.
    
    Args:
        symbol: Stock ticker
        timeout: Request timeout in seconds
        
    Returns:
        Parsed fundamentals dict or None
    """
    import yfinance as yf
    
    try:
        # Try with .NS suffix for NSE stocks
        ticker_symbol = f"{symbol}.NS" if not symbol.endswith((".NS", ".BO")) else symbol
        logger.info(f"Attempting yfinance fallback for {ticker_symbol}")
        
        # Configure yfinance to use our robust session
        stock = yf.Ticker(ticker_symbol, session=_get_session())
        info = stock.info
        
        if not info or info.get("regularMarketPrice") is None:
            logger.warning(f"yfinance returned empty/invalid data for {ticker_symbol}")
            return None
        
        mcap_cr = info.get("marketCap")
        if mcap_cr:
            mcap_cr = mcap_cr / 1e7  # Convert to crores
        
        debt_cr = info.get("totalDebt")
        if debt_cr:
            debt_cr = debt_cr / 1e7
        
        parsed = {
            "symbol": symbol.upper(),
            "company_name": info.get("longName") or info.get("shortName") or symbol,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "source": "yfinance",
            "fetched_at": datetime.now().isoformat(),
            
            "ratios": {
                "market_cap_cr": mcap_cr,
                "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "pe_ratio": info.get("trailingPE"),
                "book_value": info.get("bookValue"),
                "dividend_yield": (info.get("dividendYield") or 0) * 100 if info.get("dividendYield") else None,
                "roce": (info.get("returnOnAssets") or 0) * 100 if info.get("returnOnAssets") else None,
                "roe": (info.get("returnOnEquity") or 0) * 100 if info.get("returnOnEquity") else None,
                "face_value": info.get("faceValue"),
                "industry_pe": None,
                "debt_cr": debt_cr,
                "eps": info.get("trailingEps"),
                "promoter_holding": None,
                "high_low": f"{info.get('fiftyTwoWeekHigh', 'N/A')} / {info.get('fiftyTwoWeekLow', 'N/A')}",
            },
            
            "pros": [],
            "cons": [],
            
            # Additional yfinance data
            "quarterly_results": [],
            "profit_loss": [],
            "balance_sheet": [],
            "cash_flow": [],
            "shareholding": [],
            "peers": [],
            
            # Extra yfinance-specific fields
            "yfinance_extra": {
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "profit_margins": info.get("profitMargins"),
                "operating_margins": info.get("operatingMargins"),
                "gross_margins": info.get("grossMargins"),
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "current_ratio": info.get("currentRatio"),
                "debt_to_equity": info.get("debtToEquity"),
                "free_cashflow": info.get("freeCashflow"),
                "business_summary": info.get("longBusinessSummary"),
            }
        }
        
        logger.info(f"Successfully fetched yfinance fallback data for {symbol}")
        return parsed
        
    except Exception as e:
        logger.error(f"yfinance fallback failed for {symbol}: {e}")
        return None
