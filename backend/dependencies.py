from __future__ import annotations

import re
from enum import Enum
from typing import Optional, List

from functools import lru_cache

from fastapi import Header, HTTPException, Query

from backend.services.screener_service import ScreenerService

MAX_SYMBOLS_PER_REQUEST = 20
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9_&]{1,20}$")


def get_access_token(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """Extract bearer token from Authorization header."""
    if authorization and authorization.startswith("Bearer "):
        return authorization[7:]
    return None


def parse_symbols(
    symbols: str = Query(
        default="RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK",
        description="Comma-separated NSE symbols (max 20)",
        examples=["RELIANCE,TCS,INFY"],
    ),
) -> List[str]:
    parts = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    if len(parts) > MAX_SYMBOLS_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_SYMBOLS_PER_REQUEST} symbols per request",
        )

    for sym in parts:
        if not SYMBOL_PATTERN.match(sym):
            raise HTTPException(status_code=400, detail=f"Invalid symbol format: '{sym}'")

    return parts


class CandleInterval(str, Enum):
    ONE_MINUTE = "1minute"
    FIVE_MINUTE = "5minute"
    FIFTEEN_MINUTE = "15minute"
    THIRTY_MINUTE = "30minute"
    ONE_HOUR = "1hour"
    ONE_DAY = "1day"
    ONE_WEEK = "1week"
    ONE_MONTH = "1month"


@lru_cache(maxsize=1)
def get_screener_service() -> ScreenerService:
    return ScreenerService()
