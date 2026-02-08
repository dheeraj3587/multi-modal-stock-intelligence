"""
SQLite database layer for company fundamentals.

Stores parsed Screener.in / yfinance data in a local SQLite database
for fast retrieval and RAG context.
"""

import json
import sqlite3
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Default DB path
DEFAULT_DB_PATH = Path("data/processed/fundamentals.db")


class FundamentalsDB:
    """SQLite database for company fundamentals storage and retrieval."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Create tables if they don't exist."""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS company_fundamentals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    company_name TEXT,
                    sector TEXT,
                    industry TEXT,
                    source TEXT DEFAULT 'screener.in',
                    fetched_at TEXT NOT NULL,
                    
                    -- Key Ratios (stored as floats)
                    market_cap_cr REAL,
                    current_price REAL,
                    pe_ratio REAL,
                    book_value REAL,
                    dividend_yield REAL,
                    roce REAL,
                    roe REAL,
                    face_value REAL,
                    industry_pe REAL,
                    debt_cr REAL,
                    eps REAL,
                    promoter_holding REAL,
                    high_low TEXT,
                    
                    -- yfinance extras (nullable)
                    forward_pe REAL,
                    peg_ratio REAL,
                    price_to_book REAL,
                    profit_margins REAL,
                    operating_margins REAL,
                    gross_margins REAL,
                    revenue_growth REAL,
                    earnings_growth REAL,
                    current_ratio REAL,
                    debt_to_equity REAL,
                    free_cashflow REAL,
                    business_summary TEXT,
                    
                    -- Qualitative (JSON)
                    pros TEXT,       -- JSON array
                    cons TEXT,       -- JSON array
                    
                    -- Financial tables (JSON)
                    quarterly_results TEXT,  -- JSON array
                    profit_loss TEXT,        -- JSON array
                    balance_sheet TEXT,      -- JSON array
                    cash_flow TEXT,          -- JSON array
                    shareholding TEXT,       -- JSON array
                    peers TEXT,              -- JSON array
                    
                    -- Full raw JSON for flexibility
                    raw_json TEXT,
                    
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now')),
                    
                    UNIQUE(symbol, fetched_at)
                );
                
                CREATE INDEX IF NOT EXISTS idx_fundamentals_symbol 
                ON company_fundamentals(symbol);
                
                CREATE INDEX IF NOT EXISTS idx_fundamentals_fetched 
                ON company_fundamentals(fetched_at);
                
                CREATE INDEX IF NOT EXISTS idx_fundamentals_sector 
                ON company_fundamentals(sector);
            """)
            conn.commit()
            logger.info(f"Fundamentals DB initialized at {self.db_path}")
        finally:
            conn.close()
    
    def upsert(self, data: Dict) -> int:
        """
        Insert or update company fundamentals from parsed screener/yfinance data.
        
        Args:
            data: Parsed fundamentals dict from screener_parser
            
        Returns:
            Row ID of inserted/updated record
        """
        conn = self._get_conn()
        try:
            ratios = data.get("ratios", {})
            yf_extra = data.get("yfinance_extra", {})
            
            # Check if we already have data for this symbol today
            today = datetime.now().strftime("%Y-%m-%d")
            existing = conn.execute(
                "SELECT id FROM company_fundamentals WHERE symbol = ? AND date(fetched_at) = ?",
                (data["symbol"], today)
            ).fetchone()
            
            values = {
                "symbol": data["symbol"],
                "company_name": data.get("company_name"),
                "sector": data.get("sector"),
                "industry": data.get("industry"),
                "source": data.get("source", "screener.in"),
                "fetched_at": data.get("fetched_at", datetime.now().isoformat()),
                "market_cap_cr": ratios.get("market_cap_cr"),
                "current_price": ratios.get("current_price"),
                "pe_ratio": ratios.get("pe_ratio"),
                "book_value": ratios.get("book_value"),
                "dividend_yield": ratios.get("dividend_yield"),
                "roce": ratios.get("roce"),
                "roe": ratios.get("roe"),
                "face_value": ratios.get("face_value"),
                "industry_pe": ratios.get("industry_pe"),
                "debt_cr": ratios.get("debt_cr"),
                "eps": ratios.get("eps"),
                "promoter_holding": ratios.get("promoter_holding"),
                "high_low": ratios.get("high_low"),
                "forward_pe": yf_extra.get("forward_pe"),
                "peg_ratio": yf_extra.get("peg_ratio"),
                "price_to_book": yf_extra.get("price_to_book"),
                "profit_margins": yf_extra.get("profit_margins"),
                "operating_margins": yf_extra.get("operating_margins"),
                "gross_margins": yf_extra.get("gross_margins"),
                "revenue_growth": yf_extra.get("revenue_growth"),
                "earnings_growth": yf_extra.get("earnings_growth"),
                "current_ratio": yf_extra.get("current_ratio"),
                "debt_to_equity": yf_extra.get("debt_to_equity"),
                "free_cashflow": yf_extra.get("free_cashflow"),
                "business_summary": yf_extra.get("business_summary"),
                "pros": json.dumps(data.get("pros", [])),
                "cons": json.dumps(data.get("cons", [])),
                "quarterly_results": json.dumps(data.get("quarterly_results", [])),
                "profit_loss": json.dumps(data.get("profit_loss", [])),
                "balance_sheet": json.dumps(data.get("balance_sheet", [])),
                "cash_flow": json.dumps(data.get("cash_flow", [])),
                "shareholding": json.dumps(data.get("shareholding", [])),
                "peers": json.dumps(data.get("peers", [])),
                "raw_json": json.dumps(data),
            }
            
            if existing:
                # Update existing record
                row_id = existing["id"]
                set_clause = ", ".join(f"{k} = ?" for k in values.keys())
                conn.execute(
                    f"UPDATE company_fundamentals SET {set_clause}, updated_at = datetime('now') WHERE id = ?",
                    list(values.values()) + [row_id]
                )
                logger.info(f"Updated fundamentals for {data['symbol']} (id={row_id})")
            else:
                # Insert new record
                cols = ", ".join(values.keys())
                placeholders = ", ".join("?" for _ in values)
                cursor = conn.execute(
                    f"INSERT INTO company_fundamentals ({cols}) VALUES ({placeholders})",
                    list(values.values())
                )
                row_id = cursor.lastrowid
                logger.info(f"Inserted fundamentals for {data['symbol']} (id={row_id})")
            
            conn.commit()
            return row_id
        finally:
            conn.close()
    
    def get_latest(self, symbol: str, allow_stale: bool = True) -> Optional[Dict]:
        """
        Get the most recent fundamentals for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            allow_stale: If True, return stale data if available
            
        Returns:
            Dict with fundamentals data, or None
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT * FROM company_fundamentals 
                   WHERE symbol = ? 
                   ORDER BY fetched_at DESC LIMIT 1""",
                (symbol.upper(),)
            ).fetchone()
            
            if row:
                data = self._row_to_dict(row)
                
                # Check if data is stale
                if not allow_stale and self.is_stale(symbol, max_age_hours=12):
                    logger.warning(f"Cached data for {symbol} is stale (> 12h old)")
                    return None
                
                return data
            return None
        except Exception as e:
            logger.error(f"Error retrieving fundamentals for {symbol}: {e}")
            return None
        finally:
            conn.close()
    
    def get_all_latest(self) -> List[Dict]:
        """Get latest fundamentals for all symbols."""
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT f.* FROM company_fundamentals f
                INNER JOIN (
                    SELECT symbol, MAX(fetched_at) as max_date
                    FROM company_fundamentals
                    GROUP BY symbol
                ) latest ON f.symbol = latest.symbol AND f.fetched_at = latest.max_date
                ORDER BY f.market_cap_cr DESC NULLS LAST
            """).fetchall()
            
            return [self._row_to_dict(r) for r in rows]
        finally:
            conn.close()
    
    def get_by_sector(self, sector: str) -> List[Dict]:
        """Get latest fundamentals filtered by sector."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT * FROM company_fundamentals 
                   WHERE sector = ? 
                   ORDER BY market_cap_cr DESC NULLS LAST""",
                (sector,)
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        finally:
            conn.close()
    
    def is_stale(self, symbol: str, max_age_hours: int = 24) -> bool:
        """
        Check if data for a symbol is stale and needs refresh.
        
        Args:
            symbol: Stock ticker
            max_age_hours: Maximum age in hours before data is stale
            
        Returns:
            True if stale or not found, False if fresh
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT fetched_at FROM company_fundamentals WHERE symbol = ? ORDER BY fetched_at DESC LIMIT 1",
                (symbol.upper(),)
            ).fetchone()
            
            if not row:
                logger.info(f"No cached data found for {symbol}")
                return True
            
            try:
                fetched = datetime.fromisoformat(row["fetched_at"])
                age = datetime.now() - fetched
                is_stale = age > timedelta(hours=max_age_hours)
                
                if is_stale:
                    logger.info(f"Cached data for {symbol} is stale (age: {age.total_seconds() / 3600:.1f}h)")
                else:
                    logger.info(f"Using fresh cached data for {symbol} (age: {age.total_seconds() / 3600:.1f}h)")
                
                return is_stale
            except (ValueError, TypeError) as e:
                logger.error(f"Error parsing fetch date for {symbol}: {e}")
                return True
        except Exception as e:
            logger.error(f"Error checking staleness for {symbol}: {e}")
            return True
        finally:
            conn.close()
    
    def search(self, query: str) -> List[Dict]:
        """Search fundamentals by company name, symbol, or sector."""
        conn = self._get_conn()
        try:
            pattern = f"%{query}%"
            rows = conn.execute(
                """SELECT * FROM company_fundamentals 
                   WHERE symbol LIKE ? OR company_name LIKE ? OR sector LIKE ? OR industry LIKE ?
                   ORDER BY fetched_at DESC LIMIT 20""",
                (pattern, pattern, pattern, pattern)
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        finally:
            conn.close()
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        """Convert a database row to a structured dict."""
        d = dict(row)
        
        # Parse JSON fields back — ensure None is replaced with empty list
        for json_field in ["pros", "cons", "quarterly_results", "profit_loss", 
                          "balance_sheet", "cash_flow", "shareholding", "peers"]:
            raw = d.get(json_field)
            if raw and isinstance(raw, str):
                try:
                    d[json_field] = json.loads(raw)
                except json.JSONDecodeError:
                    d[json_field] = []
            elif not raw:
                d[json_field] = []
        
        # Build ratios sub-dict for compatibility
        d["ratios"] = {
            "market_cap_cr": d.get("market_cap_cr"),
            "current_price": d.get("current_price"),
            "pe_ratio": d.get("pe_ratio"),
            "book_value": d.get("book_value"),
            "dividend_yield": d.get("dividend_yield"),
            "roce": d.get("roce"),
            "roe": d.get("roe"),
            "face_value": d.get("face_value"),
            "industry_pe": d.get("industry_pe"),
            "debt_cr": d.get("debt_cr"),
            "eps": d.get("eps"),
            "promoter_holding": d.get("promoter_holding"),
            "high_low": d.get("high_low"),
        }
        
        return d


# Global DB instance
fundamentals_db = FundamentalsDB()
