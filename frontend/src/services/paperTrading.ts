/**
 * Paper Trading Service
 *
 * Manages a simulated portfolio stored in localStorage.
 * Quantities + avg prices are persisted locally.
 * Current prices come from the live Upstox feed (or REST fallback).
 */

const STORAGE_KEY = 'paper_portfolio';
const HISTORY_KEY = 'paper_trade_history';

/* ── Types ── */

export interface PaperHolding {
  symbol: string;
  name: string;
  sector: string;
  quantity: number;
  avgPrice: number;
  /** instrument_key for Upstox WS subscription */
  instrumentKey: string;
}

export interface PaperTrade {
  id: string;
  symbol: string;
  name: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  total: number;
  timestamp: number;
}

export type PaperPortfolio = Record<string, PaperHolding>;

/* ── Default seed portfolio (for first-time users) ── */

const SEED_PORTFOLIO: PaperPortfolio = {
  RELIANCE: {
    symbol: 'RELIANCE',
    name: 'Reliance Industries',
    sector: 'Energy',
    quantity: 10,
    avgPrice: 2450,
    instrumentKey: 'NSE_EQ|INE002A01018',
  },
  TCS: {
    symbol: 'TCS',
    name: 'Tata Consultancy Services',
    sector: 'IT',
    quantity: 5,
    avgPrice: 3800,
    instrumentKey: 'NSE_EQ|INE467B01029',
  },
  HDFCBANK: {
    symbol: 'HDFCBANK',
    name: 'HDFC Bank',
    sector: 'Banking',
    quantity: 15,
    avgPrice: 1560,
    instrumentKey: 'NSE_EQ|INE040A01034',
  },
  INFY: {
    symbol: 'INFY',
    name: 'Infosys',
    sector: 'IT',
    quantity: 12,
    avgPrice: 1480,
    instrumentKey: 'NSE_EQ|INE009A01021',
  },
  ICICIBANK: {
    symbol: 'ICICIBANK',
    name: 'ICICI Bank',
    sector: 'Banking',
    quantity: 20,
    avgPrice: 990,
    instrumentKey: 'NSE_EQ|INE090A01021',
  },
};

/* ── Persistence helpers ── */

function load<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function save<T>(key: string, data: T) {
  localStorage.setItem(key, JSON.stringify(data));
}

/* ── Service ── */

class PaperTradingService {
  private portfolio: PaperPortfolio;
  private history: PaperTrade[];
  private listeners: Set<() => void> = new Set();

  constructor() {
    this.portfolio = load<PaperPortfolio>(STORAGE_KEY, SEED_PORTFOLIO);
    this.history = load<PaperTrade[]>(HISTORY_KEY, []);
  }

  /* ── Reads ── */

  getPortfolio(): PaperPortfolio {
    return { ...this.portfolio };
  }

  getHoldings(): PaperHolding[] {
    return Object.values(this.portfolio).filter((h) => h.quantity > 0);
  }

  getHolding(symbol: string): PaperHolding | undefined {
    return this.portfolio[symbol];
  }

  getHistory(): PaperTrade[] {
    return [...this.history];
  }

  getInstrumentKeys(): string[] {
    return this.getHoldings().map((h) => h.instrumentKey);
  }

  /* ── Writes ── */

  /**
   * Execute a simulated trade at the given LTP.
   * Returns the PaperTrade record, or throws if invalid.
   */
  executeTrade(params: {
    symbol: string;
    name: string;
    sector: string;
    instrumentKey: string;
    side: 'buy' | 'sell';
    quantity: number;
    price: number;
  }): PaperTrade {
    const { symbol, name, sector, instrumentKey, side, quantity, price } = params;
    if (quantity <= 0) throw new Error('Quantity must be positive');
    if (price <= 0) throw new Error('Price must be positive');

    const existing = this.portfolio[symbol];

    if (side === 'sell') {
      if (!existing || existing.quantity < quantity) {
        throw new Error(`Cannot sell ${quantity} shares of ${symbol} — only hold ${existing?.quantity ?? 0}`);
      }
      existing.quantity -= quantity;
      if (existing.quantity === 0) {
        delete this.portfolio[symbol];
      }
    } else {
      // Buy
      if (existing) {
        const totalCost = existing.avgPrice * existing.quantity + price * quantity;
        existing.quantity += quantity;
        existing.avgPrice = totalCost / existing.quantity;
      } else {
        this.portfolio[symbol] = {
          symbol,
          name,
          sector,
          quantity,
          avgPrice: price,
          instrumentKey,
        };
      }
    }

    const trade: PaperTrade = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      symbol,
      name,
      side,
      quantity,
      price,
      total: price * quantity,
      timestamp: Date.now(),
    };

    this.history.unshift(trade);
    // Keep last 100 trades
    if (this.history.length > 100) this.history.length = 100;

    this.persist();
    this.notify();
    return trade;
  }

  /** Reset to seed portfolio */
  reset() {
    this.portfolio = { ...SEED_PORTFOLIO };
    this.history = [];
    this.persist();
    this.notify();
  }

  /* ── Subscriptions ── */

  subscribe(listener: () => void) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /* ── Internal ── */

  private persist() {
    save(STORAGE_KEY, this.portfolio);
    save(HISTORY_KEY, this.history);
  }

  private notify() {
    this.listeners.forEach((fn) => fn());
  }
}

/** Singleton */
export const paperTradingService = new PaperTradingService();
