const BASE = import.meta.env.VITE_API_URL || '';

export const AUTH_TOKEN_STORAGE_KEY = 'stocksense_upstox_token';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const token = localStorage.getItem(AUTH_TOKEN_STORAGE_KEY);
  const headers: Record<string, string> = { 'Content-Type': 'application/json', ...(init?.headers as Record<string, string>) };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  const res = await fetch(`${BASE}${path}`, { ...init, headers });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(body || `HTTP ${res.status}`);
  }
  return res.json();
}

export interface IndexQuote {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
}

export interface Stock {
  symbol: string;
  name: string;
  sector: string;
  instrument_key: string;
}

export interface StockQuote {
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

export interface StockWithSentiment {
  symbol: string;
  name: string;
  sector: string;
  price: number;
  change: number;
  changePercent: number;
  forecastConfidence: number;
  sentiment: string;
  riskLevel: string;
  lastUpdated: string;
}

export interface CandleData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface RealTimeSentiment {
  symbol: string;
  name: string;
  sentiment_score: number;
  sentiment_label: string;
  confidence: number;
  risk_level: string;
  reasoning: string;
  last_updated: string | null;
  cached: boolean;
}

export interface NewsItem {
  title: string;
  source: string;
  url: string;
  published_at: string;
  sentiment_score: number;
  description: string;
}

export interface SentimentResponse {
  symbol: string;
  overall_sentiment: number;
  confidence: number;
  reasoning: string;
  key_themes: string;
  risk_level: string;
  article_count: number;
  news: NewsItem[];
  model: string;
}

export interface StockAnalysis {
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
  recentNews: { title: string; source: string; published: string; sentiment: string; url?: string }[];
  newsCount: number;
}

export interface FundamentalAnalysis {
  about?: string;
  ratios?: Record<string, string | number>;
  pros?: string[];
  cons?: string[];
  [key: string]: unknown;
}

export interface LeaderboardStock {
  rank: number;
  symbol: string;
  name: string;
  sector: string;
  growthScore: number;
  sentiment: string;
  forecastPercent: number;
}

export interface ScorecardCategory {
  name: string;
  icon: string;
  description: string;
  weight: number;
  score: number;
  max_score: number;
  verdict: string;
  components: Record<string, unknown>;
  details: Record<string, unknown>;
}

export interface ScorecardResponse {
  symbol: string;
  company_name: string;
  sector: string | null;
  industry: string | null;
  source: string | null;
  generated_at: string;
  overall_score: number;
  overall_max: number;
  overall_verdict: string;
  overall_badge: string;
  categories: Record<string, ScorecardCategory>;
  key_stats: Record<string, unknown>;
  pros: string[];
  cons: string[];
  strengths: { category: string; score: number; verdict: string }[];
  weaknesses: { category: string; score: number; verdict: string }[];
  peers: Record<string, unknown>[];
  news_summary: { sentiment: string | null; key_themes: string | null; article_count: number };
  ai_summary: string | null;
}

export interface ScorecardListItem {
  symbol: string;
  company_name: string;
  sector: string | null;
  overall_score: number;
  overall_verdict: string;
  overall_badge: string;
  pe_ratio: number | null;
  roce: number | null;
  market_cap_cr: number | null;
}

export interface CompanySearch {
  symbol: string;
  name: string;
  sector?: string;
}

export const api = {
  getIndices: () => request<IndexQuote[]>('/indices'),
  getStocks: () => request<Stock[]>('/stocks'),
  getStocksWithSentiment: () => request<StockWithSentiment[]>('/stocks/with-sentiment'),
  getQuotes: (symbols: string | string[]) => {
    const s = Array.isArray(symbols) ? symbols.join(',') : symbols;
    return request<StockQuote[]>(`/stocks/quotes?symbols=${encodeURIComponent(s)}`);
  },
  getHistorical: (symbol: string, interval = '1day', days = 30) =>
    request<CandleData[]>(`/stocks/${encodeURIComponent(symbol)}/historical?interval=${interval}&days=${days}`),
  getAnalysis: (symbol: string) =>
    request<StockAnalysis>(`/stocks/${encodeURIComponent(symbol)}/analysis`),
  getFundamentals: (symbol: string) =>
    request<FundamentalAnalysis>(`/stocks/${encodeURIComponent(symbol)}/fundamental-analysis`),
  getLeaderboard: () => request<LeaderboardStock[]>('/leaderboard'),
  getNews: (symbol: string) =>
    request<SentimentResponse>(`/news/${encodeURIComponent(symbol)}`),
  getScorecard: (symbol: string, detailed = false) =>
    request<ScorecardResponse>(`/scorecard/${encodeURIComponent(symbol)}${detailed ? '?detailed=true' : ''}`),
  getAllScorecards: () => request<ScorecardListItem[]>('/scorecard/'),
  chat: (message: string, symbol?: string, history?: { role: string; content: string }[], deep = false) =>
    request<{ reply: string; timestamp: string; cached: boolean; model: string; deep: boolean }>('/chat/', {
      method: 'POST',
      body: JSON.stringify({ message, symbol: symbol || undefined, history: history || [], deep }),
    }),
  exchangeUpstoxCode: (code: string) =>
    request<{ access_token: string }>(`/auth/upstox/callback?code=${encodeURIComponent(code)}`),
  searchCompanies: (query: string) =>
    request<CompanySearch[]>(`/scorecard/search/${encodeURIComponent(query)}`),
};

export function getLoginUrl(): string {
  return `${BASE}/auth/upstox/login`;
}
