/**
 * Market API Service - Fetches real data from backend
 */

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

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

export interface IndexQuote {
    symbol: string;
    name: string;
    price: number;
    change: number;
    changePercent: number;
}

export interface CandleData {
    timestamp: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

export interface StockWithSentiment {
    symbol: string;
    name: string;
    sector: string;
    price: number;
    change: number;
    changePercent: number;
    forecastConfidence: number;
    sentiment: 'bullish' | 'neutral' | 'bearish';
    riskLevel: 'low' | 'medium' | 'high';
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

class MarketApiService {
    private getAuthHeader(): HeadersInit {
        const token = localStorage.getItem('upstox_access_token');
        const headers: HeadersInit = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        };
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }
        return headers;
    }

    async getStockQuotes(symbols: string[] = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'SBIN']): Promise<StockQuote[]> {
        try {
            const response = await fetch(
                `${API_URL}/stocks/quotes?symbols=${symbols.join(',')}`,
                { headers: this.getAuthHeader() }
            );
            if (!response.ok) throw new Error('Failed to fetch quotes');
            return await response.json();
        } catch (error) {
            console.error('Error fetching stock quotes:', error);
            return [];
        }
    }

    async getIndices(): Promise<IndexQuote[]> {
        try {
            const response = await fetch(
                `${API_URL}/indices`,
                { headers: this.getAuthHeader() }
            );
            if (!response.ok) throw new Error('Failed to fetch indices');
            return await response.json();
        } catch (error) {
            console.error('Error fetching indices:', error);
            return [];
        }
    }

    async getHistoricalData(symbol: string, interval: string = '1minute', days: number = 7): Promise<CandleData[]> {
        try {
            const response = await fetch(
                `${API_URL}/stocks/${symbol}/historical?interval=${interval}&days=${days}`,
                { headers: this.getAuthHeader() }
            );
            if (!response.ok) throw new Error('Failed to fetch historical data');
            return await response.json();
        } catch (error) {
            console.error('Error fetching historical data:', error);
            return [];
        }
    }

    async getStocksWithSentiment(): Promise<StockWithSentiment[]> {
        try {
            const response = await fetch(
                `${API_URL}/stocks/with-sentiment`,
                { headers: this.getAuthHeader() }
            );
            if (!response.ok) throw new Error('Failed to fetch stocks with sentiment');
            return await response.json();
        } catch (error) {
            console.error('Error fetching stocks with sentiment:', error);
            return [];
        }
    }

    async getLeaderboard(): Promise<LeaderboardStock[]> {
        try {
            const response = await fetch(
                `${API_URL}/leaderboard`,
                { headers: this.getAuthHeader() }
            );
            if (!response.ok) throw new Error('Failed to fetch leaderboard');
            return await response.json();
        } catch (error) {
            console.error('Error fetching leaderboard:', error);
            return [];
        }
    }

    async getAllStocks(): Promise<{ symbol: string; name: string; sector: string; instrument_key: string }[]> {
        try {
            const response = await fetch(
                `${API_URL}/stocks`,
                { headers: this.getAuthHeader() }
            );
            if (!response.ok) throw new Error('Failed to fetch stocks');
            return await response.json();
        } catch (error) {
            console.error('Error fetching stocks:', error);
            return [];
        }
    }
}

export const marketApi = new MarketApiService();
