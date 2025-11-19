export interface PriceTick {
    symbol: string;
    timestamp: number;
    price: number;
    volume: number;
    side: 'buy' | 'sell';
}

export interface Candle {
    symbol: string;
    start: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
    predictedClose?: number;
    confidenceLower?: number;
    confidenceUpper?: number;
}

export interface SentimentItem {
    id: string;
    source: string;
    published_at: string;
    title: string;
    score: number;
}

export interface GrowthItem {
    ticker: string;
    name: string;
    score: number;
    sector: string;
    sparkline: number[];
}

export type Timeframe = '1m' | '5m' | '1h' | '1d' | '1w';
