import { Candle, GrowthItem, PriceTick, SentimentItem } from './types';
import { subMinutes } from 'date-fns';

const TICKERS = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA', 'NVDA', 'META', 'NFLX'];
const SECTORS = ['Tech', 'Consumer', 'Finance', 'Healthcare', 'Energy'];

export const generateCandles = (count: number = 50): Candle[] => {
    const candles: Candle[] = [];
    let price = 150 + Math.random() * 50;
    const now = new Date();

    for (let i = count; i > 0; i--) {
        const start = subMinutes(now, i).getTime();
        const volatility = price * 0.005;
        const change = (Math.random() - 0.5) * volatility;
        const open = price;
        const close = price + change;
        const high = Math.max(open, close) + Math.random() * volatility * 0.5;
        const low = Math.min(open, close) - Math.random() * volatility * 0.5;

        price = close;

        // Prediction simulation
        const predictedClose = close * (1 + (Math.random() - 0.5) * 0.01);
        const confidenceBand = close * 0.02;

        candles.push({
            symbol: 'AAPL',
            start,
            open,
            high,
            low,
            close,
            volume: Math.floor(Math.random() * 10000),
            predictedClose,
            confidenceLower: predictedClose - confidenceBand,
            confidenceUpper: predictedClose + confidenceBand,
        });
    }
    return candles;
};

export const generateTick = (): PriceTick => {
    return {
        symbol: TICKERS[Math.floor(Math.random() * TICKERS.length)],
        timestamp: Date.now(),
        price: 100 + Math.random() * 200,
        volume: Math.floor(Math.random() * 500),
        side: Math.random() > 0.5 ? 'buy' : 'sell',
    };
};

export const generateSentiment = (): SentimentItem[] => {
    return [
        { id: '1', source: 'Bloomberg', published_at: new Date().toISOString(), title: 'Tech stocks rally on new AI chips', score: 0.8 },
        { id: '2', source: 'Reuters', published_at: new Date().toISOString(), title: 'Fed signals interest rate pause', score: 0.4 },
        { id: '3', source: 'CNBC', published_at: new Date().toISOString(), title: 'Oil prices dip slightly', score: -0.2 },
        { id: '4', source: 'WSJ', published_at: new Date().toISOString(), title: 'Market awaits earnings report', score: 0.1 },
    ];
};

export const generateGrowthLeaderboard = (): GrowthItem[] => {
    return TICKERS.map((ticker) => ({
        ticker,
        name: `${ticker} Inc.`,
        score: Math.floor(Math.random() * 40) + 60,
        sector: SECTORS[Math.floor(Math.random() * SECTORS.length)],
        sparkline: Array.from({ length: 10 }, () => Math.random() * 100),
    })).sort((a, b) => b.score - a.score);
};
