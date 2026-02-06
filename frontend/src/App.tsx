import { useEffect, useState, useCallback } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { motion } from 'framer-motion';
import Layout from './components/Layout/Layout';
import TradingChart from './components/TradingChart/TradingChart';
import Watchlist from './components/Watchlist/Watchlist';
import OrderBook from './components/OrderBook/OrderBook';
import MarketIndices from './components/MarketIndices/MarketIndices';
import SentimentPanel from './components/SentimentPanel/SentimentPanel';
import AuthCallback from './components/Auth/AuthCallback';
import ProtectedRoute from './components/Auth/ProtectedRoute';
import StockDashboard from './components/StockDashboard';
import { AuthProvider } from './contexts/AuthContext';
import { upstoxService } from './services/upstoxService';
import { Time } from 'lightweight-charts';

// Types
interface CandleData {
    time: Time;
    open: number;
    high: number;
    low: number;
    close: number;
    volume?: number;
}

interface WatchlistItem {
    symbol: string;
    name: string;
    price: number;
    change: number;
    instrumentKey: string;
}

interface Order {
    price: number;
    size: number;
    total: number;
}

interface SentimentItem {
    id: string;
    title: string;
    score: number;
    source: string;
    published_at: string;
}

// Default Indian stocks watchlist
const defaultWatchlist: WatchlistItem[] = [
    { symbol: 'RELIANCE', name: 'Reliance Industries', price: 2456.75, change: 1.24, instrumentKey: 'NSE_EQ|INE002A01018' },
    { symbol: 'TCS', name: 'Tata Consultancy', price: 3876.50, change: -0.56, instrumentKey: 'NSE_EQ|INE467B01029' },
    { symbol: 'HDFCBANK', name: 'HDFC Bank Ltd', price: 1678.30, change: 0.89, instrumentKey: 'NSE_EQ|INE040A01034' },
    { symbol: 'INFY', name: 'Infosys Ltd', price: 1456.25, change: 2.15, instrumentKey: 'NSE_EQ|INE009A01021' },
    { symbol: 'ICICIBANK', name: 'ICICI Bank Ltd', price: 987.45, change: -0.32, instrumentKey: 'NSE_EQ|INE090A01021' },
    { symbol: 'SBIN', name: 'State Bank of India', price: 634.80, change: 1.67, instrumentKey: 'NSE_EQ|INE062A01020' },
];

// Market indices
const defaultIndices = [
    { name: 'NIFTY 50', symbol: 'NIFTY50', price: 23456.78, change: 145.32, changePercent: 0.62 },
    { name: 'SENSEX', symbol: 'SENSEX', price: 77234.56, change: 432.18, changePercent: 0.56 },
    { name: 'BANK NIFTY', symbol: 'BANKNIFTY', price: 49876.45, change: -123.45, changePercent: -0.25 },
];

// Generate realistic mock order book
const generateOrderBook = (basePrice: number) => {
    const asks: Order[] = [];
    const bids: Order[] = [];
    let askTotal = 0;
    let bidTotal = 0;

    for (let i = 0; i < 10; i++) {
        const askSize = Math.floor(Math.random() * 500) + 100;
        askTotal += askSize;
        asks.push({
            price: basePrice + (i + 1) * 0.05,
            size: askSize,
            total: askTotal
        });

        const bidSize = Math.floor(Math.random() * 500) + 100;
        bidTotal += bidSize;
        bids.push({
            price: basePrice - (i + 1) * 0.05,
            size: bidSize,
            total: bidTotal
        });
    }

    return { asks, bids, spread: asks[0].price - bids[0].price };
};

// Generate mock candle data
const generateMockCandles = (count: number = 100): CandleData[] => {
    const candles: CandleData[] = [];
    let price = 2450 + Math.random() * 50;
    const now = Math.floor(Date.now() / 1000);

    for (let i = count; i > 0; i--) {
        const time = (now - i * 60) as Time; // 1 minute candles
        const volatility = price * 0.002;
        const change = (Math.random() - 0.5) * volatility * 2;
        const open = price;
        const close = price + change;
        const high = Math.max(open, close) + Math.random() * volatility;
        const low = Math.min(open, close) - Math.random() * volatility;
        const volume = Math.floor(Math.random() * 10000) + 1000;

        price = close;

        candles.push({ time, open, high, low, close, volume });
    }

    return candles;
};

function Dashboard() {
    const [isConnected, setIsConnected] = useState(false);
    const [isConnecting, setIsConnecting] = useState(false);
    const [selectedSymbol, setSelectedSymbol] = useState('RELIANCE');
    const [candles, setCandles] = useState<CandleData[]>([]);
    const [watchlist, setWatchlist] = useState<WatchlistItem[]>(defaultWatchlist);
    const [indices, setIndices] = useState(defaultIndices);
    const [orderBook, setOrderBook] = useState(() => generateOrderBook(2456.75));
    const [sentiment, setSentiment] = useState<SentimentItem[]>([]);
    const [isLoading, setIsLoading] = useState(true);

    // Check for existing token on load
    useEffect(() => {
        const token = localStorage.getItem('upstox_access_token');
        if (token) {
            setIsConnected(true);
            upstoxService.setAccessToken(token);
        }
    }, []);

    // Load initial data
    useEffect(() => {
        setIsLoading(true);
        // Generate mock data for demo
        setCandles(generateMockCandles(100));
        setIsLoading(false);

        // Fetch sentiment
        const fetchSentiment = async () => {
            try {
                const data = await upstoxService.getSentiment(selectedSymbol);
                const items: SentimentItem[] = data.news.map((n: any, i: number) => ({
                    id: i.toString(),
                    title: n.title,
                    score: n.sentiment_score,
                    source: n.source,
                    published_at: n.published_at
                }));
                setSentiment(items);
            } catch (e) {
                console.error('Failed to fetch sentiment:', e);
            }
        };
        fetchSentiment();
    }, [selectedSymbol]);

    // Simulate real-time updates
    useEffect(() => {
        const interval = setInterval(() => {
            // Update candles
            setCandles(prev => {
                if (prev.length === 0) return prev;
                const lastCandle = prev[prev.length - 1];
                const change = (Math.random() - 0.5) * 2;
                const newClose = lastCandle.close + change;
                const updatedCandle = {
                    ...lastCandle,
                    close: newClose,
                    high: Math.max(lastCandle.high, newClose),
                    low: Math.min(lastCandle.low, newClose),
                    volume: (lastCandle.volume || 0) + Math.floor(Math.random() * 100)
                };
                return [...prev.slice(0, -1), updatedCandle];
            });

            // Update order book
            setOrderBook(generateOrderBook(2456.75 + (Math.random() - 0.5) * 10));

            // Update watchlist prices
            setWatchlist(prev => prev.map(item => ({
                ...item,
                price: item.price * (1 + (Math.random() - 0.5) * 0.002),
                change: item.change + (Math.random() - 0.5) * 0.1
            })));

            // Update indices
            setIndices(prev => prev.map(idx => ({
                ...idx,
                price: idx.price * (1 + (Math.random() - 0.5) * 0.0005),
                change: idx.change + (Math.random() - 0.5) * 5,
                changePercent: idx.changePercent + (Math.random() - 0.5) * 0.02
            })));
        }, 2000);

        return () => clearInterval(interval);
    }, []);

    const handleConnect = useCallback(() => {
        setIsConnecting(true);
        window.location.href = `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/auth/upstox/login`;
    }, []);

    const handleDisconnect = useCallback(() => {
        upstoxService.disconnect();
        localStorage.removeItem('upstox_access_token');
        setIsConnected(false);
    }, []);

    const handleSymbolSelect = useCallback((symbol: string, instrumentKey?: string) => {
        setSelectedSymbol(symbol);
        if (instrumentKey) {
            // Instrument key tracking removed
        }
        setIsLoading(true);
        // Simulate loading new data
        setTimeout(() => {
            setCandles(generateMockCandles(100));
            setIsLoading(false);
        }, 500);
    }, []);

    return (
        <Layout
            isConnected={isConnected}
            isConnecting={isConnecting}
            onConnect={handleConnect}
            onDisconnect={handleDisconnect}
            marketData={{
                nifty50: { price: indices[0]?.price || 0, change: indices[0]?.changePercent || 0 },
                sensex: { price: indices[1]?.price || 0, change: indices[1]?.changePercent || 0 }
            }}
        >
            {/* Main Trading Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr 320px', gap: '16px', height: 'calc(100vh - 120px)' }}>
                {/* Left Panel - Watchlist */}
                <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.1 }}
                >
                    <Watchlist
                        items={watchlist}
                        activeSymbol={selectedSymbol}
                        onSelect={handleSymbolSelect}
                    />
                </motion.div>

                {/* Center Panel - Chart & Indices */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    {/* Market Indices */}
                    <motion.div
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.2 }}
                    >
                        <MarketIndices indices={indices} />
                    </motion.div>

                    {/* Main Chart */}
                    <motion.div
                        style={{ flex: 1 }}
                        initial={{ opacity: 0, scale: 0.98 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.3 }}
                    >
                        <TradingChart
                            symbol={selectedSymbol}
                            data={candles}
                            isLoading={isLoading}
                        />
                    </motion.div>
                </div>

                {/* Right Panel - Order Book & Sentiment */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <motion.div
                        style={{ flex: 1 }}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.4 }}
                    >
                        <OrderBook
                            asks={orderBook.asks}
                            bids={orderBook.bids}
                            spread={orderBook.spread}
                        />
                    </motion.div>

                    <motion.div
                        style={{ height: '300px' }}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.5 }}
                    >
                        <SentimentPanel items={sentiment} overallScore={0.65} />
                    </motion.div>
                </div>
            </div>
        </Layout>
    );
}

function App() {
    return (
        <Router>
            <AuthProvider>
                <Routes>
                    <Route path="/auth/callback" element={<AuthCallback />} />
                    <Route path="/" element={
                        <ProtectedRoute>
                            <StockDashboard />
                        </ProtectedRoute>
                    } />
                    <Route path="/trading" element={
                        <ProtectedRoute>
                            <Dashboard />
                        </ProtectedRoute>
                    } />
                </Routes>
            </AuthProvider>
        </Router>
    );
}

export default App;
