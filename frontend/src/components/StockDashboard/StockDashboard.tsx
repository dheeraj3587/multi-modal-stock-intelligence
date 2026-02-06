import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import Navbar from './Navbar/Navbar';
import StockActionPanel from './StockActionPanel/StockActionPanel';
import ForecastChart from './ForecastChart/ForecastChart';
import AIInsightsPanel, { AIInsightsData } from './AIInsightsPanel/AIInsightsPanel';
import MarketLeaderboard from './MarketLeaderboard/MarketLeaderboard';
import { marketApi, LeaderboardStock } from '../../services/marketApi';
import styles from './StockDashboard.module.css';

// Map API response to component types
type SentimentType = 'bullish' | 'neutral' | 'bearish';
type RiskLevel = 'low' | 'medium' | 'high';

interface StockData {
    symbol: string;
    name: string;
    price: number;
    change: number;
    changePercent: number;
    forecastConfidence: number;
    sentiment: SentimentType;
    riskLevel: RiskLevel;
}

interface StockDashboardProps {
    initialStock?: string;
}

const StockDashboard: React.FC<StockDashboardProps> = ({ initialStock }) => {
    const [activeNav, setActiveNav] = useState('dashboard');
    const [stocks, setStocks] = useState<StockData[]>([]);
    const [leaderboard, setLeaderboard] = useState<LeaderboardStock[]>([]);
    const [selectedStock, setSelectedStock] = useState<StockData | null>(null);
    const [isConnected, setIsConnected] = useState(false);
    const [isLoading, setIsLoading] = useState(true);

    // Check for existing token
    useEffect(() => {
        const token = localStorage.getItem('upstox_access_token');
        if (token) {
            setIsConnected(true);
        }
    }, []);

    // Fetch real data on mount
    useEffect(() => {
        const fetchData = async () => {
            setIsLoading(true);
            try {
                // Fetch stocks with sentiment
                const stocksData = await marketApi.getStocksWithSentiment();
                if (stocksData && stocksData.length > 0) {
                    const mappedStocks: StockData[] = stocksData.map(s => ({
                        symbol: s.symbol,
                        name: s.name,
                        price: s.price,
                        change: s.change,
                        changePercent: s.changePercent,
                        forecastConfidence: s.forecastConfidence,
                        sentiment: s.sentiment as SentimentType,
                        riskLevel: s.riskLevel as RiskLevel,
                    }));
                    setStocks(mappedStocks);

                    // Set initial selected stock
                    if (mappedStocks.length > 0) {
                        const initial = initialStock
                            ? mappedStocks.find(s => s.symbol === initialStock) || mappedStocks[0]
                            : mappedStocks[0];
                        setSelectedStock(initial);
                    }
                }

                // Fetch leaderboard
                const leaderboardData = await marketApi.getLeaderboard();
                if (leaderboardData) {
                    setLeaderboard(leaderboardData);
                }
            } catch (error) {
                console.error('Error fetching data:', error);
                // We could set an error state here to show a message to the user
            } finally {
                setIsLoading(false);
            }
        };

        fetchData();

        // Refresh data (faster when broker token is connected)
        const intervalMs = isConnected ? 5000 : 30000;
        const interval = setInterval(fetchData, intervalMs);
        return () => clearInterval(interval);
    }, [initialStock, isConnected]);

    // Generate AI insights based on selected stock
    const getAIInsights = (stock: StockData): AIInsightsData => ({
        growthScore: stock.forecastConfidence,
        buyConfidence: Math.max(40, Math.min(95, stock.forecastConfidence + (stock.sentiment === 'bullish' ? 5 : -5))),
        sentimentIndex: stock.sentiment === 'bullish' ? 4 : stock.sentiment === 'bearish' ? 2 : 3,
        volatility: stock.riskLevel === 'high' ? 28.5 : stock.riskLevel === 'medium' ? 18.2 : 12.4,
        riskLevel: stock.riskLevel,
        sector: getSector(stock.symbol),
    });

    const getSector = (symbol: string): string => {
        const sectors: Record<string, string> = {
            'RELIANCE': 'Energy',
            'TCS': 'IT',
            'HDFCBANK': 'Banking',
            'INFY': 'IT',
            'ICICIBANK': 'Banking',
            'HINDUNILVR': 'FMCG',
            'SBIN': 'Banking',
            'BHARTIARTL': 'Telecom',
            'ITC': 'FMCG',
            'KOTAKBANK': 'Banking',
            'LT': 'Engineering',
            'AXISBANK': 'Banking',
            'WIPRO': 'IT',
            'SUNPHARMA': 'Pharma',
            'TATAMOTORS': 'Auto',
            'MARUTI': 'Auto',
            'HCLTECH': 'IT',
            'ASIANPAINT': 'Consumer',
            'BAJFINANCE': 'Finance',
            'TITAN': 'Consumer',
        };
        return sectors[symbol] || 'Other';
    };

    const handleStockSelect = (stock: StockData) => {
        setSelectedStock(stock);
    };

    const handleConnectBroker = useCallback(() => {
        if (isConnected) {
            localStorage.removeItem('upstox_access_token');
            setIsConnected(false);
        } else {
            // Redirect to Upstox OAuth
            window.location.href = `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/auth/upstox/login`;
        }
    }, [isConnected]);

    const handleLeaderboardClick = (symbol: string) => {
        const stock = stocks.find(s => s.symbol === symbol);
        if (stock) {
            setSelectedStock(stock);
        }
    };

    // Loading state
    if (isLoading && stocks.length === 0) {
        return (
            <div className={styles.dashboard}>
                <div className={styles.backgroundOverlay} />
                <Navbar
                    activeNav={activeNav}
                    onNavChange={setActiveNav}
                    onConnectBroker={handleConnectBroker}
                    isConnected={isConnected}
                />
                <main className={styles.main}>
                    <div className={styles.loadingContainer}>
                        <div className={styles.loader} />
                        <p>Loading market data...</p>
                    </div>
                </main>
            </div>
        );
    }

    const currentStock = selectedStock || stocks[0];

    return (
        <div className={styles.dashboard}>
            {/* Background with AI streaks */}
            <div className={styles.backgroundOverlay} />

            {/* Navigation */}
            <Navbar
                activeNav={activeNav}
                onNavChange={setActiveNav}
                onConnectBroker={handleConnectBroker}
                isConnected={isConnected}
            />

            {/* Main Content */}
            <main className={styles.main}>
                <div className={styles.grid}>
                    {/* Left Panel - Stock Action */}
                    <motion.aside
                        className={styles.leftPanel}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.4 }}
                    >
                        <StockActionPanel
                            stocks={stocks}
                            selectedStock={currentStock}
                            onStockSelect={handleStockSelect}
                        />
                    </motion.aside>

                    {/* Center Panel - Chart & Leaderboard */}
                    <div className={styles.centerPanel}>
                        <motion.div
                            className={styles.chartSection}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.4, delay: 0.1 }}
                        >
                            <ForecastChart
                                symbol={currentStock.symbol}
                                name={currentStock.name}
                                currentPrice={currentStock.price}
                                changePercent={currentStock.changePercent}
                            />
                        </motion.div>

                        <motion.div
                            className={styles.leaderboardSection}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.4, delay: 0.2 }}
                        >
                            <MarketLeaderboard
                                stocks={leaderboard}
                                onStockClick={handleLeaderboardClick}
                            />
                        </motion.div>
                    </div>

                    {/* Right Panel - AI Insights */}
                    <motion.aside
                        className={styles.rightPanel}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.4, delay: 0.15 }}
                    >
                        <AIInsightsPanel data={getAIInsights(currentStock)} />
                    </motion.aside>
                </div>
            </main>
        </div>
    );
};

export default StockDashboard;
