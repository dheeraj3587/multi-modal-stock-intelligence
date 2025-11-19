import { useEffect, useState } from 'react';
import Layout from './components/Layout/Layout';
import PriceChart from './components/PriceChart/PriceChart';
import LiveTickStream from './components/LiveTickStream/LiveTickStream';
import SentimentPanel from './components/SentimentPanel/SentimentPanel';
import BuyIndicator from './components/Indicator/BuyIndicator';
import FearIndicator from './components/Indicator/FearIndicator';
import Leaderboard from './components/Leaderboard/Leaderboard';
import { generateCandles, generateGrowthLeaderboard, generateSentiment, generateTick } from './utils/mockData';
import { Candle, GrowthItem, PriceTick, SentimentItem } from './utils/types';
import { upstoxService } from './services/upstoxService';
import styles from './components/Layout/Layout.module.css';

function App() {
    const [candles, setCandles] = useState<Candle[]>([]);
    const [ticks, setTicks] = useState<PriceTick[]>([]);
    const [sentiment, setSentiment] = useState<SentimentItem[]>([]);
    const [growth, setGrowth] = useState<GrowthItem[]>([]);
    const [selectedTicker, setSelectedTicker] = useState('NSE_EQ|INE002A01018'); // Reliance
    const [isConnected, setIsConnected] = useState(false);

    const handleConnect = async () => {
        const token = prompt("Enter Upstox Access Token:");
        if (token) {
            upstoxService.connect(token);

            // Fetch historical data
            const history = await upstoxService.getHistoricalCandles(selectedTicker);
            if (history.length > 0) {
                setCandles(history);
            }

            upstoxService.onTick((tick) => {
                setTicks(prev => [tick, ...prev].slice(0, 50));
                // Optionally update last candle with live price
            });
            setIsConnected(true);
        }
    };

    useEffect(() => {
        setCandles(generateCandles(50));
        setSentiment(generateSentiment());
        setGrowth(generateGrowthLeaderboard());
    }, []);

    useEffect(() => {
        if (isConnected) return; // Stop mock simulation if connected

        const interval = setInterval(() => {
            const newTick = generateTick();
            setTicks(prev => [newTick, ...prev].slice(0, 50));

            setCandles(prev => {
                const last = prev[prev.length - 1];
                if (!last) return prev;

                const newClose = last.close + (Math.random() - 0.5) * 2;
                const updatedLast = {
                    ...last,
                    close: newClose,
                    high: Math.max(last.high, newClose),
                    low: Math.min(last.low, newClose),
                    volume: last.volume + Math.floor(Math.random() * 100)
                };
                return [...prev.slice(0, -1), updatedLast];
            });

        }, 1000);

        return () => clearInterval(interval);
    }, []);

    return (
        <Layout>
            <div className={styles.controls}>
                <button onClick={handleConnect} style={{ padding: '8px 16px', borderRadius: '8px', border: 'none', background: isConnected ? '#10B981' : '#3B82F6', color: 'white', cursor: 'pointer' }}>
                    {isConnected ? 'Live' : 'Connect Upstox'}
                </button>
                <input type="text" placeholder="Search ticker..." className={styles.search} />
            </div>
            <div className={styles.contentGrid}>
                <div className={styles.leftCol}>
                    <div style={{ height: 400 }}>
                        <PriceChart data={candles} symbol={selectedTicker} />
                    </div>
                    <LiveTickStream ticks={ticks} />
                </div>
                <div className={styles.rightCol}>
                    <div className={styles.indicatorsRow}>
                        <BuyIndicator value={78} reason="High Vol + Sentiment" />
                        <FearIndicator value={42} reason="Market Neutral" />
                    </div>
                    <SentimentPanel items={sentiment} overallScore={0.65} />
                    <div style={{ flex: 1, minHeight: 300 }}>
                        <Leaderboard items={growth} onSelect={setSelectedTicker} />
                    </div>
                </div>
            </div>
        </Layout>
    );
}

export default App;
