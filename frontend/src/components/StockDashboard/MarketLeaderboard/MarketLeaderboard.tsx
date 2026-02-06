import React from 'react';
import { motion } from 'framer-motion';
import { Trophy, TrendingUp, TrendingDown } from 'lucide-react';
import styles from './MarketLeaderboard.module.css';

interface LeaderboardStock {
    rank: number;
    symbol: string;
    name: string;
    sector: string;
    growthScore: number;
    sentiment: string;
    forecastPercent: number;
}

interface MarketLeaderboardProps {
    stocks?: LeaderboardStock[];
    onStockClick?: (symbol: string) => void;
}

const MarketLeaderboard: React.FC<MarketLeaderboardProps> = ({
    stocks = [],
    onStockClick
}) => {
    const getSentimentClass = (sentiment: string) => {
        switch (sentiment) {
            case 'bullish': return styles.bullish;
            case 'bearish': return styles.bearish;
            default: return styles.neutral;
        }
    };

    const getSectorColor = (sector: string): string => {
        const colors: Record<string, string> = {
            'IT': '#3B82F6',
            'Banking': '#8B5CF6',
            'Energy': '#F59E0B',
            'Pharma': '#10B981',
            'Auto': '#EF4444',
            'FMCG': '#EC4899',
            'Telecom': '#06B6D4',
            'Metals': '#6B7280',
            'Engineering': '#F97316',
            'Finance': '#A855F7',
            'Consumer': '#14B8A6',
        };
        return colors[sector] || '#6B7280';
    };

    // Show loading state if no stocks
    if (!stocks || stocks.length === 0) {
        return (
            <div className={styles.container}>
                <div className={styles.header}>
                    <div className={styles.headerTitle}>
                        <Trophy size={18} className={styles.headerIcon} />
                        <h3>Market Intelligence</h3>
                    </div>
                    <span className={styles.subtitle}>Top Growth-Ranked Stocks</span>
                </div>
                <div className={styles.loading}>
                    <p>Loading leaderboard...</p>
                </div>
            </div>
        );
    }

    return (
        <div className={styles.container}>
            <div className={styles.header}>
                <div className={styles.headerTitle}>
                    <Trophy size={18} className={styles.headerIcon} />
                    <h3>Market Intelligence</h3>
                </div>
                <span className={styles.subtitle}>Top Growth-Ranked Stocks (Live)</span>
            </div>

            <div className={styles.tableWrapper}>
                <table className={styles.table}>
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Symbol</th>
                            <th>Sector</th>
                            <th>Growth Score</th>
                            <th>Sentiment</th>
                            <th>Change</th>
                        </tr>
                    </thead>
                    <tbody>
                        {stocks.map((stock, index) => (
                            <motion.tr
                                key={stock.symbol}
                                className={styles.row}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: index * 0.05 }}
                                onClick={() => onStockClick?.(stock.symbol)}
                            >
                                <td className={styles.rank}>
                                    <span className={stock.rank <= 3 ? styles.topRank : ''}>
                                        {stock.rank}
                                    </span>
                                </td>
                                <td className={styles.symbolCell}>
                                    <div className={styles.stockInfo}>
                                        <span className={styles.symbol}>{stock.symbol}</span>
                                        <span className={styles.name}>{stock.name}</span>
                                    </div>
                                </td>
                                <td>
                                    <span
                                        className={styles.sectorTag}
                                        style={{ '--sector-color': getSectorColor(stock.sector) } as React.CSSProperties}
                                    >
                                        {stock.sector}
                                    </span>
                                </td>
                                <td>
                                    <div className={styles.scoreCell}>
                                        <div className={styles.scoreBar}>
                                            <motion.div
                                                className={styles.scoreFill}
                                                initial={{ width: 0 }}
                                                animate={{ width: `${stock.growthScore}%` }}
                                                transition={{ duration: 0.5, delay: index * 0.05 }}
                                            />
                                        </div>
                                        <span className={styles.scoreValue}>{stock.growthScore.toFixed(0)}</span>
                                    </div>
                                </td>
                                <td>
                                    <span className={`${styles.sentimentBadge} ${getSentimentClass(stock.sentiment)}`}>
                                        {stock.sentiment.charAt(0).toUpperCase() + stock.sentiment.slice(1)}
                                    </span>
                                </td>
                                <td className={styles.forecastCell}>
                                    <span className={stock.forecastPercent >= 0 ? styles.positive : styles.negative}>
                                        {stock.forecastPercent >= 0 ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                                        {stock.forecastPercent >= 0 ? '+' : ''}{stock.forecastPercent.toFixed(1)}%
                                    </span>
                                </td>
                            </motion.tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default MarketLeaderboard;
export type { LeaderboardStock };
