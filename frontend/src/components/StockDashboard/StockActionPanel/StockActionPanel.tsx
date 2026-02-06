import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { ChevronDown, TrendingUp, TrendingDown, AlertTriangle, Target, Zap } from 'lucide-react';
import styles from './StockActionPanel.module.css';

// Types
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

interface StockActionPanelProps {
    stocks: StockData[];
    selectedStock: StockData | null;
    onStockSelect: (stock: StockData) => void;
    onAnalyze?: () => void;
}

const StockActionPanel: React.FC<StockActionPanelProps> = ({
    stocks,
    selectedStock,
    onStockSelect,
    onAnalyze
}) => {
    const [isDropdownOpen, setIsDropdownOpen] = useState(false);

    // Show loading state if no stocks
    if (!stocks || stocks.length === 0) {
        return (
            <div className={styles.panel}>
                <div className={styles.header}>
                    <Zap size={18} className={styles.headerIcon} />
                    <h3>Stock Action</h3>
                </div>
                <div className={styles.loading}>
                    <p>Loading stocks...</p>
                </div>
            </div>
        );
    }

    const currentStock = selectedStock || stocks[0];

    const getSentimentColor = (sentiment: SentimentType) => {
        switch (sentiment) {
            case 'bullish': return styles.bullish;
            case 'bearish': return styles.bearish;
            default: return styles.neutral;
        }
    };

    const getRiskColor = (risk: RiskLevel) => {
        switch (risk) {
            case 'low': return styles.riskLow;
            case 'high': return styles.riskHigh;
            default: return styles.riskMedium;
        }
    };

    return (
        <div className={styles.panel}>
            <div className={styles.header}>
                <Zap size={18} className={styles.headerIcon} />
                <h3>Stock Action</h3>
            </div>

            {/* Stock Selector */}
            <div className={styles.selector}>
                <label className={styles.label}>Select Stock</label>
                <div className={styles.dropdown}>
                    <button
                        className={styles.dropdownBtn}
                        onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                    >
                        <div className={styles.stockInfo}>
                            <span className={styles.stockSymbol}>{currentStock.symbol}</span>
                            <span className={styles.stockName}>{currentStock.name}</span>
                        </div>
                        <ChevronDown size={18} className={isDropdownOpen ? styles.rotated : ''} />
                    </button>
                    {isDropdownOpen && (
                        <motion.div
                            className={styles.dropdownList}
                            initial={{ opacity: 0, y: -10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                        >
                            {stocks.map((stock) => (
                                <button
                                    key={stock.symbol}
                                    className={`${styles.dropdownItem} ${currentStock.symbol === stock.symbol ? styles.selected : ''}`}
                                    onClick={() => {
                                        onStockSelect(stock);
                                        setIsDropdownOpen(false);
                                    }}
                                >
                                    <span className={styles.stockSymbol}>{stock.symbol}</span>
                                    <span className={styles.stockName}>{stock.name}</span>
                                </button>
                            ))}
                        </motion.div>
                    )}
                </div>
            </div>

            {/* Price Summary */}
            <div className={styles.priceSection}>
                <div className={styles.priceRow}>
                    <span className={styles.label}>Last Price</span>
                    <span className={styles.price}>
                        {currentStock.price > 0
                            ? `₹${currentStock.price.toLocaleString('en-IN', { minimumFractionDigits: 2 })}`
                            : 'Market Closed'}
                    </span>
                </div>
                <div className={styles.priceRow}>
                    <span className={styles.label}>Change</span>
                    <span className={`${styles.change} ${currentStock.changePercent >= 0 ? styles.positive : styles.negative}`}>
                        {currentStock.changePercent >= 0 ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                        {currentStock.changePercent >= 0 ? '+' : ''}{currentStock.changePercent.toFixed(2)}%
                    </span>
                </div>
            </div>

            {/* Forecast Confidence */}
            <div className={styles.metricCard}>
                <div className={styles.metricHeader}>
                    <Target size={16} />
                    <span>AI Forecast Confidence</span>
                </div>
                <div className={styles.progressBar}>
                    <motion.div
                        className={styles.progressFill}
                        initial={{ width: 0 }}
                        animate={{ width: `${currentStock.forecastConfidence}%` }}
                        transition={{ duration: 0.8, ease: 'easeOut' }}
                    />
                </div>
                <span className={styles.progressValue}>{currentStock.forecastConfidence.toFixed(0)}%</span>
            </div>

            {/* Sentiment Badge */}
            <div className={styles.badgeRow}>
                <span className={styles.label}>AI Sentiment</span>
                <span className={`${styles.badge} ${getSentimentColor(currentStock.sentiment)}`}>
                    {currentStock.sentiment.charAt(0).toUpperCase() + currentStock.sentiment.slice(1)}
                </span>
            </div>

            {/* Risk Level */}
            <div className={styles.badgeRow}>
                <span className={styles.label}>Risk Level</span>
                <span className={`${styles.riskBadge} ${getRiskColor(currentStock.riskLevel)}`}>
                    <AlertTriangle size={12} />
                    {currentStock.riskLevel.charAt(0).toUpperCase() + currentStock.riskLevel.slice(1)}
                </span>
            </div>

            {/* Action Button */}
            <motion.button
                className={styles.actionBtn}
                onClick={onAnalyze}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
            >
                Analyze Stock
            </motion.button>
        </div>
    );
};

export default StockActionPanel;
export type { StockData, SentimentType, RiskLevel };
