import React from 'react';
import { motion } from 'framer-motion';
import styles from './MarketIndices.module.css';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface IndexData {
    name: string;
    symbol: string;
    price: number;
    change: number;
    changePercent: number;
}

interface MarketIndicesProps {
    indices: IndexData[];
    onSelect?: (symbol: string) => void;
}

const MarketIndices: React.FC<MarketIndicesProps> = ({ indices, onSelect }) => {
    return (
        <div className={styles.container}>
            {indices.map((index, i) => {
                const isUp = index.change >= 0;
                return (
                    <motion.div
                        key={index.symbol}
                        className={`${styles.card} ${isUp ? styles.up : styles.down}`}
                        onClick={() => onSelect?.(index.symbol)}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.1 }}
                        whileHover={{ scale: 1.01 }}
                    >
                        <div className={styles.cardHeader}>
                            <span className={styles.indexName}>{index.name}</span>
                            <div className={styles.liveIndicator}>
                                <span className={styles.liveDot} />
                                LIVE
                            </div>
                        </div>

                        <div className={styles.cardBody}>
                            <div className={styles.priceSection}>
                                <motion.span
                                    className={styles.price}
                                    key={index.price}
                                    initial={{ scale: 1 }}
                                    animate={{ scale: [1, 1.02, 1] }}
                                    transition={{ duration: 0.3 }}
                                >
                                    {index.price.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                                </motion.span>
                                <div className={styles.changeSection}>
                                    <span className={styles.changeValue}>
                                        {isUp ? '+' : ''}{index.change.toFixed(2)}
                                    </span>
                                    <span className={styles.changePercent}>
                                        {isUp ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                                        {isUp ? '+' : ''}{index.changePercent.toFixed(2)}%
                                    </span>
                                </div>
                            </div>

                            {/* Mini Sparkline SVG */}
                            <svg className={styles.sparkline} viewBox="0 0 80 40">
                                <defs>
                                    <linearGradient id={`gradient-${index.symbol}`} x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="0%" stopColor={isUp ? '#00ff88' : '#ff3b5c'} stopOpacity="0.3" />
                                        <stop offset="100%" stopColor={isUp ? '#00ff88' : '#ff3b5c'} stopOpacity="0" />
                                    </linearGradient>
                                </defs>
                                <path
                                    d={isUp
                                        ? "M0,35 L10,30 L20,32 L30,25 L40,20 L50,22 L60,15 L70,10 L80,5"
                                        : "M0,5 L10,10 L20,8 L30,15 L40,20 L50,18 L60,25 L70,30 L80,35"
                                    }
                                    fill="none"
                                    stroke={isUp ? '#00ff88' : '#ff3b5c'}
                                    strokeWidth="2"
                                />
                                <path
                                    d={isUp
                                        ? "M0,35 L10,30 L20,32 L30,25 L40,20 L50,22 L60,15 L70,10 L80,5 L80,40 L0,40 Z"
                                        : "M0,5 L10,10 L20,8 L30,15 L40,20 L50,18 L60,25 L70,30 L80,35 L80,40 L0,40 Z"
                                    }
                                    fill={`url(#gradient-${index.symbol})`}
                                />
                            </svg>
                        </div>
                    </motion.div>
                );
            })}
        </div>
    );
};

export default MarketIndices;
