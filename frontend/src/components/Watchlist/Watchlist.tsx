import React from 'react';
import { motion } from 'framer-motion';
import styles from './Watchlist.module.css';
import { Star, Plus, TrendingUp, TrendingDown } from 'lucide-react';

interface WatchlistItem {
    symbol: string;
    name: string;
    price: number;
    change: number;
    instrumentKey?: string;
}

interface WatchlistProps {
    items: WatchlistItem[];
    activeSymbol?: string;
    onSelect: (symbol: string, instrumentKey?: string) => void;
    onAdd?: () => void;
}

const Watchlist: React.FC<WatchlistProps> = ({ items, activeSymbol, onSelect, onAdd }) => {
    return (
        <div className={styles.container}>
            <div className={styles.header}>
                <h3 className={styles.title}>
                    <Star size={16} />
                    Watchlist
                </h3>
                <motion.button
                    className={styles.addBtn}
                    onClick={onAdd}
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                >
                    <Plus size={16} />
                </motion.button>
            </div>

            <div className={styles.list}>
                {items.length === 0 ? (
                    <div className={styles.empty}>
                        <Star size={32} className={styles.emptyIcon} />
                        <span>No stocks in watchlist</span>
                    </div>
                ) : (
                    items.map((item, index) => (
                        <motion.div
                            key={item.symbol}
                            className={`${styles.item} ${activeSymbol === item.symbol ? styles.active : ''}`}
                            onClick={() => onSelect(item.symbol, item.instrumentKey)}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.05 }}
                            whileHover={{ x: 4 }}
                        >
                            <div className={styles.itemLeft}>
                                <span className={styles.symbol}>{item.symbol}</span>
                                <span className={styles.name}>{item.name}</span>
                            </div>
                            <div className={styles.itemRight}>
                                <span className={styles.price}>
                                    ₹{item.price.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                                </span>
                                <span className={`${styles.change} ${item.change >= 0 ? styles.changeUp : styles.changeDown}`}>
                                    {item.change >= 0 ? <TrendingUp size={10} /> : <TrendingDown size={10} />}
                                    {item.change >= 0 ? '+' : ''}{item.change.toFixed(2)}%
                                </span>
                            </div>
                        </motion.div>
                    ))
                )}
            </div>
        </div>
    );
};

export default Watchlist;
