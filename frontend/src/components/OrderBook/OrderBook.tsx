import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import styles from './OrderBook.module.css';
import { BookOpen } from 'lucide-react';

interface Order {
    price: number;
    size: number;
    total: number;
}

interface Trade {
    price: number;
    size: number;
    time: string;
    side: 'buy' | 'sell';
}

interface OrderBookProps {
    asks: Order[];
    bids: Order[];
    trades?: Trade[];
    spread?: number;
}

const OrderBook: React.FC<OrderBookProps> = ({
    asks = [],
    bids = [],
    trades = [],
    spread = 0
}) => {
    const [activeTab, setActiveTab] = useState<'book' | 'trades'>('book');

    const maxTotal = Math.max(
        ...asks.map(o => o.total),
        ...bids.map(o => o.total),
        1
    );

    return (
        <div className={styles.container}>
            <div className={styles.header}>
                <h3 className={styles.title}>
                    <BookOpen size={16} />
                    Order Book
                </h3>
                <div className={styles.tabs}>
                    <button
                        className={`${styles.tab} ${activeTab === 'book' ? styles.active : ''}`}
                        onClick={() => setActiveTab('book')}
                    >
                        Book
                    </button>
                    <button
                        className={`${styles.tab} ${activeTab === 'trades' ? styles.active : ''}`}
                        onClick={() => setActiveTab('trades')}
                    >
                        Trades
                    </button>
                </div>
            </div>

            <div className={styles.content}>
                {activeTab === 'book' ? (
                    <>
                        <div className={styles.columnHeaders}>
                            <span>Price (₹)</span>
                            <span>Size</span>
                            <span>Total</span>
                        </div>

                        {/* Asks (Sell Orders) */}
                        <div className={`${styles.orderList} ${styles.askSide}`}>
                            <AnimatePresence>
                                {asks.slice(0, 8).map((order) => (
                                    <motion.div
                                        key={`ask-${order.price}`}
                                        className={`${styles.orderRow} ${styles.askRow}`}
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        exit={{ opacity: 0 }}
                                    >
                                        <div
                                            className={styles.depthBar}
                                            style={{ width: `${(order.total / maxTotal) * 100}%` }}
                                        />
                                        <span className={styles.price}>{order.price.toFixed(2)}</span>
                                        <span>{order.size.toLocaleString()}</span>
                                        <span>{order.total.toLocaleString()}</span>
                                    </motion.div>
                                ))}
                            </AnimatePresence>
                        </div>

                        {/* Spread */}
                        <div className={styles.spread}>
                            <span className={styles.spreadLabel}>Spread</span>
                            <span className={styles.spreadValue}>₹{spread.toFixed(2)}</span>
                            <span className={styles.spreadLabel}>({((spread / (bids[0]?.price || 1)) * 100).toFixed(3)}%)</span>
                        </div>

                        {/* Bids (Buy Orders) */}
                        <div className={`${styles.orderList} ${styles.bidSide}`}>
                            <AnimatePresence>
                                {bids.slice(0, 8).map((order) => (
                                    <motion.div
                                        key={`bid-${order.price}`}
                                        className={`${styles.orderRow} ${styles.bidRow}`}
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        exit={{ opacity: 0 }}
                                    >
                                        <div
                                            className={styles.depthBar}
                                            style={{ width: `${(order.total / maxTotal) * 100}%` }}
                                        />
                                        <span className={styles.price}>{order.price.toFixed(2)}</span>
                                        <span>{order.size.toLocaleString()}</span>
                                        <span>{order.total.toLocaleString()}</span>
                                    </motion.div>
                                ))}
                            </AnimatePresence>
                        </div>
                    </>
                ) : (
                    <>
                        <div className={styles.columnHeaders}>
                            <span>Price (₹)</span>
                            <span>Size</span>
                            <span>Time</span>
                        </div>
                        <div className={styles.orderList}>
                            {trades.map((trade, i) => (
                                <div
                                    key={i}
                                    className={`${styles.tradeRow} ${trade.side}`}
                                >
                                    <span className={styles.price}>{trade.price.toFixed(2)}</span>
                                    <span>{trade.size.toLocaleString()}</span>
                                    <span className={styles.time}>{trade.time}</span>
                                </div>
                            ))}
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};

export default OrderBook;
