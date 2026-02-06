import React from 'react';
import { motion } from 'framer-motion';
import styles from './Layout.module.css';
import {
    LayoutDashboard,
    Activity,
    TrendingUp,
    Wallet,
    Settings,
    Bell,
    Search
} from 'lucide-react';

interface LayoutProps {
    children: React.ReactNode;
    isConnected?: boolean;
    isConnecting?: boolean;
    onConnect?: () => void;
    onDisconnect?: () => void;
    marketData?: {
        nifty50: { price: number; change: number };
        sensex: { price: number; change: number };
    };
}

const Layout: React.FC<LayoutProps> = ({
    children,
    isConnected = false,
    isConnecting = false,
    onConnect,
    onDisconnect,
    marketData
}) => {
    const navItems = [
        { icon: LayoutDashboard, label: 'Dashboard', active: true },
        { icon: Activity, label: 'Trade', active: false },
        { icon: TrendingUp, label: 'Markets', active: false },
        { icon: Wallet, label: 'Portfolio', active: false },
    ];

    return (
        <div className={styles.appContainer}>
            {/* Sidebar */}
            <aside className={styles.sidebar}>
                <motion.div
                    className={styles.logo}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                >
                    S
                </motion.div>

                <nav className={styles.nav}>
                    {navItems.map((item, index) => (
                        <motion.div
                            key={item.label}
                            className={`${styles.navItem} ${item.active ? styles.active : ''}`}
                            title={item.label}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.1 }}
                        >
                            <item.icon size={20} />
                        </motion.div>
                    ))}
                </nav>

                <motion.div
                    className={styles.navItem}
                    title="Settings"
                    whileHover={{ scale: 1.05 }}
                >
                    <Settings size={20} />
                </motion.div>
            </aside>

            {/* Main Content */}
            <main className={styles.main}>
                {/* Header */}
                <header className={styles.header}>
                    <div className={styles.headerLeft}>
                        <h1 className={styles.pageTitle}>Trading Dashboard</h1>

                        {/* Market Ticker */}
                        <div className={styles.marketTicker}>
                            <div className={styles.tickerItem}>
                                <span className={styles.tickerLabel}>NIFTY 50</span>
                                <span className={styles.tickerValue}>
                                    {marketData?.nifty50?.price?.toLocaleString() || '23,456.78'}
                                </span>
                                <span className={`${styles.tickerChange} ${(marketData?.nifty50?.change || 0) >= 0 ? styles.up : styles.down}`}>
                                    {(marketData?.nifty50?.change || 0) >= 0 ? '+' : ''}{(marketData?.nifty50?.change || 0.45).toFixed(2)}%
                                </span>
                            </div>
                            <div className={styles.tickerItem}>
                                <span className={styles.tickerLabel}>SENSEX</span>
                                <span className={styles.tickerValue}>
                                    {marketData?.sensex?.price?.toLocaleString() || '77,234.56'}
                                </span>
                                <span className={`${styles.tickerChange} ${(marketData?.sensex?.change || 0) >= 0 ? styles.up : styles.down}`}>
                                    {(marketData?.sensex?.change || 0) >= 0 ? '+' : ''}{(marketData?.sensex?.change || 0.32).toFixed(2)}%
                                </span>
                            </div>
                        </div>
                    </div>

                    <div className={styles.headerRight}>
                        <div style={{ position: 'relative' }}>
                            <Search size={16} style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-tertiary)' }} />
                            <input
                                type="text"
                                placeholder="Search markets..."
                                className={styles.search}
                                style={{ paddingLeft: 36 }}
                            />
                        </div>

                        <motion.div whileHover={{ scale: 1.05 }} style={{ cursor: 'pointer', color: 'var(--text-secondary)' }}>
                            <Bell size={20} />
                        </motion.div>

                        {/* Connection Status */}
                        {isConnecting ? (
                            <div className={`${styles.connectionStatus} ${styles.connecting}`}>
                                <span className={styles.liveIndicator} style={{ background: 'var(--color-warning)' }} />
                                Connecting...
                            </div>
                        ) : isConnected ? (
                            <motion.button
                                className={`${styles.connectionStatus} ${styles.connected}`}
                                onClick={onDisconnect}
                                whileHover={{ scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                            >
                                <span className="live-indicator" />
                                Live
                            </motion.button>
                        ) : (
                            <motion.button
                                className={`${styles.controlBtn} ${styles.primary}`}
                                onClick={onConnect}
                                whileHover={{ scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                            >
                                Connect
                            </motion.button>
                        )}
                    </div>
                </header>

                {/* Content */}
                <div className={styles.content}>
                    {children}
                </div>
            </main>
        </div>
    );
};

export default Layout;
