import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
    BrainCircuit,
    LayoutDashboard,
    LineChart,
    TrendingUp,
    BarChart3,
    Briefcase,
    Bell,
    Settings,
    User,
    ChevronDown
} from 'lucide-react';
import styles from './Navbar.module.css';

interface NavbarProps {
    activeNav?: string;
    onNavChange?: (nav: string) => void;
    onConnectBroker?: () => void;
    isConnected?: boolean;
}

const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'forecasts', label: 'Forecasts', icon: LineChart },
    { id: 'sentiment', label: 'Sentiment', icon: TrendingUp },
    { id: 'growth', label: 'Growth Scores', icon: BarChart3 },
    { id: 'portfolio', label: 'Portfolio', icon: Briefcase },
];

const Navbar: React.FC<NavbarProps> = ({
    activeNav = 'dashboard',
    onNavChange,
    onConnectBroker,
    isConnected = false
}) => {
    const [market, setMarket] = useState<'NSE' | 'BSE'>('NSE');
    const [showMarketDropdown, setShowMarketDropdown] = useState(false);

    return (
        <nav className={styles.navbar}>
            {/* Logo */}
            <div className={styles.logoSection}>
                <motion.div
                    className={styles.logo}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                >
                    <BrainCircuit size={24} />
                </motion.div>
                <span className={styles.logoText}>Stock Intelligence AI</span>
            </div>

            {/* Navigation Items */}
            <div className={styles.navItems}>
                {navItems.map((item) => (
                    <motion.button
                        key={item.id}
                        className={`${styles.navItem} ${activeNav === item.id ? styles.active : ''}`}
                        onClick={() => onNavChange?.(item.id)}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                    >
                        <item.icon size={16} />
                        <span>{item.label}</span>
                    </motion.button>
                ))}
            </div>

            {/* Right Controls */}
            <div className={styles.controls}>
                {/* Market Selector */}
                <div className={styles.marketSelector}>
                    <button
                        className={styles.marketBtn}
                        onClick={() => setShowMarketDropdown(!showMarketDropdown)}
                    >
                        <span className={styles.marketDot} />
                        {market}
                        <ChevronDown size={14} />
                    </button>
                    {showMarketDropdown && (
                        <div className={styles.marketDropdown}>
                            <button
                                className={market === 'NSE' ? styles.active : ''}
                                onClick={() => { setMarket('NSE'); setShowMarketDropdown(false); }}
                            >
                                NSE
                            </button>
                            <button
                                className={market === 'BSE' ? styles.active : ''}
                                onClick={() => { setMarket('BSE'); setShowMarketDropdown(false); }}
                            >
                                BSE
                            </button>
                        </div>
                    )}
                </div>

                {/* Data Status Indicator */}
                <div className={styles.dataStatus}>
                    <div className={`${styles.statusPulse} ${isConnected ? styles.live : styles.delayed}`} />
                    <span className={styles.statusText}>
                        {isConnected ? 'Real-time' : 'Delayed (15m)'}
                    </span>
                </div>

                {/* Icons */}
                <motion.button
                    className={styles.iconBtn}
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.95 }}
                >
                    <Bell size={18} />
                </motion.button>
                <motion.button
                    className={styles.iconBtn}
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.95 }}
                >
                    <Settings size={18} />
                </motion.button>
                <motion.button
                    className={styles.iconBtn}
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.95 }}
                >
                    <User size={18} />
                </motion.button>

                {/* Connect Broker CTA */}
                <motion.button
                    className={`${styles.connectBtn} ${isConnected ? styles.connected : ''}`}
                    onClick={onConnectBroker}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                >
                    {isConnected ? 'Connected' : 'Connect Broker'}
                </motion.button>
            </div>
        </nav>
    );
};

export default Navbar;
