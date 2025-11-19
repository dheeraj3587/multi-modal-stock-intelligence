import React from 'react';
import styles from './Layout.module.css';
import { LayoutDashboard, Activity, TrendingUp, PieChart, Settings } from 'lucide-react';

interface LayoutProps {
    children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
    return (
        <div className={styles.appContainer}>
            <aside className={styles.sidebar}>
                <div className={styles.logo}>S</div>
                <nav className={styles.nav}>
                    <div className={`${styles.navItem} ${styles.active}`} title="Overview">
                        <LayoutDashboard size={24} />
                    </div>
                    <div className={styles.navItem} title="Live Feed">
                        <Activity size={24} />
                    </div>
                    <div className={styles.navItem} title="Growth">
                        <TrendingUp size={24} />
                    </div>
                    <div className={styles.navItem} title="Portfolio">
                        <PieChart size={24} />
                    </div>
                </nav>
                <div style={{ marginTop: 'auto', marginBottom: 24 }}>
                    <div className={styles.navItem} title="Settings">
                        <Settings size={24} />
                    </div>
                </div>
            </aside>
            <main className={styles.main}>
                <header className={styles.header}>
                    <h1 className={styles.pageTitle}>Dashboard</h1>
                    <div className={styles.controls}>
                        <input type="text" placeholder="Search ticker..." className={styles.search} />
                        <div style={{ width: 32, height: 32, background: '#E5E7EB', borderRadius: '50%' }} />
                    </div>
                </header>
                {children}
            </main>
        </div>
    );
};

export default Layout;
