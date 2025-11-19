import React from 'react';
import styles from './Indicators.module.css';
import { TrendingUp } from 'lucide-react';

interface BuyIndicatorProps {
    value: number;
    reason: string;
}

const BuyIndicator: React.FC<BuyIndicatorProps> = ({ value, reason }) => {
    const radius = 50;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (value / 100) * circumference;

    return (
        <div className={styles.card}>
            <div className={styles.title}>Buy Pressure</div>
            <div className={styles.ringContainer}>
                <svg width="120" height="120" viewBox="0 0 120 120">
                    <circle
                        cx="60"
                        cy="60"
                        r={radius}
                        fill="none"
                        stroke="#E5E7EB"
                        strokeWidth="8"
                    />
                    <circle
                        cx="60"
                        cy="60"
                        r={radius}
                        fill="none"
                        stroke="#8EC5FF"
                        strokeWidth="8"
                        strokeDasharray={circumference}
                        strokeDashoffset={offset}
                        strokeLinecap="round"
                        transform="rotate(-90 60 60)"
                        style={{ transition: 'stroke-dashoffset 0.5s ease' }}
                    />
                </svg>
                <div className={styles.value}>{value}</div>
            </div>
            <div className={styles.reason}>
                <TrendingUp size={14} style={{ display: 'inline', marginRight: 4 }} />
                {reason}
            </div>
        </div>
    );
};

export default BuyIndicator;
