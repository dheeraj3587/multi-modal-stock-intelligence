import React from 'react';
import styles from './Indicators.module.css';
import { AlertTriangle } from 'lucide-react';

interface FearIndicatorProps {
    value: number;
    reason: string;
}

const FearIndicator: React.FC<FearIndicatorProps> = ({ value, reason }) => {
    const radius = 50;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (value / 100) * circumference;

    const color = value > 50 ? '#EF4444' : '#10B981';

    return (
        <div className={styles.card}>
            <div className={styles.title}>Fear Index</div>
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
                        stroke={color}
                        strokeWidth="8"
                        strokeDasharray={circumference}
                        strokeDashoffset={offset}
                        strokeLinecap="round"
                        transform="rotate(-90 60 60)"
                        style={{ transition: 'stroke-dashoffset 0.5s ease, stroke 0.5s ease' }}
                    />
                </svg>
                <div className={styles.value}>{value}</div>
            </div>
            <div className={styles.reason}>
                <AlertTriangle size={14} style={{ display: 'inline', marginRight: 4 }} />
                {reason}
            </div>
        </div>
    );
};

export default FearIndicator;
