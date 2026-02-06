import React from 'react';
import { motion } from 'framer-motion';
import { BrainCircuit, Target, TrendingUp, Activity, Shield, Tag } from 'lucide-react';
import styles from './AIInsightsPanel.module.css';

type RiskLevel = 'low' | 'medium' | 'high';

interface AIInsightsData {
    growthScore: number;
    buyConfidence: number;
    sentimentIndex: number; // 1-5
    volatility: number;
    riskLevel: RiskLevel;
    sector: string;
}

interface AIInsightsPanelProps {
    data: AIInsightsData;
}

// Growth Score Gauge Component
const GrowthScoreGauge: React.FC<{ score: number }> = ({ score }) => {
    const circumference = 2 * Math.PI * 45;
    const strokeDashoffset = circumference - (score / 100) * circumference;
    
    return (
        <div className={styles.gaugeContainer}>
            <svg className={styles.gaugeSvg} viewBox="0 0 100 100">
                {/* Background circle */}
                <circle
                    cx="50"
                    cy="50"
                    r="45"
                    fill="none"
                    stroke="var(--bg-hover)"
                    strokeWidth="8"
                />
                {/* Progress circle */}
                <motion.circle
                    cx="50"
                    cy="50"
                    r="45"
                    fill="none"
                    stroke="url(#gaugeGradient)"
                    strokeWidth="8"
                    strokeLinecap="round"
                    strokeDasharray={circumference}
                    initial={{ strokeDashoffset: circumference }}
                    animate={{ strokeDashoffset }}
                    transition={{ duration: 1, ease: 'easeOut' }}
                    transform="rotate(-90 50 50)"
                    style={{ filter: 'drop-shadow(0 0 6px rgba(46, 229, 157, 0.5))' }}
                />
                <defs>
                    <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#2EE59D" />
                        <stop offset="100%" stopColor="#6FFFD2" />
                    </linearGradient>
                </defs>
            </svg>
            <div className={styles.gaugeValue}>
                <motion.span
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.5 }}
                >
                    {score}
                </motion.span>
            </div>
        </div>
    );
};

// Confidence Meter Component
const ConfidenceMeter: React.FC<{ value: number; label: string; icon: React.ReactNode }> = ({ value, label, icon }) => {
    return (
        <div className={styles.meterContainer}>
            <div className={styles.meterHeader}>
                {icon}
                <span>{label}</span>
            </div>
            <div className={styles.meterBar}>
                <motion.div
                    className={styles.meterFill}
                    initial={{ width: 0 }}
                    animate={{ width: `${value}%` }}
                    transition={{ duration: 0.8, ease: 'easeOut' }}
                />
            </div>
            <span className={styles.meterValue}>{value}%</span>
        </div>
    );
};

// Sentiment Index Component (5 dots)
const SentimentIndex: React.FC<{ value: number }> = ({ value }) => {
    return (
        <div className={styles.sentimentContainer}>
            <div className={styles.sentimentHeader}>
                <TrendingUp size={14} />
                <span>Market Sentiment</span>
            </div>
            <div className={styles.sentimentDots}>
                {[1, 2, 3, 4, 5].map((dot) => (
                    <motion.div
                        key={dot}
                        className={`${styles.sentimentDot} ${dot <= value ? styles.active : ''}`}
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: dot * 0.1 }}
                    />
                ))}
            </div>
            <span className={styles.sentimentLabel}>
                {value <= 2 ? 'Bearish' : value >= 4 ? 'Bullish' : 'Neutral'}
            </span>
        </div>
    );
};

const AIInsightsPanel: React.FC<AIInsightsPanelProps> = ({ data }) => {
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
                <BrainCircuit size={18} className={styles.headerIcon} />
                <h3>AI Insights</h3>
            </div>

            {/* Growth Score Gauge */}
            <div className={styles.section}>
                <div className={styles.sectionLabel}>Growth Score</div>
                <GrowthScoreGauge score={data.growthScore} />
            </div>

            {/* Buy Confidence */}
            <ConfidenceMeter
                value={data.buyConfidence}
                label="Buy Confidence"
                icon={<Target size={14} />}
            />

            {/* Sentiment Index */}
            <SentimentIndex value={data.sentimentIndex} />

            {/* Volatility */}
            <div className={styles.metricRow}>
                <div className={styles.metricLabel}>
                    <Activity size={14} />
                    <span>Volatility</span>
                </div>
                <span className={styles.metricValue}>
                    {data.volatility.toFixed(1)}%
                </span>
            </div>

            {/* Risk Level */}
            <div className={styles.metricRow}>
                <div className={styles.metricLabel}>
                    <Shield size={14} />
                    <span>Risk</span>
                </div>
                <span className={`${styles.riskBadge} ${getRiskColor(data.riskLevel)}`}>
                    {data.riskLevel.toUpperCase()}
                </span>
            </div>

            {/* Sector */}
            <div className={styles.metricRow}>
                <div className={styles.metricLabel}>
                    <Tag size={14} />
                    <span>Sector</span>
                </div>
                <span className={styles.sectorTag}>{data.sector}</span>
            </div>
        </div>
    );
};

export default AIInsightsPanel;
export type { AIInsightsData };
