import React from 'react';
import { SentimentItem } from '../../utils/types';
import styles from './SentimentPanel.module.css';
import { Newspaper } from 'lucide-react';

interface SentimentPanelProps {
    items: SentimentItem[];
    overallScore: number;
}

const SentimentPanel: React.FC<SentimentPanelProps> = ({ items, overallScore }) => {
    const markerPosition = ((overallScore + 1) / 2) * 100;

    const getSentimentClass = (score: number) => {
        if (score > 0.2) return styles.pos;
        if (score < -0.2) return styles.neg;
        return styles.neu;
    };

    const getSentimentLabel = (score: number) => {
        if (score > 0.2) return 'POS';
        if (score < -0.2) return 'NEG';
        return 'NEU';
    };

    return (
        <div className={styles.container}>
            <div className={styles.header}>
                <h3 className={styles.title}>
                    <Newspaper size={18} style={{ marginRight: 8, display: 'inline-block', verticalAlign: 'text-bottom' }} />
                    Market Sentiment
                </h3>
                <span style={{ fontWeight: 600 }}>{(overallScore * 100).toFixed(0)}</span>
            </div>

            <div className={styles.scoreBar}>
                <div className={styles.marker} style={{ left: `${markerPosition}%` }} />
            </div>

            <div className={styles.newsList}>
                {items.map((item) => (
                    <div key={item.id} className={styles.newsItem}>
                        <div className={styles.source}>{item.source}</div>
                        <div style={{ flex: 1 }}>
                            <div className={styles.headline}>{item.title}</div>
                        </div>
                        <div className={`${styles.chip} ${getSentimentClass(item.score)}`}>
                            {getSentimentLabel(item.score)}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default SentimentPanel;
