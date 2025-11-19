import React from 'react';
import { GrowthItem } from '../../utils/types';
import styles from './Leaderboard.module.css';
import { Trophy } from 'lucide-react';
import { Line, LineChart, ResponsiveContainer } from 'recharts';

interface LeaderboardProps {
    items: GrowthItem[];
    onSelect: (ticker: string) => void;
}

const Leaderboard: React.FC<LeaderboardProps> = ({ items, onSelect }) => {
    return (
        <div className={styles.container}>
            <div className={styles.header}>
                <h3 className={styles.title}>
                    <Trophy size={18} color="#F59E0B" /> Top Growth
                </h3>
            </div>
            <div className={styles.list}>
                {items.map((item, index) => (
                    <div
                        key={item.ticker}
                        className={styles.row}
                        onClick={() => onSelect(item.ticker)}
                        role="button"
                        tabIndex={0}
                        onKeyDown={(e) => e.key === 'Enter' && onSelect(item.ticker)}
                    >
                        <div className={styles.rank}>{index + 1}</div>
                        <div className={styles.info}>
                            <span className={styles.ticker}>{item.ticker}</span>
                            <span className={styles.name}>{item.name}</span>
                        </div>
                        <div className={styles.sparkline}>
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={item.sparkline.map((val, i) => ({ i, val }))}>
                                    <Line
                                        type="monotone"
                                        dataKey="val"
                                        stroke="#10B981"
                                        strokeWidth={2}
                                        dot={false}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                        <div className={styles.score}>{item.score}</div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default Leaderboard;
