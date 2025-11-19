import React, { useEffect, useRef } from 'react';
import { PriceTick } from '../../utils/types';
import styles from './LiveTickStream.module.css';
import { format } from 'date-fns';
import { Activity } from 'lucide-react';

interface LiveTickStreamProps {
    ticks: PriceTick[];
}

const LiveTickStream: React.FC<LiveTickStreamProps> = ({ ticks }) => {
    const listRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (listRef.current) {
            listRef.current.scrollTop = 0;
        }
    }, [ticks]);

    return (
        <div className={styles.container}>
            <h3 className={styles.title}>
                <Activity size={16} /> Live Trades
            </h3>
            <div className={styles.list} ref={listRef}>
                {ticks.map((tick, i) => (
                    <div key={`${tick.timestamp}-${i}`} className={styles.row}>
                        <span className={styles.time}>{format(new Date(tick.timestamp), 'HH:mm:ss')}</span>
                        <span className={tick.side === 'buy' ? styles.buy : styles.sell}>
                            {tick.price.toFixed(2)}
                        </span>
                        <span>{tick.volume}</span>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default LiveTickStream;
