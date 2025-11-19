import React from 'react';
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Candle } from '../../utils/types';
import styles from './PriceChart.module.css';
import { format } from 'date-fns';

interface PriceChartProps {
    data: Candle[];
    symbol: string;
}

const PriceChart: React.FC<PriceChartProps> = ({ data, symbol }) => {
    return (
        <div className={styles.container}>
            <div className={styles.header}>
                <h2 className={styles.title}>{symbol} Price Prediction</h2>
            </div>
            <div className={styles.chartArea}>
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data}>
                        <defs>
                            <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#8EC5FF" stopOpacity={0.3} />
                                <stop offset="95%" stopColor="#8EC5FF" stopOpacity={0} />
                            </linearGradient>
                            <linearGradient id="colorConfidence" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#C6FFD9" stopOpacity={0.2} />
                                <stop offset="95%" stopColor="#C6FFD9" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(0,0,0,0.05)" />
                        <XAxis
                            dataKey="start"
                            tickFormatter={(tick) => format(new Date(tick), 'HH:mm')}
                            stroke="#9CA3AF"
                            tick={{ fontSize: 12 }}
                            tickLine={false}
                            axisLine={false}
                        />
                        <YAxis
                            domain={['auto', 'auto']}
                            stroke="#9CA3AF"
                            tick={{ fontSize: 12 }}
                            tickLine={false}
                            axisLine={false}
                        />
                        <Tooltip
                            contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,0.05)' }}
                            labelFormatter={(label) => format(new Date(label), 'HH:mm:ss')}
                        />
                        <Area
                            type="monotone"
                            dataKey="confidenceUpper"
                            stroke="none"
                            fill="url(#colorConfidence)"
                        />
                        <Area
                            type="monotone"
                            dataKey="confidenceLower"
                            stroke="none"
                            fill="#fff"
                            fillOpacity={0.5}
                        />
                        <Area
                            type="monotone"
                            dataKey="close"
                            stroke="#8EC5FF"
                            strokeWidth={3}
                            fill="url(#colorPrice)"
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default PriceChart;
