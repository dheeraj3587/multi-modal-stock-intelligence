import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
    AreaChart,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine
} from 'recharts';
import { TrendingUp, TrendingDown } from 'lucide-react';
import styles from './ForecastChart.module.css';

type Timeframe = '1D' | '1W' | '1M' | '6M' | '1Y';

interface ChartDataPoint {
    date: string;
    price: number;
    predicted?: number;
    confidenceLower?: number;
    confidenceUpper?: number;
}

interface ForecastChartProps {
    symbol: string;
    name: string;
    currentPrice: number;
    changePercent: number;
    data?: ChartDataPoint[];
}

// Generate mock chart data with AI predictions
const generateMockData = (basePrice: number, timeframe: Timeframe): ChartDataPoint[] => {
    const points: ChartDataPoint[] = [];
    let historicalDays = 0;
    let predictionDays = 0;

    switch (timeframe) {
        case '1D': historicalDays = 24; predictionDays = 6; break;
        case '1W': historicalDays = 7; predictionDays = 3; break;
        case '1M': historicalDays = 30; predictionDays = 7; break;
        case '6M': historicalDays = 180; predictionDays = 30; break;
        case '1Y': historicalDays = 365; predictionDays = 60; break;
    }

    let price = basePrice * 0.92;

    // Historical data
    for (let i = 0; i < historicalDays; i++) {
        const volatility = (Math.random() - 0.45) * (basePrice * 0.015);
        price = Math.max(price + volatility, basePrice * 0.85);
        
        const date = timeframe === '1D'
            ? `${String(i).padStart(2, '0')}:00`
            : new Date(Date.now() - (historicalDays - i) * 24 * 60 * 60 * 1000).toLocaleDateString('en-IN', { day: '2-digit', month: 'short' });
        
        points.push({
            date,
            price: parseFloat(price.toFixed(2)),
        });
    }

    // AI Predicted data
    const lastPrice = points[points.length - 1].price;
    let predictedPrice = lastPrice;
    const trend = Math.random() > 0.4 ? 1 : -1; // 60% chance bullish

    for (let i = 0; i < predictionDays; i++) {
        const volatility = trend * (Math.random() * basePrice * 0.008) + (Math.random() - 0.5) * basePrice * 0.005;
        predictedPrice = predictedPrice + volatility;
        
        const date = timeframe === '1D'
            ? `${String(historicalDays + i).padStart(2, '0')}:00`
            : new Date(Date.now() + (i + 1) * 24 * 60 * 60 * 1000).toLocaleDateString('en-IN', { day: '2-digit', month: 'short' });
        
        const confidence = 0.03 + (i * 0.005); // Confidence interval widens over time
        
        points.push({
            date,
            price: parseFloat(lastPrice.toFixed(2)), // Last known price for reference
            predicted: parseFloat(predictedPrice.toFixed(2)),
            confidenceLower: parseFloat((predictedPrice * (1 - confidence)).toFixed(2)),
            confidenceUpper: parseFloat((predictedPrice * (1 + confidence)).toFixed(2)),
        });
    }

    return points;
};

const timeframes: Timeframe[] = ['1D', '1W', '1M', '6M', '1Y'];

const ForecastChart: React.FC<ForecastChartProps> = ({
    symbol,
    name,
    currentPrice,
    changePercent,
    data
}) => {
    const [activeTimeframe, setActiveTimeframe] = useState<Timeframe>('1M');

    const chartData = useMemo(() => {
        return data || generateMockData(currentPrice, activeTimeframe);
    }, [data, currentPrice, activeTimeframe]);

    const isPositive = changePercent >= 0;
    const lastPredicted = chartData.find(d => d.predicted)?.predicted;
    const forecastChange = lastPredicted ? ((lastPredicted - currentPrice) / currentPrice * 100) : 0;

    const CustomTooltip = ({ active, payload, label }: any) => {
        if (active && payload && payload.length) {
            return (
                <div className={styles.tooltip}>
                    <p className={styles.tooltipDate}>{label}</p>
                    {payload.map((entry: any, index: number) => (
                        <p key={index} className={styles.tooltipValue} style={{ color: entry.color }}>
                            {entry.name}: ₹{entry.value?.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                        </p>
                    ))}
                </div>
            );
        }
        return null;
    };

    return (
        <div className={styles.container}>
            {/* Header */}
            <div className={styles.header}>
                <div className={styles.stockInfo}>
                    <div className={styles.symbolRow}>
                        <span className={styles.symbol}>{symbol}</span>
                        <span className={styles.name}>{name}</span>
                    </div>
                    <div className={styles.priceRow}>
                        <span className={styles.price}>
                            ₹{currentPrice.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                        </span>
                        <span className={`${styles.change} ${isPositive ? styles.positive : styles.negative}`}>
                            {isPositive ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                            {isPositive ? '+' : ''}{changePercent.toFixed(2)}%
                        </span>
                        {lastPredicted && (
                            <span className={styles.forecast}>
                                Forecast: {forecastChange >= 0 ? '+' : ''}{forecastChange.toFixed(1)}%
                            </span>
                        )}
                    </div>
                </div>

                {/* Timeframe Selector */}
                <div className={styles.timeframeTabs}>
                    {timeframes.map((tf) => (
                        <motion.button
                            key={tf}
                            className={`${styles.timeframeBtn} ${activeTimeframe === tf ? styles.active : ''}`}
                            onClick={() => setActiveTimeframe(tf)}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                        >
                            {tf}
                        </motion.button>
                    ))}
                </div>
            </div>

            {/* Chart */}
            <div className={styles.chartWrapper}>
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                        <defs>
                            <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#2EE59D" stopOpacity={0.3} />
                                <stop offset="95%" stopColor="#2EE59D" stopOpacity={0} />
                            </linearGradient>
                            <linearGradient id="predictedGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#6FFFD2" stopOpacity={0.2} />
                                <stop offset="95%" stopColor="#6FFFD2" stopOpacity={0} />
                            </linearGradient>
                            <linearGradient id="confidenceGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#2EE59D" stopOpacity={0.1} />
                                <stop offset="95%" stopColor="#2EE59D" stopOpacity={0.02} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid 
                            strokeDasharray="3 3" 
                            stroke="rgba(255,255,255,0.05)" 
                            vertical={false}
                        />
                        <XAxis 
                            dataKey="date" 
                            axisLine={false}
                            tickLine={false}
                            tick={{ fill: '#8A8F98', fontSize: 11 }}
                            dy={10}
                        />
                        <YAxis 
                            axisLine={false}
                            tickLine={false}
                            tick={{ fill: '#8A8F98', fontSize: 11 }}
                            tickFormatter={(value) => `₹${value.toLocaleString()}`}
                            domain={['auto', 'auto']}
                            dx={-10}
                        />
                        <Tooltip content={<CustomTooltip />} />
                        
                        {/* Confidence Zone */}
                        <Area
                            type="monotone"
                            dataKey="confidenceUpper"
                            stroke="none"
                            fill="url(#confidenceGradient)"
                            fillOpacity={1}
                        />
                        <Area
                            type="monotone"
                            dataKey="confidenceLower"
                            stroke="none"
                            fill="var(--bg-tertiary)"
                            fillOpacity={1}
                        />
                        
                        {/* Historical Price */}
                        <Area
                            type="monotone"
                            dataKey="price"
                            stroke="#2EE59D"
                            strokeWidth={2}
                            fill="url(#priceGradient)"
                            dot={false}
                            activeDot={{ r: 4, fill: '#2EE59D', stroke: '#0B0F0D', strokeWidth: 2 }}
                            name="Price"
                        />
                        
                        {/* Predicted Price */}
                        <Area
                            type="monotone"
                            dataKey="predicted"
                            stroke="#6FFFD2"
                            strokeWidth={2}
                            strokeDasharray="5 5"
                            fill="url(#predictedGradient)"
                            dot={false}
                            activeDot={{ r: 4, fill: '#6FFFD2', stroke: '#0B0F0D', strokeWidth: 2 }}
                            name="Predicted"
                        />
                        
                        {/* Current Price Reference Line */}
                        <ReferenceLine 
                            y={currentPrice} 
                            stroke="#2EE59D" 
                            strokeDasharray="3 3"
                            strokeOpacity={0.5}
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>

            {/* Legend */}
            <div className={styles.legend}>
                <div className={styles.legendItem}>
                    <span className={styles.legendLine} style={{ background: '#2EE59D' }} />
                    <span>Historical</span>
                </div>
                <div className={styles.legendItem}>
                    <span className={styles.legendLineDashed} />
                    <span>AI Predicted</span>
                </div>
                <div className={styles.legendItem}>
                    <span className={styles.legendArea} />
                    <span>Confidence Zone</span>
                </div>
            </div>
        </div>
    );
};

export default ForecastChart;
