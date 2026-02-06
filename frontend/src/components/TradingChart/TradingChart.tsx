import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi, CandlestickData, Time } from 'lightweight-charts';
import { motion } from 'framer-motion';
import styles from './TradingChart.module.css';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface CandleData {
    time: Time;
    open: number;
    high: number;
    low: number;
    close: number;
    volume?: number;
}

interface TradingChartProps {
    symbol: string;
    data: CandleData[];
    onTimeframeChange?: (timeframe: string) => void;
    isLoading?: boolean;
}

const timeframes = ['1m', '5m', '15m', '1H', '4H', '1D'];

const TradingChart: React.FC<TradingChartProps> = ({
    symbol,
    data,
    onTimeframeChange,
    isLoading = false
}) => {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
    const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);
    const [activeTimeframe, setActiveTimeframe] = useState('15m');
    const [currentPrice, setCurrentPrice] = useState<number>(0);
    const [priceChange, setPriceChange] = useState<number>(0);

    useEffect(() => {
        if (!chartContainerRef.current) return;

        // Create chart
        const chart = createChart(chartContainerRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: '#1e222d' },
                textColor: '#848e9c',
            },
            grid: {
                vertLines: { color: 'rgba(255, 255, 255, 0.05)' },
                horzLines: { color: 'rgba(255, 255, 255, 0.05)' },
            },
            crosshair: {
                mode: 1,
                vertLine: {
                    color: '#00ff88',
                    width: 1,
                    style: 2,
                    labelBackgroundColor: '#00ff88',
                },
                horzLine: {
                    color: '#00ff88',
                    width: 1,
                    style: 2,
                    labelBackgroundColor: '#00ff88',
                },
            },
            rightPriceScale: {
                borderColor: '#2b3139',
                scaleMargins: {
                    top: 0.1,
                    bottom: 0.2,
                },
            },
            timeScale: {
                borderColor: '#2b3139',
                timeVisible: true,
                secondsVisible: false,
            },
        });

        chartRef.current = chart;

        // Add candlestick series
        const candleSeries = chart.addCandlestickSeries({
            upColor: '#00ff88',
            downColor: '#ff3b5c',
            borderUpColor: '#00ff88',
            borderDownColor: '#ff3b5c',
            wickUpColor: '#00ff88',
            wickDownColor: '#ff3b5c',
        });
        candleSeriesRef.current = candleSeries;

        // Add volume series
        const volumeSeries = chart.addHistogramSeries({
            color: '#26a69a',
            priceFormat: {
                type: 'volume',
            },
            priceScaleId: '',
        });
        volumeSeriesRef.current = volumeSeries;

        // Configure volume scale
        chart.priceScale('').applyOptions({
            scaleMargins: {
                top: 0.85,
                bottom: 0,
            },
        });

        // Handle resize
        const handleResize = () => {
            if (chartContainerRef.current) {
                chart.applyOptions({
                    width: chartContainerRef.current.clientWidth,
                    height: chartContainerRef.current.clientHeight,
                });
            }
        };

        window.addEventListener('resize', handleResize);
        handleResize();

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
        };
    }, []);

    // Update chart data
    useEffect(() => {
        if (!candleSeriesRef.current || !volumeSeriesRef.current || data.length === 0) return;

        const candleData: CandlestickData[] = data.map(d => ({
            time: d.time,
            open: d.open,
            high: d.high,
            low: d.low,
            close: d.close,
        }));

        const volumeData = data.map(d => ({
            time: d.time,
            value: d.volume || 0,
            color: d.close >= d.open ? 'rgba(0, 255, 136, 0.3)' : 'rgba(255, 59, 92, 0.3)',
        }));

        candleSeriesRef.current.setData(candleData);
        volumeSeriesRef.current.setData(volumeData);

        // Calculate current price and change
        if (data.length > 0) {
            const lastCandle = data[data.length - 1];
            const firstCandle = data[0];
            setCurrentPrice(lastCandle.close);
            const change = ((lastCandle.close - firstCandle.open) / firstCandle.open) * 100;
            setPriceChange(change);
        }

        // Fit content
        chartRef.current?.timeScale().fitContent();
    }, [data]);

    const handleTimeframeClick = (tf: string) => {
        setActiveTimeframe(tf);
        onTimeframeChange?.(tf);
    };

    const isPositive = priceChange >= 0;

    return (
        <div className={styles.container}>
            <div className={styles.header}>
                <div className={styles.symbolInfo}>
                    <span className={styles.symbol}>{symbol}</span>
                    <span className={`${styles.price} ${isPositive ? styles.priceUp : styles.priceDown}`}>
                        ₹{currentPrice.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </span>
                    <motion.span
                        className={`${styles.change} ${isPositive ? styles.changeUp : styles.changeDown}`}
                        initial={{ scale: 1 }}
                        animate={{ scale: [1, 1.05, 1] }}
                        transition={{ duration: 0.3 }}
                    >
                        {isPositive ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                        {isPositive ? '+' : ''}{priceChange.toFixed(2)}%
                    </motion.span>
                </div>

                <div className={styles.controls}>
                    <div className={styles.timeframeTabs}>
                        {timeframes.map(tf => (
                            <button
                                key={tf}
                                className={`${styles.timeframeBtn} ${activeTimeframe === tf ? styles.active : ''}`}
                                onClick={() => handleTimeframeClick(tf)}
                            >
                                {tf}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            <div className={styles.chartContainer}>
                {isLoading && (
                    <div className={styles.loading}>
                        <div className={styles.spinner} />
                        <span>Loading chart data...</span>
                    </div>
                )}
                <div ref={chartContainerRef} className={styles.chartWrapper} />
            </div>

            <div className={styles.statsBar}>
                <div className={styles.statItem}>
                    <span className={styles.statLabel}>24h High</span>
                    <span className={`${styles.statValue} ${styles.statValueGreen}`}>
                        ₹{(currentPrice * 1.02).toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                    </span>
                </div>
                <div className={styles.statItem}>
                    <span className={styles.statLabel}>24h Low</span>
                    <span className={`${styles.statValue} ${styles.statValueRed}`}>
                        ₹{(currentPrice * 0.98).toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                    </span>
                </div>
                <div className={styles.statItem}>
                    <span className={styles.statLabel}>24h Volume</span>
                    <span className={styles.statValue}>₹2.4Cr</span>
                </div>
                <div className={styles.statItem}>
                    <span className={styles.statLabel}>Market Cap</span>
                    <span className={styles.statValue}>₹15.2L Cr</span>
                </div>
            </div>
        </div>
    );
};

export default TradingChart;
