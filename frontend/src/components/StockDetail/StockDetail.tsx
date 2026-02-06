import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ChevronLeft, TrendingUp, TrendingDown, AlertTriangle, Zap, Target, BookOpen } from 'lucide-react';
import styles from './StockDetail.module.css';

interface StockAnalysisData {
    symbol: string;
    name: string;
    sector: string;
    currentPrice: number;
    change: number;
    changePercent: number;
    sentiment: string;
    sentimentScore: number;
    sentimentConfidence: number;
    sentimentReasoning: string;
    riskLevel: string;
    riskFactors: string[];
    growthPotential: string;
    debtLevel: string;
    predictedPrice: number;
    forecastConfidence: number;
    shortTermOutlook: string;
    longTermOutlook: string;
    recommendation: string;
    recentNews: any[];
    newsCount: number;
}

const StockDetail: React.FC = () => {
    const { symbol } = useParams<{ symbol: string }>();
    const navigate = useNavigate();
    const [analysis, setAnalysis] = useState<StockAnalysisData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchAnalysis = async () => {
            if (!symbol) return;
            
            try {
                setLoading(true);
                const response = await fetch(`http://localhost:8000/stocks/${symbol}/analysis`, {
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('upstox_access_token')}`
                    }
                });
                if (!response.ok) throw new Error('Failed to fetch analysis');
                const data = await response.json();
                setAnalysis(data);
            } catch (err: any) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };
        fetchAnalysis();
    }, [symbol]);

    if (loading) {
        return (
            <div className={styles.container}>
                <button className={styles.backButton} onClick={() => navigate('/')}>
                    <ChevronLeft size={20} /> Back
                </button>
                <div className={styles.loading}>Loading analysis...</div>
            </div>
        );
    }

    if (error || !analysis) {
        return (
            <div className={styles.container}>
                <button className={styles.backButton} onClick={() => navigate('/')}>
                    <ChevronLeft size={20} /> Back
                </button>
                <div className={styles.error}>
                    <AlertTriangle size={40} />
                    <p>{error || 'Failed to load analysis'}</p>
                </div>
            </div>
        );
    }

    const priceChange = analysis.change >= 0 ? 'gain' : 'loss';
    const sentimentColor = analysis.sentiment === 'bullish' ? 'bullish' : (analysis.sentiment === 'bearish' ? 'bearish' : 'neutral');
    const riskColor = analysis.riskLevel === 'low' ? 'low' : (analysis.riskLevel === 'high' ? 'high' : 'medium');
    const recommendationColor = analysis.recommendation === 'buy' ? 'buy' : (analysis.recommendation === 'sell' ? 'sell' : 'hold');

    return (
        <motion.div 
            className={styles.container}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
        >
            {/* Header */}
            <div className={styles.header}>
                <button className={styles.backButton} onClick={() => navigate('/')}>
                    <ChevronLeft size={20} /> Back to Dashboard
                </button>
                <div className={styles.headerContent}>
                    <div>
                        <h1 className={styles.title}>{analysis.name}</h1>
                        <p className={styles.symbol}>{analysis.symbol} • {analysis.sector}</p>
                    </div>
                </div>
            </div>

            {/* Main Grid */}
            <div className={styles.gridContainer}>

                {/* Price Section */}
                <motion.div 
                    className={styles.priceCard}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                >
                    <div className={styles.priceHeader}>
                        <h2>Current Price</h2>
                        <div className={`${styles.priceChange} ${styles[priceChange]}`}>
                            {priceChange === 'gain' ? <TrendingUp size={18} /> : <TrendingDown size={18} />}
                            {Math.abs(analysis.change).toFixed(2)} ({Math.abs(analysis.changePercent).toFixed(2)}%)
                        </div>
                    </div>
                    <div className={styles.price}>₹{analysis.currentPrice.toFixed(2)}</div>
                </motion.div>

                {/* Recommendation */}
                <motion.div 
                    className={`${styles.recommendationCard} ${styles[recommendationColor]}`}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.15 }}
                >
                    <div className={styles.recommendationLabel}>Recommendation</div>
                    <div className={styles.recommendationValue}>{analysis.recommendation.toUpperCase()}</div>
                    <div className={styles.confidence}>
                        Confidence: {analysis.forecastConfidence.toFixed(1)}%
                    </div>
                </motion.div>

                {/* Sentiment Analysis */}
                <motion.div 
                    className={`${styles.card} ${styles.sentimentCard}`}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                >
                    <h3 className={styles.cardTitle}>
                        <Zap size={18} /> AI Sentiment Analysis
                    </h3>
                    <div className={`${styles.sentimentBadge} ${styles[sentimentColor]}`}>
                        {analysis.sentiment.toUpperCase()}
                    </div>
                    <div className={styles.sentimentScore}>
                        Score: {(analysis.sentimentScore * 100).toFixed(1)}%
                        <span className={styles.confidence}>(Confidence: {(analysis.sentimentConfidence * 100).toFixed(1)}%)</span>
                    </div>
                    <p className={styles.reasoning}>"{analysis.sentimentReasoning}"</p>
                </motion.div>

                {/* Risk Assessment */}
                <motion.div 
                    className={`${styles.card} ${styles.riskCard}`}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.25 }}
                >
                    <h3 className={styles.cardTitle}>
                        <AlertTriangle size={18} /> Risk Assessment
                    </h3>
                    <div className={`${styles.riskBadge} ${styles[riskColor]}`}>
                        {analysis.riskLevel.toUpperCase()}
                    </div>
                    <div className={styles.riskFactors}>
                        {analysis.riskFactors.length > 0 ? (
                            <>
                                <p className={styles.factLabel}>Risk Factors:</p>
                                <ul>
                                    {analysis.riskFactors.map((factor, i) => (
                                        <li key={i}>• {factor}</li>
                                    ))}
                                </ul>
                            </>
                        ) : (
                            <p className={styles.noRisks}>No significant risk factors identified</p>
                        )}
                    </div>
                </motion.div>

                {/* Growth & Forecast */}
                <motion.div 
                    className={`${styles.card} ${styles.forecastCard}`}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                >
                    <h3 className={styles.cardTitle}>
                        <Target size={18} /> Growth & Forecast
                    </h3>
                    
                    <div className={styles.forecastRow}>
                        <div className={styles.forecastItem}>
                            <span className={styles.label}>Growth Potential</span>
                            <span className={styles.value}>{analysis.growthPotential}</span>
                        </div>
                        <div className={styles.forecastItem}>
                            <span className={styles.label}>Debt Level</span>
                            <span className={styles.value}>{analysis.debtLevel}</span>
                        </div>
                    </div>

                    <div className={styles.priceTarget}>
                        <span className={styles.label}>Price Target (12M)</span>
                        <div className={styles.targetPrice}>
                            ₹{analysis.predictedPrice.toFixed(2)}
                            <span className={styles.upside}>
                                {((analysis.predictedPrice - analysis.currentPrice) / analysis.currentPrice * 100).toFixed(1)}% upside
                            </span>
                        </div>
                    </div>

                    <div className={styles.outlookRow}>
                        <div className={styles.outlookItem}>
                            <span className={styles.label}>Short Term</span>
                            <span className={`${styles.outlook} ${styles[analysis.shortTermOutlook]}`}>
                                {analysis.shortTermOutlook.toUpperCase()}
                            </span>
                        </div>
                        <div className={styles.outlookItem}>
                            <span className={styles.label}>Long Term</span>
                            <span className={`${styles.outlook} ${styles[analysis.longTermOutlook]}`}>
                                {analysis.longTermOutlook.toUpperCase()}
                            </span>
                        </div>
                    </div>
                </motion.div>

                {/* Recent News */}
                {analysis.newsCount > 0 && (
                    <motion.div 
                        className={`${styles.card} ${styles.newsCard}`}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.35 }}
                    >
                        <h3 className={styles.cardTitle}>
                            <BookOpen size={18} /> Recent News ({analysis.newsCount})
                        </h3>
                        <div className={styles.newsList}>
                            {analysis.recentNews.map((news, i) => (
                                <div key={i} className={styles.newsItem}>
                                    <div className={styles.newsHeader}>
                                        <p className={styles.newsTitle}>{news.title}</p>
                                        <span className={`${styles.newsSentiment} ${styles[news.sentiment]}`}>
                                            {news.sentiment}
                                        </span>
                                    </div>
                                    <p className={styles.newsSource}>{news.source}</p>
                                </div>
                            ))}
                        </div>
                    </motion.div>
                )}

            </div>
        </motion.div>
    );
};

export default StockDetail;
