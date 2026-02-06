import { useEffect } from 'react';
import { motion } from 'framer-motion';
import { useAuth } from '../../contexts/AuthContext';
import styles from './Login.module.css';

const Login = () => {
    const { login, isLoading } = useAuth();

    useEffect(() => {
        // Prevent scrolling when login modal is shown
        document.body.style.overflow = 'hidden';
        return () => {
            document.body.style.overflow = 'unset';
        };
    }, []);

    return (
        <div className={styles.container}>
            {/* Background gradient */}
            <div className={styles.background}>
                <div className={styles.blob1}></div>
                <div className={styles.blob2}></div>
                <div className={styles.blob3}></div>
            </div>

            {/* Login Content */}
            <motion.div
                className={styles.content}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3 }}
            >
                <div className={styles.card}>
                    {/* Logo Section */}
                    <motion.div
                        className={styles.logoSection}
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.1, duration: 0.4 }}
                    >
                        <div className={styles.logoIcon}>📊</div>
                        <h1 className={styles.title}>Stock Intelligence</h1>
                        <p className={styles.subtitle}>AI-Powered Market Analysis</p>
                    </motion.div>

                    {/* Features Section */}
                    <motion.div
                        className={styles.features}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.2, duration: 0.4 }}
                    >
                        <div className={styles.feature}>
                            <span className={styles.featureIcon}>🚀</span>
                            <span>Real-time Market Data</span>
                        </div>
                        <div className={styles.feature}>
                            <span className={styles.featureIcon}>🧠</span>
                            <span>AI Sentiment Analysis</span>
                        </div>
                        <div className={styles.feature}>
                            <span className={styles.featureIcon}>📈</span>
                            <span>Price Forecasting</span>
                        </div>
                        <div className={styles.feature}>
                            <span className={styles.featureIcon}>💡</span>
                            <span>Strategic Insights</span>
                        </div>
                    </motion.div>

                    {/* Login Section */}
                    <motion.div
                        className={styles.loginSection}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3, duration: 0.4 }}
                    >
                        <button
                            className={styles.loginBtn}
                            onClick={login}
                            disabled={isLoading}
                        >
                            {isLoading ? (
                                <>
                                    <span className={styles.spinner}></span>
                                    Connecting...
                                </>
                            ) : (
                                <>
                                    <span className={styles.upstoxIcon}>📱</span>
                                    Login with Upstox
                                </>
                            )}
                        </button>
                        <p className={styles.disclaimer}>
                            Secure login powered by Upstox OAuth 2.0
                        </p>
                    </motion.div>

                    {/* Benefits Section */}
                    <motion.div
                        className={styles.benefits}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.4, duration: 0.4 }}
                    >
                        <p className={styles.benefitTitle}>Why Stock Intelligence?</p>
                        <ul className={styles.benefitList}>
                            <li>✓ Multi-level sentiment analysis from global news sources</li>
                            <li>✓ Deep learning models for price forecasting</li>
                            <li>✓ Real-time order book visualization</li>
                            <li>✓ Growth scoring using fundamental analysis</li>
                            <li>✓ Seamless Upstox broker integration</li>
                        </ul>
                    </motion.div>
                </div>
            </motion.div>

            {/* Footer */}
            <motion.div
                className={styles.footer}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5, duration: 0.4 }}
            >
                <p>© 2026 Multi-Modal Stock Intelligence. All rights reserved.</p>
            </motion.div>
        </div>
    );
};

export default Login;
