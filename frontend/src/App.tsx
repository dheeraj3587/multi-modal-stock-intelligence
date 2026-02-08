import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import { AppLayout } from './components/layout/AppLayout';
import { LoginPage } from './pages/LoginPage';
import { AuthCallbackPage } from './pages/AuthCallbackPage';
import { HomePage } from './pages/HomePage';
import { DashboardPage } from './pages/DashboardPage';
import { ScreenerPage } from './pages/ScreenerPage';
import { StockDetailPage } from './pages/StockDetailPage';
import { ScorecardPage } from './pages/ScorecardPage';
import { AllScorecardsPage } from './pages/AllScorecardsPage';
import { ForecastsPage } from './pages/ForecastsPage';
import { SentimentPage } from './pages/SentimentPage';
import { GrowthLeaderboardPage } from './pages/GrowthLeaderboardPage';
import { PortfolioPage } from './pages/PortfolioPage';
import { TradingPage } from './pages/TradingPage';
import { AIChatPage } from './pages/AIChatPage';

export default function App() {
  return (
    <BrowserRouter>
      <AnimatePresence mode="wait">
        <Routes>
          {/* Login — standalone, no sidebar */}
          <Route path="/login" element={<LoginPage />} />
          <Route path="/auth/callback" element={<AuthCallbackPage />} />

          {/* Main app — with sidebar layout */}
          <Route element={<AppLayout />}>
            <Route path="/" element={<HomePage />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/screener" element={<ScreenerPage />} />
            <Route path="/stock/:symbol" element={<StockDetailPage />} />
            <Route path="/scorecard/:symbol" element={<ScorecardPage />} />
            <Route path="/scorecards" element={<AllScorecardsPage />} />
            <Route path="/forecasts" element={<ForecastsPage />} />
            <Route path="/sentiment" element={<SentimentPage />} />
            <Route path="/growth" element={<GrowthLeaderboardPage />} />
            <Route path="/portfolio" element={<PortfolioPage />} />
            <Route path="/trading" element={<TradingPage />} />
            <Route path="/chat" element={<AIChatPage />} />
          </Route>
        </Routes>
      </AnimatePresence>
    </BrowserRouter>
  );
}
