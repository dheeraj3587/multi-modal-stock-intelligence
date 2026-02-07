import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  TrendingUp,
  TrendingDown,
  Search,
  Bell,
  Settings,
  Activity,
} from 'lucide-react';
import { PageTransition } from '../components/shared/PageTransition';
import { Badge } from '../components/ui/Badge';
import { api } from '../lib/api';
import { useApi } from '../hooks/useApi';
import { cn, formatCurrency, formatPercent } from '../lib/utils';
import { useMemo, useState } from 'react';
import { CompanyLogo } from '../components/shared/CompanyLogo';

const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.05 } },
};
const fadeUp = {
  hidden: { opacity: 0, y: 10 },
  show: { opacity: 1, y: 0, transition: { duration: 0.25 } },
};

const TIMEFRAMES = ['1h', '8h', '1d', '1w', '1m', '6m', '1y'];

export function HomePage() {
  const navigate = useNavigate();
  const [activeTimeframe, setActiveTimeframe] = useState('1d');

  const { data: indices, loading: indicesLoading } = useApi(() => api.getIndices(), [], { refetchInterval: 30000 });
  const { data: leaderboard } = useApi(() => api.getLeaderboard(), []);
  const { data: stocks, loading: stocksLoading } = useApi(() => api.getStocksWithSentiment(), []);

  const sentimentSummary = useMemo(() => {
    if (!stocks) return null;
    const bullish = stocks.filter((s) => s.sentiment === 'bullish');
    const neutral = stocks.filter((s) => s.sentiment === 'neutral');
    const bearish = stocks.filter((s) => s.sentiment === 'bearish');
    return { bullish: bullish.length, neutral: neutral.length, bearish: bearish.length, total: stocks.length };
  }, [stocks]);

  const topMovers = useMemo(() => {
    if (!stocks) return [];
    return [...stocks].sort((a, b) => Math.abs(b.changePercent) - Math.abs(a.changePercent)).slice(0, 6);
  }, [stocks]);

  const totalMarketValue = useMemo(() => {
    if (!indices) return 0;
    return indices.reduce((s, idx) => s + idx.price, 0);
  }, [indices]);

  const avgChange = useMemo(() => {
    if (!indices || indices.length === 0) return 0;
    return indices.reduce((s, idx) => s + idx.changePercent, 0) / indices.length;
  }, [indices]);

  // Generate random bar heights for background visual
  const bars = useMemo(() => Array.from({ length: 80 }, () => Math.floor(Math.random() * 70) + 15), []);

  return (
    <PageTransition>
      <div className="flex h-full gap-0">
        {/* ═══════ LEFT PANEL (RED ACCENT) ═══════ */}
        <div className="flex-[2.2] bg-accent rounded-3xl p-8 flex flex-col relative overflow-hidden">
          {/* Market Overview & Controls */}
          <div className="flex justify-between items-end mb-6 relative z-10">
            <div>
              <p className="text-white/70 text-sm mb-1">Market Overview</p>
              <div className="flex items-center gap-3">
                <h2 className="text-4xl lg:text-5xl font-semibold text-white tracking-tight">
                  {indicesLoading ? (
                    <span className="inline-block w-48 h-12 bg-white/10 rounded-lg animate-pulse" />
                  ) : (
                    formatCurrency(totalMarketValue)
                  )}
                </h2>
                {!indicesLoading && (
                  <span className="bg-black/20 text-white px-2.5 py-1 rounded-full text-sm flex items-center gap-1">
                    {avgChange >= 0 ? (
                      <TrendingUp className="w-3 h-3" />
                    ) : (
                      <TrendingDown className="w-3 h-3" />
                    )}
                    {formatPercent(avgChange)}
                  </span>
                )}
              </div>
            </div>

            <div className="flex flex-col items-end gap-3">
              <div className="flex gap-2">
                <button
                  onClick={() => navigate('/screener')}
                  className="w-8 h-8 rounded border border-white/30 flex items-center justify-center text-white hover:bg-white/10 transition"
                >
                  <Search className="w-4 h-4" />
                </button>
                <button
                  onClick={() => navigate('/sentiment')}
                  className="w-8 h-8 rounded border border-white/30 flex items-center justify-center text-white hover:bg-white/10 transition"
                >
                  <Settings className="w-4 h-4" />
                </button>
              </div>
              <div className="flex bg-black/10 rounded p-0.5 gap-0.5">
                {TIMEFRAMES.map((tf) => (
                  <button
                    key={tf}
                    onClick={() => setActiveTimeframe(tf)}
                    className={cn(
                      'px-2.5 py-1 text-xs rounded transition-colors',
                      activeTimeframe === tf
                        ? 'text-white bg-white/20 shadow-sm'
                        : 'text-white/60 hover:text-white'
                    )}
                  >
                    {tf}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Stats Grid */}
          <motion.div
            className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-6 relative z-10"
            variants={stagger}
            initial="hidden"
            animate="show"
          >
            {indicesLoading
              ? Array.from({ length: 4 }).map((_, i) => (
                  <motion.div key={i} variants={fadeUp} className="bg-black/10 rounded-xl p-4 h-24 animate-pulse" />
                ))
              : indices?.slice(0, 4).map((idx) => {
                  const isPos = idx.changePercent >= 0;
                  return (
                    <motion.div key={idx.symbol} variants={fadeUp} className="bg-black/10 rounded-xl p-4">
                      <p className="text-xs text-white/60 mb-1.5 truncate">{idx.name}</p>
                      <p className="text-lg font-semibold text-white mb-1 font-mono">{formatCurrency(idx.price)}</p>
                      <div className={cn('flex items-center gap-1 text-[11px] font-medium', isPos ? 'text-white/90' : 'text-white/70')}>
                        {isPos ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                        {formatPercent(idx.changePercent)} Today
                      </div>
                    </motion.div>
                  );
                })}
          </motion.div>

          {/* Chart Area (Visual) */}
          <div className="flex-1 relative w-full overflow-hidden rounded-xl">
            {/* SVG Chart Lines */}
            <svg className="absolute bottom-0 left-0 w-full h-[70%] z-10" preserveAspectRatio="none" viewBox="0 0 100 100">
              <defs>
                <linearGradient id="lineGrad" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="rgba(255,255,255,0.4)" />
                  <stop offset="100%" stopColor="rgba(255,255,255,0.9)" />
                </linearGradient>
              </defs>
              {/* Secondary (yellow) line */}
              <path
                d="M0,80 Q10,80 15,60 T30,70 T45,65 T60,50 T75,65 T85,35 T100,35"
                fill="none"
                stroke="#FBBF24"
                strokeWidth="0.5"
                opacity="0.6"
              />
              {/* Primary (white) line */}
              <path
                d="M0,60 Q10,75 15,70 T30,65 T45,70 T60,45 T75,60 T90,70 T100,55"
                fill="none"
                stroke="url(#lineGrad)"
                strokeWidth="0.8"
              />
            </svg>

            {/* Background Bars */}
            <div className="absolute bottom-0 left-0 w-full h-full flex items-end justify-between gap-[2px] opacity-20" style={{ maskImage: 'linear-gradient(to bottom, transparent, black 20%)' }}>
              {bars.map((h, i) => (
                <div
                  key={i}
                  className="flex-1 rounded-t-sm"
                  style={{ height: `${h}%`, background: 'rgba(0,0,0,0.3)' }}
                />
              ))}
            </div>

            {/* Tooltip Mockup */}
            <motion.div
              className="absolute top-[15%] left-[30%] z-20"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5, duration: 0.3 }}
            >
              <div className="w-3.5 h-3.5 rounded-full border-[2px] border-white bg-white/20 backdrop-blur-sm shadow-lg mb-1.5" />
              {indices && indices.length > 0 && (
                <>
                  <div className="text-white font-bold text-base">
                    {avgChange >= 0 ? '+' : ''}{formatCurrency(indices[0]?.change || 0)}
                  </div>
                  <div className="text-white/60 text-xs">{formatPercent(avgChange)}</div>
                </>
              )}
              <div className="w-[1px] h-40 bg-white/20 absolute top-[14px] left-[6px]" />
            </motion.div>
          </div>
        </div>

        {/* ═══════ RIGHT PANEL (DARK) ═══════ */}
        <div className="flex-1 bg-surface-0 p-6 flex flex-col overflow-y-auto no-scrollbar min-w-[320px]">
          {/* Top Bar */}
          <div className="flex items-center gap-2.5 mb-6">
            <div className="bg-surface-1 rounded-full px-3 py-2 flex items-center gap-2 flex-1 border border-border-subtle">
              <div className="w-6 h-6 rounded-full bg-accent overflow-hidden flex items-center justify-center text-white text-[10px] font-bold">
                S
              </div>
              <span className="text-text-primary text-xs font-mono truncate">StockSense Intelligence</span>
            </div>
            <button
              onClick={() => navigate('/screener')}
              className="w-9 h-9 rounded-full bg-surface-1 flex items-center justify-center text-text-tertiary hover:text-text-primary border border-border-subtle transition"
            >
              <Search className="w-4 h-4" />
            </button>
            <button className="w-9 h-9 rounded-full bg-surface-1 flex items-center justify-center text-text-tertiary hover:text-text-primary border border-border-subtle relative transition">
              <Bell className="w-4 h-4" />
              <span className="absolute top-2 right-2 w-2 h-2 bg-accent rounded-full" />
            </button>
          </div>

          {/* Sentiment Risk Score */}
          <div className="mb-6">
            <div className="flex justify-between items-end mb-2">
              <h3 className="text-lg font-bold text-text-primary leading-tight">
                Market<br />Sentiment
              </h3>
              <div className="text-right">
                <span className="text-[10px] text-text-tertiary block">Updated:</span>
                <span className="text-[10px] text-text-secondary">Just Now</span>
              </div>
            </div>

            {sentimentSummary && (
              <div className="mt-3 space-y-2">
                {[
                  { label: 'Bullish', count: sentimentSummary.bullish, color: 'bg-positive' },
                  { label: 'Neutral', count: sentimentSummary.neutral, color: 'bg-warning' },
                  { label: 'Bearish', count: sentimentSummary.bearish, color: 'bg-negative' },
                ].map(({ label, count, color }) => (
                  <div key={label}>
                    <div className="flex items-center justify-between text-[11px] mb-0.5">
                      <span className="text-text-secondary">{label}</span>
                      <span className="text-text-tertiary font-mono">{count}/{sentimentSummary.total}</span>
                    </div>
                    <div className="h-1.5 rounded-full bg-surface-2 overflow-hidden">
                      <motion.div
                        className={cn('h-full rounded-full', color)}
                        initial={{ width: 0 }}
                        animate={{ width: `${(count / sentimentSummary.total) * 100}%` }}
                        transition={{ duration: 0.6, ease: 'easeOut' }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Top Movers Header */}
          <div className="flex justify-between items-center mb-3">
            <h4 className="text-text-primary font-medium text-sm">Top Movers</h4>
            <button onClick={() => navigate('/screener')} className="text-xs text-text-tertiary hover:text-text-primary transition">View All</button>
          </div>

          {/* Top Movers Grid */}
          <motion.div
            className="grid grid-cols-2 gap-2.5 pb-4"
            variants={stagger}
            initial="hidden"
            animate="show"
          >
            {stocksLoading
              ? Array.from({ length: 4 }).map((_, i) => (
                  <motion.div key={i} variants={fadeUp} className="bg-surface-1 h-28 rounded-2xl animate-pulse border border-border-subtle" />
                ))
              : topMovers.slice(0, 2).map((stock) => {
                  const isPos = stock.changePercent >= 0;
                  return (
                    <motion.div
                      key={stock.symbol}
                      variants={fadeUp}
                      onClick={() => navigate(`/stock/${stock.symbol}`)}
                      className="bg-surface-1 p-3 rounded-2xl border border-border-subtle hover:bg-surface-2 transition cursor-pointer"
                    >
                      <div className="flex justify-between items-start mb-2">
                        <CompanyLogo symbol={stock.symbol} className="w-8 h-8" />
                        <div className="text-right">
                          <div className="text-[10px] text-text-tertiary">{stock.sentiment}</div>
                        </div>
                      </div>
                      <div className="text-xs text-text-tertiary">{stock.symbol}</div>
                      <div className={cn('font-bold text-sm', isPos ? 'text-positive' : 'text-negative')}>
                        {formatPercent(stock.changePercent)}
                      </div>
                      <div className="text-[10px] text-text-tertiary mt-1 flex items-center gap-1">
                        {formatCurrency(stock.price)}
                      </div>
                    </motion.div>
                  );
                })}

            {/* Wide card for 3rd mover */}
            {topMovers[2] && (
              <motion.div
                variants={fadeUp}
                onClick={() => navigate(`/stock/${topMovers[2].symbol}`)}
                className="col-span-2 bg-surface-1 p-3 rounded-2xl border border-border-subtle flex justify-between items-center cursor-pointer hover:bg-surface-2 transition overflow-hidden relative"
              >
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <CompanyLogo symbol={topMovers[2].symbol} className="w-6 h-6" />
                    <span className="text-[10px] text-text-tertiary">{topMovers[2].sentiment}</span>
                  </div>
                  <div className="text-xs text-text-tertiary">{topMovers[2].symbol}</div>
                  <div className={cn('font-bold text-lg', topMovers[2].changePercent >= 0 ? 'text-positive' : 'text-negative')}>
                    {formatPercent(topMovers[2].changePercent)}
                  </div>
                  <div className="text-[10px] text-text-tertiary mt-0.5">{formatCurrency(topMovers[2].price)}</div>
                </div>
                {/* Mini Sparkline */}
                <svg className="w-24 h-12 text-text-tertiary/30" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 50 20">
                  <path d="M0,15 L10,10 L20,15 L30,5 L40,10 L50,0" />
                </svg>
              </motion.div>
            )}

            {/* Growth leaderboard preview */}
            {topMovers[3] && (
              <motion.div
                variants={fadeUp}
                onClick={() => navigate(`/stock/${topMovers[3].symbol}`)}
                className="col-span-2 bg-surface-1 p-3 rounded-2xl border border-border-subtle flex justify-between items-center cursor-pointer hover:bg-surface-2 transition"
              >
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <CompanyLogo symbol={topMovers[3].symbol} className="w-6 h-6" />
                    <span className="text-[10px] text-text-tertiary">Growth · {topMovers[3].sector}</span>
                  </div>
                  <div className="text-xs text-text-tertiary">{topMovers[3].symbol}</div>
                  <div className={cn('font-bold text-lg', topMovers[3].changePercent >= 0 ? 'text-positive' : 'text-negative')}>
                    {formatPercent(topMovers[3].changePercent)}
                  </div>
                  <div className="text-[10px] text-text-tertiary mt-0.5">{formatCurrency(topMovers[3].price)}</div>
                </div>
              </motion.div>
            )}

            {/* Featured Accent Card (ETH-inspired — highlight top growth stock) */}
            {leaderboard && leaderboard[0] && (
              <motion.div
                variants={fadeUp}
                onClick={() => navigate(`/stock/${leaderboard[0].symbol}`)}
                className="col-span-2 bg-accent p-4 rounded-2xl shadow-lg relative overflow-hidden cursor-pointer"
              >
                <div className="flex justify-between items-start z-10 relative">
                  <div className="flex items-center gap-2">
                    <div className="w-8 h-8 bg-white rounded-full flex items-center justify-center">
                      <TrendingUp className="w-4 h-4 text-accent" />
                    </div>
                    <div>
                      <div className="text-xs text-white/80">Top Growth</div>
                      <div className="text-[10px] text-white/60">{leaderboard[0].sector}</div>
                    </div>
                  </div>
                  <Badge variant="positive" size="sm" className="bg-white/20 text-white border-0">
                    #{leaderboard[0].rank}
                  </Badge>
                </div>
                <div className="mt-3 z-10 relative">
                  <div className="text-xs text-white/60">{leaderboard[0].symbol}</div>
                  <div className="text-2xl font-bold text-white">{leaderboard[0].name}</div>
                  <div className="text-sm text-white/80 font-mono mt-0.5">
                    Score: {leaderboard[0].growthScore.toFixed(0)}
                  </div>
                  <div className="text-[10px] text-white/80 mt-1 flex items-center gap-1">
                    <Activity className="w-3 h-3" />
                    {leaderboard[0].sentiment} sentiment
                  </div>
                </div>

                {/* Background decorative bars */}
                <div className="absolute bottom-0 right-0 w-full h-1/2 opacity-20 flex items-end gap-[1px]">
                  {Array.from({ length: 40 }, (_, i) => (
                    <div
                      key={i}
                      className="w-1 bg-black rounded-t-sm"
                      style={{ height: `${Math.floor(Math.random() * 80) + 10}%` }}
                    />
                  ))}
                </div>
                <svg className="absolute bottom-3 right-0 w-full h-8 text-yellow-400 opacity-60" fill="none" stroke="currentColor" strokeWidth="1" viewBox="0 0 100 20">
                  <path d="M0,10 Q20,20 40,5 T60,15 T80,5 T100,10" />
                </svg>
              </motion.div>
            )}

            {/* Quick action cards */}
            {topMovers.slice(4, 6).map((stock) => (
              <motion.div
                key={stock.symbol}
                variants={fadeUp}
                onClick={() => navigate(`/stock/${stock.symbol}`)}
                className="bg-surface-1 p-3 rounded-2xl border border-border-subtle hover:bg-surface-2 transition cursor-pointer relative overflow-hidden"
              >
                <div className="flex justify-between items-start mb-2 relative z-10">
                  <CompanyLogo symbol={stock.symbol} className="w-8 h-8" />
                  <div className="text-right">
                    <div className="text-[10px] text-text-tertiary">{stock.sentiment}</div>
                    <div className="text-[9px] text-text-tertiary">{stock.sector}</div>
                  </div>
                </div>
                <div className="text-xs text-text-tertiary">{stock.symbol}</div>
                <div className={cn('font-bold text-sm', stock.changePercent >= 0 ? 'text-positive' : 'text-negative')}>
                  {formatPercent(stock.changePercent)}
                </div>
                <div className="text-[10px] text-text-tertiary flex items-center gap-1 mt-1">
                  {formatCurrency(stock.price)}
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>
    </PageTransition>
  );
}
