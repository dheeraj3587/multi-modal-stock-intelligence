import { useNavigate } from 'react-router-dom';
import { useMemo, useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  TrendingUp,
  TrendingDown,
  ArrowUpRight,
  ArrowDownRight,
  Download,
  ExternalLink,
  Newspaper,
} from 'lucide-react';
import { PageTransition } from '../components/shared/PageTransition';
import { api, type NewsItem } from '../lib/api';
import { useApi } from '../hooks/useApi';
import { cn, formatCurrency, formatPercent } from '../lib/utils';
import { CompanyLogo } from '../components/shared/CompanyLogo';

/* ── Animations ── */
const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.05 } },
};
const fadeUp = {
  hidden: { opacity: 0, y: 10 },
  show: { opacity: 1, y: 0, transition: { duration: 0.25 } },
};

const TIMEFRAMES = ['1h', '1d', '1w', '1m', '1y'] as const;

/* ── Mini SVG sparkline helper ── */
function Sparkline({ positive }: { positive: boolean }) {
  const path = positive
    ? 'M0,30 C8,28 16,20 24,22 C32,24 40,12 48,14 C56,16 64,8 72,6 C80,4 88,8 96,2'
    : 'M0,6 C8,10 16,8 24,14 C32,20 40,16 48,22 C56,26 64,24 72,28 C80,26 88,30 96,32';
  return (
    <svg viewBox="0 0 96 36" fill="none" className="w-full h-full">
      <path d={path} stroke={positive ? '#22c55e' : '#ef4444'} strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

/* ── Semi-circle arc for risk score (RIGHT side) ── */
function RiskArc({ percent, color }: { percent: number; color: string }) {
  const r = 40;
  const circumference = Math.PI * r;
  const offset = circumference - (percent / 100) * circumference;
  return (
    <div className="relative w-20 h-12 shrink-0">
      <svg viewBox="0 0 100 55" className="w-full h-full">
        <path d="M10,50 A40,40 0 0,1 90,50" fill="none" stroke="var(--surface-2)" strokeWidth="7" strokeLinecap="round" />
        <path
          d="M10,50 A40,40 0 0,1 90,50"
          fill="none"
          stroke={color}
          strokeWidth="7"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-700"
        />
      </svg>
    </div>
  );
}

export function DashboardPage() {
  const navigate = useNavigate();
  const [activeTimeframe, setActiveTimeframe] = useState<string>('1d');

  /* ── Real data from backend ── */
  const { data: indices, loading: indicesLoading } = useApi(() => api.getIndices(), [], { refetchInterval: 30_000 });
  const { data: stocks, loading: stocksLoading } = useApi(() => api.getStocksWithSentiment(), [], { refetchInterval: 30_000 });
  const { data: leaderboard } = useApi(() => api.getLeaderboard(), []);

  /* ── News — fetch for top mover dynamically ── */
  const [news, setNews] = useState<NewsItem[]>([]);
  const topNewsSymbol = useMemo(() => {
    if (!stocks || stocks.length === 0) return null;
    return [...stocks].sort((a, b) => Math.abs(b.changePercent) - Math.abs(a.changePercent))[0]?.symbol ?? null;
  }, [stocks]);

  useEffect(() => {
    if (!topNewsSymbol) return;
    let cancelled = false;
    (async () => {
      try {
        const res = await api.getNews(topNewsSymbol);
        if (!cancelled) setNews(res.news?.slice(0, 3) ?? []);
      } catch {
        /* silent */
      }
    })();
    return () => { cancelled = true; };
  }, [topNewsSymbol]);

  /* ── Derived data ── */
  const totalMarketValue = useMemo(() => {
    if (!indices) return 0;
    return indices.reduce((s, idx) => s + idx.price, 0);
  }, [indices]);

  const avgChange = useMemo(() => {
    if (!indices || indices.length === 0) return 0;
    return indices.reduce((s, idx) => s + idx.changePercent, 0) / indices.length;
  }, [indices]);

  const sentimentSummary = useMemo(() => {
    if (!stocks) return null;
    const bullish = stocks.filter((s) => s.sentiment === 'bullish').length;
    const bearish = stocks.filter((s) => s.sentiment === 'bearish').length;
    return { bullish, bearish, total: stocks.length };
  }, [stocks]);

  const topMovers = useMemo(() => {
    if (!stocks) return [];
    return [...stocks].sort((a, b) => Math.abs(b.changePercent) - Math.abs(a.changePercent)).slice(0, 4);
  }, [stocks]);

  /* ── Stats row for hero card bottom ── */
  const statsRow = useMemo(() => {
    if (!indices || indices.length === 0) return [];
    const nifty = indices.find((i) => i.symbol.includes('NIFTY'));
    const sensex = indices.find((i) => i.symbol.includes('SENSEX'));
    return [
      { label: 'Realized P&L', value: formatCurrency(nifty?.change ?? 0), pct: formatPercent(nifty?.changePercent ?? 0), positive: (nifty?.changePercent ?? 0) >= 0 },
      { label: 'Unrealized P&L', value: formatCurrency(sensex?.change ?? 0), pct: formatPercent(sensex?.changePercent ?? 0), positive: (sensex?.changePercent ?? 0) >= 0 },
      { label: 'Net Change', value: formatCurrency((nifty?.change ?? 0) + (sensex?.change ?? 0)), pct: formatPercent(avgChange), positive: avgChange >= 0 },
      { label: 'Projected', value: formatCurrency(totalMarketValue * 1.02), pct: '+2.00%', positive: true },
    ];
  }, [indices, avgChange, totalMarketValue]);

  /* background bars for chart area */
  const bars = useMemo(() => Array.from({ length: 40 }, () => Math.floor(Math.random() * 70) + 15), []);

  /* ── Risk percentages ── */
  const highRiskPct = useMemo(() => {
    if (!sentimentSummary || sentimentSummary.total === 0) return 34;
    return Math.round((sentimentSummary.bearish / sentimentSummary.total) * 100);
  }, [sentimentSummary]);
  const lowRiskPct = useMemo(() => {
    if (!sentimentSummary || sentimentSummary.total === 0) return 57;
    return Math.round((sentimentSummary.bullish / sentimentSummary.total) * 100);
  }, [sentimentSummary]);

  /* ── Recent activities — merge live movers + leaderboard growth signals ── */
  const recentActivities = useMemo(() => {
    const items: { symbol: string; action: string; price: string; change: string; positive: boolean; time: string }[] = [];

    // Top growth signals from leaderboard
    if (leaderboard && leaderboard.length > 0) {
      leaderboard.slice(0, 2).forEach((lb) => {
        items.push({
          symbol: lb.symbol,
          action: lb.sentiment === 'bullish' ? 'Growth Buy' : 'Watch',
          price: `Score ${lb.growthScore.toFixed(0)}`,
          change: `#${lb.rank}`,
          positive: lb.sentiment === 'bullish',
          time: lb.sector,
        });
      });
    }

    // Live stock movers
    if (stocks) {
      [...stocks]
        .sort((a, b) => new Date(b.lastUpdated).getTime() - new Date(a.lastUpdated).getTime())
        .slice(0, 4 - items.length)
        .forEach((s) => {
          items.push({
            symbol: s.symbol,
            action: s.changePercent >= 0 ? 'Buy Signal' : 'Sell Signal',
            price: formatCurrency(s.price),
            change: formatPercent(s.changePercent),
            positive: s.changePercent >= 0,
            time: new Date(s.lastUpdated).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' }),
          });
        });
    }

    return items.slice(0, 4);
  }, [stocks, leaderboard]);

  return (
    <PageTransition>


      {/* ══════════════════════════════════════════════════
          MAIN GRID — 12 cols
      ══════════════════════════════════════════════════ */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

        {/* ═══════════════════════════════════════
            LEFT COLUMN — 8 cols
        ═══════════════════════════════════════ */}
        <div className="lg:col-span-8 flex flex-col gap-6">

          {/* ─── RED HERO CARD ─── */}
          <div className="bg-accent rounded-3xl p-6 md:p-8 text-white relative overflow-hidden shadow-2xl flex flex-col min-h-[520px]">
            {/* Top row: balance + controls */}
            <div className="flex flex-wrap items-start justify-between gap-4 mb-2 relative z-20">
              <div>
                <p className="text-white/60 text-sm font-medium mb-1">Total Balance</p>
                {indicesLoading ? (
                  <div className="w-48 h-12 bg-white/10 rounded-lg animate-pulse" />
                ) : (
                  <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
                    {formatCurrency(totalMarketValue)}
                  </h1>
                )}
                {!indicesLoading && (
                  <div className="flex items-center gap-2 mt-2">
                    <span className={cn(
                      'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold',
                      avgChange >= 0 ? 'bg-white/15 text-white' : 'bg-red-900/40 text-red-200',
                    )}>
                      {avgChange >= 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                      {formatPercent(avgChange)} vs last session
                    </span>
                  </div>
                )}
              </div>
              {/* Timeframe selector */}
              <div className="flex bg-black/15 rounded-xl p-1">
                {TIMEFRAMES.map((tf) => (
                  <button
                    key={tf}
                    onClick={() => setActiveTimeframe(tf)}
                    className={cn(
                      'px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors',
                      activeTimeframe === tf
                        ? 'bg-white text-accent shadow-sm'
                        : 'text-white/50 hover:text-white/80',
                    )}
                  >
                    {tf}
                  </button>
                ))}
              </div>
            </div>

            {/* ── Chart Area ── */}
            <div className="relative flex-grow min-h-[260px] w-full my-4">
              {/* Tooltip bubble */}
              {indices && indices.length > 0 && (
                <motion.div
                  className="absolute left-[55%] top-[10%] flex flex-col items-center z-30 pointer-events-none"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.5, duration: 0.3 }}
                >
                  <div className="bg-white text-accent px-3 py-1.5 rounded-lg text-sm font-bold shadow-lg">
                    {avgChange >= 0 ? '+' : ''}{formatCurrency(indices[0]?.change || 0)}
                  </div>
                  <div className="w-0 h-0 border-l-[6px] border-r-[6px] border-t-[6px] border-transparent border-t-white" />
                  <div className="w-3 h-3 bg-yellow-400 rounded-full ring-4 ring-yellow-400/30 mt-1" />
                  <div className="w-px h-32 bg-gradient-to-b from-yellow-400/80 to-transparent" />
                </motion.div>
              )}

              {/* SVG lines */}
              <svg className="absolute inset-0 w-full h-full z-20 overflow-visible" preserveAspectRatio="none">
                {/* Yellow trend line */}
                <path
                  d="M0,200 C40,195 80,180 120,185 C160,190 200,160 240,150 C280,140 320,155 360,140 C400,125 440,135 480,120 C520,105 560,115 600,100 C640,85 680,100 720,90 C760,80 800,70 840,60 C880,50 920,55 960,40 L1200,30"
                  fill="none" stroke="#FBBF24" strokeWidth="2.5" vectorEffect="non-scaling-stroke"
                />
                {/* White secondary line */}
                <path
                  d="M0,160 C40,170 80,150 120,160 C160,170 200,155 240,165 C280,175 320,160 360,170 C400,165 440,175 480,170 C520,160 560,170 600,180 C640,175 680,165 720,175 C760,180 800,170 840,165 C880,170 920,175 960,180 L1200,185"
                  fill="none" stroke="white" strokeWidth="2" opacity="0.5" vectorEffect="non-scaling-stroke"
                />
              </svg>

              {/* Bars */}
              <div className="absolute inset-0 flex items-end justify-between gap-[3px] z-10 opacity-25 px-1">
                {bars.map((h, i) => (
                  <div
                    key={i}
                    className="flex-1 bg-black/50 rounded-t-sm hover:bg-black/70 transition-colors"
                    style={{ height: `${h}%` }}
                  />
                ))}
              </div>
            </div>

            {/* ── Bottom Stats Row ── */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 relative z-20 mt-auto pt-4 border-t border-white/10">
              {statsRow.map((stat) => (
                <div key={stat.label}>
                  <p className="text-white/50 text-xs font-medium mb-1">{stat.label}</p>
                  <p className="text-lg font-bold">{stat.value}</p>
                  <p className={cn('text-xs font-medium mt-0.5', stat.positive ? 'text-white/80' : 'text-red-200')}>
                    {stat.pct} today
                  </p>
                </div>
              ))}
            </div>

            {/* Decorative blurs */}
            <div className="absolute -top-20 -right-20 w-80 h-80 bg-red-400 rounded-full blur-3xl opacity-20 pointer-events-none" />
            <div className="absolute -bottom-20 -left-20 w-80 h-80 bg-orange-500 rounded-full blur-3xl opacity-15 pointer-events-none" />
          </div>

          {/* ─── BELOW HERO: Latest News + Top Movers ─── */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

            {/* Latest News */}
            <div className="bg-surface-1 rounded-2xl p-5 border border-border-subtle">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-base font-bold text-text-primary flex items-center gap-2">
                  <Newspaper className="w-4 h-4 text-accent" />
                  Latest News
                </h3>
                <button onClick={() => navigate('/sentiment')} className="text-xs text-text-tertiary hover:text-accent transition font-medium">
                  View All
                </button>
              </div>
              <div className="space-y-4">
                {news.length === 0
                  ? Array.from({ length: 3 }).map((_, i) => (
                      <div key={i} className="flex gap-3 animate-pulse">
                        <div className="w-16 h-16 rounded-xl bg-surface-2 shrink-0" />
                        <div className="flex-1 space-y-2">
                          <div className="h-3 bg-surface-2 rounded w-full" />
                          <div className="h-3 bg-surface-2 rounded w-3/4" />
                          <div className="h-2 bg-surface-2 rounded w-1/2" />
                        </div>
                      </div>
                    ))
                  : news.map((item, i) => (
                      <a
                        key={i}
                        href={item.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex gap-3 group cursor-pointer"
                      >
                        {/* News thumbnail placeholder */}
                        <div className="w-16 h-16 rounded-xl bg-surface-2 shrink-0 flex items-center justify-center overflow-hidden">
                          <span className="material-symbols-outlined text-text-tertiary text-2xl">article</span>
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-semibold text-text-primary group-hover:text-accent transition line-clamp-2 leading-tight">
                            {item.title}
                          </p>
                          <div className="flex items-center gap-2 mt-1.5">
                            <span className="text-[11px] text-text-tertiary">{item.source}</span>
                            <span className="text-[11px] text-text-tertiary">·</span>
                            <span className={cn(
                              'text-[11px] font-semibold',
                              item.sentiment_score > 0.2 ? 'text-positive' : item.sentiment_score < -0.2 ? 'text-negative' : 'text-warning',
                            )}>
                              {item.sentiment_score > 0.2 ? 'Bullish' : item.sentiment_score < -0.2 ? 'Bearish' : 'Neutral'}
                            </span>
                          </div>
                        </div>
                        <ExternalLink className="w-3.5 h-3.5 text-text-tertiary opacity-0 group-hover:opacity-100 transition shrink-0 mt-1" />
                      </a>
                    ))}
              </div>
            </div>

            {/* Top Movers */}
            <div className="bg-surface-1 rounded-2xl p-5 border border-border-subtle">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-base font-bold text-text-primary">Top Movers</h3>
                <button onClick={() => navigate('/screener')} className="text-xs text-text-tertiary hover:text-accent transition font-medium">
                  See All
                </button>
              </div>
              <motion.div
                className="grid grid-cols-2 gap-3"
                variants={stagger}
                initial="hidden"
                animate="show"
              >
                {stocksLoading
                  ? Array.from({ length: 4 }).map((_, i) => (
                      <motion.div key={i} variants={fadeUp} className="h-[100px] rounded-xl bg-surface-2 animate-pulse" />
                    ))
                  : topMovers.map((stock) => {
                      const isPos = stock.changePercent >= 0;
                      return (
                        <motion.div
                          key={stock.symbol}
                          variants={fadeUp}
                          onClick={() => navigate(`/stock/${stock.symbol}`)}
                          className="bg-surface-0 rounded-xl p-3 border border-border-subtle hover:shadow-elevated transition cursor-pointer group"
                        >
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <CompanyLogo symbol={stock.symbol} className="w-7 h-7" />
                              <div>
                                <p className="text-xs font-bold text-text-primary leading-none">{stock.symbol}</p>
                                <p className="text-[10px] text-text-tertiary mt-0.5">NSE</p>
                              </div>
                            </div>
                          </div>
                          <div className="h-8 mb-2">
                            <Sparkline positive={isPos} />
                          </div>
                          <div className="flex items-end justify-between">
                            <span className="text-xs font-bold text-text-primary">{formatCurrency(stock.price)}</span>
                            <span className={cn(
                              'text-[11px] font-bold flex items-center gap-0.5',
                              isPos ? 'text-positive' : 'text-negative',
                            )}>
                              {isPos ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
                              {formatPercent(stock.changePercent)}
                            </span>
                          </div>
                        </motion.div>
                      );
                    })}
              </motion.div>
            </div>
          </div>
        </div>

        {/* ═══════════════════════════════════════
            RIGHT COLUMN — 4 cols
        ═══════════════════════════════════════ */}
        <div className="lg:col-span-4 flex flex-col gap-6">

          {/* ─── PORTFOLIO RISK SCORE ─── */}
          <div className="bg-surface-1 rounded-3xl p-6 border border-border-subtle relative overflow-hidden">
            {/* Decorative bottom-right glow */}
            <div className="absolute -bottom-16 -right-16 w-48 h-48 bg-accent/10 rounded-full blur-3xl pointer-events-none" />

            {/* Header row */}
            <div className="flex items-start justify-between mb-1 relative z-10">
              <h3 className="text-2xl font-bold text-text-primary leading-tight">
                Portfolio<br />Risk Score
              </h3>
              <button className="w-10 h-10 rounded-xl bg-surface-2 flex items-center justify-center text-text-tertiary hover:text-text-primary transition">
                <span className="material-symbols-outlined text-[20px]">settings</span>
              </button>
            </div>
            <p className="text-sm text-text-tertiary mb-6 relative z-10">Updated: Just Now</p>

            {/* Risk progress bar — red fill + dashed middle + gray end */}
            <div className="relative h-3 w-full bg-surface-2 rounded-full overflow-hidden mb-2 flex relative z-10">
              {/* Red filled portion */}
              <motion.div
                className="h-full bg-accent rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${highRiskPct}%` }}
                transition={{ duration: 0.8, ease: 'easeOut' }}
              />
              {/* Dashed/striped middle section */}
              <div
                className="h-full"
                style={{
                  width: '15%',
                  backgroundImage: 'repeating-linear-gradient(90deg, var(--surface-2) 0px, var(--surface-2) 3px, transparent 3px, transparent 6px)',
                  backgroundSize: '6px 100%',
                }}
              />
            </div>
            <div className="flex justify-between text-xs text-text-tertiary mb-8 relative z-10">
              <span>Low Risk</span>
              <span>High Risk</span>
            </div>

            {/* ── HIGH RISK ASSETS ── */}
            <div className="mb-6 relative z-10">
              <div className="flex items-center gap-2 mb-1">
                <span className="w-2.5 h-2.5 rounded-full bg-accent" />
                <span className="text-base font-bold text-text-primary">High Risk Assets</span>
              </div>
              <p className="text-sm text-text-tertiary mb-3 ml-[18px]">Within the recommended norm</p>
              <div className="flex items-center justify-between">
                <span className="text-5xl font-bold text-text-primary tracking-tight">{highRiskPct}%</span>
                <RiskArc percent={highRiskPct} color="var(--accent)" />
              </div>
            </div>

            {/* ── LOW RISK ASSETS ── */}
            <div className="mb-6 relative z-10">
              <div className="flex items-center gap-2 mb-1">
                <span className="w-2.5 h-2.5 rounded-full bg-text-tertiary" />
                <span className="text-base font-bold text-text-primary">Low Risk Assets</span>
              </div>
              <p className="text-sm text-text-tertiary mb-3 ml-[18px]">Safe investments strategy</p>
              <div className="flex items-center justify-between">
                <span className="text-5xl font-bold text-text-primary tracking-tight">{lowRiskPct}%</span>
                <RiskArc percent={lowRiskPct} color="var(--text-tertiary)" />
              </div>
            </div>

            {/* Divider */}
            <div className="h-px bg-border-subtle mb-4 relative z-10" />

            {/* Bottom action row */}
            <div className="flex items-center justify-between relative z-10">
              <button
                onClick={() => navigate('/screener')}
                className="text-sm text-text-tertiary hover:text-accent transition font-medium"
              >
                Review Suggestions
              </button>
              <div className="flex items-center gap-2">
                <button className="w-10 h-10 rounded-full bg-surface-2 flex items-center justify-center text-text-tertiary hover:text-text-primary transition">
                  <TrendingUp className="w-4 h-4" />
                </button>
                <button className="w-10 h-10 rounded-full bg-accent flex items-center justify-center text-white shadow-lg hover:bg-accent/90 transition">
                  <span className="material-symbols-outlined text-[18px]">bar_chart</span>
                </button>
              </div>
            </div>
          </div>

          {/* ─── TRANSACTIONS / RECENT ACTIVITY ─── */}
          <div className="bg-surface-1 rounded-2xl p-5 border border-border-subtle flex-1">
            <div className="flex items-center justify-between mb-5">
              <h3 className="text-base font-bold text-text-primary">Transactions</h3>
              <button className="text-[11px] text-text-tertiary hover:text-accent transition font-medium flex items-center gap-1">
                <Download className="w-3 h-3" />
                Export History
              </button>
            </div>

            <div className="space-y-4">
              {recentActivities.length === 0
                ? Array.from({ length: 3 }).map((_, i) => (
                    <div key={i} className="flex items-center gap-3 animate-pulse">
                      <div className="w-10 h-10 rounded-full bg-surface-2" />
                      <div className="flex-1 space-y-2">
                        <div className="h-3 bg-surface-2 rounded w-2/3" />
                        <div className="h-2 bg-surface-2 rounded w-1/3" />
                      </div>
                    </div>
                  ))
                : recentActivities.map((act, i) => (
                    <div key={i} className="flex items-center gap-3">
                      <div className={cn(
                        'w-10 h-10 rounded-full flex items-center justify-center shrink-0',
                        act.positive ? 'bg-positive/10' : 'bg-negative/10',
                      )}>
                        {act.positive ? (
                          <ArrowUpRight className="w-4 h-4 text-positive" />
                        ) : (
                          <ArrowDownRight className="w-4 h-4 text-negative" />
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-semibold text-text-primary">{act.action} · {act.symbol}</p>
                        <p className="text-[11px] text-text-tertiary">{act.time}</p>
                      </div>
                      <div className="text-right shrink-0">
                        <p className="text-sm font-bold text-text-primary">{act.price}</p>
                        <p className={cn('text-[11px] font-semibold', act.positive ? 'text-positive' : 'text-negative')}>
                          {act.change}
                        </p>
                      </div>
                    </div>
                  ))}
            </div>
          </div>
        </div>
      </div>
    </PageTransition>
  );
}
