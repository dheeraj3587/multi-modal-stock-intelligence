import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  TrendingUp,
  ArrowUpRight,
  ArrowDownRight,
  Filter,
  Download,
  Plus,
  ChevronLeft,
  ChevronRight,
  MoreVertical,
  Shield,
} from 'lucide-react';
import { PageTransition } from '../components/shared/PageTransition';
import { api } from '../lib/api';
import { useApi } from '../hooks/useApi';
import { cn, formatCurrency, formatPercent, formatCompactNumber } from '../lib/utils';
import { CompanyLogo } from '../components/shared/CompanyLogo';

const PAGE_SIZE = 8;

/* ── Score → letter grade ── */
function scoreToGrade(score: number): string {
  if (score >= 8.5) return 'A+';
  if (score >= 7.5) return 'A';
  if (score >= 6.5) return 'B+';
  if (score >= 5.5) return 'B';
  if (score >= 4.5) return 'C+';
  if (score >= 3.5) return 'C';
  if (score >= 2.5) return 'C-';
  return 'D';
}

/* ── Badge color helper for verdict/sentiment ── */
function verdictBadge(verdict: string) {
  const v = verdict.toLowerCase();
  if (v.includes('excellent') || v.includes('strong') || v.includes('buy') || v.includes('bullish') || v.includes('greed'))
    return 'bg-positive/10 text-positive';
  if (v.includes('poor') || v.includes('below') || v.includes('sell') || v.includes('fear') || v.includes('bearish') || v.includes('overbought'))
    return 'bg-negative/10 text-negative';
  return 'bg-surface-2 text-text-secondary';
}

/* ── Risk score color per value ── */
function riskScoreColor(score: number): string {
  if (score <= 30) return 'text-positive';
  if (score <= 60) return 'text-warning';
  return 'text-negative';
}
function riskBarColor(score: number): string {
  if (score <= 30) return 'bg-positive';
  if (score <= 60) return 'bg-warning';
  return 'bg-negative';
}

export function AllScorecardsPage() {
  const navigate = useNavigate();
  const { data: scorecards, loading } = useApi(() => api.getAllScorecards(), []);
  const { data: stocks } = useApi(() => api.getStocksWithSentiment(), [], { refetchInterval: 30_000 });

  const [search, setSearch] = useState('');
  const [page, setPage] = useState(1);

  /* ── Merge scorecard + live stock data ── */
  const stockMap = useMemo(() => {
    if (!stocks) return new Map<string, NonNullable<typeof stocks>[0]>();
    const m = new Map<string, (typeof stocks)[0]>();
    for (const s of stocks) m.set(s.symbol, s);
    return m;
  }, [stocks]);

  /* ── Filter + paginate ── */
  const filtered = useMemo(() => {
    if (!scorecards) return [];
    let result = [...scorecards];
    if (search) {
      const q = search.toLowerCase();
      result = result.filter((s) => s.symbol.toLowerCase().includes(q) || s.company_name.toLowerCase().includes(q));
    }
    result.sort((a, b) => b.overall_score - a.overall_score);
    return result;
  }, [scorecards, search]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const paginated = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  /* ── Top performer ── */
  const topPerformer = useMemo(() => {
    if (!stocks || stocks.length === 0) return null;
    return [...stocks].sort((a, b) => b.changePercent - a.changePercent)[0];
  }, [stocks]);

  /* ── Avg risk score (from scorecard scores — invert so higher score = lower risk) ── */
  const avgRisk = useMemo(() => {
    if (!scorecards || scorecards.length === 0) return 34;
    const avg = scorecards.reduce((s, c) => s + c.overall_score, 0) / scorecards.length;
    return Math.round(100 - avg * 10); // convert 0-10 score to 0-100 risk
  }, [scorecards]);

  /* ── Sentiment breakdown for risk card ── */
  const sentimentSummary = useMemo(() => {
    if (!stocks) return { bullish: 57, bearish: 34, total: 100 };
    const bullish = stocks.filter((s) => s.sentiment === 'bullish').length;
    const bearish = stocks.filter((s) => s.sentiment === 'bearish').length;
    const total = stocks.length || 1;
    return {
      bullish: Math.round((bullish / total) * 100),
      bearish: Math.round((bearish / total) * 100),
      total,
    };
  }, [stocks]);

  /* ── Recent activity from stock movers ── */
  const recentActivity = useMemo(() => {
    if (!stocks) return [];
    return [...stocks]
      .sort((a, b) => Math.abs(b.changePercent) - Math.abs(a.changePercent))
      .slice(0, 3)
      .map((s) => ({
        symbol: s.symbol,
        name: s.name,
        action: s.changePercent >= 0 ? 'Buy Signal' : 'Sell Signal',
        value: formatCurrency(Math.abs(s.change)),
        positive: s.changePercent >= 0,
        time: new Date(s.lastUpdated).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' }),
        status: Math.abs(s.changePercent) > 3 ? 'Confirmed' : s.changePercent >= 0 ? 'Completed' : 'Pending',
      }));
  }, [stocks]);

  return (
    <PageTransition>
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8">
        {/* ════════════════════════════════════════
            LEFT — 9 cols
        ════════════════════════════════════════ */}
        <section className="lg:col-span-9 flex flex-col gap-6">
          {/* ── Header row ── */}
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
            <div>
              <h1 className="text-2xl font-bold text-text-primary mb-1">All Scorecards</h1>
              <p className="text-sm text-text-tertiary">
                Compare risk, growth potential, and technical indicators across all tracked assets.
              </p>
            </div>
            <div className="flex items-center gap-3">
              <button className="flex items-center gap-2 px-4 py-2.5 bg-surface-1 border border-border-subtle rounded-xl hover:border-accent/40 text-sm font-medium transition-all">
                <Filter className="w-4 h-4" /> Filter
              </button>
              <button className="flex items-center gap-2 px-4 py-2.5 bg-surface-1 border border-border-subtle rounded-xl hover:border-accent/40 text-sm font-medium transition-all">
                <Download className="w-4 h-4" /> Export
              </button>
              <button
                onClick={() => navigate('/screener')}
                className="px-4 py-2.5 bg-accent text-white rounded-xl shadow-lg shadow-accent/20 hover:bg-accent/90 transition-all text-sm font-medium"
              >
                <span className="flex items-center gap-1"><Plus className="w-4 h-4" /> Add Asset</span>
              </button>
            </div>
          </div>

          {/* ── 3 Summary cards ── */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Top Performer */}
            <div className="bg-accent text-white p-6 rounded-2xl relative overflow-hidden shadow-xl shadow-accent/20 group">
              <div className="absolute -right-4 -top-4 opacity-10 transform rotate-12 group-hover:scale-110 transition-transform duration-500">
                <TrendingUp className="w-24 h-24" />
              </div>
              <p className="text-white/70 text-sm font-medium mb-1">Top Performer (24h)</p>
              <h3 className="text-3xl font-bold mb-2">
                {topPerformer ? formatPercent(topPerformer.changePercent) : '+0.00%'}
              </h3>
              <div className="flex items-center gap-2">
                <div className="bg-white/20 p-1 rounded-md">
                  <span className="material-symbols-outlined text-sm">trending_up</span>
                </div>
                <span className="font-medium">{topPerformer?.name ?? 'Loading...'}</span>
                {topPerformer && (
                  <span className="text-xs bg-white/20 px-2 py-0.5 rounded-full">
                    {topPerformer.riskLevel === 'low' ? 'Low Risk' : topPerformer.riskLevel === 'medium' ? 'Med Risk' : 'High Risk'}
                  </span>
                )}
              </div>
            </div>

            {/* Avg Portfolio Risk */}
            <div className="bg-surface-1 p-6 rounded-2xl border border-border-subtle">
              <p className="text-text-tertiary text-sm font-medium mb-1">Avg Portfolio Risk</p>
              <div className="flex items-end gap-2 mb-2">
                <h3 className="text-3xl font-bold text-text-primary">{avgRisk}/100</h3>
                <span className="text-positive text-sm font-medium mb-1.5 flex items-center">
                  <ArrowDownRight className="w-3.5 h-3.5" /> 2.1%
                </span>
              </div>
              <div className="w-full h-2 bg-surface-2 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-positive rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${avgRisk}%` }}
                  transition={{ duration: 0.6, ease: 'easeOut' }}
                />
              </div>
              <p className="text-xs text-text-tertiary mt-2">Within recommended safety norms</p>
            </div>

            {/* Total Assets Tracked */}
            <div className="bg-surface-1 p-6 rounded-2xl border border-border-subtle">
              <p className="text-text-tertiary text-sm font-medium mb-1">Total Assets Tracked</p>
              <div className="flex items-end gap-2 mb-2">
                <h3 className="text-3xl font-bold text-text-primary">{scorecards?.length ?? 0}</h3>
                <span className="text-accent text-sm font-medium mb-1.5 flex items-center">
                  <Plus className="w-3.5 h-3.5" /> {scorecards ? Math.min(4, scorecards.length) : 0} New
                </span>
              </div>
              {/* Avatar stack */}
              <div className="flex -space-x-2 mt-2">
                {(scorecards ?? []).slice(0, 3).map((sc) => (
                  <CompanyLogo key={sc.symbol} symbol={sc.symbol} className="w-8 h-8 border-2 border-surface-1" />
                ))}
                {(scorecards?.length ?? 0) > 3 && (
                  <div className="w-8 h-8 rounded-full border-2 border-surface-1 bg-surface-2 flex items-center justify-center text-xs font-medium text-text-tertiary">
                    +{(scorecards?.length ?? 0) - 3}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* ── Search bar ── */}
          <div className="flex items-center gap-3">
            <div className="relative flex-1 max-w-sm">
              <span className="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-text-tertiary text-[18px]">search</span>
              <input
                type="text"
                placeholder="Search companies..."
                value={search}
                onChange={(e) => { setSearch(e.target.value); setPage(1); }}
                className="h-10 w-full rounded-xl border border-border-subtle bg-surface-1 pl-10 pr-3 text-sm text-text-primary placeholder:text-text-tertiary focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent transition-all"
              />
            </div>
          </div>

          {/* ── Data Table ── */}
          <div className="bg-surface-1 rounded-2xl border border-border-subtle overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-surface-2/50 border-b border-border-subtle text-xs uppercase tracking-wider text-text-tertiary font-semibold">
                    <th className="p-4 pl-6">Asset Name</th>
                    <th className="p-4 text-right">Price</th>
                    <th className="p-4 text-center">Risk Score</th>
                    <th className="p-4 text-center">Technical</th>
                    <th className="p-4 text-center">Sentiment</th>
                    <th className="p-4 text-center">Growth</th>
                    <th className="p-4 text-right">24h Change</th>
                    <th className="p-4 pr-6 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border-subtle text-sm">
                  {loading
                    ? Array.from({ length: 5 }).map((_, i) => (
                        <tr key={i}>
                          <td className="p-4 pl-6"><div className="flex items-center gap-3"><div className="w-10 h-10 rounded-full bg-surface-2 animate-pulse" /><div className="space-y-1"><div className="w-20 h-3 bg-surface-2 rounded animate-pulse" /><div className="w-12 h-2 bg-surface-2 rounded animate-pulse" /></div></div></td>
                          <td className="p-4"><div className="w-16 h-3 bg-surface-2 rounded animate-pulse ml-auto" /></td>
                          <td className="p-4"><div className="w-12 h-3 bg-surface-2 rounded animate-pulse mx-auto" /></td>
                          <td className="p-4"><div className="w-14 h-5 bg-surface-2 rounded-full animate-pulse mx-auto" /></td>
                          <td className="p-4"><div className="w-14 h-5 bg-surface-2 rounded-full animate-pulse mx-auto" /></td>
                          <td className="p-4"><div className="w-6 h-3 bg-surface-2 rounded animate-pulse mx-auto" /></td>
                          <td className="p-4"><div className="w-12 h-3 bg-surface-2 rounded animate-pulse ml-auto" /></td>
                          <td className="p-4 pr-6"><div className="w-5 h-5 bg-surface-2 rounded animate-pulse ml-auto" /></td>
                        </tr>
                      ))
                    : paginated.map((sc) => {
                        const stock = stockMap.get(sc.symbol);
                        const riskScore = Math.round(100 - sc.overall_score * 10);
                        const grade = scoreToGrade(sc.overall_score);
                        const changePercent = stock?.changePercent ?? 0;
                        const isPos = changePercent >= 0;

                        // Technical verdict from score
                        const technical = sc.overall_score >= 7.5 ? 'Strong Buy' : sc.overall_score >= 6 ? 'Buy' : sc.overall_score >= 4 ? 'Hold' : sc.overall_score >= 2.5 ? 'Overbought' : 'Sell';

                        // Sentiment from stock data
                        const sentiment = stock?.sentiment ?? 'neutral';
                        const sentimentLabel = sentiment === 'bullish'
                          ? (stock?.forecastConfidence ?? 0) > 0.8 ? 'Extreme Greed' : 'Bullish'
                          : sentiment === 'bearish'
                            ? (stock?.forecastConfidence ?? 0) > 0.8 ? 'Fear' : 'Bearish'
                            : 'Neutral';

                        // Highlight top risk row
                        const isHighRisk = riskScore >= 70;

                        return (
                          <tr
                            key={sc.symbol}
                            onClick={() => navigate(`/scorecard/${sc.symbol}`)}
                            className={cn(
                              'group hover:bg-surface-2/30 transition-colors cursor-pointer',
                              isHighRisk && 'bg-accent/5 border-l-4 border-l-accent',
                            )}
                          >
                            {/* Asset Name */}
                            <td className="p-4 pl-6">
                              <div className="flex items-center gap-3">
                                <CompanyLogo symbol={sc.symbol} className="w-10 h-10" />
                                <div>
                                  <div className="font-bold text-text-primary">{sc.company_name}</div>
                                  <div className="text-xs text-text-tertiary">{sc.symbol}</div>
                                </div>
                              </div>
                            </td>

                            {/* Price */}
                            <td className="p-4 text-right font-mono text-text-secondary">
                              {stock ? formatCurrency(stock.price) : sc.market_cap_cr ? `₹${formatCompactNumber(sc.market_cap_cr * 1e7)}` : '—'}
                            </td>

                            {/* Risk Score */}
                            <td className="p-4">
                              <div className="flex flex-col items-center gap-1">
                                <span className={cn('font-bold', riskScoreColor(riskScore))}>{riskScore}/100</span>
                                <div className="w-16 h-1 bg-surface-2 rounded-full overflow-hidden">
                                  <div className={cn('h-full rounded-full', riskBarColor(riskScore))} style={{ width: `${riskScore}%` }} />
                                </div>
                              </div>
                            </td>

                            {/* Technical */}
                            <td className="p-4 text-center">
                              <span className={cn('px-2.5 py-1 rounded-full text-xs font-medium', verdictBadge(technical))}>
                                {technical}
                              </span>
                            </td>

                            {/* Sentiment */}
                            <td className="p-4 text-center">
                              <span className={cn('px-2.5 py-1 rounded-full text-xs font-medium', verdictBadge(sentimentLabel))}>
                                {sentimentLabel}
                              </span>
                            </td>

                            {/* Growth Grade */}
                            <td className="p-4 text-center">
                              <span className={cn('font-bold', sc.overall_score >= 6 ? 'text-accent' : 'text-text-secondary')}>
                                {grade}
                              </span>
                            </td>

                            {/* 24h Change */}
                            <td className={cn('p-4 text-right font-medium', isPos ? 'text-positive' : 'text-negative')}>
                              {stock ? formatPercent(changePercent) : '—'}
                            </td>

                            {/* Action */}
                            <td className="p-4 pr-6 text-right">
                              <button
                                onClick={(e) => { e.stopPropagation(); }}
                                className="text-text-tertiary hover:text-text-primary transition-colors"
                              >
                                <MoreVertical className="w-4 h-4" />
                              </button>
                            </td>
                          </tr>
                        );
                      })}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            <div className="bg-surface-2/30 border-t border-border-subtle p-4 flex items-center justify-between">
              <span className="text-xs text-text-tertiary">
                Showing {((page - 1) * PAGE_SIZE) + 1}-{Math.min(page * PAGE_SIZE, filtered.length)} of {filtered.length} assets
              </span>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page <= 1}
                  className="w-8 h-8 flex items-center justify-center rounded-lg border border-border-subtle text-text-tertiary hover:bg-surface-2 disabled:opacity-40 transition"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                {Array.from({ length: Math.min(totalPages, 5) }, (_, i) => i + 1).map((p) => (
                  <button
                    key={p}
                    onClick={() => setPage(p)}
                    className={cn(
                      'w-8 h-8 flex items-center justify-center rounded-lg text-xs font-medium transition',
                      p === page
                        ? 'bg-accent text-white shadow-md shadow-accent/20'
                        : 'border border-border-subtle text-text-tertiary hover:bg-surface-2',
                    )}
                  >
                    {p}
                  </button>
                ))}
                {totalPages > 5 && <span className="text-text-tertiary text-xs">...</span>}
                <button
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={page >= totalPages}
                  className="w-8 h-8 flex items-center justify-center rounded-lg border border-border-subtle text-text-tertiary hover:bg-surface-2 disabled:opacity-40 transition"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </section>

        {/* ════════════════════════════════════════
            RIGHT SIDEBAR — 3 cols
        ════════════════════════════════════════ */}
        <aside className="lg:col-span-3 flex flex-col gap-6">
          {/* ── Portfolio Risk Score (Red gradient card) ── */}
          <div className="bg-accent rounded-2xl p-6 text-white shadow-lg relative overflow-hidden">
            {/* texture overlay */}
            <div className="absolute inset-0 opacity-[0.04] bg-[repeating-linear-gradient(45deg,transparent,transparent_2px,rgba(255,255,255,0.1)_2px,rgba(255,255,255,0.1)_4px)]" />
            <div className="relative z-10">
              <div className="flex justify-between items-start mb-6">
                <h2 className="text-lg font-semibold leading-tight">Portfolio<br />Risk Score</h2>
                <div className="text-xs opacity-70 text-right">Updated:<br />Just Now</div>
              </div>

              {/* Labels */}
              <div className="mb-1 flex justify-between text-xs opacity-80 font-medium">
                <span>Low Risk</span>
                <span>High Risk</span>
              </div>
              {/* Risk bar with dashes */}
              <div className="w-full h-3 bg-black/20 rounded-full mb-6 overflow-hidden relative">
                {/* Divider dashes */}
                <div className="absolute inset-0 flex justify-between px-1 items-center pointer-events-none">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <div key={i} className="w-px h-full bg-white/10" />
                  ))}
                </div>
                <motion.div
                  className="h-full bg-white rounded-full shadow-[0_0_10px_rgba(255,255,255,0.5)]"
                  initial={{ width: 0 }}
                  animate={{ width: `${sentimentSummary.bearish}%` }}
                  transition={{ duration: 0.8, ease: 'easeOut' }}
                />
              </div>

              {/* High Risk Assets */}
              <div className="mb-4">
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-2 h-2 rounded-full bg-red-900 border border-red-300 shadow-[0_0_5px_rgba(255,0,0,0.8)]" />
                  <span className="text-sm font-medium">High Risk Assets</span>
                </div>
                <p className="text-xs opacity-70 mb-1">Within the recommended norm</p>
                <div className="flex justify-between items-end">
                  <span className="text-3xl font-bold">{sentimentSummary.bearish}%</span>
                  <div className="w-8 h-8 rounded-full border-2 border-white/30 border-t-white transform -rotate-45" />
                </div>
              </div>

              <div className="w-full h-px bg-white/10 mb-4" />

              {/* Low Risk Assets */}
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-2 h-2 rounded-full bg-white" />
                  <span className="text-sm font-medium">Low Risk Assets</span>
                </div>
                <p className="text-xs opacity-70 mb-1">Safe investments strategy</p>
                <div className="flex justify-between items-end">
                  <span className="text-3xl font-bold">{sentimentSummary.bullish}%</span>
                  <div className="w-8 h-8 rounded-full border-2 border-white/30 border-t-white border-r-white transform rotate-12" />
                </div>
              </div>
            </div>
          </div>

          {/* ── Recent Activity ── */}
          <div className="bg-surface-1 rounded-2xl border border-border-subtle p-6 flex-1">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-bold text-text-primary">Recent Activity</h3>
              <button onClick={() => navigate('/sentiment')} className="text-xs text-accent hover:underline font-medium">View All</button>
            </div>
            <div className="space-y-4">
              {recentActivity.map((act, i) => (
                <div key={i} className="flex items-center justify-between group cursor-pointer">
                  <div className="flex items-center gap-3">
                    <div className={cn(
                      'w-10 h-10 rounded-full flex items-center justify-center transition-colors',
                      act.positive ? 'bg-positive/10 text-positive group-hover:bg-positive/20' : 'bg-negative/10 text-negative group-hover:bg-negative/20',
                    )}>
                      {act.positive
                        ? <ArrowDownRight className="w-4 h-4" />
                        : <ArrowUpRight className="w-4 h-4" />
                      }
                    </div>
                    <div>
                      <div className="text-sm font-semibold text-text-primary">{act.action} {act.symbol}</div>
                      <div className="text-xs text-text-tertiary">{act.time}</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={cn('text-sm font-bold', act.positive ? 'text-positive' : 'text-text-primary')}>
                      {act.positive ? '+' : '-'}{act.value}
                    </div>
                    <div className="text-xs text-text-tertiary">{act.status}</div>
                  </div>
                </div>
              ))}

              {/* Upgrade to Pro */}
              <div className="mt-6 p-4 rounded-xl bg-surface-2/50 border border-border-subtle relative overflow-hidden">
                <div className="relative z-10">
                  <h4 className="text-sm font-bold text-text-primary mb-1">Upgrade to Pro</h4>
                  <p className="text-xs text-text-tertiary mb-3">Get advanced risk analytics and real-time alerts.</p>
                  <button className="text-xs font-semibold text-accent hover:text-accent/80 transition-colors">
                    Learn More →
                  </button>
                </div>
                <div className="absolute -right-4 -bottom-4 opacity-5">
                  <Shield className="w-20 h-20" />
                </div>
              </div>
            </div>
          </div>
        </aside>
      </div>
    </PageTransition>
  );
}
