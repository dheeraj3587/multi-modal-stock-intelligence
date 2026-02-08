import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useApi } from '../hooks/useApi';
import { api } from '../lib/api';
import type { LeaderboardStock } from '../lib/api';
import { CompanyLogo } from '../components/shared/CompanyLogo';
import { cn, formatCompactNumber, formatCurrency, formatPercent } from '../lib/utils';
import { Skeleton } from '../components/ui/Skeleton';
import {
  ArrowUp,
  ArrowDown,
  ArrowRight,
  RefreshCcw,
  Plus,
  SlidersHorizontal,
  LayoutGrid,
  FileText,
  MessageSquare,
  Globe,
} from 'lucide-react';

/* ─── helpers ───────────────────────────────────────────── */

const CELL_BG = [
  'bg-[#DA2929]',     // 0 – primary
  'bg-[#A81D1D]',     // 1
  'bg-[#FF4D4D]',     // 2
  'bg-[#4A0A0A]',     // 3
  'bg-[#751111]',     // 4
  'bg-[#2C0505]',     // 5
  'bg-[#E32929]',     // 6
  'bg-[#5E0E0E]',     // 7
];

const CELL_HOVER = [
  'hover:bg-red-500',
  'hover:bg-red-600',
  'hover:bg-red-400',
  'hover:bg-gray-800',
  'hover:bg-red-700',
  'hover:bg-gray-800',
  'hover:bg-red-400',
  'hover:bg-red-800',
];

const GRID_CLASS = [
  'col-span-2 row-span-2',   // 0
  'col-span-1 row-span-2',   // 1
  'col-span-1 row-span-1',   // 2
  'col-span-1 row-span-1',   // 3
  'col-span-2 row-span-1',   // 4
  'col-span-1 row-span-1',   // 5
  'col-span-1 row-span-1',   // 6
  'col-span-2 row-span-1',   // 7
];

function inrMoney(value: number) {
  const f = formatCurrency(value, 'INR');
  return f.startsWith('₹') ? f : `₹${f}`;
}

function timeAgo(iso: string) {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return '';
  const s = Math.floor((Date.now() - t) / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

/* ─── component ─────────────────────────────────────────── */

export function GrowthLeaderboardPage() {
  const navigate = useNavigate();

  /* — data hooks — */
  const {
    data: stocks,
    loading,
    refetch: refetchStocks,
  } = useApi(() => api.getStocksWithSentiment(), [], { refetchInterval: 30_000 });

  const { data: leaderboard } = useApi(() => api.getLeaderboard(), []);

  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const [timeframe, setTimeframe] = useState('1d');

  /* merge growth scores into stocks */
  const growthMap = useMemo(() => {
    const m = new Map<string, LeaderboardStock>();
    leaderboard?.forEach((l) => m.set(l.symbol, l));
    return m;
  }, [leaderboard]);

  /* top 8 for heatmap */
  const topStocks = useMemo(() => {
    if (!stocks) return [];
    const sorted = [...stocks].sort((a, b) => {
      const ga = growthMap.get(a.symbol)?.growthScore ?? 0;
      const gb = growthMap.get(b.symbol)?.growthScore ?? 0;
      return gb - ga;
    });
    return sorted.slice(0, 8);
  }, [stocks, growthMap]);

  /* quotes for prices  */
  const symbolsParam = useMemo(() => {
    const syms = new Set<string>();
    topStocks.forEach((s) => syms.add(s.symbol));
    if (selectedSymbol) syms.add(selectedSymbol);
    return Array.from(syms).slice(0, 20).join(',');
  }, [topStocks, selectedSymbol]);

  const {
    data: quotes,
    refetch: refetchQuotes,
  } = useApi(() => api.getQuotes(symbolsParam), [symbolsParam], {
    enabled: symbolsParam.length > 0,
    refetchInterval: 30_000,
  });

  const quoteMap = useMemo(() => {
    const m = new Map<string, any>();
    if (!quotes) return m;
    for (const q of quotes as any[]) m.set(q.symbol, q);
    return m;
  }, [quotes]);

  /* merge quotes into topStocks */
  const heatmapStocks = useMemo(
    () =>
      topStocks.map((s) => {
        const q: any = quoteMap.get(s.symbol);
        return {
          ...s,
          price: q?.price ?? s.price,
          changePercent: q?.changePercent ?? s.changePercent,
          change: q?.change ?? s.change,
          volume: q?.volume ?? 0,
          growthScore: growthMap.get(s.symbol)?.growthScore ?? 0,
          forecastPercent: growthMap.get(s.symbol)?.forecastPercent ?? 0,
        };
      }),
    [topStocks, quoteMap, growthMap],
  );

  /* picked stock for right panel */
  const baseSelected = useMemo(() => {
    if (!stocks?.length) return null;
    if (selectedSymbol) return stocks.find((s) => s.symbol === selectedSymbol) || stocks[0];
    return stocks[0];
  }, [stocks, selectedSymbol]);

  const selectedStock = useMemo(() => {
    if (!baseSelected) return null;
    const q: any = quoteMap.get(baseSelected.symbol);
    return {
      ...baseSelected,
      price: q?.price ?? baseSelected.price,
      changePercent: q?.changePercent ?? baseSelected.changePercent,
      change: q?.change ?? baseSelected.change,
      volume: q?.volume ?? 0,
      growthScore: growthMap.get(baseSelected.symbol)?.growthScore ?? 0,
    };
  }, [baseSelected, quoteMap, growthMap]);

  /* sentiment summary across all stocks */
  const sentimentSummary = useMemo(() => {
    if (!stocks?.length) return null;
    const bullish = stocks.filter((s) => s.sentiment === 'bullish').length;
    const bearish = stocks.filter((s) => s.sentiment === 'bearish').length;
    const neutral = stocks.filter((s) => s.sentiment === 'neutral').length;
    const total = stocks.length;
    const avg = total > 0 ? (bullish - bearish) / total : 0;
    const score = Math.round(((avg + 1) / 2) * 100);
    const label =
      score >= 75 ? 'Extreme Greed' : score >= 60 ? 'Greed' : score >= 40 ? 'Neutral' : score >= 25 ? 'Fear' : 'Extreme Fear';
    return { bullish, bearish, neutral, total, score, label };
  }, [stocks]);

  /* aggregate metrics */
  const totalTradedValue = useMemo(() => {
    if (!quotes) return 0;
    return (quotes as any[]).reduce((s, q) => s + (Number(q.volume) || 0) * (Number(q.price) || 0), 0);
  }, [quotes]);

  const avgAbsChange = useMemo(() => {
    if (!quotes || !(quotes as any[]).length) return 0;
    const list = quotes as any[];
    return list.reduce((s, q) => s + Math.abs(Number(q.changePercent) || 0), 0) / list.length;
  }, [quotes]);

  const avgChange = useMemo(() => {
    if (!quotes || !(quotes as any[]).length) return 0;
    const list = quotes as any[];
    return list.reduce((s, q) => s + (Number(q.changePercent) || 0), 0) / list.length;
  }, [quotes]);

  /* news for selected stock */
  const {
    data: newsData,
    loading: newsLoading,
    refetch: refetchNews,
  } = useApi(() => api.getNews(selectedStock?.symbol || ''), [selectedStock?.symbol], {
    enabled: Boolean(selectedStock?.symbol),
  });

  const handleRefresh = () => {
    refetchStocks();
    refetchQuotes();
    refetchNews();
  };

  /* ── loading state ── */
  if (loading || !stocks) {
    return (
      <div className="space-y-8">
        <Skeleton className="h-64 w-full rounded-3xl" />
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <Skeleton className="lg:col-span-8 h-96 rounded-3xl" />
          <Skeleton className="lg:col-span-4 h-96 rounded-3xl" />
        </div>
      </div>
    );
  }

  /* ─── render ─── */
  return (
    <motion.div
      initial={{ opacity: 0, y: 12, filter: 'blur(4px)' }}
      animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
      transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
      className="font-sans text-text-primary min-h-screen"
    >
      <main className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* ───── LEFT COL (8) ───── */}
        <div className="lg:col-span-8 flex flex-col gap-6">

          {/* Hero Card */}
          <div className="bg-[#DA2929] text-white rounded-3xl p-8 relative overflow-hidden shadow-xl shadow-red-900/20">
            <div className="absolute -right-20 -top-20 w-96 h-96 bg-white opacity-5 rounded-full blur-3xl pointer-events-none" />
            <div className="absolute left-10 bottom-0 w-full h-32 bg-gradient-to-t from-black/20 to-transparent pointer-events-none" />

            <div className="relative z-10">
              {/* Title row */}
              <div className="flex justify-between items-start mb-6">
                <div>
                  <p className="text-red-100 text-sm font-medium mb-1">Market Sentiment Score</p>
                  <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
                    {sentimentSummary?.label ?? '—'}
                    <span className="text-lg opacity-80 font-normal align-middle ml-2">
                      {sentimentSummary?.score ?? '—'}/100
                    </span>
                  </h1>
                </div>
                <div className="flex gap-2">
                  <button className="w-10 h-10 rounded-xl bg-white/10 hover:bg-white/20 flex items-center justify-center backdrop-blur-sm transition">
                    <Plus className="w-5 h-5" />
                  </button>
                  <button className="w-10 h-10 rounded-xl bg-white/10 hover:bg-white/20 flex items-center justify-center backdrop-blur-sm transition">
                    <SlidersHorizontal className="w-5 h-5" />
                  </button>
                </div>
              </div>

              {/* Metric cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                {[
                  { label: 'Trending Volume', value: totalTradedValue ? inrMoney(totalTradedValue) : '—', change: avgChange },
                  { label: 'Social Mentions', value: newsLoading ? '…' : String(newsData?.article_count ?? 0), change: selectedStock?.changePercent ?? 0 },
                  { label: 'Volatility Index', value: avgAbsChange.toFixed(1), change: 0 },
                  { label: 'Market Cap', value: '—', change: 0 },
                ].map((item, i) => (
                  <div key={i} className="bg-black/20 rounded-2xl p-4 backdrop-blur-sm">
                    <p className="text-xs text-red-100 opacity-70 mb-1">{item.label}</p>
                    <p className="text-xl font-semibold">{item.value}</p>
                    <span
                      className={cn(
                        'text-xs flex items-center mt-1 w-fit px-1.5 py-0.5 rounded',
                        item.change >= 0 ? 'text-green-300 bg-green-900/30' : 'text-red-200 bg-red-900/30',
                      )}
                    >
                      {item.change >= 0 ? <ArrowUp className="w-3 h-3 mr-1" /> : <ArrowDown className="w-3 h-3 mr-1" />}
                      {Math.abs(item.change).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>

              {/* Timeframe pills */}
              <div className="flex items-center gap-1 bg-black/20 p-1 rounded-xl w-fit backdrop-blur-sm">
                {['1h', '8h', '1d', '1w', '1m'].map((tf) => (
                  <button
                    key={tf}
                    onClick={() => setTimeframe(tf)}
                    className={cn(
                      'px-4 py-1.5 rounded-lg text-sm transition',
                      tf === timeframe ? 'bg-white text-[#DA2929] font-bold shadow-sm' : 'hover:bg-white/10 text-red-100',
                    )}
                  >
                    {tf}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Heatmap */}
          <div className="rounded-3xl bg-surface-light dark:bg-surface-dark p-6 border border-gray-200 dark:border-gray-800 shadow-sm">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-semibold flex items-center gap-2 text-text-primary">
                <LayoutGrid className="text-[#DA2929] w-5 h-5" />
                Sector Heatmap
              </h2>
              <div className="text-xs text-gray-500 flex items-center gap-2">
                <span>Updated: Just Now</span>
                <button
                  className="p-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-full transition"
                  onClick={handleRefresh}
                >
                  <RefreshCcw className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* CSS Grid (4×4) */}
            <div className="grid grid-cols-4 grid-rows-4 gap-1 h-[600px] w-full rounded-xl overflow-hidden text-white font-medium">
              {heatmapStocks.map((stock, i) => (
                <div
                  key={stock.symbol}
                  onClick={() => setSelectedSymbol(stock.symbol)}
                  className={cn(
                    CELL_BG[i] ?? 'bg-[#5E0E0E]',
                    CELL_HOVER[i] ?? 'hover:bg-red-800',
                    GRID_CLASS[i] ?? 'col-span-1 row-span-1',
                    'p-4 relative group cursor-pointer transition-colors flex flex-col justify-between',
                  )}
                >
                  <div className="flex justify-between items-start">
                    <div className="flex items-center gap-2">
                      {i === 0 && <CompanyLogo symbol={stock.symbol} className="w-8 h-8 rounded-full bg-white p-0.5" chars={3} />}
                      <span className={cn('font-bold', i === 0 ? 'text-lg' : 'text-sm', i === 3 || i === 5 ? 'text-gray-300' : '')}>{stock.symbol}</span>
                    </div>
                    <span className={cn('text-xs opacity-90', i === 3 || i === 5 ? 'text-gray-400' : '')}>
                      {formatPercent(stock.changePercent)}
                    </span>
                  </div>

                  <div className="mt-auto">
                    <p className={cn('font-bold tracking-tight', i === 0 ? 'text-2xl' : 'text-sm', i === 3 ? 'text-gray-200' : '')}>
                      {inrMoney(stock.price)}
                    </p>
                    {i < 3 && <p className="text-[10px] opacity-70">Vol: {formatCompactNumber(stock.volume)}</p>}
                    {i === 7 && <p className="text-xs text-white/60">Top growth potential</p>}
                  </div>

                  {/* mini sparkline on cell 4 */}
                  {i === 4 && (
                    <div className="mt-2 h-8 w-full opacity-50">
                      <svg className="w-full h-full stroke-current fill-none" viewBox="0 0 100 30" preserveAspectRatio="none">
                        <path d="M0,25 C20,25 20,10 40,15 C60,20 60,5 100,0" strokeWidth="2" />
                      </svg>
                    </div>
                  )}

                  {/* hover overlay */}
                  <div className="absolute inset-0 bg-black/20 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                    <span className="bg-white text-[#DA2929] text-xs px-3 py-1 rounded-full font-bold">
                      View Details
                    </span>
                  </div>
                </div>
              ))}
            </div>

            {/* Legend */}
            <div className="mt-4 flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 bg-[#DA2929] rounded-sm" /> High Positive
                <span className="w-3 h-3 bg-[#5E0E0E] rounded-sm ml-2" /> Neutral
                <span className="w-3 h-3 bg-[#2C0505] rounded-sm ml-2" /> Negative
              </div>
              <span>Data provided by StockSense</span>
            </div>
          </div>
        </div>

        {/* ───── RIGHT COL (4) ───── */}
        <div className="lg:col-span-4 flex flex-col gap-6">
          {selectedStock ? (
            <div className="bg-surface-light dark:bg-surface-dark rounded-3xl p-6 border border-gray-200 dark:border-gray-800 shadow-lg h-full">
              {/* Stock header */}
              <div className="flex items-center justify-between mb-8">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 rounded-full bg-orange-100 dark:bg-orange-900/20 flex items-center justify-center p-2 overflow-hidden">
                    <CompanyLogo symbol={selectedStock.symbol} className="w-9 h-9" chars={3} />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white leading-none">{selectedStock.name}</h3>
                    <span className="text-sm text-gray-500 dark:text-gray-400">{selectedStock.symbol}/INR</span>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-lg font-bold text-gray-900 dark:text-white">{inrMoney(selectedStock.price)}</div>
                  <div className={cn('text-xs font-medium', selectedStock.changePercent >= 0 ? 'text-green-500' : 'text-red-500')}>
                    {formatPercent(selectedStock.changePercent)}
                  </div>
                </div>
              </div>

              {/* Social Heat bar */}
              <div className="mb-8">
                <div className="flex justify-between items-end mb-2">
                  <h4 className="text-lg font-semibold text-gray-900 dark:text-white">Social Heat</h4>
                  <span
                    className={cn(
                      'text-xs font-bold',
                      selectedStock.sentiment === 'bullish' ? 'text-green-500' : 'text-[#DA2929]',
                    )}
                  >
                    {selectedStock.sentiment === 'bullish' ? 'Extremely Hot' : selectedStock.sentiment === 'bearish' ? 'Cold' : 'Warm'}
                  </span>
                </div>
                <div className="h-3 w-full bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden flex">
                  <div
                    className="h-full bg-[#DA2929] rounded-full shadow-[0_0_10px_rgba(218,41,41,0.6)]"
                    style={{
                      width: `${Math.max(10, Math.min(100, Number(selectedStock.forecastConfidence) || 50))}%`,
                    }}
                  />
                </div>
                <div className="flex justify-between mt-1 text-[10px] text-gray-400 uppercase tracking-wider font-medium">
                  <span>Cold</span>
                  <span>Neutral</span>
                  <span>Viral</span>
                </div>
              </div>

              {/* Positive / Negative */}
              <div className="grid grid-cols-2 gap-4 mb-8">
                <div className="bg-gray-50 dark:bg-black p-4 rounded-2xl border border-gray-100 dark:border-gray-800">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="w-2 h-2 rounded-full bg-green-500" />
                    <span className="text-xs font-medium text-gray-500 dark:text-gray-400">Positive Posts</span>
                  </div>
                  <div className="text-2xl font-bold text-gray-900 dark:text-white">
                    {sentimentSummary ? `${Math.round((sentimentSummary.bullish / sentimentSummary.total) * 100)}%` : '—'}
                  </div>
                  <div className="text-xs text-gray-400 mt-1">Within expected norm</div>
                </div>
                <div className="bg-gray-50 dark:bg-black p-4 rounded-2xl border border-gray-100 dark:border-gray-800">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="w-2 h-2 rounded-full bg-red-500" />
                    <span className="text-xs font-medium text-gray-500 dark:text-gray-400">Negative Posts</span>
                  </div>
                  <div className="text-2xl font-bold text-gray-900 dark:text-white">
                    {sentimentSummary ? `${Math.round((sentimentSummary.bearish / sentimentSummary.total) * 100)}%` : '—'}
                  </div>
                  <div className="text-xs text-gray-400 mt-1">Lower than average</div>
                </div>
              </div>

              {/* Top Sentiment Drivers */}
              <div className="space-y-4">
                <h4 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Top Sentiment Drivers
                </h4>
                {newsLoading ? (
                  <div className="space-y-2">
                    <Skeleton className="h-14 w-full rounded-xl" />
                    <Skeleton className="h-14 w-full rounded-xl" />
                    <Skeleton className="h-14 w-full rounded-xl" />
                  </div>
                ) : newsData?.news?.length ? (
                  newsData.news.slice(0, 3).map((item) => {
                    const src = (item.source || '').toLowerCase();
                    const tag = src.includes('reddit') ? 'Reddit' : src.includes('twitter') || src.includes('x.com') ? 'Twitter' : 'News';
                    const Icon = tag === 'Reddit' ? MessageSquare : tag === 'Twitter' ? Globe : FileText;
                    const tagClasses =
                      tag === 'Reddit'
                        ? 'bg-orange-100 text-orange-700 dark:bg-orange-900/40 dark:text-orange-300'
                        : tag === 'Twitter'
                          ? 'bg-gray-200 text-gray-700 dark:bg-gray-800 dark:text-gray-300'
                          : 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300';
                    const iconWrap =
                      tag === 'Reddit'
                        ? 'bg-orange-100 dark:bg-orange-900/20'
                        : tag === 'Twitter'
                          ? 'bg-black dark:bg-gray-800'
                          : 'bg-blue-100 dark:bg-blue-900/20';
                    const iconColor = tag === 'Twitter' ? 'text-white' : tag === 'Reddit' ? 'text-orange-500' : 'text-blue-500';

                    return (
                      <a
                        key={item.url}
                        href={item.url}
                        target="_blank"
                        rel="noreferrer"
                        className="group flex gap-4 p-3 rounded-xl hover:bg-gray-50 dark:hover:bg-gray-900 transition-colors cursor-pointer border border-transparent hover:border-gray-100 dark:hover:border-gray-800"
                      >
                        <div className={cn('w-10 h-10 rounded-full flex items-center justify-center shrink-0', iconWrap)}>
                          <Icon className={cn('w-4 h-4', iconColor)} />
                        </div>
                        <div>
                          <p className="text-sm font-medium text-gray-900 dark:text-gray-100 line-clamp-1 group-hover:text-[#DA2929] transition-colors">
                            {item.title}
                          </p>
                          <div className="flex items-center gap-2 mt-1">
                            <span className={cn('text-[10px] px-1.5 py-0.5 rounded', tagClasses)}>{tag}</span>
                            <span className="text-[10px] text-gray-400">{timeAgo(item.published_at)}</span>
                          </div>
                        </div>
                      </a>
                    );
                  })
                ) : (
                  <div className="text-sm text-gray-500 dark:text-gray-400">No recent news found.</div>
                )}
              </div>

              {/* Trade button */}
              <button
                onClick={() => navigate(`/stock/${selectedStock.symbol}`)}
                className="w-full mt-8 bg-[#DA2929] hover:bg-[#B91C1C] text-white font-semibold py-3.5 rounded-xl shadow-lg shadow-red-500/30 transition-all flex items-center justify-center gap-2"
              >
                <span>Trade {selectedStock.symbol}</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <Skeleton className="h-full rounded-3xl" />
          )}
        </div>
      </main>
    </motion.div>
  );
}
