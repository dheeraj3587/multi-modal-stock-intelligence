import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  ResponsiveContainer,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Bar,
  ComposedChart,
} from 'recharts';
import {
  TrendingUp,
  TrendingDown,
  Search,
  Bell,
  CheckCircle,
  XCircle,
  RefreshCw,
  ArrowUpRight,
  ArrowDownRight,
  BarChart3,
  Activity,
  Star,
  Plus,
  SlidersHorizontal,
} from 'lucide-react';
import { PageTransition } from '../components/shared/PageTransition';
import { CompanyLogo } from '../components/shared/CompanyLogo';
import { api, type StockAnalysis, type CandleData } from '../lib/api';
import { useApi } from '../hooks/useApi';
import { cn, formatCurrency, formatPercent } from '../lib/utils';

/* ── Animations ── */
const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.06 } },
};
const fadeUp = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.3 } },
};

const TIMEFRAMES = ['1h', '8h', '1d', '1w', '1m', '6m', '1y'] as const;

/* ── Types ── */
interface ForecastSetup {
  symbol: string;
  name: string;
  action: 'buy' | 'sell' | 'hold';
  predictedPrice: number;
  currentPrice: number;
  changePercent: number;
  confidence: number;
  sentiment: string;
  sector: string;
  status: 'confirmed' | 'pending' | 'failed';
  timestamp: string;
}

/* ── Deterministic sparkline from symbol ── */
function generateMiniSparkline(symbol: string, positive: boolean): string {
  let hash = 0;
  for (let i = 0; i < symbol.length; i++) hash = symbol.charCodeAt(i) + ((hash << 5) - hash);
  const pts: number[] = [];
  for (let i = 0; i < 20; i++) {
    const base = positive ? 30 - (i / 19) * 20 : 10 + (i / 19) * 20;
    const noise = Math.sin(hash + i * 2.1) * 8 + Math.cos(hash * 0.7 + i * 1.3) * 5;
    pts.push(Math.max(2, Math.min(48, base + noise)));
  }
  const step = 320 / (pts.length - 1);
  return pts.map((y, i) => `${i === 0 ? 'M' : 'L'}${(i * step).toFixed(1)},${y.toFixed(1)}`).join(' ');
}

/* ── Deterministic bar heights ── */
function barHeights(seed: string, count: number): number[] {
  let h = 0;
  for (let i = 0; i < seed.length; i++) h = seed.charCodeAt(i) + ((h << 5) - h);
  return Array.from({ length: count }, (_, i) =>
    Math.abs(Math.sin(h + i * 1.7) * 0.7 + Math.cos(h * 0.3 + i * 2.3) * 0.3),
  );
}

const TIMEFRAMES_TO_DAYS: Record<string, number> = {
  '1h': 1, '8h': 1, '1d': 7, '1w': 30, '1m': 90, '6m': 180, '1y': 365,
};

/* ═════════════════════════════════════════
   Forecasts Page
═════════════════════════════════════════ */
export function ForecastsPage() {
  const navigate = useNavigate();
  const [activeTimeframe, setActiveTimeframe] = useState<typeof TIMEFRAMES[number]>('1d');
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);

  /* ── Data ── */
  const { data: stocks, loading: stocksLoading } = useApi(
    () => api.getStocksWithSentiment(),
    [],
    { refetchInterval: 30_000 },
  );

  /* ── Fetch analyses for top movers ── */
  const [analyses, setAnalyses] = useState<Map<string, StockAnalysis>>(new Map());
  const [analysesLoading, setAnalysesLoading] = useState(false);
  const fetchedRef = useRef(false);

  const topStocks = useMemo(() => {
    if (!stocks) return [];
    return [...stocks].sort((a, b) => Math.abs(b.changePercent) - Math.abs(a.changePercent)).slice(0, 12);
  }, [stocks]);

  /* Auto-select first stock */
  useEffect(() => {
    if (!selectedSymbol && topStocks.length) setSelectedSymbol(topStocks[0].symbol);
  }, [topStocks, selectedSymbol]);

  useEffect(() => {
    if (!topStocks.length || fetchedRef.current) return;
    fetchedRef.current = true;
    setAnalysesLoading(true);
    (async () => {
      const map = new Map<string, StockAnalysis>();
      for (let i = 0; i < topStocks.length; i += 4) {
        const batch = topStocks.slice(i, i + 4);
        const settled = await Promise.allSettled(batch.map((s) => api.getAnalysis(s.symbol)));
        settled.forEach((r, idx) => {
          if (r.status === 'fulfilled' && r.value) map.set(batch[idx].symbol, r.value);
        });
      }
      setAnalyses(map);
      setAnalysesLoading(false);
    })();
  }, [topStocks]);

  /* ── Aggregated metrics ── */
  const metrics = useMemo(() => {
    if (!stocks?.length) return null;
    const totalValue = stocks.reduce((s, st) => s + st.price, 0);
    const avgChange = stocks.reduce((s, st) => s + st.changePercent, 0) / stocks.length;
    let avgConf = stocks.reduce((s, st) => s + st.forecastConfidence, 0) / stocks.length;
    if (analyses.size > 0) {
      const confs = Array.from(analyses.values()).map((a) => a.forecastConfidence);
      avgConf = confs.reduce((a, b) => a + b, 0) / confs.length;
    }
    const realizedPL = stocks.filter((s) => s.change > 0).reduce((a, s) => a + s.change, 0);
    const unrealizedPL = stocks.filter((s) => s.change < 0).reduce((a, s) => a + s.change, 0);
    let projGrowth = 0;
    analyses.forEach((a) => {
      if (a.predictedPrice > a.currentPrice) projGrowth += a.predictedPrice - a.currentPrice;
    });
    return { totalValue, avgChange, avgConf, realizedPL, unrealizedPL, projGrowth };
  }, [stocks, analyses]);

  /* ── Setup cards ── */
  const setups = useMemo<ForecastSetup[]>(() => {
    if (!topStocks.length) return [];
    return topStocks.map((stock) => {
      const a = analyses.get(stock.symbol);
      const action: 'buy' | 'sell' | 'hold' = a
        ? (a.recommendation as 'buy' | 'sell' | 'hold')
        : stock.changePercent > 1 ? 'buy' : stock.changePercent < -1 ? 'sell' : 'hold';
      const conf = (a?.forecastConfidence ?? stock.forecastConfidence) * 100;
      const status: ForecastSetup['status'] = conf >= 70 ? 'confirmed' : conf >= 40 ? 'pending' : 'failed';
      return {
        symbol: stock.symbol,
        name: stock.name,
        action,
        predictedPrice: a?.predictedPrice ?? stock.price * (1 + stock.changePercent / 100),
        currentPrice: stock.price,
        changePercent: stock.changePercent,
        confidence: conf,
        sentiment: stock.sentiment,
        sector: stock.sector,
        status,
        timestamp: stock.lastUpdated || new Date().toISOString(),
      };
    });
  }, [topStocks, analyses]);

  /* ── Risk score ── */
  const riskScore = useMemo(() => {
    if (!stocks?.length) return 25;
    const hi = stocks.filter((s) => s.riskLevel === 'high').length;
    const md = stocks.filter((s) => s.riskLevel === 'medium').length;
    return Math.round(((hi * 3 + md * 1.5) / stocks.length) * 33);
  }, [stocks]);

  /* ── Featured card ── */
  const featuredSetup = setups.find((s) => s.action === 'buy' && s.confidence >= 70) ?? setups[0];

  /* ══════════════════════════════════════════
     INTERACTIVE CHART — fetch real historical
  ══════════════════════════════════════════ */
  const [historical, setHistorical] = useState<CandleData[]>([]);
  const [histLoading, setHistLoading] = useState(false);
  const histCacheRef = useRef<Map<string, CandleData[]>>(new Map());

  /* Fetch historical when selectedSymbol or timeframe changes */
  useEffect(() => {
    if (!selectedSymbol) return;
    const days = TIMEFRAMES_TO_DAYS[activeTimeframe] ?? 90;
    const cacheKey = `${selectedSymbol}_${days}`;

    if (histCacheRef.current.has(cacheKey)) {
      setHistorical(histCacheRef.current.get(cacheKey)!);
      return;
    }

    setHistLoading(true);
    api.getHistorical(selectedSymbol, '1day', days)
      .then((data) => {
        histCacheRef.current.set(cacheKey, data);
        setHistorical(data);
      })
      .catch(() => setHistorical([]))
      .finally(() => setHistLoading(false));
  }, [selectedSymbol, activeTimeframe]);

  /* Build recharts-ready data: historical + forecast extension */
  const chartData = useMemo(() => {
    if (!historical.length) return [];

    const analysis = selectedSymbol ? analyses.get(selectedSymbol) : null;
    const selectedStock = selectedSymbol ? stocks?.find((s) => s.symbol === selectedSymbol) : null;

    // Historical points
    const hist = historical.map((c) => ({
      date: new Date(c.timestamp).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' }),
      price: c.close,
      volume: c.volume,
      forecast: undefined as number | undefined,
      upper: undefined as number | undefined,
      lower: undefined as number | undefined,
    }));

    // Bridge: last historical point also starts the forecast
    if (hist.length > 0) {
      hist[hist.length - 1].forecast = hist[hist.length - 1].price;
    }

    // Forecast extension (5 synthetic points projected from predicted price)
    const lastPrice = historical[historical.length - 1]?.close ?? 0;
    const predictedPrice = analysis?.predictedPrice ?? (selectedStock ? selectedStock.price * (1 + selectedStock.changePercent / 200) : lastPrice);
    const confidence = analysis?.forecastConfidence ?? (selectedStock?.forecastConfidence ?? 0.5);
    const bandWidth = lastPrice * (1 - confidence) * 0.15;

    if (lastPrice > 0 && predictedPrice > 0) {
      const delta = (predictedPrice - lastPrice) / 5;
      const lastDate = new Date(historical[historical.length - 1].timestamp);
      for (let i = 1; i <= 5; i++) {
        const d = new Date(lastDate);
        d.setDate(d.getDate() + i * 7);
        const p = lastPrice + delta * i;
        hist.push({
          date: d.toLocaleDateString('en-IN', { day: 'numeric', month: 'short' }),
          price: undefined as unknown as number,
          volume: 0,
          forecast: p,
          upper: p + bandWidth * i * 0.5,
          lower: Math.max(0, p - bandWidth * i * 0.5),
        });
      }
    }

    return hist;
  }, [historical, selectedSymbol, analyses, stocks]);

  /* Selected stock's analysis for stat cards */
  const selectedAnalysis = selectedSymbol ? analyses.get(selectedSymbol) : null;
  const selectedStock = selectedSymbol ? stocks?.find((s) => s.symbol === selectedSymbol) : null;

  /* Handle card click — update chart instead of navigating */
  const handleSetupClick = useCallback((symbol: string) => {
    setSelectedSymbol(symbol);
  }, []);

  /* ════════════════════════════════
     RENDER
  ════════════════════════════════ */
  return (
    <PageTransition>
      <div className="w-full h-[calc(100vh-2rem)] grid grid-cols-12 gap-0 bg-surface-0 rounded-3xl overflow-hidden shadow-2xl relative">

        {/* ═══════════════════════════════
            LEFT — 8 cols, red hero
        ═══════════════════════════════ */}
        <div className="col-span-12 lg:col-span-8 bg-accent relative p-6 md:p-10 text-white flex flex-col overflow-hidden">
          {/* bg overlay */}
          <div className="absolute inset-0 opacity-10 pointer-events-none">
            <div className="absolute bottom-0 left-0 w-full h-1/2 bg-gradient-to-t from-black/20 to-transparent" />
          </div>

          {/* ─ Top actions ─ */}
          <div className="flex items-center justify-end mb-6 z-10">
            <div className="flex items-center gap-3">
              <button className="p-2 bg-white/10 hover:bg-white/20 rounded-lg backdrop-blur-sm transition">
                <Plus className="w-5 h-5" />
              </button>
              <button className="p-2 bg-white/10 hover:bg-white/20 rounded-lg backdrop-blur-sm transition">
                <SlidersHorizontal className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* ─ Projected Value ─ */}
          <div className="z-10 mb-6">
            <p className="text-white/70 text-sm font-medium mb-1">
              {selectedSymbol ? `${selectedSymbol.replace('.NS', '')} — ${selectedStock?.name ?? 'Loading...'}` : 'Projected Wallet Value'}
            </p>
            <div className="flex items-baseline gap-4 flex-wrap">
              {stocksLoading ? (
                <div className="h-14 w-64 bg-white/10 rounded-xl animate-pulse" />
              ) : (
                <>
                  <h2 className="text-5xl md:text-6xl font-bold tracking-tight">
                    {formatCurrency(selectedStock?.price ?? metrics?.totalValue ?? 0)}
                  </h2>
                  <div className="flex items-center gap-1 bg-black/20 px-3 py-1 rounded-full text-sm font-medium backdrop-blur-md">
                    {(selectedStock?.changePercent ?? metrics?.avgChange ?? 0) >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                    <span>{formatPercent(selectedStock?.changePercent ?? metrics?.avgChange ?? 0)}</span>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* ─ Timeframes ─ */}
          <div className="flex gap-2 mb-5 z-10 overflow-x-auto pb-1 no-scrollbar">
            {TIMEFRAMES.map((tf) => (
              <button
                key={tf}
                onClick={() => setActiveTimeframe(tf)}
                className={cn(
                  'px-4 py-1.5 rounded-full text-sm font-medium transition whitespace-nowrap',
                  activeTimeframe === tf
                    ? 'bg-white text-accent shadow-lg'
                    : 'bg-white/10 hover:bg-white/20 text-white/80',
                )}
              >
                {tf}
              </button>
            ))}
          </div>

          {/* ─ Stat cards — per-stock when selected ─ */}
          <motion.div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6 z-10" variants={stagger} initial="hidden" animate="show">
            <motion.div variants={fadeUp} className="p-4 rounded-xl bg-white/10 backdrop-blur-sm border border-white/10">
              <p className="text-white/60 text-xs mb-1">Realized PL</p>
              <p className="text-xl font-semibold">
                {selectedStock
                  ? `${selectedStock.change >= 0 ? '+' : ''}${formatCurrency(selectedStock.change)}`
                  : `+${formatCurrency(metrics?.realizedPL ?? 0)}`}
              </p>
              <p className="text-white/60 text-xs mt-1 flex items-center gap-1">
                <ArrowUpRight className="w-2.5 h-2.5" /> {formatPercent(selectedStock?.changePercent ?? 4.9)} Today
              </p>
            </motion.div>
            <motion.div variants={fadeUp} className="p-4 rounded-xl bg-white/10 backdrop-blur-sm border border-white/10">
              <p className="text-white/60 text-xs mb-1">Unrealized PL</p>
              <p className="text-xl font-semibold">
                {selectedAnalysis
                  ? `${(selectedAnalysis.predictedPrice - selectedAnalysis.currentPrice) >= 0 ? '+' : ''}${formatCurrency(selectedAnalysis.predictedPrice - selectedAnalysis.currentPrice)}`
                  : formatCurrency(metrics?.unrealizedPL ?? 0)}
              </p>
              <p className="text-white/60 text-xs mt-1 flex items-center gap-1">
                <ArrowDownRight className="w-2.5 h-2.5" /> Predicted delta
              </p>
            </motion.div>
            <motion.div variants={fadeUp} className="p-4 rounded-xl bg-white/10 backdrop-blur-sm border border-white/10">
              <p className="text-white/60 text-xs mb-1">ML Confidence</p>
              <p className="text-xl font-semibold">
                {selectedAnalysis
                  ? `${(selectedAnalysis.forecastConfidence * 100).toFixed(1)}%`
                  : analysesLoading ? '—' : `${((metrics?.avgConf ?? 0) * 100).toFixed(1)}%`}
              </p>
              <p className="text-white/60 text-xs mt-1 flex items-center gap-1">
                <CheckCircle className="w-2.5 h-2.5" />
                {(() => {
                  const c = selectedAnalysis?.forecastConfidence ?? metrics?.avgConf ?? 0;
                  return c >= 0.7 ? 'High' : c >= 0.4 ? 'Medium' : 'Low';
                })()}
              </p>
            </motion.div>
            <motion.div variants={fadeUp} className="p-4 rounded-xl bg-white/10 backdrop-blur-sm border border-white/10">
              <p className="text-white/60 text-xs mb-1">Projected Growth</p>
              <p className="text-xl font-semibold">
                {selectedAnalysis && selectedAnalysis.predictedPrice > 0
                  ? `+${formatCurrency(Math.max(0, selectedAnalysis.predictedPrice - selectedAnalysis.currentPrice))}`
                  : `+${formatCurrency(metrics?.projGrowth ?? 0)}`}
              </p>
              <p className="text-white/60 text-xs mt-1 flex items-center gap-1">
                <Activity className="w-2.5 h-2.5" />
                {selectedAnalysis?.recommendation
                  ? `Signal: ${selectedAnalysis.recommendation.toUpperCase()}`
                  : '2.9% Today'}
              </p>
            </motion.div>
          </motion.div>

          {/* ─ Main Chart (interactive recharts) ─ */}
          <div className="flex-grow relative z-10 w-full min-h-[250px]">
            {stocksLoading || histLoading ? (
              <div className="w-full h-full bg-white/5 rounded-xl animate-pulse flex items-center justify-center">
                <RefreshCw className="w-6 h-6 text-white/30 animate-spin" />
              </div>
            ) : chartData.length === 0 ? (
              <div className="w-full h-full flex items-center justify-center text-white/40 text-sm">
                Select a stock from the right panel
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="fcPriceFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="rgba(255,255,255,0.25)" />
                      <stop offset="100%" stopColor="rgba(255,255,255,0)" />
                    </linearGradient>
                    <linearGradient id="fcForecastFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="rgba(255,193,7,0.25)" />
                      <stop offset="100%" stopColor="rgba(255,193,7,0)" />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis
                    dataKey="date"
                    tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                    interval={Math.max(0, Math.floor(chartData.length / 8))}
                  />
                  <YAxis
                    tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                    width={55}
                    domain={['auto', 'auto']}
                    tickFormatter={(v: number) => `₹${v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(0)}`}
                  />
                  <RechartsTooltip
                    contentStyle={{ background: 'rgba(0,0,0,0.85)', border: '1px solid rgba(255,255,255,0.15)', borderRadius: 10, fontSize: 12 }}
                    itemStyle={{ color: '#fff' }}
                    labelStyle={{ color: 'rgba(255,255,255,0.7)', marginBottom: 4 }}
                    formatter={(value: number, name: string) => {
                      const labels: Record<string, string> = { price: 'Price', forecast: 'Forecast', upper: 'Upper Band', lower: 'Lower Band' };
                      return [`₹${value?.toFixed(2) ?? '—'}`, labels[name] ?? name];
                    }}
                  />
                  {/* Volume bars */}
                  <Bar dataKey="volume" fill="rgba(255,255,255,0.08)" radius={[2, 2, 0, 0]} yAxisId="vol" />
                  {/* Confidence bands */}
                  <Area type="monotone" dataKey="upper" stroke="none" fill="rgba(255,193,7,0.08)" connectNulls={false} dot={false} />
                  <Area type="monotone" dataKey="lower" stroke="none" fill="rgba(255,193,7,0.08)" connectNulls={false} dot={false} />
                  {/* Historical price line */}
                  <Area type="monotone" dataKey="price" stroke="white" strokeWidth={2.5} fill="url(#fcPriceFill)" connectNulls={false} dot={false} />
                  {/* Forecast line */}
                  <Area type="monotone" dataKey="forecast" stroke="#FFC107" strokeWidth={2.5} strokeDasharray="6 3" fill="url(#fcForecastFill)" connectNulls={false} dot={false} />
                  {/* Hidden volume axis */}
                  <YAxis yAxisId="vol" orientation="right" hide domain={[0, (dm: number) => dm * 5]} />
                </ComposedChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* ═══════════════════════════════
            RIGHT — 4 cols, dark panel
        ═══════════════════════════════ */}
        <div className="col-span-12 lg:col-span-4 bg-[#0A0A0A] text-white p-6 flex flex-col relative overflow-hidden">

          {/* ─ Top bar ─ */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3 bg-[#1E1E1E] p-2 pr-4 rounded-full border border-gray-800">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-orange-400 to-red-500 flex items-center justify-center">
                <BarChart3 className="w-4 h-4 text-white" />
              </div>
              <span className="text-sm font-mono text-gray-400 truncate w-24">StockSense</span>
            </div>
            <div className="flex gap-2">
              <button className="w-10 h-10 rounded-full border border-gray-800 flex items-center justify-center hover:bg-gray-800 transition">
                <Search className="w-4 h-4 text-gray-400" />
              </button>
              <button className="w-10 h-10 rounded-full border border-gray-800 flex items-center justify-center hover:bg-gray-800 transition relative">
                <Bell className="w-4 h-4 text-gray-400" />
                <span className="absolute top-2 right-2 w-2 h-2 bg-accent rounded-full" />
              </button>
            </div>
          </div>

          {/* ─ Portfolio Risk Score ─ */}
          <div className="mb-6">
            <div className="flex justify-between items-end mb-2">
              <h3 className="text-2xl font-bold leading-tight">Portfolio<br />Risk Score</h3>
              <div className="text-right">
                <p className="text-xs text-gray-500">Updated:</p>
                <p className="text-xs text-white">Just Now</p>
              </div>
            </div>
            <div className="h-12 w-full flex items-center gap-1">
              <div className="h-2 bg-accent rounded-l-full transition-all duration-700" style={{ width: `${Math.max(5, riskScore)}%` }} />
              <div className="h-full flex-grow flex items-center justify-between overflow-hidden">
                {Array.from({ length: 30 }).map((_, i) => (
                  <div key={i} className="w-[2px] h-3 bg-gray-800" />
                ))}
              </div>
            </div>
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>Low Risk</span>
              <span>High Risk</span>
            </div>
          </div>

          {/* ─ Header ─ */}
          <div className="flex justify-between items-center mb-4">
            <h4 className="font-semibold text-lg">High Probability Setups</h4>
            <button onClick={() => navigate('/screener')} className="text-xs text-gray-500 hover:text-white transition">View All</button>
          </div>

          {/* ─ Scrollable setup cards ─ */}
          <div className="flex-grow overflow-y-auto pr-2 space-y-3" style={{ scrollbarWidth: 'thin', scrollbarColor: 'rgba(156,163,175,0.5) transparent' }}>
            {stocksLoading
              ? Array.from({ length: 4 }).map((_, i) => (
                  <div key={i} className="bg-[#1E1E1E] p-4 rounded-xl border border-gray-800 h-32 animate-pulse" />
                ))
              : setups.map((setup, idx) => {
                  const isBuy = setup.action === 'buy';
                  const isFeatured = setup === featuredSetup;
                  const sparkPath = generateMiniSparkline(setup.symbol, isBuy);
                  const timeStr = new Date(setup.timestamp).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' });
                  const dateStr = new Date(setup.timestamp).toLocaleDateString('en-IN', { day: '2-digit', month: '2-digit', year: 'numeric' });
                  const stCfg = {
                    confirmed: { Icon: CheckCircle, text: 'Confirmed', color: 'text-green-400' },
                    pending: { Icon: RefreshCw, text: 'Pending', color: 'text-gray-400' },
                    failed: { Icon: XCircle, text: 'Failed', color: 'text-red-400' },
                  }[setup.status];

                  /* ── Featured (red) card ── */
                  if (isFeatured) {
                    return (
                      <motion.div
                        key={setup.symbol}
                        initial={{ opacity: 0, y: 8 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: idx * 0.05 }}
                        onClick={() => handleSetupClick(setup.symbol)}
                        className={cn(
                          'p-4 rounded-xl border shadow-lg group relative overflow-hidden cursor-pointer transition-all',
                          setup.symbol === selectedSymbol
                            ? 'bg-accent border-yellow-400 ring-2 ring-yellow-400/50'
                            : 'bg-accent border-red-600',
                        )}
                      >
                        <div className="absolute inset-0 bg-gradient-to-br from-white/10 to-transparent pointer-events-none" />
                        <div className="flex justify-between items-start mb-3 relative z-10">
                          <div className="flex gap-3">
                            <div className="relative">
                              <CompanyLogo symbol={setup.symbol} className="w-10 h-10 bg-white" />
                              <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-yellow-500 rounded-full flex items-center justify-center border border-red-500">
                                <Star className="w-2 h-2 text-white" />
                              </div>
                            </div>
                            <div>
                              <p className="text-sm font-bold text-white">
                                {isBuy ? 'Buy' : setup.action === 'sell' ? 'Sell' : 'Hold'} {setup.symbol.replace('.NS', '')}
                              </p>
                              <p className="text-xs text-white/70">{dateStr}</p>
                            </div>
                          </div>
                          <div className="text-right text-white">
                            <p className="text-xs text-white/70">{setup.name}</p>
                            <p className="text-xs font-mono">{isBuy ? '+' : ''}{formatPercent(setup.changePercent)}</p>
                          </div>
                        </div>
                        <div className="h-12 w-full relative overflow-hidden mb-2 z-10">
                          <svg className="w-full h-full" viewBox="0 0 320 50" preserveAspectRatio="none">
                            <defs>
                              <linearGradient id="featFill" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="rgba(255,255,255,0.2)" />
                                <stop offset="100%" stopColor="rgba(255,255,255,0)" />
                              </linearGradient>
                            </defs>
                            <path d={sparkPath} fill="none" stroke="white" strokeWidth="2" />
                            <path d={`${sparkPath} V50 H0 Z`} fill="url(#featFill)" />
                          </svg>
                        </div>
                        <div className="flex justify-between items-center relative z-10">
                          <div>
                            <p className="text-xs text-white/70">{setup.symbol.replace('.NS', '')}</p>
                            <p className="text-sm font-semibold text-white">{isBuy ? '+' : ''}{formatCurrency(Math.abs(setup.predictedPrice - setup.currentPrice))}</p>
                          </div>
                          <div className="flex items-center gap-1 text-xs text-white">
                            <stCfg.Icon className="w-3 h-3" />
                            <span>{stCfg.text}</span>
                          </div>
                        </div>
                      </motion.div>
                    );
                  }

                  /* ── Regular card ── */
                  return (
                    <motion.div
                      key={setup.symbol}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: idx * 0.05 }}
                      onClick={() => handleSetupClick(setup.symbol)}
                      className={cn(
                        'bg-[#1E1E1E] p-4 rounded-xl border transition group cursor-pointer',
                        setup.symbol === selectedSymbol
                          ? 'border-accent ring-2 ring-accent/40 bg-accent/10'
                          : 'border-gray-800 hover:border-gray-700',
                      )}
                    >
                      <div className="flex justify-between items-start mb-3">
                        <div className="flex gap-3">
                          <div className="relative">
                            <CompanyLogo symbol={setup.symbol} className="w-10 h-10" />
                            <div className={cn(
                              'absolute -bottom-1 -right-1 w-4 h-4 rounded-full flex items-center justify-center border border-[#1E1E1E]',
                              isBuy ? 'bg-blue-500' : setup.action === 'sell' ? 'bg-red-500' : 'bg-gray-600',
                            )}>
                              {isBuy
                                ? <ArrowDownRight className="w-2.5 h-2.5 text-white rotate-180" />
                                : setup.action === 'sell'
                                  ? <ArrowUpRight className="w-2.5 h-2.5 text-white rotate-180" />
                                  : <Activity className="w-2.5 h-2.5 text-white" />}
                            </div>
                          </div>
                          <div>
                            <p className="text-sm font-bold">
                              {isBuy ? 'Buy' : setup.action === 'sell' ? 'Sell' : 'Hold'} {setup.symbol.replace('.NS', '')}
                            </p>
                            <p className="text-xs text-gray-500">{timeStr}</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="text-xs text-gray-400">{setup.name}</p>
                          <p className={cn('text-xs font-mono',
                            isBuy ? 'text-green-400' : setup.action === 'sell' ? 'text-red-400' : 'text-white',
                          )}>
                            {isBuy ? '+' : ''}{formatPercent(setup.changePercent)}
                          </p>
                        </div>
                      </div>
                      {/* sparkline + bars */}
                      <div className="h-10 w-full relative overflow-hidden">
                        <svg className="w-full h-full" viewBox="0 0 320 50" preserveAspectRatio="none">
                          <path
                            d={sparkPath}
                            fill="none"
                            stroke={isBuy ? '#4ADE80' : setup.action === 'sell' ? '#F87171' : '#607D8B'}
                            strokeWidth="2"
                            className="opacity-50 group-hover:opacity-100 transition-opacity"
                          />
                          {barHeights(setup.symbol, 20).map((h, i) => (
                            <rect key={i} x={i * 16 + 5} y={50 - h * 40} width="2" height={h * 40} fill="#333" />
                          ))}
                        </svg>
                      </div>
                      {/* footer */}
                      <div className="flex justify-between items-center mt-2">
                        <div>
                          <p className="text-xs text-gray-500">{setup.symbol.replace('.NS', '')}</p>
                          <p className="text-sm font-semibold">{isBuy ? '+' : ''}{formatCurrency(Math.abs(setup.predictedPrice - setup.currentPrice))}</p>
                        </div>
                        <div className={cn('flex items-center gap-1 text-xs', stCfg.color)}>
                          <stCfg.Icon className={cn('w-3 h-3', setup.status === 'pending' && 'animate-spin')} />
                          <span>{stCfg.text}</span>
                        </div>
                      </div>
                    </motion.div>
                  );
                })}
          </div>
        </div>
      </div>
    </PageTransition>
  );
}
