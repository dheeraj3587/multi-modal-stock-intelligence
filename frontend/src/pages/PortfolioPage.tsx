import { useMemo, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Plus,
  RotateCcw,
} from 'lucide-react';
import { usePaperPortfolio } from '../hooks/usePaperPortfolio';
import { useUpstoxFeed } from '../hooks/useUpstoxFeed';
import { useApi } from '../hooks/useApi';
import { api } from '../lib/api';
import { CompanyLogo } from '../components/shared/CompanyLogo';
import { Skeleton } from '../components/ui/Skeleton';
import { cn, formatCurrency, formatPercent } from '../lib/utils';

/* ─── risk helpers ─── */

function riskLevel(pnlPct: number): number {
  const abs = Math.abs(pnlPct);
  if (abs > 8) return 4;
  if (abs > 4) return 3;
  if (abs > 1.5) return 2;
  return 1;
}

function RiskBars({ level }: { level: number }) {
  return (
    <div className="flex space-x-1">
      {[1, 2, 3, 4].map((i) => (
        <div
          key={i}
          className={cn(
            'w-1.5 h-3 rounded-sm',
            i <= level ? 'bg-[#E13838]' : 'bg-gray-200 dark:bg-gray-700',
          )}
        />
      ))}
    </div>
  );
}

/* ─── mini sparkline (deterministic from symbol hash) ─── */

function MiniSparkline({ positive, flat }: { positive: boolean; flat?: boolean }) {
  if (flat) {
    return (
      <svg className="w-full h-full overflow-visible" viewBox="0 0 100 40">
        <path d="M0,20 H100" fill="none" stroke="#9ca3af" strokeDasharray="4 2" strokeWidth="2" />
      </svg>
    );
  }
  const color = positive ? '#22c55e' : '#ef4444';
  const path = positive
    ? 'M0,35 L10,32 L20,36 L30,25 L40,28 L50,15 L60,20 L70,10 L80,15 L90,5 L100,10'
    : 'M0,20 L10,25 L20,15 L30,20 L40,30 L50,25 L60,35 L70,30 L80,38 L90,35 L100,40';
  const fill = positive
    ? 'M0,35 L10,32 L20,36 L30,25 L40,28 L50,15 L60,20 L70,10 L80,15 L90,5 L100,10 V40 H0 Z'
    : 'M0,20 L10,25 L20,15 L30,20 L40,30 L50,25 L60,35 L70,30 L80,38 L90,35 L100,40 V40 H0 Z';

  return (
    <svg className="w-full h-full overflow-visible" viewBox="0 0 100 40">
      <path d={path} fill="none" stroke={color} strokeWidth="2" />
      <path d={fill} fill={color} fillOpacity="0.1" stroke="none" />
    </svg>
  );
}

/* ─── icon map (sector → lucide-like colour) ─── */

const SECTOR_ICON_BG: Record<string, string> = {
  Energy: 'bg-orange-100 dark:bg-orange-900/30 text-orange-500',
  IT: 'bg-blue-100 dark:bg-blue-900/30 text-blue-500',
  Banking: 'bg-purple-100 dark:bg-purple-900/30 text-purple-500',
  Finance: 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-500',
  FMCG: 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-600',
  Pharma: 'bg-pink-100 dark:bg-pink-900/30 text-pink-500',
  Auto: 'bg-red-100 dark:bg-red-900/30 text-red-500',
  Metals: 'bg-gray-200 dark:bg-gray-800 text-gray-500',
  Telecom: 'bg-cyan-100 dark:bg-cyan-900/30 text-cyan-500',
};

function sectorBg(sector: string) {
  return SECTOR_ICON_BG[sector] ?? 'bg-gray-100 dark:bg-gray-800 text-gray-500';
}

/* ═══════════════════════════════════════════════════════════ */
/* Component                                                   */
/* ═══════════════════════════════════════════════════════════ */

export function PortfolioPage() {
  const navigate = useNavigate();
  const { holdings, reset } = usePaperPortfolio();

  /* ── prices ── */
  const instrumentKeys = useMemo(() => holdings.map((h) => h.instrumentKey), [holdings]);
  const { prices: livePrices } = useUpstoxFeed(instrumentKeys);

  const symbolsCSV = useMemo(() => holdings.map((h) => h.symbol).join(','), [holdings]);
  const { data: restQuotes, loading } = useApi(
    () => (symbolsCSV ? api.getQuotes(symbolsCSV) : Promise.resolve([])),
    [symbolsCSV],
    { refetchInterval: 30_000 },
  );

  const currentPrices = useMemo(() => {
    const map: Record<string, { price: number; changePercent: number }> = {};
    restQuotes?.forEach((q) => {
      if (q.price > 0) map[q.symbol] = { price: q.price, changePercent: q.changePercent };
    });
    holdings.forEach((h) => {
      const wsPrice = livePrices[h.instrumentKey];
      if (wsPrice && wsPrice > 0) {
        const existing = map[h.symbol];
        map[h.symbol] = { price: wsPrice, changePercent: existing?.changePercent ?? 0 };
      }
    });
    return map;
  }, [holdings, livePrices, restQuotes]);

  /* ── enriched holdings ── */
  const enriched = useMemo(
    () =>
      holdings.map((h) => {
        const info = currentPrices[h.symbol];
        const ltp = info?.price ?? h.avgPrice;
        const changePct = info?.changePercent ?? 0;
        const value = ltp * h.quantity;
        const invested = h.avgPrice * h.quantity;
        const pnl = value - invested;
        const pnlPercent = invested > 0 ? (pnl / invested) * 100 : 0;
        return { ...h, ltp, changePct, value, invested, pnl, pnlPercent };
      }),
    [holdings, currentPrices],
  );

  const totalValue = enriched.reduce((s, h) => s + h.value, 0);
  const totalInvested = enriched.reduce((s, h) => s + h.invested, 0);
  const totalPnl = totalValue - totalInvested;
  const totalPnlPercent = totalInvested > 0 ? (totalPnl / totalInvested) * 100 : 0;

  /* daily P&L: approximate from changePct × value */
  const dailyPnl = enriched.reduce((s, h) => {
    const dayChange = (h.changePct / 100) * h.value;
    return s + dayChange;
  }, 0);
  const dailyPnlPct = totalValue > 0 ? (dailyPnl / totalValue) * 100 : 0;

  /* risk score — % of portfolio in "risky" positions (high volatility) */
  const highRiskPct = useMemo(() => {
    const highRisk = enriched.filter((h) => riskLevel(h.changePct) >= 3);
    const hrValue = highRisk.reduce((s, h) => s + h.value, 0);
    return totalValue > 0 ? Math.round((hrValue / totalValue) * 100) : 0;
  }, [enriched, totalValue]);

  /* split: first 4 are asset cards, 5th+ go into bottom row */
  const topAssets = enriched.slice(0, 4);
  const extraAssets = enriched.slice(4);

  const handleReset = useCallback(() => {
    if (window.confirm('Reset paper portfolio to default seed holdings?')) reset();
  }, [reset]);

  /* ── loading ── */
  if (loading && !restQuotes) {
    return (
      <div className="space-y-8">
        <Skeleton className="h-10 w-64 rounded-lg" />
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          <Skeleton className="lg:col-span-5 h-[520px] rounded-3xl" />
          <Skeleton className="lg:col-span-7 h-[520px] rounded-3xl" />
        </div>
      </div>
    );
  }

  /* ═══════ render ═══════ */
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="font-sans text-gray-900 dark:text-gray-100 min-h-screen"
    >
      {/* Header */}
      <header className="mb-8 flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">Portfolio Holdings</h1>
          <p className="text-gray-500 dark:text-gray-400">Manage your assets and analyze risk distribution.</p>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={() => navigate('/trading')}
            className="flex items-center justify-center w-10 h-10 rounded-xl border border-gray-200 dark:border-gray-800 text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          >
            <Plus className="w-4 h-4" />
          </button>
          <button
            onClick={handleReset}
            className="flex items-center justify-center w-10 h-10 rounded-xl border border-gray-200 dark:border-gray-800 text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            title="Reset portfolio"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* Main grid: left 5 / right 7 */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* ─── LEFT COLUMN ─── */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          {/* Portfolio Value Card */}
          <div className="bg-[#E13838] text-white rounded-3xl p-8 relative overflow-hidden shadow-[0_0_20px_rgba(225,56,56,0.3)] h-full flex flex-col justify-between min-h-[500px]">
            {/* Background bars decoration */}
            <div className="absolute inset-0 opacity-10 pointer-events-none">
              <div className="h-full w-full flex items-end justify-around px-2">
                {[12, 24, 16, 32, 8, 20, 40, 28, 10, 36, 14, 24].map((h, i) => (
                  <div key={i} className="w-1 bg-white rounded-t" style={{ height: `${h * 2.5}px` }} />
                ))}
              </div>
            </div>

            <div className="relative z-10">
              <div className="flex justify-between items-start mb-6">
                <div>
                  <h3 className="text-white/80 font-medium mb-1">Total Portfolio Value</h3>
                  <div className="text-5xl font-bold tracking-tight mb-2">
                    {formatCurrency(totalValue)}
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="flex items-center bg-white/20 px-2 py-1 rounded-lg text-sm font-medium">
                      {totalPnlPercent >= 0 ? (
                        <TrendingUp className="w-4 h-4 mr-1" />
                      ) : (
                        <TrendingDown className="w-4 h-4 mr-1" />
                      )}
                      {formatPercent(totalPnlPercent)}
                    </span>
                    <span className="text-white/60 text-sm">vs invested</span>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 mt-8">
                <div className="bg-white/10 rounded-xl p-4 backdrop-blur-sm">
                  <p className="text-white/70 text-sm mb-1">Daily P&L</p>
                  <p className="text-2xl font-bold">
                    {dailyPnl >= 0 ? '+' : ''}{formatCurrency(dailyPnl)}
                  </p>
                  <p className="text-white/50 text-xs mt-1">
                    {dailyPnlPct >= 0 ? '+' : ''}{dailyPnlPct.toFixed(1)}% Today
                  </p>
                </div>
                <div className="bg-white/10 rounded-xl p-4 backdrop-blur-sm">
                  <p className="text-white/70 text-sm mb-1">Unrealized P&L</p>
                  <p className="text-2xl font-bold">
                    {totalPnl >= 0 ? '+' : ''}{formatCurrency(totalPnl)}
                  </p>
                  <p className="text-white/50 text-xs mt-1">Since Inception</p>
                </div>
              </div>
            </div>

            {/* Chart decoration */}
            <div className="relative z-10 mt-auto h-48 w-full">
              <svg className="w-full h-full overflow-visible drop-shadow-md" viewBox="0 0 400 150">
                <defs>
                  <linearGradient id="lineGradient" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0%" stopColor="rgba(255,255,255,0.8)" />
                    <stop offset="100%" stopColor="rgba(255,255,255,0)" />
                  </linearGradient>
                </defs>
                <g className="opacity-30" fill="white">
                  {[
                    { x: 10, h: 70 }, { x: 30, h: 90 }, { x: 50, h: 50 }, { x: 70, h: 110 },
                    { x: 90, h: 80 }, { x: 110, h: 60 }, { x: 130, h: 100 }, { x: 150, h: 120 },
                    { x: 170, h: 90 }, { x: 190, h: 70 }, { x: 210, h: 105 }, { x: 230, h: 85 },
                    { x: 250, h: 130 }, { x: 270, h: 95 }, { x: 290, h: 75 }, { x: 310, h: 110 },
                    { x: 330, h: 90 }, { x: 350, h: 65 }, { x: 370, h: 120 }, { x: 390, h: 100 },
                  ].map((bar) => (
                    <rect key={bar.x} x={bar.x} y={150 - bar.h} width={4} height={bar.h} rx={2} />
                  ))}
                </g>
                <path
                  d="M0,120 Q50,110 80,90 T150,80 T220,50 T300,60 T400,20"
                  fill="none"
                  stroke="#FFD740"
                  strokeWidth="3"
                />
                <circle cx="220" cy="50" r="4" fill="white" stroke="rgba(255,255,255,0.5)" strokeWidth="4" />
                <line x1="220" x2="220" y1="50" y2="150" stroke="white" strokeDasharray="4 2" strokeWidth="1" className="opacity-60" />
              </svg>
              <div className="absolute top-[20%] left-[55%] bg-white text-[#E13838] px-3 py-1 rounded-lg text-xs font-bold shadow-lg transform -translate-x-1/2">
                Highest Value
              </div>
            </div>
          </div>

          {/* Portfolio Risk Score */}
          <div className="bg-surface-light dark:bg-surface-dark rounded-3xl p-6 shadow-sm border border-gray-100 dark:border-gray-800">
            <div className="flex justify-between items-center mb-4">
              <h4 className="font-semibold text-gray-900 dark:text-white">Portfolio Risk Score</h4>
              <span className="text-xs text-gray-400">Updated: Just Now</span>
            </div>

            {/* Risk bar */}
            <div className="relative pt-2 pb-4">
              <div className="h-2 bg-gray-200 dark:bg-gray-800 rounded-full overflow-hidden flex">
                <div
                  className="bg-[#E13838] h-full transition-all duration-500"
                  style={{ width: `${Math.min(100, highRiskPct)}%` }}
                />
                <div className="flex-1 relative">
                  <div className="absolute inset-0 flex justify-between px-1">
                    {Array.from({ length: 6 }).map((_, i) => (
                      <div key={i} className="w-px h-full bg-gray-300 dark:bg-gray-700" />
                    ))}
                  </div>
                </div>
              </div>
              <div className="flex justify-between text-xs text-gray-400 mt-2 font-medium">
                <span>Low Risk</span>
                <span>High Risk</span>
              </div>
            </div>

            {/* High-risk stat */}
            <div className="flex items-center gap-4 mt-2">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-2 h-2 rounded-full bg-[#E13838]" />
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">High Risk Assets</span>
                </div>
                <p className="text-xs text-gray-500">Within recommended norm</p>
              </div>
              <div className="text-2xl font-bold text-gray-900 dark:text-white">{highRiskPct}%</div>
              <div className="w-10 h-10 relative">
                <svg className="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
                  <path
                    className="text-gray-200 dark:text-gray-800"
                    d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="text-[#E13838]"
                    d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    fill="none"
                    stroke="currentColor"
                    strokeDasharray={`${highRiskPct}, 100`}
                    strokeWidth="4"
                  />
                </svg>
              </div>
            </div>
          </div>
        </div>

        {/* ─── RIGHT COLUMN ─── */}
        <div className="lg:col-span-7">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white">Your Assets</h2>
            <button
              onClick={() => navigate('/trading')}
              className="text-sm font-medium text-[#E13838] hover:text-red-500 transition-colors"
            >
              View All
            </button>
          </div>

          {enriched.length === 0 ? (
            <div className="text-center py-16 text-gray-500 dark:text-gray-400">
              No holdings yet. Buy stocks from the Trading page.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Top 4 asset cards */}
              {topAssets.map((h) => {
                const isPositive = h.changePct > 0;
                const isFlat = h.changePct === 0;

                return (
                  <motion.div
                    key={h.symbol}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.2 }}
                    onClick={() => navigate(`/stock/${h.symbol}`)}
                    className="bg-surface-light dark:bg-surface-dark p-5 rounded-3xl border border-gray-100 dark:border-gray-800 hover:border-[#E13838]/50 dark:hover:border-[#E13838]/50 transition-all cursor-pointer group"
                  >
                    {/* Header: logo + name + change */}
                    <div className="flex justify-between items-start mb-4">
                      <div className="flex items-center space-x-3">
                        <div className={cn('w-10 h-10 rounded-full flex items-center justify-center overflow-hidden', sectorBg(h.sector))}>
                          <CompanyLogo symbol={h.symbol} className="w-8 h-8" chars={2} />
                        </div>
                        <div>
                          <h4 className="font-bold text-gray-900 dark:text-white group-hover:text-[#E13838] transition-colors">
                            {h.name}
                          </h4>
                          <span className="text-xs text-gray-500 uppercase">{h.symbol}</span>
                        </div>
                      </div>
                      <div className="text-right">
                        <p
                          className={cn(
                            'text-sm font-medium flex items-center justify-end',
                            isPositive ? 'text-green-500' : isFlat ? 'text-gray-500' : 'text-red-500',
                          )}
                        >
                          {isPositive ? (
                            <TrendingUp className="w-3 h-3 mr-0.5" />
                          ) : isFlat ? (
                            <Minus className="w-3 h-3 mr-0.5" />
                          ) : (
                            <TrendingDown className="w-3 h-3 mr-0.5" />
                          )}
                          {Math.abs(h.changePct).toFixed(1)}%
                        </p>
                        <p className="text-xs text-gray-400">24h</p>
                      </div>
                    </div>

                    {/* Holdings amount */}
                    <div className="mb-4">
                      <p className="text-2xl font-bold text-gray-900 dark:text-white">
                        {h.quantity} {h.symbol}
                      </p>
                      <p className="text-sm text-gray-500">≈ {formatCurrency(h.value)}</p>
                    </div>

                    {/* Sparkline + Risk */}
                    <div className="flex items-end justify-between">
                      <div className="h-10 w-24">
                        <MiniSparkline positive={isPositive} flat={isFlat} />
                      </div>
                      <div className="flex flex-col items-end">
                        <span className="text-[10px] text-gray-400 mb-1">Risk Level</span>
                        <RiskBars level={riskLevel(h.changePct)} />
                      </div>
                    </div>
                  </motion.div>
                );
              })}

              {/* Extra assets — wide row cards */}
              {extraAssets.map((h) => (
                <motion.div
                  key={h.symbol}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.2 }}
                  onClick={() => navigate(`/stock/${h.symbol}`)}
                  className="bg-surface-light dark:bg-surface-dark p-5 rounded-3xl border border-gray-100 dark:border-gray-800 hover:border-[#E13838]/50 dark:hover:border-[#E13838]/50 transition-all cursor-pointer group md:col-span-2 flex flex-col md:flex-row md:items-center justify-between"
                >
                  <div className="flex items-center space-x-4 mb-4 md:mb-0">
                    <div className={cn('w-12 h-12 rounded-full flex items-center justify-center overflow-hidden', sectorBg(h.sector))}>
                      <CompanyLogo symbol={h.symbol} className="w-9 h-9" chars={2} />
                    </div>
                    <div>
                      <h4 className="font-bold text-gray-900 dark:text-white text-lg group-hover:text-[#E13838] transition-colors">
                        {h.name}
                      </h4>
                      <span className="text-xs text-gray-500 uppercase">{h.symbol}</span>
                    </div>
                  </div>
                  <div className="flex items-center space-x-8">
                    <div>
                      <p className="text-xs text-gray-400 mb-1">Holdings</p>
                      <p className="font-bold text-gray-900 dark:text-white text-lg">
                        {h.quantity} {h.symbol}
                      </p>
                    </div>
                    <div className="hidden sm:block">
                      <p className="text-xs text-gray-400 mb-1">Value</p>
                      <p className="font-bold text-gray-900 dark:text-white text-lg">{formatCurrency(h.value)}</p>
                    </div>
                    <div className="hidden sm:block">
                      <p className="text-xs text-gray-400 mb-1">24h Change</p>
                      <p
                        className={cn(
                          'font-bold text-lg',
                          h.changePct >= 0 ? 'text-green-500' : 'text-red-500',
                        )}
                      >
                        {formatPercent(h.changePct)}
                      </p>
                    </div>
                    <div className="flex flex-col items-end pl-4 border-l border-gray-200 dark:border-gray-700">
                      <span className="text-[10px] text-gray-400 mb-1">Risk</span>
                      <RiskBars level={riskLevel(h.changePct)} />
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
