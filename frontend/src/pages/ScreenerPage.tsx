import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Search,
  SlidersHorizontal,
  TrendingUp,
  TrendingDown,
  Minus,
  LayoutGrid,
  List,
  ChevronLeft,
  ChevronRight,
  Shield,
  ArrowRight,
} from 'lucide-react';
import { ResponsiveContainer, AreaChart, Area, Tooltip as RechartsTooltip } from 'recharts';
import { PageTransition } from '../components/shared/PageTransition';
import { CompanyLogo } from '../components/shared/CompanyLogo';
import { api, type StockWithSentiment } from '../lib/api';
import { useApi } from '../hooks/useApi';
import { cn, formatCurrency, formatPercent } from '../lib/utils';

/* ── Constants ── */
const PAGE_SIZE = 6;

const SECTORS = [
  'IT', 'Banking', 'Finance', 'Insurance', 'Energy', 'Power',
  'FMCG', 'Consumer', 'Auto', 'Pharma', 'Healthcare', 'Metals',
  'Telecom', 'Cement', 'Infrastructure', 'Conglomerate', 'Mining',
];

type SortOption = 'risk-desc' | 'gainers' | 'losers' | 'price-desc' | 'price-asc' | 'name';
type ViewMode = 'grid' | 'list';

/* ── Risk badge helper ── */
function riskBadge(level: string) {
  switch (level) {
    case 'high':
      return { label: 'Risk: High', bg: 'bg-negative/10', border: 'border-negative/30', text: 'text-negative' };
    case 'medium':
      return { label: 'Risk: Med', bg: 'bg-warning/10', border: 'border-warning/30', text: 'text-warning' };
    case 'low':
      return { label: 'Risk: Low', bg: 'bg-positive/10', border: 'border-positive/30', text: 'text-positive' };
    default:
      return { label: 'Risk: —', bg: 'bg-surface-2', border: 'border-border', text: 'text-text-secondary' };
  }
}

/* ── Generate realistic-looking sparkline data from symbol + price + change ── */
function generateSparklineData(symbol: string, price: number, changePercent: number): { value: number }[] {
  let hash = 0;
  for (let i = 0; i < symbol.length; i++) hash = symbol.charCodeAt(i) + ((hash << 5) - hash);

  const points = 30;
  const data: { value: number }[] = [];
  const basePrice = price > 0 ? price / (1 + changePercent / 100) : 100;
  const amplitude = basePrice * 0.02; // 2% noise

  for (let i = 0; i < points; i++) {
    const progress = i / (points - 1);
    const trendComponent = basePrice + (price - basePrice) * progress;
    const noise = Math.sin(hash + i * 1.7) * amplitude + Math.cos(hash * 0.3 + i * 2.3) * amplitude * 0.6;
    data.push({ value: Math.max(0, trendComponent + noise) });
  }
  return data;
}

/* ════════════════════════════════════════
   Main Component
════════════════════════════════════════ */
export function ScreenerPage() {
  const navigate = useNavigate();
  const { data: stocks, loading } = useApi(() => api.getStocksWithSentiment(), [], { refetchInterval: 30_000 });

  /* ── Filter state ── */
  const [search, setSearch] = useState('');
  const [riskFilters, setRiskFilters] = useState<Set<string>>(new Set());
  const [sectorFilters, setSectorFilters] = useState<Set<string>>(new Set());
  const [sortBy, setSortBy] = useState<SortOption>('risk-desc');
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [page, setPage] = useState(1);

  /* ── Toggle helpers ── */
  const toggleRisk = (level: string) => {
    setRiskFilters((prev) => {
      const next = new Set(prev);
      next.has(level) ? next.delete(level) : next.add(level);
      return next;
    });
    setPage(1);
  };
  const toggleSector = (sector: string) => {
    setSectorFilters((prev) => {
      const next = new Set(prev);
      next.has(sector) ? next.delete(sector) : next.add(sector);
      return next;
    });
    setPage(1);
  };

  /* ── Unique sectors from data ── */
  const availableSectors = useMemo(() => {
    if (!stocks) return SECTORS;
    const present = new Set(stocks.map((s) => s.sector));
    return SECTORS.filter((s) => present.has(s));
  }, [stocks]);

  /* ── Filter + sort ── */
  const filtered = useMemo(() => {
    if (!stocks) return [];
    let result = [...stocks];

    if (search) {
      const q = search.toLowerCase();
      result = result.filter((s) => s.symbol.toLowerCase().includes(q) || s.name.toLowerCase().includes(q));
    }
    if (riskFilters.size > 0) result = result.filter((s) => riskFilters.has(s.riskLevel));
    if (sectorFilters.size > 0) result = result.filter((s) => sectorFilters.has(s.sector));

    switch (sortBy) {
      case 'risk-desc': {
        const riskOrder: Record<string, number> = { high: 0, medium: 1, low: 2 };
        result.sort((a, b) => (riskOrder[a.riskLevel] ?? 1) - (riskOrder[b.riskLevel] ?? 1));
        break;
      }
      case 'gainers': result.sort((a, b) => b.changePercent - a.changePercent); break;
      case 'losers': result.sort((a, b) => a.changePercent - b.changePercent); break;
      case 'price-desc': result.sort((a, b) => b.price - a.price); break;
      case 'price-asc': result.sort((a, b) => a.price - b.price); break;
      case 'name': result.sort((a, b) => a.name.localeCompare(b.name)); break;
    }
    return result;
  }, [stocks, search, riskFilters, sectorFilters, sortBy]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const paginated = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  const activeFilterCount = riskFilters.size + sectorFilters.size;

  const applyFilters = () => setPage(1);

  const clearFilters = () => {
    setRiskFilters(new Set());
    setSectorFilters(new Set());
    setSearch('');
    setPage(1);
  };

  return (
    <PageTransition>
      <div className="flex flex-col md:flex-row gap-0 min-h-[calc(100vh-80px)] -m-6 lg:-m-8 2xl:-m-10">

        {/* ════════════════════════════════════════
            LEFT SIDEBAR — Filters
        ════════════════════════════════════════ */}
        <aside className="w-full md:w-64 lg:w-72 shrink-0 p-6 border-r border-border bg-surface-0">
          <div className="sticky top-6 space-y-8">
            <div>
              <h2 className="text-lg font-bold mb-5 flex items-center gap-2 text-text-primary">
                <SlidersHorizontal className="w-5 h-5 text-accent" /> Filters
                {activeFilterCount > 0 && (
                  <span className="ml-auto text-xs bg-accent/10 text-accent px-2 py-0.5 rounded-full font-medium">
                    {activeFilterCount}
                  </span>
                )}
              </h2>

              {/* Risk Level */}
              <div className="mb-6">
                <label className="text-[11px] font-semibold uppercase tracking-wider text-text-tertiary mb-3 block">
                  Risk Level
                </label>
                <div className="space-y-2">
                  {(['low', 'medium', 'high'] as const).map((level) => (
                    <label key={level} className="flex items-center gap-3 cursor-pointer group">
                      <input
                        type="checkbox"
                        checked={riskFilters.has(level)}
                        onChange={() => toggleRisk(level)}
                        className="w-4 h-4 rounded bg-surface-2 border-border text-accent focus:ring-accent focus:ring-offset-0"
                      />
                      <span className={cn(
                        'text-sm transition-colors group-hover:text-accent',
                        riskFilters.has(level) ? 'text-accent font-medium' : 'text-text-secondary'
                      )}>
                        {level === 'low' ? 'Low Risk' : level === 'medium' ? 'Medium Risk' : 'High Risk'}
                      </span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Sector */}
              <div className="mb-6">
                <label className="text-[11px] font-semibold uppercase tracking-wider text-text-tertiary mb-3 block">
                  Sector
                </label>
                <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
                  {availableSectors.map((sec) => (
                    <label key={sec} className="flex items-center gap-3 cursor-pointer group">
                      <input
                        type="checkbox"
                        checked={sectorFilters.has(sec)}
                        onChange={() => toggleSector(sec)}
                        className="w-4 h-4 rounded bg-surface-2 border-border text-accent focus:ring-accent focus:ring-offset-0"
                      />
                      <span className={cn(
                        'text-sm transition-colors group-hover:text-accent',
                        sectorFilters.has(sec) ? 'text-accent font-medium' : 'text-text-secondary'
                      )}>
                        {sec}
                      </span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Apply / Clear */}
              <div className="space-y-2">
                <button
                  onClick={applyFilters}
                  className="w-full py-3 bg-accent hover:bg-accent-hover text-white font-semibold rounded-xl shadow-lg shadow-accent/20 transition-all active:scale-[0.98]"
                >
                  Apply Filters
                </button>
                {activeFilterCount > 0 && (
                  <button
                    onClick={clearFilters}
                    className="w-full py-2 text-xs text-accent hover:underline font-medium"
                  >
                    Clear all filters
                  </button>
                )}
              </div>
            </div>

            {/* Upgrade Card */}
            <div className="p-5 rounded-2xl bg-surface-1 border border-border-subtle relative overflow-hidden">
              <div className="absolute -right-3 -bottom-3 opacity-5">
                <Shield className="w-20 h-20" />
              </div>
              <div className="relative z-10">
                <h3 className="text-text-primary font-bold mb-1 text-sm">Upgrade to Pro</h3>
                <p className="text-xs text-text-tertiary mb-3">Get advanced risk metrics and real-time alerts.</p>
                <button className="text-xs font-bold text-accent flex items-center gap-1 hover:underline">
                  Learn more <ArrowRight className="w-3 h-3" />
                </button>
              </div>
            </div>
          </div>
        </aside>

        {/* ════════════════════════════════════════
            MAIN CONTENT
        ════════════════════════════════════════ */}
        <main className="flex-1 p-6 lg:p-8">
          {/* Header */}
          <header className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
            <div>
              <h1 className="text-3xl font-bold text-text-primary mb-1">Stock Screener</h1>
              <p className="text-text-tertiary text-sm">
                Real-time analysis of{' '}
                <span className="text-accent font-semibold">
                  {stocks ? stocks.length.toLocaleString() : '—'}
                </span>{' '}
                active assets
              </p>
            </div>
            <div className="flex items-center gap-3">
              {/* Search */}
              <div className="relative hidden sm:block">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-tertiary" />
                <input
                  type="text"
                  value={search}
                  onChange={(e) => { setSearch(e.target.value); setPage(1); }}
                  placeholder="Search symbol..."
                  className="w-56 pl-10 pr-4 py-2.5 bg-surface-2 border-none rounded-full text-sm text-text-primary placeholder:text-text-tertiary focus:ring-2 focus:ring-accent focus:outline-none transition-all"
                />
              </div>
              {/* Sort */}
              <div className="flex items-center gap-2 bg-surface-0 border border-border-subtle rounded-xl px-3 py-2">
                <span className="text-xs text-text-tertiary whitespace-nowrap">Sort by:</span>
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as SortOption)}
                  className="bg-transparent border-none p-0 text-sm font-medium text-text-primary focus:ring-0 cursor-pointer"
                >
                  <option value="risk-desc">Risk Score (High to Low)</option>
                  <option value="gainers">Top Gainers</option>
                  <option value="losers">Top Losers</option>
                  <option value="price-desc">Price (High to Low)</option>
                  <option value="price-asc">Price (Low to High)</option>
                  <option value="name">Name (A-Z)</option>
                </select>
              </div>
              {/* View toggle */}
              <button
                onClick={() => setViewMode('grid')}
                className={cn(
                  'p-2 rounded-xl border transition-colors',
                  viewMode === 'grid'
                    ? 'bg-accent/10 border-accent/30 text-accent'
                    : 'bg-surface-0 border-border-subtle text-text-tertiary hover:border-accent hover:text-accent'
                )}
              >
                <LayoutGrid className="w-5 h-5" />
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={cn(
                  'p-2 rounded-xl border transition-colors',
                  viewMode === 'list'
                    ? 'bg-accent/10 border-accent/30 text-accent'
                    : 'bg-surface-0 border-border-subtle text-text-tertiary hover:border-accent hover:text-accent'
                )}
              >
                <List className="w-5 h-5" />
              </button>
            </div>
          </header>

          {/* ── Stock Cards Grid ── */}
          {loading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
              {Array.from({ length: 6 }).map((_, i) => (
                <div key={i} className="bg-surface-0 rounded-2xl p-5 border border-border-subtle animate-pulse">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-10 h-10 rounded-full bg-surface-2" />
                    <div className="space-y-2 flex-1">
                      <div className="h-3 bg-surface-2 rounded w-24" />
                      <div className="h-2 bg-surface-2 rounded w-16" />
                    </div>
                    <div className="h-5 bg-surface-2 rounded w-16" />
                  </div>
                  <div className="h-20 bg-surface-2 rounded mb-4" />
                  <div className="flex justify-between border-t border-border-subtle pt-4">
                    <div className="space-y-2">
                      <div className="h-2 bg-surface-2 rounded w-16" />
                      <div className="h-5 bg-surface-2 rounded w-20" />
                    </div>
                    <div className="space-y-2 text-right">
                      <div className="h-2 bg-surface-2 rounded w-16 ml-auto" />
                      <div className="h-5 bg-surface-2 rounded w-14 ml-auto" />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : filtered.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-20 text-text-tertiary">
              <Search className="w-12 h-12 mb-4 opacity-30" />
              <p className="text-lg font-medium mb-1">No stocks match your filters</p>
              <p className="text-sm">Try adjusting your filters or search term.</p>
              <button onClick={clearFilters} className="mt-4 text-sm text-accent hover:underline font-medium">
                Clear all filters
              </button>
            </div>
          ) : viewMode === 'grid' ? (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
              {paginated.map((stock) => (
                <StockCard key={stock.symbol} stock={stock} onClick={() => navigate(`/stock/${stock.symbol}`)} />
              ))}
            </div>
          ) : (
            /* ── List View ── */
            <div className="bg-surface-0 rounded-2xl border border-border-subtle overflow-hidden">
              <div className="grid grid-cols-[1fr_100px_120px_100px] gap-4 px-5 py-3 border-b border-border text-[11px] uppercase tracking-wider font-semibold text-text-tertiary bg-surface-1">
                <span>Stock</span>
                <span className="text-center">Risk</span>
                <span className="text-right">Price</span>
                <span className="text-right">24h Change</span>
              </div>
              {paginated.map((stock) => {
                const badge = riskBadge(stock.riskLevel);
                const isPositive = stock.changePercent >= 0;
                return (
                  <div
                    key={stock.symbol}
                    onClick={() => navigate(`/stock/${stock.symbol}`)}
                    className="grid grid-cols-[1fr_100px_120px_100px] gap-4 px-5 py-4 border-b border-border-subtle hover:bg-surface-2/30 cursor-pointer transition-colors items-center"
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <CompanyLogo symbol={stock.symbol} className="w-10 h-10 shrink-0" />
                      <div className="min-w-0">
                        <div className="font-bold text-text-primary text-sm truncate">{stock.name}</div>
                        <div className="text-xs text-text-tertiary">{stock.sector}</div>
                      </div>
                    </div>
                    <div className="text-center">
                      <span className={cn('inline-flex items-center px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wide border', badge.bg, badge.border, badge.text)}>
                        {badge.label}
                      </span>
                    </div>
                    <div className="text-right font-mono text-sm text-text-primary font-medium">
                      {formatCurrency(stock.price)}
                    </div>
                    <div className={cn('text-right font-medium text-sm flex items-center justify-end gap-1', isPositive ? 'text-positive' : stock.changePercent === 0 ? 'text-text-tertiary' : 'text-negative')}>
                      {isPositive ? <TrendingUp className="w-3.5 h-3.5" /> : stock.changePercent === 0 ? <Minus className="w-3.5 h-3.5" /> : <TrendingDown className="w-3.5 h-3.5" />}
                      {formatPercent(stock.changePercent)}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* ── Pagination ── */}
          {filtered.length > PAGE_SIZE && (
            <div className="mt-10 flex justify-center">
              <nav className="flex items-center gap-2">
                <button
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page <= 1}
                  className="p-2 rounded-xl bg-surface-0 border border-border-subtle text-text-tertiary hover:text-accent disabled:opacity-40 transition-colors"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>

                {Array.from({ length: Math.min(totalPages, 5) }, (_, i) => {
                  let pageNum: number;
                  if (totalPages <= 5) {
                    pageNum = i + 1;
                  } else if (page <= 3) {
                    pageNum = i + 1;
                  } else if (page >= totalPages - 2) {
                    pageNum = totalPages - 4 + i;
                  } else {
                    pageNum = page - 2 + i;
                  }
                  return (
                    <button
                      key={pageNum}
                      onClick={() => setPage(pageNum)}
                      className={cn(
                        'w-8 h-8 rounded-xl text-sm font-medium transition-all',
                        page === pageNum
                          ? 'bg-accent text-white font-bold shadow-lg shadow-accent/30'
                          : 'bg-surface-0 border border-border-subtle text-text-tertiary hover:border-accent hover:text-accent'
                      )}
                    >
                      {pageNum}
                    </button>
                  );
                })}

                {totalPages > 5 && page < totalPages - 2 && (
                  <span className="text-text-tertiary text-sm">...</span>
                )}

                <button
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={page >= totalPages}
                  className="p-2 rounded-xl bg-surface-0 border border-border-subtle text-text-tertiary hover:text-accent disabled:opacity-40 transition-colors"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </nav>
            </div>
          )}
        </main>
      </div>
    </PageTransition>
  );
}

/* ════════════════════════════════════════
   Stock Card Component
════════════════════════════════════════ */
function StockCard({ stock, onClick }: { stock: StockWithSentiment; onClick: () => void }) {
  const badge = riskBadge(stock.riskLevel);
  const isPositive = stock.changePercent >= 0;
  const isFlat = Math.abs(stock.changePercent) < 0.05;
  const sparkData = useMemo(() => generateSparklineData(stock.symbol, stock.price, stock.changePercent), [stock.symbol, stock.price, stock.changePercent]);

  const chartColor = isFlat
    ? 'rgb(var(--text-tertiary))'
    : isPositive
      ? 'rgb(var(--positive))'
      : 'rgb(var(--negative))';

  const gradientId = `sparkGrad-${stock.symbol}`;

  return (
    <motion.article
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      onClick={onClick}
      className="bg-surface-0 rounded-2xl p-5 border border-border-subtle shadow-card hover:shadow-elevated hover:border-accent/30 transition-all duration-300 cursor-pointer group"
    >
      {/* Header */}
      <div className="flex justify-between items-start mb-4">
        <div className="flex items-center gap-3">
          <CompanyLogo symbol={stock.symbol} className="w-11 h-11" />
          <div>
            <h3 className="font-bold text-text-primary leading-tight text-sm">{stock.name}</h3>
            <span className="text-xs text-text-tertiary">{stock.sector}</span>
          </div>
        </div>
        <span className={cn(
          'inline-flex items-center px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wide border',
          badge.bg, badge.border, badge.text,
        )}>
          {badge.label}
        </span>
      </div>

      {/* Interactive Sparkline Chart */}
      <div className="h-20 w-full mb-4">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={sparkData} margin={{ top: 4, right: 4, bottom: 0, left: 4 }}>
            <defs>
              <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={chartColor} stopOpacity={0.2} />
                <stop offset="100%" stopColor={chartColor} stopOpacity={0} />
              </linearGradient>
            </defs>
            <RechartsTooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  return (
                    <div className="bg-surface-0 border border-border-subtle rounded-lg shadow-elevated px-3 py-2 text-xs">
                      <span className="font-medium text-text-primary">{formatCurrency(payload[0].value as number)}</span>
                    </div>
                  );
                }
                return null;
              }}
              cursor={{ stroke: chartColor, strokeWidth: 1, strokeDasharray: '4 4' }}
            />
            <Area
              type="monotone"
              dataKey="value"
              stroke={chartColor}
              strokeWidth={2}
              fill={`url(#${gradientId})`}
              dot={false}
              activeDot={{ r: 4, fill: chartColor, stroke: 'var(--surface-0)', strokeWidth: 2 }}
              isAnimationActive={true}
              animationDuration={800}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Footer */}
      <div className="flex items-end justify-between border-t border-border-subtle pt-4">
        <div>
          <p className="text-xs text-text-tertiary mb-1">Current Price</p>
          <p className="text-2xl font-bold text-text-primary">
            {stock.price > 0 ? formatCurrency(stock.price) : <span className="text-text-tertiary">Loading...</span>}
          </p>
        </div>
        <div className="text-right">
          <p className="text-xs text-text-tertiary mb-1">24h Change</p>
          <div className={cn(
            'flex items-center justify-end gap-1 font-bold',
            isPositive ? 'text-positive' : isFlat ? 'text-text-tertiary' : 'text-negative',
          )}>
            {isPositive ? (
              <TrendingUp className="w-4 h-4" />
            ) : isFlat ? (
              <Minus className="w-4 h-4" />
            ) : (
              <TrendingDown className="w-4 h-4" />
            )}
            <span>{formatPercent(stock.changePercent)}</span>
          </div>
        </div>
      </div>
    </motion.article>
  );
}
