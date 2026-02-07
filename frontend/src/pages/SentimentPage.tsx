import { useState, useMemo } from 'react';
import { useApi } from '../hooks/useApi';
import { api } from '../lib/api';
import { CompanyLogo } from '../components/shared/CompanyLogo';
import { cn, formatCompactNumber, formatCurrency, formatPercent } from '../lib/utils';
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
  Globe
} from 'lucide-react';
import { Skeleton } from '../components/ui/Skeleton';

// Additional icons if needed can be mapped or imported
// Replicating the HTML structure

export function SentimentPage() {
    const {
        data: stocks,
        loading,
        refetch: refetchStocks,
    } = useApi(() => api.getStocksWithSentiment(), [], { refetchInterval: 30_000 });
  const [selectedStockSymbol, setSelectedStockSymbol] = useState<string | null>(null);

  // Memoize top stocks for the grid (take first 8 to match the design roughly)
  const topStocks = useMemo(() => {
    if (!stocks) return [];
    // Sort by something meaningful if needed, e.g., market cap or just take them
    // Assuming API returns pertinent stocks
    return stocks.slice(0, 8); 
  }, [stocks]);

    const baseSelectedStock = useMemo(() => {
    if (!stocks?.length) return null;
    if (selectedStockSymbol) return stocks.find(s => s.symbol === selectedStockSymbol) || stocks[0];
    return stocks[0];
  }, [stocks, selectedStockSymbol, topStocks]);

    const symbolsForQuotes = useMemo(() => {
        const symbols = new Set<string>();
        const isAllowed = (sym: string) => /^[A-Z0-9_&]{1,20}$/.test(sym.trim().toUpperCase());
        for (const stock of topStocks) {
            if (isAllowed(stock.symbol)) symbols.add(stock.symbol);
        }
        if (baseSelectedStock?.symbol && isAllowed(baseSelectedStock.symbol)) symbols.add(baseSelectedStock.symbol);
        return Array.from(symbols).slice(0, 20);
    }, [topStocks, baseSelectedStock?.symbol]);

    const symbolsParam = useMemo(() => symbolsForQuotes.join(','), [symbolsForQuotes]);

    const {
        data: quotes,
        refetch: refetchQuotes,
    } = useApi(
        () => api.getQuotes(symbolsParam),
        [symbolsParam],
        { enabled: symbolsForQuotes.length > 0, refetchInterval: 30_000 }
    );

    const quoteBySymbol = useMemo(() => {
        const map = new Map<string, any>();
        if (!quotes) return map;
        for (const q of quotes as any[]) map.set(q.symbol, q);
        return map;
    }, [quotes]);

    const topStocksMerged = useMemo(() => {
        return topStocks.map((s) => {
            const q: any = quoteBySymbol.get(s.symbol);
            return {
                ...s,
                price: q?.price ?? s.price,
                changePercent: q?.changePercent ?? s.changePercent,
                change: q?.change ?? s.change,
                volume: q?.volume ?? 0,
            };
        });
    }, [topStocks, quoteBySymbol]);

    const selectedStock = useMemo(() => {
        if (!baseSelectedStock) return null;
        const q: any = quoteBySymbol.get(baseSelectedStock.symbol);
        return {
            ...baseSelectedStock,
            price: q?.price ?? baseSelectedStock.price,
            changePercent: q?.changePercent ?? baseSelectedStock.changePercent,
            change: q?.change ?? baseSelectedStock.change,
            volume: q?.volume ?? 0,
        };
    }, [baseSelectedStock, quoteBySymbol]);

    const sentimentSummary = useMemo(() => {
        if (!stocks?.length) return null;
        const bullish = stocks.filter((s) => s.sentiment === 'bullish').length;
        const bearish = stocks.filter((s) => s.sentiment === 'bearish').length;
        const neutral = stocks.filter((s) => s.sentiment === 'neutral').length;
        const total = stocks.length;
        const avg = total > 0 ? (bullish - bearish) / total : 0;
        const score = Math.round(((avg + 1) / 2) * 100);

        const label =
            score >= 75 ? 'Extreme Greed' :
            score >= 60 ? 'Greed' :
            score >= 40 ? 'Neutral' :
            score >= 25 ? 'Fear' :
            'Extreme Fear';

        return { bullish, bearish, neutral, total, score, label };
    }, [stocks]);

    const totalTradedValue = useMemo(() => {
        if (!quotes) return 0;
        return (quotes as any[]).reduce((sum, q) => sum + (Number(q.volume) || 0) * (Number(q.price) || 0), 0);
    }, [quotes]);

    const avgAbsChangePct = useMemo(() => {
        if (!quotes || (quotes as any[]).length === 0) return 0;
        const list = quotes as any[];
        const total = list.reduce((sum, q) => sum + Math.abs(Number(q.changePercent) || 0), 0);
        return total / list.length;
    }, [quotes]);

    const avgChangePct = useMemo(() => {
        if (!quotes || (quotes as any[]).length === 0) return 0;
        const list = quotes as any[];
        const total = list.reduce((sum, q) => sum + (Number(q.changePercent) || 0), 0);
        return total / list.length;
    }, [quotes]);

    const {
        data: newsData,
        loading: newsLoading,
        refetch: refetchNews,
    } = useApi(
        () => api.getNews(selectedStock?.symbol || ''),
        [selectedStock?.symbol],
        { enabled: Boolean(selectedStock?.symbol) }
    );

    const formatInrMoney = (value: number) => {
        const formatted = formatCurrency(value, 'INR');
        return formatted.startsWith('₹') ? formatted : `₹${formatted}`;
    };

    const timeAgo = (iso: string) => {
        const t = Date.parse(iso);
        if (Number.isNaN(t)) return '';
        const seconds = Math.floor((Date.now() - t) / 1000);
        if (seconds < 60) return `${seconds}s ago`;
        const minutes = Math.floor(seconds / 60);
        if (minutes < 60) return `${minutes}m ago`;
        const hours = Math.floor(minutes / 60);
        if (hours < 24) return `${hours}h ago`;
        const days = Math.floor(hours / 24);
        return `${days}d ago`;
    };

    const handleRefresh = () => {
        refetchStocks();
        refetchQuotes();
        refetchNews();
    };
  
  const getBgColorForCell = (index: number) => {
    // Mimic the diverse reds/browns from the design
    // The design has: Primary Red, Dark Red, Lighter Red, Brownish
    if (index === 0) return "bg-primary";
    if (index === 1) return "bg-[#A81D1D]";
    if (index === 2) return "bg-[#FF4D4D]";
    if (index === 3) return "bg-[#4A0A0A]";
    if (index === 4) return "bg-[#751111]";
    if (index === 5) return "bg-[#2C0505]";
    if (index === 6) return "bg-[#E32929]";
    return "bg-[#5E0E0E]";
  };

  // Grid classes mapping based on index
  const getGridClass = (index: number) => {
    // 0: cell-1 (2x2)
    // 1: cell-2 (1x2)
    // 2: cell-3 (1x1)
    // 3: cell-3 (1x1)
    // 4: cell-4 (2x1) - BNB in sample
    // 5: 1x1 - ADA
    // 6: 1x1 - AVAX
    // 7: 2x1 - DOGE (or whatever fills)
    switch(index) {
        case 0: return "col-span-2 row-span-2";
        case 1: return "col-span-1 row-span-2";
        case 2: return "col-span-1 row-span-1";
        case 3: return "col-span-1 row-span-1";
        case 4: return "col-span-2 row-span-1";
        case 5: return "col-span-1 row-span-1";
        case 6: return "col-span-1 row-span-1";
        case 7: return "col-span-2 row-span-1";
        default: return "col-span-1 row-span-1"; 
    }
  };

  if (loading || !stocks) {
    return (
        <div className="p-8 space-y-8">
            <Skeleton className="h-64 w-full rounded-3xl" />
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                <Skeleton className="lg:col-span-8 h-96 rounded-3xl" />
                <Skeleton className="lg:col-span-4 h-96 rounded-3xl" />
            </div>
        </div>
    );
  }

  return (
    <div className="font-sans text-gray-900 dark:text-gray-100 min-h-screen">
        {/* Main Content Grid */}
        <main className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            
            {/* Left Column: Metrics & Heatmap (span 8) */}
            <div className="lg:col-span-8 flex flex-col gap-6">
                
                {/* Market Sentinel Card */}
                <div className="bg-primary text-white rounded-3xl p-8 relative overflow-hidden shadow-xl shadow-red-900/20">
                    <div className="absolute -right-20 -top-20 w-96 h-96 bg-white opacity-5 rounded-full blur-3xl pointer-events-none"></div>
                    <div className="absolute left-10 bottom-0 w-full h-32 bg-gradient-to-t from-black/20 to-transparent pointer-events-none"></div>
                    
                    <div className="relative z-10">
                        <div className="flex justify-between items-start mb-6">
                            <div>
                                <p className="text-red-100 text-sm font-medium mb-1">Market Sentiment Score</p>
                                <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
                                    {sentimentSummary?.label || '—'}
                                    <span className="text-lg opacity-80 font-normal align-middle ml-2">{sentimentSummary?.score ?? '—'}/100</span>
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

                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                            {[
                                {
                                    label: 'Trending Volume',
                                    value: totalTradedValue ? formatInrMoney(totalTradedValue) : '—',
                                    change: avgChangePct,
                                },
                                {
                                    label: 'Social Mentions',
                                    value: newsLoading ? '…' : String(newsData?.article_count ?? 0),
                                    change: selectedStock?.changePercent ?? 0,
                                },
                                {
                                    label: 'Volatility Index',
                                    value: avgAbsChangePct.toFixed(1),
                                    change: 0,
                                },
                                {
                                    label: 'Market Cap',
                                    value: '—',
                                    change: 0,
                                },
                            ].map((item, i) => (
                                <div key={i} className="bg-black/20 rounded-2xl p-4 backdrop-blur-sm">
                                    <p className="text-xs text-red-100 opacity-70 mb-1">{item.label}</p>
                                    <p className="text-xl font-semibold">{item.value}</p>
                                    <span className={cn(
                                        "text-xs flex items-center mt-1 w-fit px-1.5 py-0.5 rounded",
                                        item.change >= 0 ? "text-green-300 bg-green-900/30" : "text-red-200 bg-red-900/30"
                                    )}>
                                        {item.change >= 0 ? <ArrowUp className="w-3 h-3 mr-1" /> : <ArrowDown className="w-3 h-3 mr-1" />}
                                        {Math.abs(item.change)}%
                                    </span>
                                </div>
                            ))}
                        </div>

                        <div className="flex items-center gap-1 bg-black/20 p-1 rounded-xl w-fit backdrop-blur-sm">
                            {['1h', '8h', '1d', '1w', '1m'].map(tf => (
                                <button key={tf} className={cn(
                                    "px-4 py-1.5 rounded-lg text-sm transition",
                                    tf === '1d' ? "bg-white text-primary font-bold shadow-sm" : "hover:bg-white/10 text-red-100"
                                )}>
                                    {tf}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Heatmap Grid */}
                <div className="rounded-3xl bg-surface-light dark:bg-surface-dark p-6 border border-gray-200 dark:border-gray-800 shadow-sm">
                    <div className="flex justify-between items-center mb-4">
                        <h2 className="text-lg font-semibold flex items-center gap-2 text-text-primary">
                            <LayoutGrid className="text-primary w-5 h-5" />
                            Sector Heatmap
                        </h2>
                        <div className="text-xs text-gray-500 flex items-center gap-2">
                            <span>Updated: Just Now</span>
                            <button className="p-1 hover:bg-gray-100 dark:hover:bg-surface-2 rounded-full transition" onClick={handleRefresh}>
                                <RefreshCcw className="w-4 h-4" />
                            </button>
                        </div>
                    </div>

                    {/* CSS Grid Implementation */}
                    <div className="grid grid-cols-4 grid-rows-4 gap-1 h-[600px] w-full rounded-xl overflow-hidden text-white font-medium">
                        {topStocksMerged.map((stock: any, i) => (
                            <div 
                                key={stock.symbol}
                                onClick={() => setSelectedStockSymbol(stock.symbol)}
                                className={cn(
                                    getBgColorForCell(i),
                                    getGridClass(i),
                                    "p-4 relative group cursor-pointer hover:opacity-90 transition-all flex flex-col justify-between"
                                )}
                            >
                                <div className="flex justify-between items-start">
                                    <div className="flex items-center gap-2">
                                        {i === 0 && (
                                            <CompanyLogo symbol={stock.symbol} className="w-8 h-8" chars={3} />
                                        )}
                                        <span className={cn("font-bold", i===0 ? "text-lg" : "text-sm")}>{stock.symbol}</span>
                                    </div>
                                    <span className="text-xs opacity-90">{formatPercent(stock.changePercent)}</span>
                                </div>
                                
                                <div className="mt-auto">
                                    <p className={cn("font-bold tracking-tight", i===0 ? "text-2xl" : "text-sm")}>
                                        {formatInrMoney(stock.price)}
                                    </p>
                                    {i < 3 && <p className="text-[10px] opacity-70">Vol: {formatCompactNumber(stock.volume || 0)}</p>}
                                </div>

                                {/* Graph overlay for one cell (BNB example) - conditionally rendering a dummy SVG for visuals */}
                                {i === 4 && (
                                     <div className="mt-2 h-8 w-full opacity-50">
                                        <svg className="w-full h-full stroke-current fill-none" viewBox="0 0 100 30" preserveAspectRatio="none">
                                            <path d="M0,25 C20,25 20,10 40,15 C60,20 60,5 100,0" strokeWidth="2"></path>
                                        </svg>
                                    </div>
                                )}

                                {/* Hover effect */}
                                <div className="absolute inset-0 bg-black/20 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                                    <span className="bg-white text-primary text-xs px-3 py-1 rounded-full font-bold">View Details</span>
                                </div>
                            </div>
                        ))}
                    </div>

                    <div className="mt-4 flex items-center justify-between text-xs text-text-tertiary">
                        <div className="flex items-center gap-2">
                            <span className="w-3 h-3 bg-primary rounded-sm"></span> High Positive
                            <span className="w-3 h-3 bg-[#5E0E0E] rounded-sm ml-2"></span> Neutral
                            <span className="w-3 h-3 bg-[#2C0505] rounded-sm ml-2"></span> Negative
                        </div>
                        <span>Data provided by StockSense</span>
                    </div>
                </div>
            </div>

            {/* Right Column: Selected Stock Detail (span 4) */}
            <div className="lg:col-span-4 flex flex-col gap-6">
                {selectedStock ? (
                    <div className="bg-surface-light dark:bg-surface-dark rounded-3xl p-6 border border-gray-200 dark:border-gray-800 shadow-lg h-full">
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
                                <div className="text-lg font-bold text-gray-900 dark:text-white">{formatInrMoney(selectedStock.price)}</div>
                                <div className={cn("text-xs font-medium", selectedStock.changePercent >= 0 ? "text-green-500" : "text-red-500")}>
                                    {formatPercent(selectedStock.changePercent)}
                                </div>
                            </div>
                        </div>

                        <div className="mb-8">
                            <div className="flex justify-between items-end mb-2">
                                <h4 className="text-lg font-semibold text-gray-900 dark:text-white">Social Heat</h4>
                                <span className={cn("text-xs font-bold", selectedStock.sentiment === 'bullish' ? "text-green-500" : "text-primary")}>
                                  {selectedStock.sentiment === 'bullish' ? 'Exploding' : 'Extremely Hot'}
                                </span>
                            </div>
                            <div className="h-3 w-full bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden flex">
                                <div 
                                    className="h-full bg-primary w-[85%] rounded-full shadow-[0_0_10px_rgba(218,41,41,0.6)]"
                                    style={{ width: `${Math.max(0, Math.min(100, Number(selectedStock.forecastConfidence) || 0))}%` }}
                                ></div>
                            </div>
                            <div className="flex justify-between mt-1 text-[10px] text-gray-400 uppercase tracking-wider font-medium">
                                <span>Cold</span>
                                <span>Neutral</span>
                                <span>Viral</span>
                            </div>
                        </div>

                        <div className="grid grid-cols-2 gap-4 mb-8">
                            <div className="bg-gray-50 dark:bg-black p-4 rounded-2xl border border-gray-100 dark:border-gray-800">
                                <div className="flex items-center gap-2 mb-2">
                                    <span className="w-2 h-2 rounded-full bg-green-500"></span>
                                    <span className="text-xs font-medium text-gray-500 dark:text-gray-400">Positive Posts</span>
                                </div>
                                                                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                                                                    {sentimentSummary ? `${Math.round((sentimentSummary.bullish / sentimentSummary.total) * 100)}%` : '—'}
                                                                </div>
                                <div className="text-xs text-gray-400 mt-1">Within expected norm</div>
                            </div>
                            <div className="bg-gray-50 dark:bg-black p-4 rounded-2xl border border-gray-100 dark:border-gray-800">
                                <div className="flex items-center gap-2 mb-2">
                                    <span className="w-2 h-2 rounded-full bg-red-500"></span>
                                    <span className="text-xs font-medium text-gray-500 dark:text-gray-400">Negative Posts</span>
                                </div>
                                                                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                                                                    {sentimentSummary ? `${Math.round((sentimentSummary.bearish / sentimentSummary.total) * 100)}%` : '—'}
                                                                </div>
                                <div className="text-xs text-gray-400 mt-1">Lower than average</div>
                            </div>
                        </div>

                        <div className="space-y-4">
                            <h4 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Top Sentiment Drivers</h4>
                                                        {newsLoading ? (
                                                            <div className="space-y-2">
                                                                <Skeleton className="h-14 w-full rounded-xl" />
                                                                <Skeleton className="h-14 w-full rounded-xl" />
                                                                <Skeleton className="h-14 w-full rounded-xl" />
                                                            </div>
                                                        ) : (newsData?.news?.length ? (
                                                            (newsData.news.slice(0, 3)).map((item) => {
                                                                const src = (item.source || '').toLowerCase();
                                                                const tag = src.includes('reddit') ? 'Reddit' : (src.includes('twitter') || src.includes('x.com') || src.includes('x ')) ? 'Twitter' : 'News';
                                                                const Icon = tag === 'Reddit' ? MessageSquare : tag === 'Twitter' ? Globe : FileText;
                                                                const tagClasses = tag === 'Reddit'
                                                                    ? 'bg-orange-100 text-orange-700 dark:bg-orange-900/40 dark:text-orange-300'
                                                                    : tag === 'Twitter'
                                                                        ? 'bg-gray-200 text-gray-700 dark:bg-gray-800 dark:text-gray-300'
                                                                        : 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300';
                                                                const iconWrap = tag === 'Reddit'
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
                                                                            <p className="text-sm font-medium text-gray-900 dark:text-gray-100 line-clamp-1 group-hover:text-primary transition-colors">
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
                                                        ))}
                        </div>

                        <button className="w-full mt-8 bg-primary hover:bg-primary-dark text-white font-semibold py-3.5 rounded-xl shadow-lg shadow-red-500/30 transition-all flex items-center justify-center gap-2">
                            <span>Trade {selectedStock.symbol}</span>
                            <ArrowRight className="w-4 h-4" />
                        </button>
                    </div>
                ) : (
                    <Skeleton className="h-full rounded-3xl" />
                )}
            </div>
        </main>
    </div>
  );
}
