import { useState, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Zap,
  Wifi,
  WifiOff,
  CheckCircle,
  XCircle,
} from 'lucide-react';
import { PageTransition } from '../components/shared/PageTransition';
import { PriceChart } from '../components/charts/PriceChart';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Select } from '../components/ui/Select';
import { Badge } from '../components/ui/Badge';
import { Tooltip } from '../components/ui/Tooltip';
import { ChartSkeleton } from '../components/ui/Skeleton';
import { api, AUTH_TOKEN_STORAGE_KEY } from '../lib/api';
import type { Stock, CandleData } from '../lib/api';
import { useApi } from '../hooks/useApi';
import { useTheme } from '../hooks/useTheme';
import { useUpstoxFeed } from '../hooks/useUpstoxFeed';
import { usePaperPortfolio } from '../hooks/usePaperPortfolio';
import { cn, formatCurrency } from '../lib/utils';

type OrderType = 'market' | 'limit';
type OrderSide = 'buy' | 'sell';

interface Toast {
  id: string;
  type: 'success' | 'error';
  message: string;
}

export function TradingPage() {
  const { isDark } = useTheme();
  const { holdings, executeTrade, history } = usePaperPortfolio();

  /* ── Stock list ── */
  const { data: stocks } = useApi(() => api.getStocks(), []);
  const stockMap = useMemo(() => {
    const m: Record<string, Stock> = {};
    stocks?.forEach((s) => { m[s.symbol] = s; });
    return m;
  }, [stocks]);
  const stockOptions = useMemo(
    () => stocks?.map((s) => ({ value: s.symbol, label: `${s.symbol} — ${s.name}` })) ?? [],
    [stocks]
  );

  /* ── Selected symbol ── */
  const [selectedSymbol, setSelectedSymbol] = useState('RELIANCE');
  const selectedStock = stockMap[selectedSymbol];
  const instrumentKey = selectedStock?.instrument_key ?? '';

  /* ── Live price ── */
  const instrumentKeys = useMemo(() => (instrumentKey ? [instrumentKey] : []), [instrumentKey]);
  const { prices: livePrices, isLive } = useUpstoxFeed(instrumentKeys);
  const livePrice = instrumentKey ? livePrices[instrumentKey] : undefined;

  /* ── Historical chart data (seed) ── */
  const { data: historical, loading: chartLoading } = useApi(
    () => api.getHistorical(selectedSymbol, '1minute', 1),
    [selectedSymbol]
  );

  // Append live tick to chart data for smooth updates
  const chartData = useMemo(() => {
    if (!historical) return [];
    if (!livePrice) return historical;
    // Replace / add last candle with live LTP
    const last = historical[historical.length - 1];
    if (!last) return historical;
    const updated: CandleData = {
      ...last,
      close: livePrice,
      high: Math.max(last.high, livePrice),
      low: Math.min(last.low, livePrice),
    };
    return [...historical.slice(0, -1), updated];
  }, [historical, livePrice]);

  /* ── REST fallback for LTP ── */
  const { data: restQuote } = useApi(
    () => api.getQuotes(selectedSymbol),
    [selectedSymbol],
    { refetchInterval: 10_000 }
  );
  const restPrice = restQuote?.[0]?.price;
  const currentPrice = livePrice ?? restPrice ?? 0;

  /* ── Order form ── */
  const [orderSide, setOrderSide] = useState<OrderSide>('buy');
  const [orderType, setOrderType] = useState<OrderType>('market');
  const [quantity, setQuantity] = useState('');
  const [limitPrice, setLimitPrice] = useState('');

  const executionPrice = orderType === 'limit' && limitPrice ? Number(limitPrice) : currentPrice;
  const qty = Number(quantity) || 0;
  const estimatedValue = qty > 0 && executionPrice > 0 ? qty * executionPrice : 0;

  const currentHolding = holdings.find((h) => h.symbol === selectedSymbol);
  const maxSellQty = currentHolding?.quantity ?? 0;

  /* ── Toast state ── */
  const [toasts, setToasts] = useState<Toast[]>([]);
  const addToast = useCallback((type: 'success' | 'error', message: string) => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    setToasts((prev) => [...prev, { id, type, message }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  }, []);

  /* ── Submit order ── */
  const handleSubmit = useCallback(() => {
    if (qty <= 0 || executionPrice <= 0) return;

    const result = executeTrade({
      symbol: selectedSymbol,
      name: selectedStock?.name ?? selectedSymbol,
      sector: selectedStock?.sector ?? 'Unknown',
      instrumentKey: instrumentKey,
      side: orderSide,
      quantity: qty,
      price: executionPrice,
    });

    if (result.success) {
      addToast('success', result.message);
      setQuantity('');
      setLimitPrice('');
    } else {
      addToast('error', result.message);
    }
  }, [qty, executionPrice, selectedSymbol, selectedStock, instrumentKey, orderSide, executeTrade, addToast]);

  /* ── Derived info ── */
  const hasToken = !!localStorage.getItem(AUTH_TOKEN_STORAGE_KEY);
  const canSell = orderSide === 'sell' ? qty > 0 && qty <= maxSellQty : true;
  const canSubmit = qty > 0 && executionPrice > 0 && canSell;

  return (
    <PageTransition>
      <div className="space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-accent" />
            <h1 className="text-lg font-bold text-text-primary">Paper Trading</h1>
          </div>
          <div className="flex items-center gap-2">
            <Tooltip content={isLive ? 'Live tick stream active' : hasToken ? 'Connecting to Upstox WS…' : 'No Upstox token — using REST prices'}>
              <Badge variant={isLive ? 'positive' : 'warning'} size="md">
                {isLive ? <Wifi className="h-3 w-3 mr-1" /> : <WifiOff className="h-3 w-3 mr-1" />}
                {isLive ? 'Live' : 'REST'}
              </Badge>
            </Tooltip>
            <Badge size="md">Paper Mode</Badge>
          </div>
        </div>

        {/* Symbol selector */}
        <div className="max-w-sm">
          <Select
            options={stockOptions.length > 0 ? stockOptions : [{ value: selectedSymbol, label: selectedSymbol }]}
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-[1fr_380px] gap-4">
          {/* Left: chart */}
          <div className="space-y-4">
            <Card variant="elevated" className="overflow-hidden">
              <CardHeader>
                <div className="flex items-center gap-3">
                  <CardTitle>{selectedSymbol} — Intraday</CardTitle>
                  {currentPrice > 0 && (
                    <span className="text-lg font-bold font-mono text-text-primary">{formatCurrency(currentPrice)}</span>
                  )}
                </div>
              </CardHeader>
              <CardContent className="-mx-4 -mb-4">
                {chartLoading || !historical ? (
                  <div className="px-4 pb-4"><ChartSkeleton /></div>
                ) : (
                  <PriceChart data={chartData} isDark={isDark} height={480} />
                )}
              </CardContent>
            </Card>

            {/* Current holding info */}
            {currentHolding && (
              <Card>
                <div className="flex items-center gap-3 text-xs">
                  <span className="text-text-tertiary">You hold</span>
                  <span className="font-mono font-medium text-text-primary">{currentHolding.quantity} shares</span>
                  <span className="text-text-tertiary">@ avg</span>
                  <span className="font-mono text-text-secondary">{formatCurrency(currentHolding.avgPrice)}</span>
                </div>
              </Card>
            )}
          </div>

          {/* Right: order panel + recent trades */}
          <div className="space-y-4">
            <Card variant="elevated">
              <CardHeader>
                <CardTitle>Place Paper Order</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {/* Buy / Sell toggle */}
                  <div className="flex gap-1 p-0.5 rounded bg-surface-2">
                    <button
                      onClick={() => setOrderSide('buy')}
                      className={cn(
                        'flex-1 rounded py-2 text-sm font-medium transition-colors',
                        orderSide === 'buy'
                          ? 'bg-positive text-white'
                          : 'text-text-secondary hover:text-text-primary'
                      )}
                    >
                      Buy
                    </button>
                    <button
                      onClick={() => setOrderSide('sell')}
                      className={cn(
                        'flex-1 rounded py-2 text-sm font-medium transition-colors',
                        orderSide === 'sell'
                          ? 'bg-negative text-white'
                          : 'text-text-secondary hover:text-text-primary'
                      )}
                    >
                      Sell
                    </button>
                  </div>

                  {/* Order type */}
                  <div>
                    <label className="text-xs text-text-tertiary mb-1.5 block">Order Type</label>
                    <div className="flex gap-1 p-0.5 rounded bg-surface-2">
                      {(['market', 'limit'] as const).map((t) => (
                        <button
                          key={t}
                          onClick={() => setOrderType(t)}
                          className={cn(
                            'flex-1 rounded py-1.5 text-xs font-medium transition-colors capitalize',
                            orderType === t ? 'bg-surface-0 text-text-primary shadow-sm' : 'text-text-tertiary'
                          )}
                        >
                          {t}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Quantity */}
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <label className="text-xs text-text-tertiary">Quantity</label>
                      {orderSide === 'sell' && maxSellQty > 0 && (
                        <button
                          onClick={() => setQuantity(String(maxSellQty))}
                          className="text-[10px] text-accent hover:underline"
                        >
                          Max: {maxSellQty}
                        </button>
                      )}
                    </div>
                    <Input
                      type="number"
                      placeholder="0"
                      min={1}
                      value={quantity}
                      onChange={(e) => setQuantity(e.target.value)}
                    />
                  </div>

                  {/* Limit price */}
                  {orderType === 'limit' && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.15 }}
                    >
                      <label className="text-xs text-text-tertiary mb-1.5 block">Limit Price</label>
                      <Input
                        type="number"
                        placeholder={currentPrice > 0 ? currentPrice.toFixed(2) : '0.00'}
                        value={limitPrice}
                        onChange={(e) => setLimitPrice(e.target.value)}
                      />
                    </motion.div>
                  )}

                  {/* Order summary */}
                  {qty > 0 && (
                    <div className="rounded bg-surface-2 p-3 space-y-1.5 text-xs">
                      <div className="flex justify-between">
                        <span className="text-text-tertiary">Price</span>
                        <span className="font-mono text-text-primary">
                          {orderType === 'market' ? (currentPrice > 0 ? formatCurrency(currentPrice) : 'Loading…') : formatCurrency(Number(limitPrice) || 0)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-text-tertiary">Quantity</span>
                        <span className="font-mono text-text-primary">{qty}</span>
                      </div>
                      <div className="flex justify-between border-t border-border-subtle pt-1.5">
                        <span className="text-text-secondary font-medium">Estimated Value</span>
                        <span className="font-mono font-medium text-text-primary">
                          {estimatedValue > 0 ? formatCurrency(estimatedValue) : '—'}
                        </span>
                      </div>
                      {orderSide === 'sell' && qty > maxSellQty && (
                        <div className="text-negative text-[10px]">
                          You only hold {maxSellQty} shares of {selectedSymbol}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Submit */}
                  <Button
                    size="lg"
                    className={cn(
                      'w-full',
                      orderSide === 'buy' ? 'bg-positive hover:bg-positive/90' : 'bg-negative hover:bg-negative/90'
                    )}
                    disabled={!canSubmit}
                    onClick={handleSubmit}
                  >
                    {orderSide === 'buy' ? 'Buy' : 'Sell'} {selectedSymbol}
                  </Button>

                  <p className="text-[10px] text-text-tertiary text-center leading-relaxed">
                    Simulated execution at {orderType === 'market' ? 'current LTP' : 'limit price'}.
                    No real orders are placed.
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Recent paper trades */}
            <Card className="p-0 overflow-hidden">
              <div className="px-4 py-3 border-b border-border">
                <CardTitle>Recent Trades</CardTitle>
              </div>
              {history.length === 0 ? (
                <div className="px-4 py-4 text-xs text-text-tertiary">No paper trades yet.</div>
              ) : (
                <div className="max-h-52 overflow-y-auto">
                  {history.slice(0, 15).map((t) => (
                    <div
                      key={t.id}
                      className="flex items-center gap-2 px-4 py-2 border-b border-border-subtle text-xs"
                    >
                      <Badge variant={t.side === 'buy' ? 'positive' : 'negative'} size="sm">
                        {t.side.toUpperCase()}
                      </Badge>
                      <span className="font-medium text-text-primary">{t.symbol}</span>
                      <span className="text-text-tertiary">{t.quantity}×{formatCurrency(t.price)}</span>
                      <span className="ml-auto font-mono text-text-secondary">{formatCurrency(t.total)}</span>
                    </div>
                  ))}
                </div>
              )}
            </Card>
          </div>
        </div>

        {/* Toast overlay */}
        <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-2">
          <AnimatePresence>
            {toasts.map((t) => (
              <motion.div
                key={t.id}
                initial={{ opacity: 0, y: 12, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -8, scale: 0.95 }}
                transition={{ duration: 0.2 }}
                className={cn(
                  'flex items-center gap-2 rounded-lg border px-4 py-3 text-sm shadow-elevated',
                  t.type === 'success'
                    ? 'border-positive/30 bg-positive/10 text-positive'
                    : 'border-negative/30 bg-negative/10 text-negative'
                )}
              >
                {t.type === 'success' ? <CheckCircle className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}
                {t.message}
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </div>
    </PageTransition>
  );
}
