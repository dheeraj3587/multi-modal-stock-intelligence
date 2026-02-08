import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion} from 'framer-motion';
import {
  ArrowLeft,
  TrendingUp,
  TrendingDown,
  Shield,
  Brain,
  FileText,
  Clock,
  ExternalLink,
  ChevronRight,
} from 'lucide-react';
import { PageTransition } from '../components/shared/PageTransition';
import { PriceChart } from '../components/charts/PriceChart';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';

import { ChartSkeleton, Skeleton } from '../components/ui/Skeleton';
import { api } from '../lib/api';
import { useApi } from '../hooks/useApi';
import { useTheme } from '../hooks/useTheme';
import { cn, formatCurrency, formatPercent } from '../lib/utils';

const TIMEFRAMES = [
  { label: '1D', interval: '1minute', days: 1 },
  { label: '1W', interval: '30minute', days: 7 },
  { label: '1M', interval: '1day', days: 30 },
  { label: '3M', interval: '1day', days: 90 },
  { label: '1Y', interval: '1day', days: 365 },
];

export function StockDetailPage() {
  const { symbol = '' } = useParams<{ symbol: string }>();
  const navigate = useNavigate();
  const { isDark } = useTheme();
  const [timeframe, setTimeframe] = useState(2); // default 1M

  const tf = TIMEFRAMES[timeframe];
  const { data: analysis, loading: analysisLoading } = useApi(() => api.getAnalysis(symbol), [symbol]);
  const { data: historical, loading: chartLoading } = useApi(
    () => api.getHistorical(symbol, tf.interval, tf.days),
    [symbol, timeframe]
  );
  const { data: fundamentals } = useApi(() => api.getFundamentals(symbol), [symbol]);

  const isPositive = (analysis?.changePercent ?? 0) >= 0;

  return (
    <PageTransition>
      <div className="space-y-6">
        {/* Back + Header */}
        <div>
          <button
            onClick={() => navigate(-1)}
            className="flex items-center gap-1.5 text-xs text-text-tertiary hover:text-text-primary transition-colors mb-3"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Back
          </button>

          {analysisLoading ? (
            <div className="space-y-2">
              <Skeleton className="h-6 w-40" />
              <Skeleton className="h-9 w-48" />
            </div>
          ) : analysis ? (
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-center gap-3">
                  <h1 className="text-2xl font-bold text-text-primary">{analysis.symbol}</h1>
                  <Badge variant={analysis.recommendation === 'buy' ? 'positive' : analysis.recommendation === 'sell' ? 'negative' : 'warning'} size="md">
                    {analysis.recommendation?.toUpperCase()}
                  </Badge>
                </div>
                <p className="text-sm text-text-secondary mt-0.5">{analysis.name} &middot; {analysis.sector}</p>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-text-primary font-mono">{formatCurrency(analysis.currentPrice)}</div>
                <div className={cn('flex items-center justify-end gap-1 text-sm font-mono', isPositive ? 'text-positive' : 'text-negative')}>
                  {isPositive ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
                  {formatCurrency(analysis.change)} ({formatPercent(analysis.changePercent)})
                </div>
              </div>
            </div>
          ) : null}
        </div>

        {/* Price Chart */}
        <Card variant="elevated" className="overflow-hidden">
          <CardHeader>
            <CardTitle>Price Chart</CardTitle>
            <div className="flex gap-1">
              {TIMEFRAMES.map((tf, i) => (
                <button
                  key={tf.label}
                  onClick={() => setTimeframe(i)}
                  className={cn(
                    'rounded px-2.5 py-1 text-xs font-medium transition-colors',
                    timeframe === i
                      ? 'bg-accent text-white'
                      : 'text-text-tertiary hover:bg-surface-2 hover:text-text-primary'
                  )}
                >
                  {tf.label}
                </button>
              ))}
            </div>
          </CardHeader>
          <CardContent className="-mx-4 -mb-4">
            {chartLoading || !historical ? (
              <div className="px-4 pb-4"><ChartSkeleton /></div>
            ) : (
              <PriceChart data={historical} isDark={isDark} />
            )}
          </CardContent>
        </Card>

        {/* AI Forecast banner */}
        {analysis && analysis.predictedPrice > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.2 }}
          >
            <Card className="bg-accent-muted border-accent/20">
              <div className="flex items-start gap-3">
                <Brain className="h-5 w-5 text-accent mt-0.5 flex-shrink-0" />
                <div>
                  <div className="text-sm font-medium text-text-primary">AI Price Forecast</div>
                  <div className="mt-1 text-sm text-text-secondary">
                    Predicted price: <span className="font-mono font-medium text-text-primary">{formatCurrency(analysis.predictedPrice)}</span>
                    {' '}&middot; Confidence: <span className="font-mono">{(analysis.forecastConfidence * 100).toFixed(0)}%</span>
                  </div>
                  {analysis.shortTermOutlook && (
                    <p className="mt-2 text-xs text-text-secondary leading-relaxed">{analysis.shortTermOutlook}</p>
                  )}
                </div>
              </div>
            </Card>
          </motion.div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* AI Sentiment */}
          <Card variant="elevated">
            <CardHeader>
              <CardTitle>
                <div className="flex items-center gap-2">
                  <Brain className="h-4 w-4 text-accent" />
                  AI Sentiment
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {analysis ? (
                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <Badge
                      variant={analysis.sentiment === 'bullish' ? 'positive' : analysis.sentiment === 'bearish' ? 'negative' : 'warning'}
                      size="md"
                    >
                      {analysis.sentiment}
                    </Badge>
                    <span className="text-sm text-text-secondary">
                      Score: <span className="font-mono">{(analysis.sentimentScore * 100).toFixed(0)}</span> / 100
                      &middot; Confidence: <span className="font-mono">{(analysis.sentimentConfidence * 100).toFixed(0)}%</span>
                    </span>
                  </div>
                  {analysis.sentimentReasoning && (
                    <p className="text-sm text-text-secondary leading-relaxed">{analysis.sentimentReasoning}</p>
                  )}
                  <div className="flex items-center gap-2 pt-1">
                    <Shield className="h-3.5 w-3.5 text-text-tertiary" />
                    <span className="text-xs text-text-tertiary">Risk: {analysis.riskLevel}</span>
                  </div>
                  {analysis.riskFactors?.length > 0 && (
                    <ul className="space-y-1">
                      {analysis.riskFactors.map((f, i) => (
                        <li key={i} className="text-xs text-text-tertiary flex items-start gap-1.5">
                          <span className="text-negative mt-0.5">-</span> {f}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              ) : (
                <div className="space-y-2">
                  <Skeleton className="h-5 w-32" />
                  <Skeleton className="h-16 w-full" />
                </div>
              )}
            </CardContent>
          </Card>

          {/* Fundamentals */}
          <Card variant="elevated">
            <CardHeader>
              <CardTitle>
                <div className="flex items-center gap-2">
                  <FileText className="h-4 w-4 text-accent" />
                  Fundamentals
                </div>
              </CardTitle>
              <button
                onClick={() => navigate(`/scorecard/${symbol}`)}
                className="flex items-center gap-1 text-xs text-accent hover:underline"
              >
                Full Scorecard <ChevronRight className="h-3 w-3" />
              </button>
            </CardHeader>
            <CardContent>
              {fundamentals ? (
                <div className="space-y-3">
                  {fundamentals.about && (
                    <p className="text-xs text-text-secondary leading-relaxed line-clamp-3">{fundamentals.about}</p>
                  )}
                  {fundamentals.ratios && (
                    <div className="grid grid-cols-2 gap-2">
                      {Object.entries(fundamentals.ratios).slice(0, 8).map(([key, val]) => (
                        <div key={key} className="flex items-center justify-between py-1.5 border-b border-border-subtle">
                          <span className="text-xs text-text-tertiary">{key}</span>
                          <span className="text-xs font-mono text-text-primary">{val}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  <div className="grid grid-cols-2 gap-4 pt-2">
                    {(fundamentals.pros?.length ?? 0) > 0 && (
                      <div>
                        <div className="text-xs font-medium text-positive mb-1">Strengths</div>
                        {fundamentals.pros!.slice(0, 3).map((p, i) => (
                          <div key={i} className="text-[11px] text-text-secondary leading-relaxed mb-1">+ {p}</div>
                        ))}
                      </div>
                    )}
                    {(fundamentals.cons?.length ?? 0) > 0 && (
                      <div>
                        <div className="text-xs font-medium text-negative mb-1">Weaknesses</div>
                        {fundamentals.cons!.slice(0, 3).map((c, i) => (
                          <div key={i} className="text-[11px] text-text-secondary leading-relaxed mb-1">- {c}</div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="space-y-2">
                  <Skeleton className="h-12 w-full" />
                  <Skeleton className="h-24 w-full" />
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Recent News */}
        {analysis?.recentNews && analysis.recentNews.length > 0 && (
          <Card variant="elevated">
            <CardHeader>
              <CardTitle>
                <div className="flex items-center gap-2">
                  <Clock className="h-4 w-4 text-text-tertiary" />
                  Recent News ({analysis.newsCount})
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {analysis.recentNews.slice(0, 5).map((news, i) => (
                  <a
                    key={i}
                    href={news.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-start gap-2 rounded p-2 -mx-2 hover:bg-surface-2 transition-colors group"
                  >
                    <div className="flex-1 min-w-0">
                      <div className="text-sm text-text-primary group-hover:text-accent transition-colors line-clamp-1">{news.title}</div>
                      <div className="text-xs text-text-tertiary mt-0.5">{news.source}</div>
                    </div>
                    <ExternalLink className="h-3.5 w-3.5 text-text-tertiary flex-shrink-0 mt-0.5 opacity-0 group-hover:opacity-100 transition-opacity" />
                  </a>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </PageTransition>
  );
}
