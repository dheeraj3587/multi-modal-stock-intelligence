import { useParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowLeft, CheckCircle, XCircle, Info } from 'lucide-react';
import { PageTransition } from '../components/shared/PageTransition';
import { ScoreRing } from '../components/ui/ScoreRing';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Skeleton } from '../components/ui/Skeleton';
import { Tooltip } from '../components/ui/Tooltip';
import { api } from '../lib/api';
import { useApi } from '../hooks/useApi';
import { cn, scoreColor } from '../lib/utils';

const CATEGORY_META: Record<string, { label: string; description: string }> = {
  valuation: { label: 'Valuation', description: 'P/E, P/B, PEG, and relative valuation metrics' },
  profitability: { label: 'Profitability', description: 'Margins, ROE, ROCE, and earnings quality' },
  growth: { label: 'Growth', description: 'Revenue and earnings growth trajectory' },
  financial_health: { label: 'Financial Health', description: 'Debt ratios, interest coverage, cash flow' },
  sentiment: { label: 'Sentiment', description: 'AI-driven market and news sentiment analysis' },
  ownership: { label: 'Ownership', description: 'Promoter holding, FII/DII patterns' },
};

export function ScorecardPage() {
  const { symbol = '' } = useParams<{ symbol: string }>();
  const navigate = useNavigate();
  const { data: sc, loading } = useApi(() => api.getScorecard(symbol, true), [symbol]);

  const badgeVariant = (badge: string) => {
    const b = badge?.toLowerCase();
    if (b === 'excellent' || b === 'good') return 'positive' as const;
    if (b === 'poor' || b === 'below_average') return 'negative' as const;
    return 'warning' as const;
  };

  return (
    <PageTransition>
      <div className="space-y-6">
        {/* Back */}
        <button
          onClick={() => navigate(-1)}
          className="flex items-center gap-1.5 text-xs text-text-tertiary hover:text-text-primary transition-colors"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Back
        </button>

        {loading || !sc ? (
          <div className="space-y-4">
            <Skeleton className="h-8 w-48" />
            <div className="flex gap-6">
              <Skeleton className="h-40 w-40 rounded-full" />
              <div className="flex-1 space-y-2">
                <Skeleton className="h-6 w-full" />
                <Skeleton className="h-6 w-3/4" />
                <Skeleton className="h-6 w-1/2" />
              </div>
            </div>
          </div>
        ) : (
          <>
            {/* Header */}
            <div>
              <div className="flex items-center gap-3">
                <h1 className="text-2xl font-bold text-text-primary">{sc.company_name}</h1>
                <Badge variant={badgeVariant(sc.overall_badge)} size="md">
                  {sc.overall_verdict}
                </Badge>
              </div>
              <p className="text-sm text-text-secondary mt-0.5">
                {sc.symbol}
                {sc.sector ? ` · ${sc.sector}` : ''}
                {sc.industry ? ` · ${sc.industry}` : ''}
              </p>
            </div>

            {/* Score ring + category bars */}
            <div className="grid grid-cols-1 md:grid-cols-[auto_1fr] gap-8 items-start">
              {/* Score ring */}
              <div className="flex flex-col items-center">
                <ScoreRing score={sc.overall_score} max={sc.overall_max} size={180} label={`of ${sc.overall_max}`} />
                <p className="mt-3 text-sm text-text-secondary text-center max-w-[200px]">{sc.overall_verdict}</p>
              </div>

              {/* Category breakdown */}
              <div className="space-y-4">
                {Object.entries(sc.categories).map(([key, cat]) => {
                  const meta = CATEGORY_META[key] || { label: key, description: '' };
                  const max = cat.max_score || 10;
                  const percent = (cat.score / max) * 100;
                  const scorePct = (cat.score / max) * 100;
                  return (
                    <div key={key}>
                      <div className="flex items-center justify-between mb-1.5">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium text-text-primary">{meta.label}</span>
                          <Tooltip content={meta.description}>
                            <Info className="h-3.5 w-3.5 text-text-tertiary cursor-help" />
                          </Tooltip>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className={cn('text-sm font-mono font-medium', scoreColor(scorePct))}>
                            {cat.score.toFixed(1)}
                          </span>
                          <span className="text-xs text-text-tertiary">/ {max}</span>
                        </div>
                      </div>
                      <div className="h-2 rounded-full bg-surface-2 overflow-hidden">
                        <motion.div
                          className={cn(
                            'h-full rounded-full',
                            cat.score >= 7 ? 'bg-positive' : cat.score >= 4 ? 'bg-warning' : 'bg-negative'
                          )}
                          initial={{ width: 0 }}
                          animate={{ width: `${percent}%` }}
                          transition={{ duration: 0.6, ease: 'easeOut', delay: 0.1 }}
                        />
                      </div>
                      <p className="text-[11px] text-text-tertiary mt-1">{cat.verdict}</p>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Key Stats */}
            {sc.key_stats && Object.keys(sc.key_stats).length > 0 && (
              <Card>
                <CardHeader><CardTitle>Key Statistics</CardTitle></CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {Object.entries(sc.key_stats).map(([key, val]) => (
                      <div key={key} className="rounded bg-surface-2 px-3 py-2">
                        <div className="text-[11px] text-text-tertiary">{key}</div>
                        <div className="text-sm font-mono font-medium text-text-primary mt-0.5">{String(val ?? '—')}</div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Strengths & Weaknesses */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {sc.strengths?.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>
                      <div className="flex items-center gap-2 text-positive">
                        <CheckCircle className="h-4 w-4" />
                        Strengths
                      </div>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-1.5">
                      {sc.strengths.map((s, i) => (
                        <li key={i} className="text-sm text-text-secondary flex items-start gap-2">
                          <span className="text-positive mt-0.5 flex-shrink-0">+</span>
                          <span className="flex-1">
                            <span className="text-text-primary font-medium">{s.category}</span>
                            <span className="text-text-tertiary"> — {s.verdict}</span>
                          </span>
                          <span className={cn('font-mono text-xs', scoreColor((s.score / 10) * 100))}>
                            {s.score.toFixed(1)}/10
                          </span>
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              )}
              {sc.weaknesses?.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>
                      <div className="flex items-center gap-2 text-negative">
                        <XCircle className="h-4 w-4" />
                        Weaknesses
                      </div>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-1.5">
                      {sc.weaknesses.map((w, i) => (
                        <li key={i} className="text-sm text-text-secondary flex items-start gap-2">
                          <span className="text-negative mt-0.5 flex-shrink-0">-</span>
                          <span className="flex-1">
                            <span className="text-text-primary font-medium">{w.category}</span>
                            <span className="text-text-tertiary"> — {w.verdict}</span>
                          </span>
                          <span className={cn('font-mono text-xs', scoreColor((w.score / 10) * 100))}>
                            {w.score.toFixed(1)}/10
                          </span>
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              )}
            </div>

            {/* AI Summary */}
            {sc.ai_summary && (
              <Card>
                <CardHeader><CardTitle>AI Summary</CardTitle></CardHeader>
                <CardContent>
                  <p className="text-sm text-text-secondary leading-relaxed whitespace-pre-line">{sc.ai_summary}</p>
                </CardContent>
              </Card>
            )}
          </>
        )}
      </div>
    </PageTransition>
  );
}
