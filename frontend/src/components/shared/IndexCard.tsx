import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { cn, formatCurrency, formatPercent } from '../../lib/utils';
import { Card } from '../ui/Card';
import type { IndexQuote } from '../../lib/api';

interface IndexCardProps {
  index: IndexQuote;
}

export function IndexCard({ index }: IndexCardProps) {
  const isPositive = index.change >= 0;
  const Icon = isPositive ? TrendingUp : index.change === 0 ? Minus : TrendingDown;

  return (
    <Card variant="elevated" className="min-w-[180px]">
      <div className="text-xs text-text-tertiary mb-1">{index.name}</div>
      <div className="text-lg font-semibold text-text-primary font-mono">
        {formatCurrency(index.price)}
      </div>
      <div className={cn('flex items-center gap-1 mt-1 text-xs font-mono', isPositive ? 'text-positive' : 'text-negative')}>
        <Icon className="h-3 w-3" />
        <span>{formatPercent(index.changePercent)}</span>
      </div>
    </Card>
  );
}
