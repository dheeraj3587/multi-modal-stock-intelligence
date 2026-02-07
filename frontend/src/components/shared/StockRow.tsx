import { useNavigate } from 'react-router-dom';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { cn, formatCurrency, formatPercent } from '../../lib/utils';
import { Badge } from '../ui/Badge';

interface StockRowProps {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  sector?: string;
  sentiment?: string;
  extra?: React.ReactNode;
}

export function StockRow({ symbol, name, price, change, changePercent, sector, sentiment, extra }: StockRowProps) {
  const navigate = useNavigate();
  const isPositive = change >= 0;

  return (
    <div
      onClick={() => navigate(`/stock/${symbol}`)}
      className="flex items-center gap-4 px-4 py-3 border-b border-border-subtle cursor-pointer transition-colors hover:bg-surface-2 group"
    >
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-text-primary">{symbol}</span>
          {sector && <Badge>{sector}</Badge>}
        </div>
        <span className="text-xs text-text-tertiary truncate block">{name}</span>
      </div>

      {sentiment && (
        <Badge
          variant={sentiment === 'bullish' ? 'positive' : sentiment === 'bearish' ? 'negative' : 'warning'}
          size="sm"
        >
          {sentiment}
        </Badge>
      )}

      <div className="text-right min-w-[100px]">
        <div className="text-sm font-medium text-text-primary font-mono">{formatCurrency(price)}</div>
        <div className={cn('flex items-center justify-end gap-1 text-xs font-mono', isPositive ? 'text-positive' : 'text-negative')}>
          {isPositive ? <TrendingUp className="h-3 w-3" /> : change === 0 ? <Minus className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
          {formatPercent(changePercent)}
        </div>
      </div>

      {extra}
    </div>
  );
}
