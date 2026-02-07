import { cn } from '../../lib/utils';

interface SentimentHeatmapProps {
  data: {
    symbol: string;
    name: string;
    sector: string;
    sentiment: string;
    score: number;
    change: number;
  }[];
  onSelect?: (symbol: string) => void;
}

function getCellColor(sentiment: string, score: number): string {
  const intensity = Math.min(Math.abs(score), 1);
  if (sentiment === 'bullish') {
    return `rgba(74, 222, 128, ${0.1 + intensity * 0.3})`;
  }
  if (sentiment === 'bearish') {
    return `rgba(248, 113, 113, ${0.1 + intensity * 0.3})`;
  }
  return `rgba(251, 191, 36, ${0.08 + intensity * 0.15})`;
}

export function SentimentHeatmap({ data, onSelect }: SentimentHeatmapProps) {
  // Group by sector
  const sectors = data.reduce<Record<string, typeof data>>((acc, item) => {
    if (!acc[item.sector]) acc[item.sector] = [];
    acc[item.sector].push(item);
    return acc;
  }, {});

  return (
    <div className="space-y-4">
      {Object.entries(sectors).map(([sector, items]) => (
        <div key={sector}>
          <div className="text-xs font-medium text-text-tertiary uppercase tracking-wider mb-2">{sector}</div>
          <div className="flex flex-wrap gap-1.5">
            {items.map((item) => (
              <button
                key={item.symbol}
                onClick={() => onSelect?.(item.symbol)}
                className="group relative rounded px-3 py-2 text-left transition-all hover:ring-1 hover:ring-accent/30"
                style={{ backgroundColor: getCellColor(item.sentiment, item.score) }}
              >
                <div className="text-xs font-medium text-text-primary">{item.symbol}</div>
                <div className={cn(
                  'text-[10px] font-mono',
                  item.sentiment === 'bullish' ? 'text-positive' : item.sentiment === 'bearish' ? 'text-negative' : 'text-warning'
                )}>
                  {item.sentiment}
                </div>
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
