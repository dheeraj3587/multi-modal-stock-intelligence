import { useState } from 'react';
import { cn } from '../../lib/utils';
import { getStockLogoUrl } from '../../lib/stockLogos';

/* ── Avatar color map for text fallback ── */
const AVATAR_COLORS = [
  'bg-orange-500/10 text-orange-500 border-orange-500/20',
  'bg-blue-500/10 text-blue-500 border-blue-500/20',
  'bg-purple-500/10 text-purple-500 border-purple-500/20',
  'bg-emerald-500/10 text-emerald-500 border-emerald-500/20',
  'bg-rose-500/10 text-rose-500 border-rose-500/20',
  'bg-indigo-500/10 text-indigo-500 border-indigo-500/20',
  'bg-amber-500/10 text-amber-500 border-amber-500/20',
  'bg-cyan-500/10 text-cyan-500 border-cyan-500/20',
];

function avatarColor(symbol: string) {
  let hash = 0;
  for (let i = 0; i < symbol.length; i++) hash = symbol.charCodeAt(i) + ((hash << 5) - hash);
  return AVATAR_COLORS[Math.abs(hash) % AVATAR_COLORS.length];
}

interface CompanyLogoProps {
  symbol: string;
  className?: string;
  /** Max characters shown in text fallback (default 3) */
  chars?: number;
}

/**
 * Company logo with automatic text-avatar fallback.
 * Uses Logo.dev CDN for high-quality brand logos.
 */
export function CompanyLogo({ symbol, className, chars = 3 }: CompanyLogoProps) {
  const [imgError, setImgError] = useState(false);
  const logoUrl = getStockLogoUrl(symbol);

  if (!logoUrl || imgError) {
    return (
      <div className={cn('rounded-full flex items-center justify-center font-bold text-xs border', avatarColor(symbol), className)}>
        {symbol.slice(0, chars)}
      </div>
    );
  }

  return (
    <div className={cn('rounded-full border border-border-subtle bg-surface-2 overflow-hidden flex items-center justify-center p-1.5', className)}>
      <img
        src={logoUrl}
        alt={symbol}
        className="w-full h-full object-contain"
        onError={() => setImgError(true)}
        loading="lazy"
        referrerPolicy="origin"
      />
    </div>
  );
}
