import { cn } from '../../lib/utils';

interface BadgeProps {
  variant?: 'default' | 'positive' | 'negative' | 'warning' | 'info';
  size?: 'sm' | 'md';
  children: React.ReactNode;
  className?: string;
}

export function Badge({ variant = 'default', size = 'sm', className, children }: BadgeProps) {
  const base = 'inline-flex items-center rounded-lg font-medium';

  const variants = {
    default: 'bg-surface-2 text-text-secondary',
    positive: 'bg-positive/10 text-positive',
    negative: 'bg-negative/10 text-negative',
    warning: 'bg-warning/10 text-warning',
    info: 'bg-info/10 text-info',
  };

  const sizes = {
    sm: 'px-1.5 py-0.5 text-[10px]',
    md: 'px-2 py-1 text-xs',
  };

  return (
    <span className={cn(base, variants[variant], sizes[size], className)}>{children}</span>
  );
}
