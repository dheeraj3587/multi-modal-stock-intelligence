import { motion, type HTMLMotionProps } from 'framer-motion';
import { cn } from '../../lib/utils';

interface CardProps extends HTMLMotionProps<'div'> {
  variant?: 'default' | 'interactive' | 'elevated';
}

export function Card({ className, variant = 'default', children, ...props }: CardProps) {
  const base = 'rounded-2xl border border-border-subtle bg-surface-0 p-4 shadow-card';
  const variants = {
    default: '',
    interactive: 'cursor-pointer transition-all duration-150 hover:bg-surface-0 hover:shadow-elevated hover:border-accent/20',
    elevated: 'shadow-elevated',
  };

  return (
    <motion.div className={cn(base, variants[variant], className)} {...props}>
      {children}
    </motion.div>
  );
}

export function CardHeader({ className, children }: { className?: string; children: React.ReactNode }) {
  return <div className={cn('mb-4 flex items-center justify-between', className)}>{children}</div>;
}

export function CardTitle({ className, children }: { className?: string; children: React.ReactNode }) {
  return (
    <h3
      className={cn(
        'text-[11px] font-semibold uppercase tracking-wider text-text-tertiary',
        className
      )}
    >
      {children}
    </h3>
  );
}

export function CardContent({ className, children }: { className?: string; children: React.ReactNode }) {
  return <div className={cn('', className)}>{children}</div>;
}
