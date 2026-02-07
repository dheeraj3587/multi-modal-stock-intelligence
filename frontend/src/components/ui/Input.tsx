import { Search } from 'lucide-react';
import { cn } from '../../lib/utils';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  icon?: boolean;
}

export function Input({ className, icon, ...props }: InputProps) {
  if (icon) {
    return (
      <div className="relative">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-tertiary" />
        <input
          className={cn(
            'h-10 w-full rounded-xl border border-border bg-surface-1 pl-9 pr-3 text-sm text-text-primary placeholder:text-text-tertiary focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent transition-all duration-150',
            className
          )}
          {...props}
        />
      </div>
    );
  }

  return (
    <input
      className={cn(
        'h-10 w-full rounded-xl border border-border bg-surface-1 px-3 text-sm text-text-primary placeholder:text-text-tertiary focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent transition-all duration-150',
        className
      )}
      {...props}
    />
  );
}
