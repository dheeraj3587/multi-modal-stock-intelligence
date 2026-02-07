import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Search,
  LineChart,
  ClipboardCheck,
  Brain,
  TrendingUp,
  Briefcase,
  BarChart3,
  Zap,
} from 'lucide-react';
import { cn } from '../../lib/utils';
import { ThemeToggle } from './ThemeToggle';

const NAV_ITEMS = [
  { to: '/', icon: LayoutDashboard, label: 'Home' },
  { to: '/dashboard', icon: BarChart3, label: 'Dashboard' },
  { to: '/screener', icon: Search, label: 'Screener' },
  { to: '/scorecards', icon: ClipboardCheck, label: 'Scorecards' },
  { to: '/forecasts', icon: LineChart, label: 'Forecasts' },
  { to: '/sentiment', icon: Brain, label: 'Sentiment' },
  { to: '/growth', icon: TrendingUp, label: 'Growth' },
  { to: '/portfolio', icon: Briefcase, label: 'Portfolio' },
  { to: '/trading', icon: Zap, label: 'Trading' },
];

export function Sidebar() {
  return (
    <aside className="fixed inset-y-0 left-0 z-30 flex w-56 flex-col border-r border-border bg-surface-0">
      {/* Logo */}
      <div className="flex h-14 items-center gap-2.5 px-5 border-b border-border">
        <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-accent">
          <BarChart3 className="h-4 w-4 text-white" />
        </div>
        <span className="text-sm font-bold text-text-primary tracking-tight">StockSense</span>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto py-3 px-2 space-y-0.5">
        {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-all duration-150',
                isActive
                  ? 'bg-accent text-white font-medium shadow-sm'
                  : 'text-text-secondary hover:bg-surface-2 hover:text-text-primary'
              )
            }
          >
            <Icon className="h-4 w-4 flex-shrink-0" />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="flex items-center justify-between border-t border-border px-4 py-3">
        <span className="text-[10px] text-text-tertiary">v1.0</span>
        <ThemeToggle />
      </div>
    </aside>
  );
}
