import { useState, useEffect } from 'react';
import { NavLink, useNavigate, useSearchParams } from 'react-router-dom';
import {
  BarChart3,
  Search,
  Bell,
  ChevronDown,
  Menu,
  X,
  Link2,
  Check,
} from 'lucide-react';
import { cn } from '../../lib/utils';
import { ThemeToggle } from './ThemeToggle';

const PRIMARY_NAV = [
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/screener', label: 'Screener' },
  { to: '/scorecards', label: 'Scorecards' },
  { to: '/portfolio', label: 'Portfolio' },
  { to: '/trading', label: 'Trade' },
  { to: '/chat', label: 'AI Chat' },
];

const MARKET_NAV = [
  { to: '/forecasts', label: 'Forecasts' },
  { to: '/sentiment', label: 'Sentiment' },
  { to: '/growth', label: 'Growth' },
];

export function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [marketOpen, setMarketOpen] = useState(false);
  const [upstoxConnected, setUpstoxConnected] = useState(false);
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  // Check if redirected from Upstox OAuth
  useEffect(() => {
    if (searchParams.get('upstox_connected') === 'true') {
      setUpstoxConnected(true);
      localStorage.setItem('upstox_connected', 'true');
      // Clean URL
      window.history.replaceState({}, '', window.location.pathname);
    } else if (localStorage.getItem('upstox_connected') === 'true') {
      setUpstoxConnected(true);
    }
  }, [searchParams]);

  const handleConnectUpstox = () => {
    window.location.href = 'http://localhost:8000/auth/upstox/login';
  };

  return (
    <nav className="sticky top-0 z-50 bg-surface-0/90 backdrop-blur-md border-b border-border">
      <div className="px-4 lg:px-6">
        <div className="flex items-center justify-between h-16">
          {/* ── Left: Logo + Nav ── */}
          <div className="flex items-center gap-8">
            {/* Logo */}
            <NavLink to="/" className="flex items-center gap-2 shrink-0">
              <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-accent">
                <BarChart3 className="h-4.5 w-4.5 text-white" />
              </div>
              <span className="text-lg font-bold text-accent tracking-tight">stocksense</span>
            </NavLink>

            {/* Desktop nav links */}
            <div className="hidden md:flex items-center gap-1">
              {PRIMARY_NAV.map(({ to, label }) => (
                <NavLink
                  key={to}
                  to={to}
                  className={({ isActive }) =>
                    cn(
                      'px-3 py-1.5 rounded-lg text-[15px] font-medium transition-colors',
                      isActive
                        ? 'text-text-primary font-semibold'
                        : 'text-text-tertiary hover:text-text-primary',
                    )
                  }
                >
                  {label}
                </NavLink>
              ))}

              {/* Market dropdown */}
              <div
                className="relative"
                onMouseEnter={() => setMarketOpen(true)}
                onMouseLeave={() => setMarketOpen(false)}
              >
                <button
                  onClick={() => setMarketOpen((o) => !o)}
                  className={cn(
                    'flex items-center gap-1 px-3 py-1.5 rounded-lg text-[15px] font-medium transition-colors',
                    marketOpen ? 'text-text-primary' : 'text-text-tertiary hover:text-text-primary',
                  )}
                >
                  Market
                  <ChevronDown className={cn('h-3.5 w-3.5 transition-transform', marketOpen && 'rotate-180')} />
                </button>

                {marketOpen && (
                  <div className="absolute top-full left-0 pt-1 z-50">
                    <div className="w-40 rounded-xl border border-border bg-surface-0 shadow-elevated py-1">
                      {MARKET_NAV.map(({ to, label }) => (
                        <NavLink
                          key={to}
                          to={to}
                          onClick={() => setMarketOpen(false)}
                          className={({ isActive }) =>
                            cn(
                              'block px-4 py-2 text-sm transition-colors',
                              isActive
                                ? 'text-accent font-medium bg-accent/5'
                                : 'text-text-secondary hover:bg-surface-2 hover:text-text-primary',
                            )
                          }
                        >
                          {label}
                        </NavLink>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* ── Right: Actions ── */}
          <div className="flex items-center gap-2">
            {/* Upstox Connect Button */}
            <button
              onClick={handleConnectUpstox}
              disabled={upstoxConnected}
              className={cn(
                'hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium transition-all',
                upstoxConnected
                  ? 'bg-green-500/20 text-green-400 cursor-default'
                  : 'bg-accent hover:bg-accent/80 text-white cursor-pointer'
              )}
            >
              {upstoxConnected ? (
                <><Check className="h-3.5 w-3.5" /> Connected</>
              ) : (
                <><Link2 className="h-3.5 w-3.5" /> Connect Upstox</>
              )}
            </button>

            {/* Search */}
            <div className="hidden sm:flex items-center gap-2 bg-surface-2 border border-border-subtle rounded-full px-3 py-1.5">
              <Search className="h-4 w-4 text-text-tertiary" />
              <input
                type="text"
                placeholder="Search stocks..."
                className="bg-transparent text-sm text-text-primary placeholder:text-text-tertiary outline-none w-36"
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && (e.target as HTMLInputElement).value) {
                    navigate(`/stock/${(e.target as HTMLInputElement).value.toUpperCase()}`);
                    (e.target as HTMLInputElement).value = '';
                  }
                }}
              />
            </div>

            {/* Notifications */}
            <button className="relative flex items-center justify-center w-9 h-9 rounded-full hover:bg-surface-2 transition-colors text-text-tertiary hover:text-text-primary">
              <Bell className="h-4 w-4" />
              <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-accent rounded-full border-2 border-surface-0" />
            </button>

            {/* Theme toggle */}
            <ThemeToggle />

            {/* Mobile hamburger */}
            <button
              className="md:hidden flex items-center justify-center w-9 h-9 rounded-full hover:bg-surface-2 transition-colors text-text-tertiary"
              onClick={() => setMobileOpen(!mobileOpen)}
            >
              {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </button>
          </div>
        </div>
      </div>

      {/* ── Mobile menu ── */}
      {mobileOpen && (
        <div className="md:hidden border-t border-border bg-surface-0 px-4 py-3 space-y-1">
          {[...PRIMARY_NAV, ...MARKET_NAV].map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              onClick={() => setMobileOpen(false)}
              className={({ isActive }) =>
                cn(
                  'block px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-accent text-white'
                    : 'text-text-secondary hover:bg-surface-2 hover:text-text-primary',
                )
              }
            >
              {label}
            </NavLink>
          ))}
        </div>
      )}
    </nav>
  );
}
