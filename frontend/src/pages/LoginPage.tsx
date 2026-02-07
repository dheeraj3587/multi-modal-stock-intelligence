import { useState, useMemo, useCallback, FormEvent } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowRight } from 'lucide-react';
import { getLoginUrl } from '../lib/api';

type AuthMode = 'login' | 'register';

export function LoginPage() {
  const [mode, setMode] = useState<AuthMode>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');

  const handleUpstoxLogin = () => {
    window.location.href = getLoginUrl();
  };

  const [formError, setFormError] = useState('');
  const [formLoading, setFormLoading] = useState(false);

  const handleSubmit = useCallback(
    (e: FormEvent) => {
      e.preventDefault();
      setFormError('');

      if (mode === 'register') {
        if (password !== confirmPassword) {
          setFormError('Passwords do not match.');
          return;
        }
        if (password.length < 6) {
          setFormError('Password must be at least 6 characters.');
          return;
        }
      }

      // Backend only supports Upstox OAuth — redirect to Upstox login
      setFormLoading(true);
      window.location.href = getLoginUrl();
    },
    [mode, email, password, name, confirmPassword],
  );

  const toggleMode = () => {
    setMode((m) => (m === 'login' ? 'register' : 'login'));
  };

  /* Decorative bar heights for left panel */
  const bars = useMemo(
    () =>
      Array.from({ length: 29 }, () => Math.floor(Math.random() * 80) + 10),
    [],
  );

  return (
    <div className="flex min-h-screen w-full flex-col lg:flex-row font-sans">
      {/* ───── Left Panel (Red) ───── */}
      <div className="relative w-full lg:w-7/12 bg-accent overflow-hidden flex flex-col justify-between p-8 lg:p-12 text-white">
        {/* Background bars */}
        <div className="absolute inset-0 opacity-20 pointer-events-none flex items-end justify-between gap-1 px-4 pb-0">
          {bars.map((h, i) => (
            <div
              key={i}
              className="w-2 rounded-t-sm bg-black/40"
              style={{ height: `${h}%` }}
            />
          ))}
        </div>

        {/* Wave SVG */}
        <svg
          className="absolute bottom-0 left-0 w-full h-64 opacity-60 pointer-events-none"
          preserveAspectRatio="none"
          viewBox="0 0 1440 320"
        >
          <defs>
            <linearGradient id="waveGrad" x1="0%" x2="0%" y1="0%" y2="100%">
              <stop offset="0%" style={{ stopColor: '#fff', stopOpacity: 1 }} />
              <stop
                offset="100%"
                style={{ stopColor: '#fff', stopOpacity: 0 }}
              />
            </linearGradient>
          </defs>
          <path
            d="M0,224L48,213.3C96,203,192,181,288,181.3C384,181,480,203,576,224C672,245,768,267,864,234.7C960,203,1056,117,1152,85.3C1248,53,1344,75,1392,85.3L1440,96L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"
            fill="url(#waveGrad)"
            fillOpacity="0.3"
          />
          <path
            d="M0,224L48,213.3C96,203,192,181,288,181.3C384,181,480,203,576,224C672,245,768,267,864,234.7C960,203,1056,117,1152,85.3C1248,53,1344,75,1392,85.3L1440,96"
            fill="none"
            stroke="white"
            strokeWidth="3"
          />
        </svg>

        {/* Logo */}
        <div className="relative z-10 flex items-center gap-2">
          <div className="h-8 w-8 rounded bg-white text-accent flex items-center justify-center font-bold text-xl select-none">
            S
          </div>
          <span className="text-2xl font-bold tracking-tight">stocksense</span>
        </div>

        {/* Hero Stats */}
        <motion.div
          className="relative z-10 my-auto animate-float"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        >
          <h2 className="text-sm font-medium opacity-80 uppercase tracking-widest mb-2">
            Portfolio Value
          </h2>
          <div className="flex items-baseline gap-4 mb-8 flex-wrap">
            <h1 className="text-5xl md:text-7xl font-bold tracking-tighter">
              ₹26,02,974.00
            </h1>
            <div className="flex items-center gap-1 bg-white/20 backdrop-blur-md px-3 py-1 rounded-full text-sm font-medium">
              <span className="material-symbols-outlined text-sm">
                arrow_outward
              </span>
              1.54%
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 max-w-md">
            <div className="bg-black/20 backdrop-blur-sm p-5 rounded-2xl border border-white/10">
              <p className="text-sm opacity-70 mb-1">Realized P&L</p>
              <p className="text-2xl font-semibold text-white">
                + ₹1,28,429.00
              </p>
              <p className="text-xs mt-2 opacity-60 flex items-center gap-1">
                <span className="material-symbols-outlined text-[14px]">
                  trending_up
                </span>
                +0.9% Today
              </p>
            </div>
            <div className="bg-black/20 backdrop-blur-sm p-5 rounded-2xl border border-white/10">
              <p className="text-sm opacity-70 mb-1">Projected Growth</p>
              <p className="text-2xl font-semibold text-white">
                + ₹35,021.10
              </p>
              <p className="text-xs mt-2 opacity-60 flex items-center gap-1">
                <span className="material-symbols-outlined text-[14px]">
                  trending_up
                </span>
                +0.6% Today
              </p>
            </div>
          </div>
        </motion.div>

        {/* Footer */}
        <div className="relative z-10 text-xs opacity-60">
          <p>
            © 2026 StockSense. NSE/BSE data may be delayed.
          </p>
        </div>
      </div>

      {/* ───── Right Panel (Form) ───── */}
      <div className="w-full lg:w-5/12 flex items-center justify-center p-8 lg:p-16 bg-surface-0 text-text-primary relative">
        <div className="w-full max-w-md space-y-8">
          {/* Heading */}
          <div className="text-center lg:text-left">
            <div className="inline-flex items-center justify-center h-12 w-12 rounded-xl bg-accent/10 text-accent mb-6 lg:hidden">
              <span className="font-bold text-xl">S</span>
            </div>
            <AnimatePresence mode="wait">
              <motion.div
                key={mode}
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                transition={{ duration: 0.2 }}
              >
                <h2 className="text-3xl lg:text-4xl font-bold tracking-tight">
                  {mode === 'login' ? 'Welcome back' : 'Create account'}
                </h2>
                <p className="mt-2 text-text-secondary">
                  {mode === 'login'
                    ? 'Access your portfolio risk score and trading tools.'
                    : 'Get started with your free StockSense account.'}
                </p>
              </motion.div>
            </AnimatePresence>
          </div>

          {/* Upstox OAuth */}
          <div className="pt-4">
            <button
              type="button"
              onClick={handleUpstoxLogin}
              className="group relative flex w-full justify-center items-center gap-3 rounded-xl bg-surface-2 px-4 py-4 text-sm font-semibold text-text-primary hover:bg-surface-3 focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2 transition-all shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 duration-200 border border-border"
            >
              <svg
                className="h-5 w-5 text-accent"
                fill="currentColor"
                viewBox="0 0 24 24"
              >
                <path d="M12 2L2 7l10 5 10-5-10-5zm0 9l2.5-1.25L12 8.5l-2.5 1.25L12 11zm0 2.5l-5-2.5-5 2.5L12 22l10-8.5-5-2.5-5 2.5z" />
              </svg>
              <span className="text-lg">Login with Upstox</span>
              <span className="absolute right-4 opacity-0 group-hover:opacity-100 transition-opacity text-accent">
                <ArrowRight className="h-5 w-5" />
              </span>
            </button>

            <div className="relative flex py-6 items-center">
              <div className="flex-grow border-t border-border" />
              <span className="flex-shrink-0 mx-4 text-text-tertiary text-xs uppercase tracking-wider">
                Or verify with email
              </span>
              <div className="flex-grow border-t border-border" />
            </div>
          </div>

          {/* Email/Password Form */}
          <form onSubmit={handleSubmit} className="space-y-5">
            <AnimatePresence mode="wait">
              {mode === 'register' && (
                <motion.div
                  key="name-field"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.2 }}
                  className="overflow-hidden"
                >
                  <label
                    htmlFor="name"
                    className="block text-sm font-medium leading-6 text-text-secondary"
                  >
                    Full Name
                  </label>
                  <div className="mt-2 relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-text-tertiary">
                      <span className="material-symbols-outlined text-[20px]">
                        person
                      </span>
                    </div>
                    <input
                      id="name"
                      name="name"
                      type="text"
                      required
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="John Doe"
                      className="block w-full rounded-xl border-0 py-3.5 pl-10 text-text-primary shadow-sm ring-1 ring-inset ring-border placeholder:text-text-tertiary focus:ring-2 focus:ring-inset focus:ring-accent bg-surface-1 sm:text-sm sm:leading-6 transition-shadow"
                    />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <div>
              <label
                htmlFor="email"
                className="block text-sm font-medium leading-6 text-text-secondary"
              >
                Email address
              </label>
              <div className="mt-2 relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-text-tertiary">
                  <span className="material-symbols-outlined text-[20px]">
                    mail
                  </span>
                </div>
                <input
                  id="email"
                  name="email"
                  type="email"
                  autoComplete="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="block w-full rounded-xl border-0 py-3.5 pl-10 text-text-primary shadow-sm ring-1 ring-inset ring-border placeholder:text-text-tertiary focus:ring-2 focus:ring-inset focus:ring-accent bg-surface-1 sm:text-sm sm:leading-6 transition-shadow"
                />
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between">
                <label
                  htmlFor="password"
                  className="block text-sm font-medium leading-6 text-text-secondary"
                >
                  Password
                </label>
                {mode === 'login' && (
                  <div className="text-sm">
                    <a
                      href="#"
                      className="font-semibold text-accent hover:text-accent-hover"
                    >
                      Forgot password?
                    </a>
                  </div>
                )}
              </div>
              <div className="mt-2 relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-text-tertiary">
                  <span className="material-symbols-outlined text-[20px]">
                    lock
                  </span>
                </div>
                <input
                  id="password"
                  name="password"
                  type="password"
                  autoComplete={
                    mode === 'login' ? 'current-password' : 'new-password'
                  }
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="block w-full rounded-xl border-0 py-3.5 pl-10 text-text-primary shadow-sm ring-1 ring-inset ring-border placeholder:text-text-tertiary focus:ring-2 focus:ring-inset focus:ring-accent bg-surface-1 sm:text-sm sm:leading-6 transition-shadow"
                />
              </div>
            </div>

            <AnimatePresence mode="wait">
              {mode === 'register' && (
                <motion.div
                  key="confirm-field"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.2 }}
                  className="overflow-hidden"
                >
                  <label
                    htmlFor="confirmPassword"
                    className="block text-sm font-medium leading-6 text-text-secondary"
                  >
                    Confirm Password
                  </label>
                  <div className="mt-2 relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-text-tertiary">
                      <span className="material-symbols-outlined text-[20px]">
                        lock
                      </span>
                    </div>
                    <input
                      id="confirmPassword"
                      name="confirmPassword"
                      type="password"
                      autoComplete="new-password"
                      required
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      className="block w-full rounded-xl border-0 py-3.5 pl-10 text-text-primary shadow-sm ring-1 ring-inset ring-border placeholder:text-text-tertiary focus:ring-2 focus:ring-inset focus:ring-accent bg-surface-1 sm:text-sm sm:leading-6 transition-shadow"
                    />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <div>
              {formError && (
                <p className="text-sm text-negative text-center font-medium">{formError}</p>
              )}
              <button
                type="submit"
                disabled={formLoading}
                className="flex w-full justify-center rounded-xl bg-accent border border-accent px-3 py-3.5 text-sm font-semibold leading-6 text-white shadow-sm hover:bg-accent/90 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent transition-colors disabled:opacity-60"
              >
                {formLoading ? 'Redirecting...' : mode === 'login' ? 'Sign In' : 'Create Account'}
              </button>
            </div>
          </form>

          {/* Toggle login / register */}
          <p className="mt-10 text-center text-sm text-text-secondary">
            {mode === 'login' ? 'Not a member?' : 'Already have an account?'}
            <button
              type="button"
              onClick={toggleMode}
              className="ml-1 font-semibold leading-6 text-accent hover:text-accent-hover"
            >
              {mode === 'login'
                ? 'Create a free account'
                : 'Sign in instead'}
            </button>
          </p>

          {/* Market Status Footer */}
          <div className="mt-8 pt-6 border-t border-border">
            <div className="flex items-center justify-between text-xs text-text-tertiary mb-2">
              <span>Market Status</span>
              <span className="flex items-center gap-1 text-positive">
                <div className="h-1.5 w-1.5 rounded-full bg-positive animate-pulse" />
                NSE • Open
              </span>
            </div>
            <div className="flex space-x-4 overflow-hidden">
              <div className="flex items-center gap-2">
                <span className="font-bold text-text-secondary">NIFTY 50</span>
                <span className="text-positive">+0.84%</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-text-secondary">SENSEX</span>
                <span className="text-positive">+0.62%</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-text-secondary">RELIANCE</span>
                <span className="text-negative">-0.21%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
