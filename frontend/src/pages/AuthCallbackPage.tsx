import { useEffect, useMemo, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { motion } from 'framer-motion';
import { AlertCircle } from 'lucide-react';
import { api, AUTH_TOKEN_STORAGE_KEY } from '../lib/api';
import { Button } from '../components/ui/Button';

export function AuthCallbackPage() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const code = params.get('code');

  const [status, setStatus] = useState<'loading' | 'error' | 'success'>('loading');
  const [message, setMessage] = useState<string>('Completing Upstox sign-in…');

  const retryUrl = useMemo(() => '/login', []);

  useEffect(() => {
    let cancelled = false;

    async function run() {
      if (!code) {
        setStatus('error');
        setMessage('Missing OAuth code. Please retry login.');
        return;
      }

      try {
        const tokenResponse = await api.exchangeUpstoxCode(code);
        const accessToken = tokenResponse?.access_token;

        if (!accessToken) {
          throw new Error('No access_token returned by backend');
        }

        localStorage.setItem(AUTH_TOKEN_STORAGE_KEY, accessToken);
        // Back-compat for older builds
        localStorage.setItem('upstox_token', accessToken);

        if (cancelled) return;
        setStatus('success');
        setMessage('Signed in. Redirecting…');

        setTimeout(() => {
          if (!cancelled) navigate('/', { replace: true });
        }, 250);
      } catch (err) {
        if (cancelled) return;
        setStatus('error');
        setMessage(err instanceof Error ? err.message : 'Login failed');
      }
    }

    run();
    return () => {
      cancelled = true;
    };
  }, [code, navigate]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-surface-0 p-4">
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2, ease: 'easeOut' }}
        className="w-full max-w-sm"
      >
        <div className="rounded-xl border border-border bg-surface-1 p-8 shadow-elevated">
          <div className="text-sm text-text-secondary">
            {status === 'error' ? (
              <div className="space-y-4">
                <div className="flex items-center gap-2 text-negative">
                  <AlertCircle className="h-4 w-4" />
                  <span className="font-medium">Authentication failed</span>
                </div>
                <p className="text-xs text-text-tertiary leading-relaxed">{message}</p>
                <Button className="w-full" onClick={() => navigate(retryUrl, { replace: true })}>
                  Back to Login
                </Button>
              </div>
            ) : (
              <div className="space-y-2">
                <div className="text-sm font-medium text-text-primary">Signing you in</div>
                <p className="text-xs text-text-tertiary leading-relaxed">{message}</p>
              </div>
            )}
          </div>
        </div>
      </motion.div>
    </div>
  );
}
