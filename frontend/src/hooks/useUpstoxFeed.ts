/**
 * useUpstoxFeed — React hook for live LTP ticks from the Upstox WS.
 *
 * Manages connect/disconnect lifecycle and provides a Map of
 * instrumentKey → latest LTP that updates in real time.
 *
 * Falls back gracefully: if there's no Upstox token the hook
 * simply stays in "disconnected" state.
 */

import { useEffect, useState, useCallback, useRef } from 'react';
import { upstoxWs, type LtpTick, type ConnectionStatus } from '../services/upstoxWebSocket';
import { AUTH_TOKEN_STORAGE_KEY } from '../lib/api';

export interface LivePrices {
  /** instrumentKey → ltp */
  [instrumentKey: string]: number;
}

interface UseUpstoxFeedReturn {
  /** Latest LTP prices keyed by instrument_key */
  prices: LivePrices;
  /** Connection status */
  status: ConnectionStatus;
  /** Whether connected and receiving data */
  isLive: boolean;
  /** Reconnect manually */
  reconnect: () => void;
}

export function useUpstoxFeed(instrumentKeys: string[]): UseUpstoxFeedReturn {
  const [prices, setPrices] = useState<LivePrices>({});
  const [status, setStatus] = useState<ConnectionStatus>('disconnected');
  const keysRef = useRef<string[]>(instrumentKeys);

  // Keep ref in sync
  keysRef.current = instrumentKeys;

  // Connect WS when token available
  const connect = useCallback(() => {
    const token = localStorage.getItem(AUTH_TOKEN_STORAGE_KEY);
    if (!token) return;
    upstoxWs.connect(token);
  }, []);

  useEffect(() => {
    connect();

    const unsubStatus = upstoxWs.onStatus(setStatus);

    const unsubTick = upstoxWs.onTick((tick: LtpTick) => {
      setPrices((prev) => {
        if (prev[tick.instrumentKey] === tick.ltp) return prev;
        return { ...prev, [tick.instrumentKey]: tick.ltp };
      });
    });

    return () => {
      unsubStatus();
      unsubTick();
    };
  }, [connect]);

  // Manage subscriptions when keys change
  useEffect(() => {
    if (instrumentKeys.length === 0) return;
    upstoxWs.subscribe(instrumentKeys);

    return () => {
      // Only unsubscribe keys that are no longer needed
      upstoxWs.unsubscribe(instrumentKeys);
    };
  }, [instrumentKeys.join(',')]); // eslint-disable-line react-hooks/exhaustive-deps

  const reconnect = useCallback(() => {
    upstoxWs.disconnect();
    connect();
  }, [connect]);

  return {
    prices,
    status,
    isLive: status === 'connected',
    reconnect,
  };
}
