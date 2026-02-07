/**
 * usePaperPortfolio — React hook wrapping the paper trading service.
 *
 * Provides reactive state for holdings, trade history, and a
 * convenient executeTrade that shows toast-style feedback.
 */

import { useCallback, useSyncExternalStore } from 'react';
import {
  paperTradingService,
  type PaperHolding,
  type PaperTrade,
} from '../services/paperTrading';

interface UsePaperPortfolioReturn {
  holdings: PaperHolding[];
  history: PaperTrade[];
  executeTrade: (params: {
    symbol: string;
    name: string;
    sector: string;
    instrumentKey: string;
    side: 'buy' | 'sell';
    quantity: number;
    price: number;
  }) => { success: boolean; message: string; trade?: PaperTrade };
  reset: () => void;
  getHolding: (symbol: string) => PaperHolding | undefined;
}

/** Internal version counter to drive re-renders */
function subscribeToService(onStoreChange: () => void) {
  return paperTradingService.subscribe(onStoreChange);
}

let snapshotId = 0;
function getSnapshot() {
  return snapshotId;
}

// Bump snapshot on every service notification
paperTradingService.subscribe(() => { snapshotId++; });

export function usePaperPortfolio(): UsePaperPortfolioReturn {
  // This forces re-render when service state changes
  useSyncExternalStore(subscribeToService, getSnapshot);

  const holdings = paperTradingService.getHoldings();
  const history = paperTradingService.getHistory();

  const executeTrade = useCallback(
    (params: {
      symbol: string;
      name: string;
      sector: string;
      instrumentKey: string;
      side: 'buy' | 'sell';
      quantity: number;
      price: number;
    }) => {
      try {
        const trade = paperTradingService.executeTrade(params);
        return {
          success: true,
          message: `Paper ${params.side} executed: ${params.quantity} × ${params.symbol} @ ₹${params.price.toFixed(2)}`,
          trade,
        };
      } catch (err) {
        return {
          success: false,
          message: err instanceof Error ? err.message : 'Trade failed',
        };
      }
    },
    []
  );

  const reset = useCallback(() => {
    paperTradingService.reset();
  }, []);

  const getHolding = useCallback(
    (symbol: string) => paperTradingService.getHolding(symbol),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [holdings]
  );

  return { holdings, history, executeTrade, reset, getHolding };
}
