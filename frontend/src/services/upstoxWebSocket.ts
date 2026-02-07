/**
 * Upstox WebSocket Service
 *
 * Connects to the Upstox v2 binary market-data feed.
 * Decodes protobuf frames → emits LTP ticks via callbacks.
 *
 * Proto schema: https://assets.upstox.com/feed/market-data-feed/v1/MarketDataFeed.proto
 * WS Endpoint:  wss://api.upstox.com/v2/feed/market-data-feed
 */

import protobuf from 'protobufjs';

/* ── Types ── */

export interface LtpTick {
  instrumentKey: string;
  ltp: number;
  ltq: number; // last traded quantity
  ltt: number; // last traded time (epoch ms)
  cp: number;  // close price (prev day)
}

export type TickHandler = (tick: LtpTick) => void;
export type StatusHandler = (status: ConnectionStatus) => void;
export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

/* ── Constants ── */

const WS_URL = 'wss://api.upstox.com/v2/feed/market-data-feed';
const PROTO_URL = 'https://assets.upstox.com/feed/market-data-feed/v1/MarketDataFeed.proto';

const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 30000;

/* ── Proto singleton ── */

let protoRootPromise: Promise<protobuf.Root> | null = null;

function loadProtoRoot(): Promise<protobuf.Root> {
  if (!protoRootPromise) {
    protoRootPromise = protobuf.load(PROTO_URL).catch((err) => {
      protoRootPromise = null; // allow retry
      throw err;
    });
  }
  return protoRootPromise;
}

/* ── Service class ── */

class UpstoxWebSocketService {
  private ws: WebSocket | null = null;
  private status: ConnectionStatus = 'disconnected';
  private token: string | null = null;
  private subscribedKeys: Set<string> = new Set();
  private reconnectAttempt = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  private tickHandlers: Set<TickHandler> = new Set();
  private statusHandlers: Set<StatusHandler> = new Set();

  private decoderType: protobuf.Type | null = null;

  /* ── Public API ── */

  /**
   * Connect to the Upstox WS feed.
   * If already connected, this is a no-op.
   * @param accessToken  Upstox access token
   */
  async connect(accessToken: string) {
    if (this.ws && this.status === 'connected' && this.token === accessToken) return;
    this.token = accessToken;

    // Pre-load proto
    try {
      const root = await loadProtoRoot();
      this.decoderType = root.lookupType('com.upstox.marketdatafeeder.rpc.proto.MarketDataFeed');
    } catch (err) {
      console.warn('[UpstoxWS] Failed to load proto schema, falling back to JSON parsing', err);
      this.decoderType = null;
    }

    this.openSocket();
  }

  /** Disconnect and clean up */
  disconnect() {
    this.clearReconnect();
    this.subscribedKeys.clear();
    if (this.ws) {
      this.ws.onclose = null; // prevent reconnect
      this.ws.close();
      this.ws = null;
    }
    this.setStatus('disconnected');
  }

  /** Subscribe to instrument keys (additive) */
  subscribe(instrumentKeys: string[]) {
    const newKeys = instrumentKeys.filter((k) => !this.subscribedKeys.has(k));
    if (newKeys.length === 0) return;
    newKeys.forEach((k) => this.subscribedKeys.add(k));
    this.sendSubscription();
  }

  /** Unsubscribe instrument keys */
  unsubscribe(instrumentKeys: string[]) {
    instrumentKeys.forEach((k) => this.subscribedKeys.delete(k));
    // Send updated subscription with only remaining keys
    this.sendSubscription();
  }

  /** Register a tick callback */
  onTick(handler: TickHandler): () => void {
    this.tickHandlers.add(handler);
    return () => this.tickHandlers.delete(handler);
  }

  /** Register a status callback */
  onStatus(handler: StatusHandler): () => void {
    this.statusHandlers.add(handler);
    handler(this.status); // fire immediately with current
    return () => this.statusHandlers.delete(handler);
  }

  getStatus(): ConnectionStatus {
    return this.status;
  }

  /* ── Internal ── */

  private openSocket() {
    this.clearReconnect();
    this.setStatus('connecting');

    this.ws = new WebSocket(WS_URL);

    // Upstox requires authorization via first binary message
    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => {
      this.reconnectAttempt = 0;
      this.setStatus('connected');
      // Send auth + subscriptions
      this.sendSubscription();
    };

    this.ws.onmessage = (event: MessageEvent) => {
      this.handleMessage(event.data);
    };

    this.ws.onerror = () => {
      this.setStatus('error');
    };

    this.ws.onclose = () => {
      this.setStatus('disconnected');
      this.scheduleReconnect();
    };
  }

  private sendSubscription() {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    const keys = Array.from(this.subscribedKeys);
    if (keys.length === 0) return;

    const payload = JSON.stringify({
      guid: `sub-${Date.now()}`,
      method: 'sub',
      data: {
        mode: 'ltpc',
        instrumentKeys: keys,
      },
    });

    // Upstox WS expects auth header; for the browser-side connection
    // we send it as a text message since custom headers aren't supported.
    // Some Upstox docs say to include Authorization in the first message as JSON.
    const authPayload = JSON.stringify({
      guid: `auth-${Date.now()}`,
      method: 'authorize',
      data: {
        token: this.token,
      },
    });

    this.ws.send(authPayload);
    // Small delay so auth processes first
    setTimeout(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(payload);
      }
    }, 100);
  }

  private handleMessage(data: ArrayBuffer | string) {
    try {
      let parsed: Record<string, unknown>;

      if (data instanceof ArrayBuffer && this.decoderType) {
        // Binary protobuf
        const uint8 = new Uint8Array(data);
        const decoded = this.decoderType.decode(uint8);
        parsed = this.decoderType.toObject(decoded, {
          longs: Number,
          enums: String,
          defaults: true,
        }) as Record<string, unknown>;
      } else if (typeof data === 'string') {
        parsed = JSON.parse(data);
      } else {
        // ArrayBuffer without proto — try raw JSON
        const text = new TextDecoder().decode(data as ArrayBuffer);
        parsed = JSON.parse(text);
      }

      this.extractTicks(parsed);
    } catch (err) {
      console.warn('[UpstoxWS] Failed to decode message', err);
    }
  }

  /**
   * Upstox feed responses have a `feeds` map keyed by instrument_key.
   * Each entry has `ltpc` (or `ff`) data.
   */
  private extractTicks(msg: Record<string, unknown>) {
    const feeds = msg['feeds'] as Record<string, Record<string, unknown>> | undefined;
    if (!feeds) return;

    for (const [instrumentKey, feedData] of Object.entries(feeds)) {
      // ltpc mode
      const ltpc = feedData['ltpc'] as Record<string, unknown> | undefined;
      if (ltpc) {
        const tick: LtpTick = {
          instrumentKey,
          ltp: Number(ltpc['ltp'] ?? 0),
          ltq: Number(ltpc['ltq'] ?? 0),
          ltt: Number(ltpc['ltt'] ?? 0),
          cp: Number(ltpc['cp'] ?? 0),
        };
        this.emitTick(tick);
        continue;
      }

      // Full feed fallback
      const ff = feedData['ff'] as Record<string, unknown> | undefined;
      if (ff) {
        const ltpcNested = ff['ltpc'] as Record<string, unknown> | undefined;
        const ltp = Number(ltpcNested?.['ltp'] ?? ff['ltpc_ltp'] ?? 0);
        const tick: LtpTick = {
          instrumentKey,
          ltp,
          ltq: Number(ltpcNested?.['ltq'] ?? 0),
          ltt: Number(ltpcNested?.['ltt'] ?? 0),
          cp: Number(ltpcNested?.['cp'] ?? 0),
        };
        this.emitTick(tick);
      }
    }
  }

  private emitTick(tick: LtpTick) {
    this.tickHandlers.forEach((fn) => {
      try { fn(tick); } catch (e) { console.error('[UpstoxWS] Tick handler error', e); }
    });
  }

  private setStatus(status: ConnectionStatus) {
    this.status = status;
    this.statusHandlers.forEach((fn) => {
      try { fn(status); } catch (e) { console.error('[UpstoxWS] Status handler error', e); }
    });
  }

  private scheduleReconnect() {
    if (!this.token) return;
    const delay = Math.min(RECONNECT_BASE_MS * 2 ** this.reconnectAttempt, RECONNECT_MAX_MS);
    this.reconnectAttempt++;
    this.reconnectTimer = setTimeout(() => this.openSocket(), delay);
  }

  private clearReconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }
}

/** Singleton */
export const upstoxWs = new UpstoxWebSocketService();
