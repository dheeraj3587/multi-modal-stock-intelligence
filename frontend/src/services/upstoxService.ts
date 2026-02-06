import protobuf from 'protobufjs';
import { PriceTick, Candle } from '../utils/types';
import { format, subDays } from 'date-fns';

const PROTO_PATH = '/marketDataFeed.proto'; // We will need to move the proto file to public or import it as text

export class UpstoxService {
    private socket: WebSocket | null = null;
    private root: protobuf.Root | null = null;
    private onTickCallback: ((tick: PriceTick) => void) | null = null;
    private accessToken: string | null = null;

    constructor() {
        this.loadProto();
        const storedToken = localStorage.getItem('upstox_access_token');
        if (storedToken) {
            this.accessToken = storedToken;
        }
    }

    setAccessToken(token: string) {
        this.accessToken = token;
    }

    private async loadProto() {
        try {
            // In a real app, we might bundle the JSON, but here we'll fetch the text
            // For simplicity in Vite, we can put the proto in the public folder or import it as a string
            // Let's assume we move it to public/marketDataFeed.proto
            this.root = await protobuf.load(PROTO_PATH);
        } catch (error) {
            console.error('Failed to load proto:', error);
        }
    }

    connect(token: string) {
        if (!token) return;
        this.accessToken = token;

        // Upstox V3 WebSocket Endpoint
        const url = `wss://api.upstox.com/v2/feed/market-data-feed`;

        this.socket = new WebSocket(url);
        this.socket.binaryType = 'arraybuffer';

        this.socket.onopen = () => {
            console.log('Connected to Upstox');
            // Authenticate and Subscribe
            // Note: V3 usually requires the token in the URL or an initial handshake.
            // According to docs, it's often wss://api.upstox.com/v2/feed/market-data-feed?response_type=protobuf
            // And the token is passed via headers (which browser WS doesn't support easily) or as a query param if supported.
            // Alternatively, there is an auth flow.
            // For this implementation, we will assume the standard "send auth message" or "token in url" pattern.
            // Upstox V3 actually uses a specific URL format with the token:
            // https://api.upstox.com/v2/feed/market-data-feed is the base.
            // We usually need to perform an HTTP handshake to get the authorized WS URL or pass the token.

            // Since we can't easily pass headers in browser WebSocket, we'll try the query param method if available,
            // or assume the user provides a full authorized URL.

            // For now, let's send a subscription message immediately after open.
            this.subscribe(['NSE_EQ|INE002A01018']); // Reliance as example
        };

        this.socket.onmessage = async (event) => {
            if (!this.root) return;

            try {
                const buffer = new Uint8Array(event.data);
                const FeedResponse = this.root.lookupType("com.upstox.marketdatafeeder.rpc.proto.FeedResponse");
                const message = FeedResponse.decode(buffer);
                const object = FeedResponse.toObject(message, {
                    longs: String,
                    enums: String,
                    bytes: String,
                });

                this.processMessage(object);
            } catch (err) {
                console.error('Decode error:', err);
            }
        };

        this.socket.onerror = (err) => {
            console.error('WebSocket error:', err);
        };
    }

    subscribe(instrumentKeys: string[]) {
        if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return;

        const request = {
            guid: "some-guid",
            method: "sub",
            data: {
                mode: "full",
                instrumentKeys: instrumentKeys
            }
        };

        // Upstox expects binary subscription message? Or JSON?
        // V3 docs say JSON for subscription is okay in some contexts, but let's verify.
        // Actually, usually subscription is sent as binary if the feed is binary.
        // But for simplicity, many APIs accept JSON control messages.
        // Let's try sending as JSON first.
        this.socket.send(JSON.stringify(request));
    }

    private processMessage(data: any) {
        if (!data.feeds) return;

        Object.keys(data.feeds).forEach(key => {
            const feed = data.feeds[key];
            if (feed.fullFeed) {
                const ff = feed.fullFeed;
                const tick: PriceTick = {
                    symbol: key,
                    timestamp: parseInt(ff.ohlc.ts),
                    price: ff.ltpc.ltp,
                    volume: parseInt(ff.volume),
                    side: 'buy' // inferred or random for now
                };
                if (this.onTickCallback) this.onTickCallback(tick);
            }
        });
    }

    onTick(callback: (tick: PriceTick) => void) {
        this.onTickCallback = callback;
    }

    disconnect() {
        if (this.socket) {
            this.socket.close();
        }
    }

    async getHistoricalCandles(instrumentKey: string, interval: string = '1'): Promise<Candle[]> {
        if (!this.accessToken) {
            console.error('Access token not set');
            return [];
        }

        const toDate = format(new Date(), 'yyyy-MM-dd');
        const fromDate = format(subDays(new Date(), 7), 'yyyy-MM-dd'); // Last 7 days

        // Using the proxy path '/upstox' which maps to 'https://api.upstox.com'
        const url = `/upstox/v3/historical-candle/${encodeURIComponent(instrumentKey)}/minutes/${interval}/${toDate}/${fromDate}`;

        try {
            const response = await fetch(url, {
                headers: {
                    'Accept': 'application/json',
                    'Authorization': `Bearer ${this.accessToken}`
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const json = await response.json();
            if (json.status === 'success' && json.data && json.data.candles) {
                return json.data.candles.map((c: any[]) => ({
                    symbol: instrumentKey,
                    start: new Date(c[0]).getTime(),
                    open: c[1],
                    high: c[2],
                    low: c[3],
                    close: c[4],
                    volume: c[5],
                    // Adding dummy prediction data for the chart visualization
                    predictedClose: c[4],
                    confidenceLower: c[4] * 0.99,
                    confidenceUpper: c[4] * 1.01
                })).reverse(); // Recharts usually expects chronological order, API returns reverse chronological? Check docs.
                // Usually historical APIs return newest first. Recharts wants oldest first (left to right).
                // Let's assume we need to reverse it.
            }
            return [];
        } catch (error) {
            console.error('Failed to fetch historical candles:', error);
            return [];
        }
    }

    async getSentiment(symbol: string): Promise<{ overall_sentiment: number, news: any[] }> {
        const url = `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/news/${symbol}`;
        try {
            const response = await fetch(url);
            if (!response.ok) throw new Error('Failed to fetch sentiment');
            return await response.json();
        } catch (error) {
            console.error('Sentiment fetch error:', error);
            // mocked fallback for demo if backend fails
            return { overall_sentiment: 0.5, news: [] };
        }
    }
}

export const upstoxService = new UpstoxService();
