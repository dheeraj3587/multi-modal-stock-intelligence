# StockSense Dashboard

A modern, glassmorphism-styled React + TypeScript dashboard for stock market intelligence.

## Features

- **Real-time Price Chart**: Interactive area chart with prediction confidence bands.
- **Live Tick Stream**: Scrolling feed of recent trades.
- **Sentiment Analysis**: Visual sentiment score and news feed.
- **Growth Leaderboard**: Top performing stocks with sparklines.
- **Indicators**: Radial progress indicators for Buy Pressure and Fear Index.
- **Glassmorphism UI**: Modern aesthetic with pastel colors and blur effects.

## Setup & Run

1.  **Install dependencies**:
    ```bash
    npm install
    ```

2.  **Run development server**:
    ```bash
    npm run dev
    ```
    Open [http://localhost:5173](http://localhost:5173) in your browser.

3.  **Build for production**:
    ```bash
    npm run build
    ```

## Connecting to Live Data (SSE)

To connect to a real Server-Sent Events (SSE) endpoint (e.g., `/stream/ticks`), you can modify `App.tsx` or create a custom hook.

**Example Implementation:**

```typescript
useEffect(() => {
  const eventSource = new EventSource('/stream/ticks');

  eventSource.onmessage = (event) => {
    const tick: PriceTick = JSON.parse(event.data);
    setTicks((prev) => [tick, ...prev].slice(0, 50));
    
    // Update chart data logic here...
  };

  eventSource.onerror = (error) => {
    console.error('SSE Error:', error);
    eventSource.close();
  };

  return () => {
    eventSource.close();
  };
}, []);
```

## Project Structure

- `src/components`: UI components (PriceChart, SentimentPanel, etc.)
- `src/styles`: Global styles and variables.
- `src/utils`: Types and mock data generators.
