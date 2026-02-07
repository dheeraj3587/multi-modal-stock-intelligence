import { useEffect, useRef, useCallback } from 'react';
import { createChart, type IChartApi, type ISeriesApi, type UTCTimestamp, ColorType, LineStyle } from 'lightweight-charts';
import type { CandleData } from '../../lib/api';

interface PriceChartProps {
  data: CandleData[];
  prediction?: { timestamp: string; close: number }[];
  height?: number;
  isDark?: boolean;
}

export function PriceChart({ data, prediction, height, isDark = true }: PriceChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Area'> | null>(null);
  const predSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);

  const initChart = useCallback(() => {
    if (!containerRef.current || data.length === 0) return;

    const getAutoHeight = () => {
      const w = containerRef.current?.clientWidth ?? 0;
      const vh = typeof window !== 'undefined' ? window.innerHeight : 900;
      // Keep a pleasing chart aspect ratio across screen sizes
      const byWidth = Math.round(w * 0.42);
      const byViewport = Math.round(vh * 0.42);
      return Math.max(360, Math.min(640, Math.min(byWidth, byViewport)));
    };

    const resolvedHeight = typeof height === 'number' ? height : getAutoHeight();

    const toUtcSeconds = (ts: unknown): UTCTimestamp | null => {
      if (ts == null) return null;
      if (typeof ts === 'number' && Number.isFinite(ts)) {
        const seconds = ts > 1_000_000_000_000 ? Math.floor(ts / 1000) : Math.floor(ts);
        return seconds as UTCTimestamp;
      }
      if (typeof ts === 'string') {
        const s = ts.trim();
        if (!s) return null;
        // Epoch seconds/millis as string
        if (/^\d+$/.test(s)) {
          const v = Number(s);
          if (!Number.isFinite(v)) return null;
          const seconds = v > 1_000_000_000_000 ? Math.floor(v / 1000) : Math.floor(v);
          return seconds as UTCTimestamp;
        }
        const ms = Date.parse(s);
        if (!Number.isFinite(ms)) return null;
        return Math.floor(ms / 1000) as UTCTimestamp;
      }
      return null;
    };

    // Cleanup previous
    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }

    const cssVar = (name: string) =>
      typeof window !== 'undefined'
        ? getComputedStyle(document.documentElement).getPropertyValue(name).trim()
        : '';

    const normalizeRgbTriplet = (raw: string) => raw.replace(/\s+/g, ' ').trim().replace(/ /g, ', ');

    const rgbFromCssVar = (name: string, fallback: string) => {
      const raw = cssVar(name);
      if (!raw) return fallback;
      // Our theme tokens are stored as space-separated RGB triplets: "R G B".
      // Some libraries can't parse CSS Color 4 space-separated rgb(), so emit legacy comma syntax.
      if (/^\d+\s+\d+\s+\d+$/.test(raw)) return `rgb(${normalizeRgbTriplet(raw)})`;
      // If the var is already a valid color string (e.g., hex), just use it.
      return raw;
    };

    const bg = rgbFromCssVar('--surface-0', isDark ? '#0f1117' : '#ffffff');
    const textColor = rgbFromCssVar('--text-tertiary', isDark ? '#6b7280' : '#868e96');
    const gridColor = rgbFromCssVar('--border-subtle', isDark ? '#1e2130' : '#f1f3f5');
    const lineColor = rgbFromCssVar('--accent', isDark ? '#6c8aff' : '#4263eb');

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: resolvedHeight,
      layout: {
        background: { type: ColorType.Solid, color: bg },
        textColor,
        fontFamily: 'Inter, system-ui, sans-serif',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: gridColor },
        horzLines: { color: gridColor },
      },
      crosshair: {
        vertLine: { labelBackgroundColor: lineColor },
        horzLine: { labelBackgroundColor: lineColor },
      },
      rightPriceScale: {
        borderColor: gridColor,
      },
      timeScale: {
        borderColor: gridColor,
        timeVisible: true,
        secondsVisible: false,
      },
      handleScroll: { vertTouchDrag: false },
    });

    // Main price area series
    const series = chart.addAreaSeries({
      lineColor,
      topColor: isDark ? 'rgba(108, 138, 255, 0.15)' : 'rgba(66, 99, 235, 0.1)',
      bottomColor: isDark ? 'rgba(108, 138, 255, 0.01)' : 'rgba(66, 99, 235, 0.01)',
      lineWidth: 2,
    });

    const mapped = data
      .map((d) => {
        const t = toUtcSeconds(d.timestamp);
        const v = typeof d.close === 'number' ? d.close : Number(d.close);
        if (t == null || !Number.isFinite(v)) return null;
        return { time: t, value: v };
      })
      .filter((p): p is { time: UTCTimestamp; value: number } => p !== null)
      // lightweight-charts expects ascending time
      .sort((a, b) => a.time - b.time);

    if (mapped.length === 0) return;
    series.setData(mapped as never);
    seriesRef.current = series;

    // Prediction overlay
    if (prediction && prediction.length > 0) {
      const predSeries = chart.addLineSeries({
        color: isDark ? '#fbbf24' : '#e67700',
        lineWidth: 2,
        lineStyle: LineStyle.Dashed,
      });
      const predMapped = prediction
        .map((d) => {
          const t = toUtcSeconds(d.timestamp);
          const v = typeof d.close === 'number' ? d.close : Number(d.close);
          if (t == null || !Number.isFinite(v)) return null;
          return { time: t, value: v };
        })
        .filter((p): p is { time: UTCTimestamp; value: number } => p !== null)
        .sort((a, b) => a.time - b.time);
      predSeries.setData(predMapped as never);
      predSeriesRef.current = predSeries;
    }

    chart.timeScale().fitContent();
    chartRef.current = chart;

    // Resize observer
    const observer = new ResizeObserver(() => {
      if (containerRef.current && chartRef.current) {
        const nextHeight = typeof height === 'number' ? height : getAutoHeight();
        chartRef.current.applyOptions({ width: containerRef.current.clientWidth, height: nextHeight });
      }
    });
    observer.observe(containerRef.current);

    return () => observer.disconnect();
  }, [data, prediction, height, isDark]);

  useEffect(() => {
    const cleanup = initChart();
    return () => {
      cleanup?.();
      chartRef.current?.remove();
    };
  }, [initChart]);

  return <div ref={containerRef} className="w-full rounded-xl overflow-hidden" />;
}
