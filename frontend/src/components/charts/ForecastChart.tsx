import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine } from 'recharts';

interface ForecastChartProps {
  historicalData: { date: string; price: number }[];
  forecastData: { date: string; price: number; upper?: number; lower?: number }[];
  isDark?: boolean;
}

export function ForecastChart({ historicalData, forecastData, isDark = true }: ForecastChartProps) {
  const combined = [
    ...historicalData.map((d) => ({ ...d, type: 'historical' as const })),
    ...forecastData.map((d) => ({ ...d, type: 'forecast' as const })),
  ];

  const lastHistorical = historicalData[historicalData.length - 1];

  const accent = isDark ? '#6c8aff' : '#4263eb';
  const forecast = isDark ? '#fbbf24' : '#e67700';
  const grid = isDark ? '#1e2130' : '#f1f3f5';
  const textColor = isDark ? '#6b7280' : '#868e96';

  return (
    <ResponsiveContainer width="100%" height={360}>
      <AreaChart data={combined} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
        <defs>
          <linearGradient id="historicalGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={accent} stopOpacity={0.12} />
            <stop offset="100%" stopColor={accent} stopOpacity={0} />
          </linearGradient>
          <linearGradient id="forecastGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={forecast} stopOpacity={0.08} />
            <stop offset="100%" stopColor={forecast} stopOpacity={0} />
          </linearGradient>
          <linearGradient id="bandGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={forecast} stopOpacity={0.06} />
            <stop offset="100%" stopColor={forecast} stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke={grid} />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 10, fill: textColor }}
          tickLine={false}
          axisLine={{ stroke: grid }}
        />
        <YAxis
          tick={{ fontSize: 10, fill: textColor }}
          tickLine={false}
          axisLine={false}
          domain={['auto', 'auto']}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: isDark ? '#161922' : '#ffffff',
            border: `1px solid ${grid}`,
            borderRadius: 8,
            fontSize: 12,
          }}
          labelStyle={{ color: textColor }}
        />
        {lastHistorical && (
          <ReferenceLine
            x={lastHistorical.date}
            stroke={textColor}
            strokeDasharray="4 2"
            strokeOpacity={0.4}
            label={{ value: 'Now', fill: textColor, fontSize: 10, position: 'top' }}
          />
        )}
        {/* Confidence band */}
        <Area type="monotone" dataKey="upper" stroke="none" fill="url(#bandGrad)" dot={false} />
        <Area type="monotone" dataKey="lower" stroke="none" fill="url(#bandGrad)" dot={false} />
        {/* Historical line */}
        <Area
          type="monotone"
          dataKey="price"
          stroke={accent}
          strokeWidth={2}
          fill="url(#historicalGrad)"
          dot={false}
          connectNulls={false}
          animationDuration={800}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
