import { motion } from 'framer-motion';
import { scoreStrokeColor } from '../../lib/utils';

interface ScoreRingProps {
  score: number;
  max?: number;
  size?: number;
  strokeWidth?: number;
  label?: string;
}

export function ScoreRing({ score, max = 100, size = 160, strokeWidth = 10, label }: ScoreRingProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const normalized = Math.min(score / max, 1);
  const offset = circumference * (1 - normalized);
  const scorePct = (score / max) * 100;
  const color = scoreStrokeColor(scorePct);
  const displayScore = max <= 10 ? score.toFixed(1) : String(Math.round(score));

  return (
    <div className="relative inline-flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="var(--border)"
          strokeWidth={strokeWidth}
        />
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1, ease: 'easeOut' }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-3xl font-bold text-text-primary">{displayScore}</span>
        {label && <span className="text-xs text-text-tertiary mt-0.5">{label}</span>}
      </div>
    </div>
  );
}
