import type { ReactNode } from 'react';

export interface BarRow {
  label: ReactNode;
  value: number;
  /** Highlighted bar (e.g. the predicted class). */
  emph?: boolean;
}

interface BarChartProps {
  rows: BarRow[];
  /** Value that maps to a full-width bar. */
  max?: number;
  format?: (v: number) => string;
}

/** Horizontal bars for probabilities / similarity scores. */
export function BarChart({
  rows,
  max = 1,
  format = (v) => `${(v * 100).toFixed(1)}%`,
}: BarChartProps) {
  return (
    <div className="barchart">
      {rows.map((r, i) => (
        <div key={i} className={`barchart-row${r.emph ? ' emph' : ''}`}>
          <div className="barchart-label">{r.label}</div>
          <div className="barchart-track">
            <div
              className="barchart-fill"
              style={{ width: `${Math.max(0, Math.min(1, r.value / max)) * 100}%` }}
            />
          </div>
          <div className="barchart-value tabular">{format(r.value)}</div>
        </div>
      ))}
    </div>
  );
}
