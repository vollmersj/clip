import { useState, type ReactNode } from 'react';

export interface MatrixViewProps {
  values: number[][];
  rowHeads: ReactNode[];
  colHeads: ReactNode[];
  /** Value range mapped to the color ramp; 'auto' uses the matrix min/max. */
  domain?: 'auto' | [number, number];
  showValues?: boolean;
  /** Draw a ring around diagonal cells (the matching pairs). */
  markDiagonal?: boolean;
  /** Dim everything except this row (used to explain row-wise softmax). */
  focusRow?: number | null;
  /** Dim everything except this column. */
  focusCol?: number | null;
  valueFormat?: (v: number) => string;
  /** Called with the hovered cell, or null when the pointer leaves. */
  onHover?: (cell: [number, number] | null) => void;
  /** Corner label, e.g. axis names. */
  corner?: ReactNode;
  legendLabel?: string;
  /** Cell size in px (content square). */
  cellSize?: number;
}

/**
 * The similarity-matrix heatmap used throughout the walkthrough.
 * Rows are images, columns are texts (like Figure 1 of the CLIP paper).
 */
export function MatrixView({
  values,
  rowHeads,
  colHeads,
  domain = 'auto',
  showValues = true,
  markDiagonal = false,
  focusRow = null,
  focusCol = null,
  valueFormat = (v) => v.toFixed(2),
  onHover,
  corner,
  legendLabel,
  cellSize = 52,
}: MatrixViewProps) {
  const [hover, setHover] = useState<[number, number] | null>(null);

  const m = values[0]?.length ?? 0;
  let lo: number;
  let hi: number;
  if (domain === 'auto') {
    lo = Infinity;
    hi = -Infinity;
    for (const row of values)
      for (const v of row) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    if (!isFinite(lo)) {
      lo = 0;
      hi = 1;
    }
  } else {
    [lo, hi] = domain;
  }
  const span = hi - lo || 1;

  const setHoverBoth = (cell: [number, number] | null) => {
    setHover(cell);
    onHover?.(cell);
  };

  const focused = focusRow !== null || focusCol !== null;

  return (
    <div className="matrix-wrap">
      <div
        className="matrix"
        style={{
          gridTemplateColumns: `auto repeat(${m}, ${cellSize}px)`,
        }}
        onMouseLeave={() => setHoverBoth(null)}
      >
        <div className="matrix-corner">{corner}</div>
        {Array.from({ length: m }, (_, j) => (
          <div
            key={`c${j}`}
            className={`matrix-colhead${hover && hover[1] === j ? ' hl' : ''}${
              focused && focusCol !== null && focusCol !== j ? ' dim' : ''
            }`}
          >
            {colHeads[j]}
          </div>
        ))}
        {values.map((row, i) => (
          <div key={`r${i}`} className="matrix-row" style={{ display: 'contents' }}>
            <div
              className={`matrix-rowhead${hover && hover[0] === i ? ' hl' : ''}${
                focused && focusRow !== null && focusRow !== i ? ' dim' : ''
              }`}
            >
              {rowHeads[i]}
            </div>
            {row.map((v, j) => {
              const t = Math.min(1, Math.max(0, (v - lo) / span));
              const pct = Math.round(t * 100);
              const isDiag = markDiagonal && i === j;
              const isHover = hover && hover[0] === i && hover[1] === j;
              const dim =
                focused &&
                ((focusRow !== null && focusRow !== i) ||
                  (focusCol !== null && focusCol !== j));
              return (
                <div
                  key={j}
                  className={`matrix-cell${isDiag ? ' diag' : ''}${
                    isHover ? ' hover' : ''
                  }${dim ? ' dim' : ''}`}
                  style={{
                    width: cellSize,
                    height: cellSize,
                    background: `color-mix(in oklab, var(--cell-hi) ${pct}%, var(--cell-lo))`,
                    color: t > 0.55 ? 'var(--cell-ink-strong)' : 'var(--cell-ink)',
                  }}
                  onMouseEnter={() => setHoverBoth([i, j])}
                >
                  {showValues && (
                    <span className="matrix-value tabular">{valueFormat(v)}</span>
                  )}
                </div>
              );
            })}
          </div>
        ))}
      </div>
      {legendLabel && (
        <div className="matrix-legend">
          <span className="tabular small">{valueFormat(lo)}</span>
          <div className="matrix-legend-ramp" />
          <span className="tabular small">{valueFormat(hi)}</span>
          <span className="small muted" style={{ marginLeft: 8 }}>
            {legendLabel}
          </span>
        </div>
      )}
    </div>
  );
}
