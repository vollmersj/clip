interface VectorStripProps {
  values: number[];
  /** How many leading dimensions to draw. */
  count?: number;
  modality: 'img' | 'txt';
  title?: string;
}

/**
 * Renders the first few dimensions of a real embedding as a strip of
 * diverging-colored squares (red = negative, blue = positive).
 */
export function VectorStrip({ values, count = 20, modality, title }: VectorStripProps) {
  const shown = values.slice(0, count);
  // Normalize color intensity to the largest magnitude in the shown slice.
  const maxAbs = Math.max(0.01, ...shown.map((v) => Math.abs(v)));
  return (
    <span className={`vector-strip ${modality}`} title={title}>
      {shown.map((v, i) => {
        const t = Math.abs(v) / maxAbs;
        const hue = v >= 0 ? 'var(--accent)' : 'var(--bad)';
        return (
          <span
            key={i}
            className="dimsq"
            style={{
              background: `color-mix(in oklab, ${hue} ${Math.round(t * 100)}%, var(--surface-2))`,
            }}
            title={`dim ${i}: ${v.toFixed(3)}`}
          />
        );
      })}
      <span className="ellipsis">… {values.length}d</span>
    </span>
  );
}
