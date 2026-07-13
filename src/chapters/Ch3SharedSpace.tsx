import { useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import { Formula } from '../components/Formula';
import { data } from '../lib/data';
import { dot } from '../lib/math';

interface Vec {
  angle: number; // radians, math convention (y up)
  len: number; // 1.0 = unit length
}

const R = 100; // px radius representing length 1
const CX = 160;
const CY = 150;

function toXY(v: Vec) {
  return {
    x: CX + Math.cos(v.angle) * v.len * R,
    y: CY - Math.sin(v.angle) * v.len * R,
  };
}

function UnitCircleDemo() {
  const [u, setU] = useState<Vec>({ angle: 1.15, len: 1.3 });
  const [v, setV] = useState<Vec>({ angle: 0.45, len: 0.7 });
  const [normalized, setNormalized] = useState(false);
  const dragging = useRef<'u' | 'v' | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const uu: Vec = normalized ? { ...u, len: 1 } : u;
  const vv: Vec = normalized ? { ...v, len: 1 } : v;

  const cosTheta = Math.cos(uu.angle - vv.angle);
  const dotProduct = uu.len * vv.len * cosTheta;
  const thetaDeg = (Math.abs(
    ((uu.angle - vv.angle + Math.PI) % (2 * Math.PI)) - Math.PI,
  ) * 180) / Math.PI;

  const handleMove = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (!dragging.current || !svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const scale = 320 / rect.width;
    const x = (e.clientX - rect.left) * scale - CX;
    const y = CY - (e.clientY - rect.top) * scale;
    const angle = Math.atan2(y, x);
    const len = Math.min(1.35, Math.max(0.35, Math.hypot(x, y) / R));
    const next = { angle, len };
    if (dragging.current === 'u') setU(next);
    else setV(next);
  };

  const startDrag = (which: 'u' | 'v') => (e: ReactPointerEvent<SVGCircleElement>) => {
    dragging.current = which;
    (e.currentTarget.ownerSVGElement ?? e.currentTarget).setPointerCapture?.(e.pointerId);
  };

  const pu = toXY(uu);
  const pv = toXY(vv);

  return (
    <div className="card">
      <p className="card-title">Angle in, similarity out</p>
      <p className="card-sub">
        Drag the two arrowheads. <span className="tag img">image vector</span>{' '}
        <span className="tag txt">text vector</span> — in 2D instead of {data.dim}D, but
        the geometry is the same.
      </p>
      <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', alignItems: 'center' }}>
        <svg
          ref={svgRef}
          viewBox="0 0 320 300"
          style={{ width: 320, maxWidth: '100%', touchAction: 'none', cursor: 'default' }}
          onPointerMove={handleMove}
          onPointerUp={() => (dragging.current = null)}
          onPointerLeave={() => (dragging.current = null)}
        >
          {/* unit circle */}
          <circle
            cx={CX}
            cy={CY}
            r={R}
            fill="none"
            stroke="var(--grid)"
            strokeDasharray="4 4"
            strokeWidth="1.5"
          />
          <text x={CX + R + 6} y={CY + 4} fontSize="11" fill="var(--muted)">
            length 1
          </text>
          {/* axes */}
          <line x1={CX - 140} y1={CY} x2={CX + 140} y2={CY} stroke="var(--grid)" strokeWidth="1" />
          <line x1={CX} y1={CY - 135} x2={CX} y2={CY + 135} stroke="var(--grid)" strokeWidth="1" />

          {/* angle arc */}
          {(() => {
            const a1 = Math.min(uu.angle, vv.angle);
            const a2 = Math.max(uu.angle, vv.angle);
            const large = a2 - a1 > Math.PI ? 1 : 0;
            const sweep = 0;
            const r = 34;
            const x1 = CX + Math.cos(a1) * r;
            const y1 = CY - Math.sin(a1) * r;
            const x2 = CX + Math.cos(a2) * r;
            const y2 = CY - Math.sin(a2) * r;
            return (
              <path
                d={`M ${x1} ${y1} A ${r} ${r} 0 ${large} ${sweep} ${x2} ${y2}`}
                fill="none"
                stroke="var(--warn)"
                strokeWidth="2.5"
              />
            );
          })()}

          {/* vectors */}
          <line x1={CX} y1={CY} x2={pu.x} y2={pu.y} stroke="var(--img)" strokeWidth="3" />
          <line x1={CX} y1={CY} x2={pv.x} y2={pv.y} stroke="var(--txt)" strokeWidth="3" />
          <circle
            cx={pu.x}
            cy={pu.y}
            r="10"
            fill="var(--img)"
            style={{ cursor: 'grab' }}
            onPointerDown={startDrag('u')}
          />
          <circle
            cx={pv.x}
            cy={pv.y}
            r="10"
            fill="var(--txt)"
            style={{ cursor: 'grab' }}
            onPointerDown={startDrag('v')}
          />
          <text x={pu.x + 14} y={pu.y + 4} fontSize="13" fontWeight="650" fill="var(--img)">
            I
          </text>
          <text x={pv.x + 14} y={pv.y + 4} fontSize="13" fontWeight="650" fill="var(--txt-strong)">
            T
          </text>
        </svg>

        <div style={{ flex: '1 1 260px', minWidth: 240 }}>
          <table className="readout-table">
            <tbody>
              <tr>
                <td>angle θ</td>
                <td className="tabular">{thetaDeg.toFixed(0)}°</td>
              </tr>
              <tr>
                <td>
                  cos θ <span className="muted small">(cosine similarity)</span>
                </td>
                <td className="tabular emph">{cosTheta.toFixed(3)}</td>
              </tr>
              <tr>
                <td>‖I‖ · ‖T‖ (lengths)</td>
                <td className="tabular">
                  {uu.len.toFixed(2)} · {vv.len.toFixed(2)}
                </td>
              </tr>
              <tr>
                <td>
                  dot product I · T
                </td>
                <td className={`tabular${normalized ? ' emph' : ''}`}>
                  {dotProduct.toFixed(3)}
                </td>
              </tr>
            </tbody>
          </table>
          <label className="small" style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 12, cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={normalized}
              onChange={(e) => setNormalized(e.target.checked)}
            />
            L2-normalize both vectors (what CLIP does)
          </label>
          <p className="small muted" style={{ marginTop: 8 }}>
            {normalized
              ? 'Lengths are now exactly 1 — the dot product equals the cosine. One multiplication per dimension, and similarity is just “how aligned are the directions”.'
              : 'With unnormalized vectors, the dot product mixes direction with length. Tick the box to see CLIP’s fix.'}
          </p>
        </div>
      </div>
    </div>
  );
}

function RealNumbersDemo() {
  // cosine range across all real image-caption pairs
  const sims: { pair: string; v: number; match: boolean }[] = [];
  data.items.forEach((a, i) => {
    data.items.forEach((b, j) => {
      sims.push({
        pair: `${a.id} ↔ “${b.caption.slice(0, 28)}…”`,
        v: dot(a.imageEmbed, b.captionEmbed),
        match: i === j,
      });
    });
  });
  const matches = sims.filter((s) => s.match).map((s) => s.v);
  const rest = sims.filter((s) => !s.match).map((s) => s.v);
  const fmt = (v: number) => v.toFixed(2);
  return (
    <div className="note">
      <span className="note-label">Reality check — real CLIP numbers are small</span>
      Across our 8 gallery pairs, <em>matching</em> image–caption cosines range from{' '}
      <strong className="tabular">{fmt(Math.min(...matches))}</strong> to{' '}
      <strong className="tabular">{fmt(Math.max(...matches))}</strong>, while{' '}
      <em>mismatched</em> ones range from{' '}
      <strong className="tabular">{fmt(Math.min(...rest))}</strong> to{' '}
      <strong className="tabular">{fmt(Math.max(...rest))}</strong>. Nothing near 1.0!
      What matters is not the absolute value but the <em>gap</em> — and the temperature
      parameter in Chapter 5 will stretch that gap enormously.
    </div>
  );
}

export function Ch3SharedSpace() {
  return (
    <>
      <h1>A Shared Embedding Space</h1>
      <p className="lede">
        An embedding is just a point — or better, a <em>direction</em> — in a{' '}
        {data.dim}-dimensional space. To train anything, we first need a number that
        says how similar two embeddings are.
      </p>

      <h2>Cosine similarity</h2>
      <p>
        CLIP measures similarity as the <strong>cosine of the angle</strong> between two
        vectors: +1 means “pointing the same way”, 0 means “unrelated / perpendicular”,
        −1 means “opposite”.
      </p>
      <Formula
        display
        tex={String.raw`\text{sim}(I, T) \;=\; \cos\theta \;=\; \frac{I \cdot T}{\lVert I \rVert \,\lVert T \rVert}`}
      />
      <dl className="formula-legend">
        <dt>I, T</dt>
        <dd>image and text embeddings ({data.dim} numbers each)</dd>
        <dt>I · T</dt>
        <dd>
          dot product: multiply the vectors element by element and sum everything up
        </dd>
        <dt>‖I‖</dt>
        <dd>the length (L2 norm) of the vector</dd>
      </dl>

      <p>
        CLIP <strong>L2-normalizes</strong> both embeddings — scales each to length 1 —
        so every embedding lives on the surface of a unit hypersphere, and the cosine
        becomes a plain dot product:
      </p>
      <Formula
        display
        tex={String.raw`\hat{I} = \frac{I}{\lVert I \rVert},\qquad \hat{T} = \frac{T}{\lVert T \rVert},\qquad \text{sim}(I,T) = \hat{I}\cdot\hat{T}`}
      />

      <UnitCircleDemo />

      <RealNumbersDemo />

      <details className="deep-dive">
        <summary>Deep dive: why are even “perfect” matches only ≈ 0.3?</summary>
        <div className="deep-dive-body">
          <p>
            In high-dimensional spaces, random vectors are almost always nearly
            orthogonal, and the learned image and text embeddings occupy somewhat
            different regions of the sphere (a “modality gap”). CLIP never needs the
            matching pair to reach cosine 1.0 — the loss only needs the correct pair to
            score <em>higher than every alternative in the batch</em>. Keep this in mind
            when you see “85%” in a CLIP demo: that will be a softmax{' '}
            <em>probability</em>, computed from these small cosines — not a cosine
            itself. Chapter 5 makes that distinction precise.
          </p>
        </div>
      </details>
    </>
  );
}
