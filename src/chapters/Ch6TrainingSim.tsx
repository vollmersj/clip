import { useEffect, useMemo, useRef, useState } from 'react';
import {
  initState,
  step,
  computeLoss,
  simMatrix,
  type SimState,
} from '../lib/contrastive';
import { MatrixView } from '../components/MatrixView';
import { Slider } from '../components/Slider';
import { mean } from '../lib/math';

const CONCEPTS = [
  { emoji: '🐱', name: 'cat' },
  { emoji: '🐶', name: 'dog' },
  { emoji: '🚗', name: 'car' },
  { emoji: '✈️', name: 'plane' },
  { emoji: '🍎', name: 'apple' },
  { emoji: '🍕', name: 'pizza' },
];
const N = CONCEPTS.length;
const ALL = Array.from({ length: N }, (_, i) => i);
const GOOD_SEEDS = [1, 2, 3, 4, 5, 6, 8, 9];

const R = 130;
const C = 165;

function CircleViz({ state }: { state: SimState }) {
  return (
    <svg viewBox="0 0 330 330" style={{ width: '100%', maxWidth: 340 }}>
      <circle cx={C} cy={C} r={R} fill="none" stroke="var(--grid)" strokeWidth="1.5" strokeDasharray="4 4" />
      {/* pair connectors */}
      {ALL.map((i) => {
        const ix = C + Math.cos(state.phis[i]) * R;
        const iy = C - Math.sin(state.phis[i]) * R;
        const tx = C + Math.cos(state.psis[i]) * R;
        const ty = C - Math.sin(state.psis[i]) * R;
        const cos = Math.cos(state.phis[i] - state.psis[i]);
        const good = Math.max(0, Math.round(cos * 100));
        return (
          <line
            key={`l${i}`}
            x1={ix}
            y1={iy}
            x2={tx}
            y2={ty}
            stroke={`color-mix(in oklab, var(--good) ${good}%, var(--baseline))`}
            strokeWidth={cos > 0.9 ? 2.5 : 1.5}
            opacity="0.75"
          />
        );
      })}
      {/* image markers: blue "photo frames" */}
      {ALL.map((i) => {
        const x = C + Math.cos(state.phis[i]) * R;
        const y = C - Math.sin(state.phis[i]) * R;
        return (
          <g key={`i${i}`}>
            <rect x={x - 13} y={y - 13} width="26" height="26" rx="6" fill="var(--img-soft)" stroke="var(--img)" strokeWidth="2" />
            <text x={x} y={y + 5} textAnchor="middle" fontSize="14">
              {CONCEPTS[i].emoji}
            </text>
          </g>
        );
      })}
      {/* text markers: green circles */}
      {ALL.map((i) => {
        const x = C + Math.cos(state.psis[i]) * R;
        const y = C - Math.sin(state.psis[i]) * R;
        return (
          <g key={`t${i}`}>
            <circle cx={x} cy={y} r="13" fill="var(--txt-soft)" stroke="var(--txt)" strokeWidth="2" />
            <text x={x} y={y + 5} textAnchor="middle" fontSize="14">
              {CONCEPTS[i].emoji}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function LossCurve({ history, batchSize }: { history: number[]; batchSize: number }) {
  const W = 300;
  const H = 130;
  const PAD = 6;
  const yMax = Math.max(Math.log(N) * 1.25, ...history);
  const guess = Math.log(batchSize);
  const toXY = (v: number, i: number) => {
    const x = PAD + (i / Math.max(1, history.length - 1)) * (W - 2 * PAD);
    const y = H - PAD - (v / yMax) * (H - 2 * PAD);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  };
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: 320 }}>
      <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="var(--baseline)" strokeWidth="1" />
      {/* random-guessing reference */}
      <line
        x1={PAD}
        y1={H - PAD - (guess / yMax) * (H - 2 * PAD)}
        x2={W - PAD}
        y2={H - PAD - (guess / yMax) * (H - 2 * PAD)}
        stroke="var(--muted)"
        strokeWidth="1"
        strokeDasharray="3 4"
      />
      <text x={W - PAD} y={H - PAD - (guess / yMax) * (H - 2 * PAD) - 4} fontSize="9.5" fill="var(--muted)" textAnchor="end">
        random guessing (ln {batchSize})
      </text>
      {history.length > 1 && (
        <polyline
          points={history.map(toXY).join(' ')}
          fill="none"
          stroke="var(--accent)"
          strokeWidth="2"
          strokeLinejoin="round"
        />
      )}
    </svg>
  );
}

export function Ch6TrainingSim() {
  const seedRef = useRef(GOOD_SEEDS[0]);
  const [state, setState] = useState<SimState>(() => initState(N, seedRef.current));
  const [running, setRunning] = useState(false);
  const [steps, setSteps] = useState(0);
  const [history, setHistory] = useState<number[]>([]);
  const [logTau, setLogTau] = useState(Math.log10(0.15));
  const [lr, setLr] = useState(0.3);
  const [batchSize, setBatchSize] = useState(N);
  const tau = 10 ** logTau;

  const doSteps = (count: number) => {
    setState((s) => {
      let cur = s;
      const newLosses: number[] = [];
      for (let t = 0; t < count; t++) {
        const idx =
          batchSize >= N
            ? ALL
            : [...ALL].sort(() => Math.random() - 0.5).slice(0, batchSize);
        cur = step(cur, tau, lr, idx);
        newLosses.push(computeLoss(cur, tau, ALL).loss);
      }
      setHistory((h) => {
        const next = [...h, ...newLosses];
        return next.length > 400 ? next.slice(next.length - 400) : next;
      });
      setSteps((n) => n + count);
      return cur;
    });
  };

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => doSteps(2), 60);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, tau, lr, batchSize]);

  const reset = () => {
    seedRef.current = Math.floor(Math.random() * 100000);
    setState(initState(N, seedRef.current));
    setHistory([]);
    setSteps(0);
    setRunning(false);
  };

  const { loss } = computeLoss(state, tau, ALL);
  const matrix = useMemo(() => simMatrix(state), [state]);
  const matchedCos = mean(ALL.map((i) => matrix[i][i]));
  const mismatchedCos = mean(
    ALL.flatMap((i) => ALL.filter((j) => j !== i).map((j) => matrix[i][j])),
  );

  return (
    <>
      <h1>Train It Yourself: A Tiny CLIP</h1>
      <p className="lede">
        Real CLIP moves {`${512}`}-dimensional embeddings with two deep encoders. This
        toy version keeps everything else identical — the exact symmetric contrastive
        loss, the temperature, real gradient descent — but lives on a 2D circle, so you
        can <em>watch</em> the loss reshape the space.
      </p>

      <div className="card wide">
        <p className="card-title">Contrastive training, live</p>
        <p className="card-sub">
          <span className="tag img">images</span> = squares ·{' '}
          <span className="tag txt">texts</span> = circles. Six concepts, twelve points.
          Press <strong>Train</strong>.
        </p>

        <div className="controls-row">
          <button className="btn primary" onClick={() => setRunning(!running)}>
            {running ? '⏸ Pause' : '▶ Train'}
          </button>
          <button className="btn" onClick={() => doSteps(1)} disabled={running}>
            Step ×1
          </button>
          <button className="btn" onClick={reset}>
            ↺ Reset
          </button>
          <span className="small muted tabular">step {steps}</span>
        </div>

        <div className="controls-row">
          <Slider
            label="temperature τ"
            value={logTau}
            min={-2}
            max={0}
            step={0.01}
            onChange={setLogTau}
            format={() => tau.toFixed(tau < 0.1 ? 3 : 2)}
          />
          <Slider
            label="learning rate"
            value={lr}
            min={0.05}
            max={1}
            step={0.05}
            onChange={setLr}
          />
          <Slider
            label="batch size"
            value={batchSize}
            min={2}
            max={N}
            step={1}
            onChange={setBatchSize}
            format={(v) => `${v} of ${N}`}
          />
        </div>

        <div className="sim-grid">
          <CircleViz state={state} />
          <div>
            <LossCurve history={history} batchSize={batchSize} />
            <table className="readout-table" style={{ marginTop: 8 }}>
              <tbody>
                <tr>
                  <td>loss (full batch)</td>
                  <td className="tabular emph">{loss.toFixed(3)}</td>
                </tr>
                <tr>
                  <td>mean cosine, matched pairs</td>
                  <td className="tabular" style={{ color: 'var(--good-text)' }}>
                    {matchedCos.toFixed(3)}
                  </td>
                </tr>
                <tr>
                  <td>mean cosine, mismatched</td>
                  <td className="tabular">{mismatchedCos.toFixed(3)}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <details className="deep-dive" style={{ marginBottom: 0 }}>
          <summary>Watch the same numbers as a similarity matrix</summary>
          <div className="deep-dive-body">
            <MatrixView
              values={matrix}
              rowHeads={CONCEPTS.map((c) => (
                <span key={c.name} title={`image of a ${c.name}`}>{c.emoji}📷</span>
              ))}
              colHeads={CONCEPTS.map((c) => (
                <span key={c.name} title={`“a photo of a ${c.name}”`} style={{ fontSize: 12 }}>
                  {c.emoji}
                </span>
              ))}
              markDiagonal
              domain={[-1, 1]}
              cellSize={44}
              corner={<span>img ↓ · txt →</span>}
              legendLabel="cosine (−1 … 1)"
            />
            <p className="small muted" style={{ marginTop: 8 }}>
              Training success = a bright green-ringed diagonal on a dark background.
            </p>
          </div>
        </details>
      </div>

      <h2>Experiments to try</h2>
      <ul>
        <li>
          <strong>Alignment and uniformity.</strong> Train with the defaults: matched
          squares and circles snap together (alignment), while different concepts
          repel until they're spread around the circle (uniformity). The contrastive
          loss is exactly this pair of forces — the pull is strongest when the match is
          uncertain, the push is strongest against confusable neighbors.
        </li>
        <li>
          <strong>Batch size matters.</strong> Reset, set batch size to 2, train. Each
          step now sees only one negative pair, so updates are noisy and the full-batch
          loss wobbles. This is why CLIP used a batch of 32,768.
        </li>
        <li>
          <strong>Temperature as force-field sharpness.</strong> At τ = 1 the forces are
          mushy — everything drifts slowly and nothing separates crisply. At τ = 0.01
          only the single nearest rival exerts meaningful push. The sweet spot in
          between is why CLIP <em>learns</em> τ instead of fixing it.
        </li>
        <li>
          <strong>Stuck? That's a local minimum.</strong> Occasionally two concepts end
          up trapped in a bad ordering — on a 1D circle, points can't pass each other
          without temporarily raising the loss. Hit Reset and it will usually converge.
          Real CLIP has 512 dimensions to maneuver in, so this near-degenerate geometry
          is far less of a problem.
        </li>
      </ul>

      <div className="note">
        <span className="note-label">What transfers to the real thing</span>
        Everything you just saw — the loss, the temperature, the batch effects — is
        computed with the same formulas as real CLIP. The only differences: real CLIP
        updates the <em>encoder weights</em> (not the points directly), and its points
        live on a 512-dimensional sphere instead of a circle.
      </div>
    </>
  );
}
