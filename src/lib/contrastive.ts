/**
 * A tiny CLIP: n (image, text) pairs whose embeddings live on the 2D unit
 * circle, parameterized by angles. Trained with the exact symmetric
 * InfoNCE loss from the paper, via analytic gradients.
 */

import { softmax } from './math.ts';

export interface SimState {
  /** Image embedding angles (radians). */
  phis: number[];
  /** Text embedding angles (radians). */
  psis: number[];
}

/** Deterministic PRNG so "reset" is reproducible but varied by seed. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function initState(n: number, seed: number): SimState {
  const rand = mulberry32(seed);
  return {
    phis: Array.from({ length: n }, () => rand() * 2 * Math.PI),
    psis: Array.from({ length: n }, () => rand() * 2 * Math.PI),
  };
}

const vec = (a: number): [number, number] => [Math.cos(a), Math.sin(a)];

/** Cosine similarity matrix of the current state (all pairs). */
export function simMatrix(state: SimState): number[][] {
  return state.phis.map((phi) =>
    state.psis.map((psi) => Math.cos(phi - psi)),
  );
}

export interface LossResult {
  loss: number;
  lossImg: number;
  lossTxt: number;
  /** Row softmax over the selected batch. */
  p: number[][];
}

/**
 * Symmetric InfoNCE over the sub-batch given by `idx`
 * (positions into state arrays).
 */
export function computeLoss(state: SimState, tau: number, idx: number[]): LossResult {
  const k = idx.length;
  const logits = idx.map((i) =>
    idx.map((j) => Math.cos(state.phis[i] - state.psis[j]) / tau),
  );
  const p = logits.map((row) => softmax(row));
  const cols = Array.from({ length: k }, (_, j) => logits.map((row) => row[j]));
  const q = cols.map((col) => softmax(col)); // q[j][i]
  let lossImg = 0;
  let lossTxt = 0;
  for (let a = 0; a < k; a++) {
    lossImg += -Math.log(Math.max(p[a][a], 1e-12));
    lossTxt += -Math.log(Math.max(q[a][a], 1e-12));
  }
  lossImg /= k;
  lossTxt /= k;
  return { loss: (lossImg + lossTxt) / 2, lossImg, lossTxt, p };
}

/**
 * One gradient-descent step on the angles, over the sub-batch `idx`.
 * Returns the new state (inputs are not mutated).
 *
 * Gradient wrt embeddings (standard softmax cross-entropy algebra):
 *   dL_img/dI_a = (1/(kτ)) Σ_b (p[a][b] − δ_ab) T_b     (rows)
 *   dL_txt/dI_a = (1/(kτ)) Σ_b (q[b][a] − δ_ab) T_b     (columns)
 * and symmetrically for T. Chain rule onto the angle uses the unit
 * tangent dI/dφ = (−sin φ, cos φ).
 */
export function step(
  state: SimState,
  tau: number,
  lr: number,
  idx: number[],
): SimState {
  const k = idx.length;
  const logits = idx.map((i) =>
    idx.map((j) => Math.cos(state.phis[i] - state.psis[j]) / tau),
  );
  const p = logits.map((row) => softmax(row));
  const cols = Array.from({ length: k }, (_, j) => logits.map((row) => row[j]));
  const q = cols.map((col) => softmax(col)); // q[j][i]

  const gradPhi = new Array(k).fill(0);
  const gradPsi = new Array(k).fill(0);
  const scale = 1 / (2 * k * tau); // 1/2 from averaging the two losses

  for (let a = 0; a < k; a++) {
    const i = idx[a];
    const [tx, ty] = [-Math.sin(state.phis[i]), Math.cos(state.phis[i])];
    let gx = 0;
    let gy = 0;
    for (let b = 0; b < k; b++) {
      const j = idx[b];
      const w = p[a][b] - (a === b ? 1 : 0) + (q[b][a] - (a === b ? 1 : 0));
      const [Tx, Ty] = vec(state.psis[j]);
      gx += w * Tx;
      gy += w * Ty;
    }
    gradPhi[a] = scale * (gx * tx + gy * ty);
  }

  for (let b = 0; b < k; b++) {
    const j = idx[b];
    const [tx, ty] = [-Math.sin(state.psis[j]), Math.cos(state.psis[j])];
    let gx = 0;
    let gy = 0;
    for (let a = 0; a < k; a++) {
      const i = idx[a];
      const w = p[a][b] - (a === b ? 1 : 0) + (q[b][a] - (a === b ? 1 : 0));
      const [Ix, Iy] = vec(state.phis[i]);
      gx += w * Ix;
      gy += w * Iy;
    }
    gradPsi[b] = scale * (gx * tx + gy * ty);
  }

  const phis = state.phis.slice();
  const psis = state.psis.slice();
  for (let a = 0; a < k; a++) {
    phis[idx[a]] -= lr * gradPhi[a];
    psis[idx[a]] -= lr * gradPsi[a];
  }
  return { phis, psis };
}
