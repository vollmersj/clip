// Numerical gradient check for the toy-CLIP training engine.
// Run: node scripts/test_grad.ts   (Node >= 23 strips types natively)
import { initState, computeLoss, step, type SimState } from '../src/lib/contrastive.ts';

function numericalGrads(state: SimState, tau: number, idx: number[]) {
  const eps = 1e-6;
  const gPhi: number[] = [];
  const gPsi: number[] = [];
  for (const i of idx) {
    for (const key of ['phis', 'psis'] as const) {
      const plus: SimState = { phis: state.phis.slice(), psis: state.psis.slice() };
      const minus: SimState = { phis: state.phis.slice(), psis: state.psis.slice() };
      plus[key][i] += eps;
      minus[key][i] -= eps;
      const g =
        (computeLoss(plus, tau, idx).loss - computeLoss(minus, tau, idx).loss) /
        (2 * eps);
      (key === 'phis' ? gPhi : gPsi).push(g);
    }
  }
  return { gPhi, gPsi };
}

function analyticGrads(state: SimState, tau: number, idx: number[], lr = 1e-7) {
  const next = step(state, tau, lr, idx);
  return {
    gPhi: idx.map((i) => (state.phis[i] - next.phis[i]) / lr),
    gPsi: idx.map((i) => (state.psis[i] - next.psis[i]) / lr),
  };
}

let failures = 0;
for (const { n, tau, idx } of [
  { n: 6, tau: 0.15, idx: [0, 1, 2, 3, 4, 5] },
  { n: 6, tau: 0.5, idx: [0, 2, 5] },
  { n: 8, tau: 0.07, idx: [1, 3, 4, 6] },
  { n: 4, tau: 1.0, idx: [0, 1, 2, 3] },
]) {
  const state = initState(n, 1234 + n);
  const num = numericalGrads(state, tau, idx);
  const ana = analyticGrads(state, tau, idx);
  let maxErr = 0;
  for (let a = 0; a < idx.length; a++) {
    maxErr = Math.max(
      maxErr,
      Math.abs(num.gPhi[a] - ana.gPhi[a]),
      Math.abs(num.gPsi[a] - ana.gPsi[a]),
    );
  }
  const ok = maxErr < 1e-4;
  if (!ok) failures++;
  console.log(
    `n=${n} tau=${tau} batch=${idx.length}: max |analytic - numerical| = ${maxErr.toExponential(2)} ${ok ? '✓' : '✗ FAIL'}`,
  );
}

// Convergence check: from a well-behaved seed, the full-batch loss should
// approach the global optimum (~0.069 for 6 concepts uniformly spread at τ=0.15).
{
  let s = initState(6, 1);
  const idx = [0, 1, 2, 3, 4, 5];
  const before = computeLoss(s, 0.15, idx).loss;
  for (let t = 0; t < 3000; t++) s = step(s, 0.15, 0.3, idx);
  const after = computeLoss(s, 0.15, idx).loss;
  const ok = after < 0.1 && after < before;
  if (!ok) failures++;
  console.log(
    `convergence: loss ${before.toFixed(3)} → ${after.toFixed(4)} after 3000 steps ${ok ? '✓' : '✗ FAIL'}`,
  );
}

if (failures > 0) {
  console.error(`${failures} check(s) failed`);
  process.exit(1);
}
console.log('All gradient checks passed.');
