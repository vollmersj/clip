/** Small numerical helpers shared across chapters. */

export function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function norm(a: number[]): number {
  return Math.sqrt(dot(a, a));
}

export function normalize(a: number[]): number[] {
  const n = norm(a);
  return n === 0 ? a.slice() : a.map((x) => x / n);
}

/** Cosine similarity of two (not necessarily normalized) vectors. */
export function cosine(a: number[], b: number[]): number {
  const na = norm(a);
  const nb = norm(b);
  if (na === 0 || nb === 0) return 0;
  return dot(a, b) / (na * nb);
}

/** Numerically stable softmax. */
export function softmax(logits: number[]): number[] {
  const m = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - m));
  const sum = exps.reduce((s, x) => s + x, 0);
  return exps.map((x) => x / sum);
}

/** rows[i][j] = cosine(a[i], b[j]) for already-normalized embeddings. */
export function similarityMatrix(a: number[][], b: number[][]): number[][] {
  return a.map((ai) => b.map((bj) => dot(ai, bj)));
}

export function argmax(xs: number[]): number {
  let best = 0;
  for (let i = 1; i < xs.length; i++) if (xs[i] > xs[best]) best = i;
  return best;
}

/** Indices of xs sorted descending by value. */
export function rankDescending(xs: number[]): number[] {
  return xs
    .map((x, i) => [x, i] as const)
    .sort((p, q) => q[0] - p[0])
    .map(([, i]) => i);
}

export function mean(xs: number[]): number {
  return xs.reduce((s, x) => s + x, 0) / xs.length;
}

export function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}
