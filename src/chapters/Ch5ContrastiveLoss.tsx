import { useMemo, useState } from 'react';
import { data, galleryUrl } from '../lib/data';
import { similarityMatrix, softmax, mean } from '../lib/math';
import { MatrixView } from '../components/MatrixView';
import { BarChart } from '../components/BarChart';
import { Formula } from '../components/Formula';
import { Slider } from '../components/Slider';

type View = 'sim' | 'logits' | 'rowP' | 'colP';

const VIEW_LABELS: Record<View, string> = {
  sim: 'cosines',
  logits: 'logits (÷ τ)',
  rowP: 'row softmax',
  colP: 'column softmax',
};

export function Ch5ContrastiveLoss() {
  // log-scale slider over temperature: τ from 0.01 to 1
  const [logTau, setLogTau] = useState(Math.log10(0.07));
  const [view, setView] = useState<View>('sim');
  const tau = 10 ** logTau;

  const sim = useMemo(
    () =>
      similarityMatrix(
        data.items.map((it) => it.imageEmbed),
        data.items.map((it) => it.captionEmbed),
      ),
    [],
  );
  const n = sim.length;

  const { logits, rowP, colP, lossImg, lossTxt } = useMemo(() => {
    const logits = sim.map((row) => row.map((v) => v / tau));
    const rowP = logits.map((row) => softmax(row));
    const cols = Array.from({ length: n }, (_, j) => logits.map((row) => row[j]));
    const colPT = cols.map((col) => softmax(col)); // colPT[j][i]
    const colP = logits.map((_, i) => colPT.map((col) => col[i]));
    const lossImg = mean(rowP.map((row, i) => -Math.log(row[i])));
    const lossTxt = mean(colPT.map((col, j) => -Math.log(col[j])));
    return { logits, rowP, colP, lossImg, lossTxt };
  }, [sim, tau, n]);

  const shown: number[][] =
    view === 'sim' ? sim : view === 'logits' ? logits : view === 'rowP' ? rowP : colP;

  const catProbs = rowP[0];

  return (
    <>
      <h1>The Contrastive Loss</h1>
      <p className="lede">
        Chapter 4 ended with a wish: <em>the diagonal should win its row and its
        column</em>. The contrastive loss makes that wish differentiable — in three
        small steps that you can drive with one slider.
      </p>

      <h2>Step 1 · From cosines to logits: the temperature τ</h2>
      <p>
        Cosine similarities are cramped into a narrow band (roughly 0.1–0.35 here).
        Before comparing them, CLIP divides everything by a small{' '}
        <strong>temperature</strong> τ, stretching tiny differences into big ones:
      </p>
      <Formula display tex={String.raw`\ell_{ij} \;=\; \frac{\hat{I}_i \cdot \hat{T}_j}{\tau}`} />
      <p>
        τ is not hand-picked — it's a <strong>learned parameter</strong>. The paper
        initializes it at 0.07 and lets gradient descent tune it (capped so the
        multiplier never exceeds 100, i.e. τ ≥ 0.01).
      </p>

      <h2>Step 2 · Each image “classifies” the captions</h2>
      <p>
        Read one row of the matrix: image I₁ against all {n} captions. Apply a{' '}
        <strong>softmax</strong> and the row becomes a probability distribution — as if
        the image were classifying which caption belongs to it:
      </p>
      <Formula
        display
        tex={String.raw`p_i(j) \;=\; \frac{e^{\ell_{ij}}}{\sum_{k=1}^{N} e^{\ell_{ik}}}`}
      />
      <div className="card">
        <p className="card-title">Row 1 as a classification problem</p>
        <p className="card-sub">
          The cat image assigns a probability to each caption. Drag τ and watch the
          softmax sharpen and flatten.
        </p>
        <div style={{ display: 'flex', gap: 18, alignItems: 'flex-start', flexWrap: 'wrap' }}>
          <img
            className="thumb"
            src={galleryUrl(data.items[0].file)}
            alt={data.items[0].caption}
            style={{ width: 96, height: 96 }}
          />
          <div style={{ flex: '1 1 320px' }}>
            <BarChart
              rows={data.items.map((it, j) => ({
                label: `“${it.caption.length > 34 ? it.caption.slice(0, 34) + '…' : it.caption}”`,
                value: catProbs[j],
                emph: j === 0,
              }))}
            />
          </div>
        </div>
        <div className="controls-row" style={{ marginTop: 16 }}>
          <Slider
            label="temperature τ"
            value={logTau}
            min={-2}
            max={0}
            step={0.01}
            onChange={setLogTau}
            format={() => tau.toFixed(tau < 0.1 ? 3 : 2)}
          />
          <span className="small muted" style={{ flex: '1 1 220px' }}>
            {tau > 0.5
              ? 'High τ → logits barely differ → nearly uniform probabilities. The model looks clueless even though the cosines are fine.'
              : tau > 0.04
                ? 'Around the paper’s initial value (0.07): the true caption clearly leads, but rivals keep visible probability.'
                : 'Tiny τ → the largest cosine takes almost all probability mass. This is roughly where CLIP’s learned τ ended up (~0.01).'}
          </span>
        </div>
      </div>

      <h2>Step 3 · Score it with cross-entropy — in both directions</h2>
      <p>
        The correct “class” for image I<sub>i</sub> is caption T<sub>i</sub>, so the loss
        for one image is simply <Formula tex={String.raw`-\log p_i(i)`} /> — near 0 when
        the true pair gets almost all the probability, large when it doesn't. Averaged
        over the batch:
      </p>
      <Formula
        display
        tex={String.raw`\mathcal{L}_{\text{img}\to\text{txt}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{\ell_{ii}}}{\sum_{k} e^{\ell_{ik}}}
        \qquad
        \mathcal{L}_{\text{txt}\to\text{img}} = -\frac{1}{N}\sum_{j=1}^{N}\log\frac{e^{\ell_{jj}}}{\sum_{k} e^{\ell_{kj}}}`}
      />
      <p>
        The same game is played down each <em>column</em> (each caption classifies the
        images), and CLIP averages the two:
      </p>
      <Formula
        display
        tex={String.raw`\mathcal{L} \;=\; \tfrac{1}{2}\left(\mathcal{L}_{\text{img}\to\text{txt}} + \mathcal{L}_{\text{txt}\to\text{img}}\right)`}
      />

      <div className="card wide">
        <p className="card-title">The loss lab</p>
        <p className="card-sub">
          The real gallery matrix under each transformation. Diagonal cells (true pairs)
          are ringed in green.
        </p>
        <div className="controls-row">
          <div className="seg" role="tablist">
            {(Object.keys(VIEW_LABELS) as View[]).map((v) => (
              <button
                key={v}
                className={view === v ? 'active' : ''}
                onClick={() => setView(v)}
              >
                {VIEW_LABELS[v]}
              </button>
            ))}
          </div>
          <Slider
            label="temperature τ"
            value={logTau}
            min={-2}
            max={0}
            step={0.01}
            onChange={setLogTau}
            format={() => tau.toFixed(tau < 0.1 ? 3 : 2)}
          />
        </div>
        <MatrixView
          values={shown}
          rowHeads={data.items.map((it, i) => (
            <span key={it.id} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              <span className="small muted tabular">I{i + 1}</span>
              <img className="thumb" src={galleryUrl(it.file)} alt={it.caption} />
            </span>
          ))}
          colHeads={data.items.map((it, j) => (
            <span key={it.id} style={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'center' }}>
              <span className="small muted tabular">T{j + 1}</span>
              <span style={{ fontSize: 11 }}>{it.className}</span>
            </span>
          ))}
          markDiagonal
          domain={view === 'rowP' || view === 'colP' ? [0, 1] : 'auto'}
          valueFormat={(v) =>
            view === 'logits' ? v.toFixed(Math.abs(v) >= 10 ? 0 : 1) : v.toFixed(2)
          }
          corner={<span>images ↓ · texts →</span>}
          legendLabel={
            view === 'sim'
              ? 'cosine similarity'
              : view === 'logits'
                ? 'logit = cosine ÷ τ'
                : 'probability'
          }
        />
        <table className="readout-table" style={{ marginTop: 16, maxWidth: 460 }}>
          <tbody>
            <tr>
              <td>
                image→text loss <span className="muted small">(rows)</span>
              </td>
              <td className="tabular">{lossImg.toFixed(3)}</td>
            </tr>
            <tr>
              <td>
                text→image loss <span className="muted small">(columns)</span>
              </td>
              <td className="tabular">{lossTxt.toFixed(3)}</td>
            </tr>
            <tr>
              <td>total contrastive loss</td>
              <td className="tabular emph">{((lossImg + lossTxt) / 2).toFixed(3)}</td>
            </tr>
          </tbody>
        </table>
        <p className="small muted" style={{ marginTop: 10 }}>
          Try it: switch to <em>row softmax</em> and pull τ down — the diagonal soaks up
          all probability and the loss collapses toward 0. Pull τ up to 1 and the loss
          approaches ln {n} ≈ {Math.log(n).toFixed(2)}, the loss of pure guessing among{' '}
          {n} options.
        </p>
      </div>

      <div className="note">
        <span className="note-label">The “85%” trap</span>
        When a CLIP demo says <em>“dog: 85%”</em>, that number is one of these softmax{' '}
        <strong>probabilities</strong> — produced from small cosines (≈ 0.2–0.3)
        stretched by 1/τ. It is <em>not</em> a cosine similarity, and calling it
        “85% similarity” muddles two different quantities you now know how to tell
        apart.
      </div>

      <details className="deep-dive">
        <summary>Deep dive: why train in both directions?</summary>
        <div className="deep-dive-body">
          <p>
            The row direction alone (“image classifies captions”) could be satisfied by
            solutions that discriminate poorly the other way — several captions could,
            for instance, collapse onto nearly identical embeddings as long as each
            image still ranks its own caption first. Adding the column direction makes
            every caption also compete to claim its image, forcing both modalities to
            spread out and carry information. In the paper's pseudocode this is
            literally <code>cross_entropy(logits, labels, axis=0)</code> and{' '}
            <code>axis=1</code>, averaged.
          </p>
        </div>
      </details>

      <details className="deep-dive">
        <summary>Deep dive: what the gradient actually does</summary>
        <div className="deep-dive-body">
          <p>
            Differentiating the row loss for image I<sub>i</sub> gives a beautifully
            interpretable update: the gradient pulls I<sub>i</sub> toward its true
            caption with strength <Formula tex={String.raw`(1 - p_i(i))`} /> and pushes
            it away from every wrong caption T<sub>j</sub> with strength{' '}
            <Formula tex={String.raw`p_i(j)`} />. Confidently correct pairs (p ≈ 1)
            generate almost no update; confident mistakes generate large ones. This is
            also why <strong>big batches help</strong>: more captions in the row means
            more — and harder — negatives to push against. You'll see these forces at
            work in the next chapter.
          </p>
        </div>
      </details>
    </>
  );
}
