import { useRef, useState } from 'react';
import { data, galleryUrl } from '../lib/data';
import { getClip, MODEL_ID, type ClipModel, type OverallProgress } from '../lib/clip';
import { dot, softmax, argmax, rankDescending } from '../lib/math';
import { BarChart } from '../components/BarChart';

const TAU = 0.01;

const DEFAULT_LABELS = [
  'a photo of a cat',
  'a photo of a dog',
  'a photo of a car',
  'a photo of a bicycle',
].join('\n');

type LoadState =
  | { phase: 'idle' }
  | { phase: 'loading'; progress: OverallProgress | null }
  | { phase: 'ready'; model: ClipModel }
  | { phase: 'error'; message: string };

function ModelLoader({
  state,
  onLoad,
}: {
  state: LoadState;
  onLoad: () => void;
}) {
  return (
    <div className="card">
      <p className="card-title">Step 1 · Load the real model</p>
      <p className="card-sub">
        <code>{MODEL_ID}</code> — the quantized ONNX export of OpenAI's ViT-B/32
        weights (~120 MB). It downloads once from the Hugging Face CDN, is cached by
        your browser, and runs <strong>entirely on your machine</strong> — no image or
        text you enter here leaves your computer.
      </p>
      {state.phase === 'idle' && (
        <button className="btn primary" onClick={onLoad}>
          ⬇ Load CLIP (~120 MB)
        </button>
      )}
      {state.phase === 'loading' && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span className="spinner" />
          <div style={{ flex: 1 }}>
            <div className="progress-outer">
              <div
                className="progress-inner"
                style={{
                  width: `${Math.round((state.progress?.fraction ?? 0) * 100)}%`,
                }}
              />
            </div>
            <p className="small muted" style={{ margin: '6px 0 0' }}>
              {state.progress
                ? `${state.progress.loadedMB.toFixed(0)} / ${state.progress.totalMB.toFixed(0)} MB`
                : 'Preparing download…'}
            </p>
          </div>
        </div>
      )}
      {state.phase === 'ready' && (
        <p style={{ color: 'var(--good-text)', fontWeight: 650, margin: 0 }}>
          ✓ Model loaded — the tools below are live.
        </p>
      )}
      {state.phase === 'error' && (
        <div>
          <p style={{ color: 'var(--bad)', margin: '0 0 10px' }}>
            Loading failed: {state.message}. Are you online? (The model comes from
            huggingface.co.)
          </p>
          <button className="btn" onClick={onLoad}>
            Retry
          </button>
        </div>
      )}
    </div>
  );
}

function LiveClassifier({ model }: { model: ClipModel }) {
  const [imageUrl, setImageUrl] = useState<string>(galleryUrl(data.items[1].file));
  const [imageBlob, setImageBlob] = useState<Blob | null>(null);
  const [labelsText, setLabelsText] = useState(DEFAULT_LABELS);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ labels: string[]; cos: number[]; probs: number[] } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const classify = async () => {
    const labels = labelsText
      .split('\n')
      .map((s) => s.trim())
      .filter(Boolean);
    if (labels.length < 2) {
      setError('Enter at least two labels (one per line).');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const [textEmbeds, imgEmbed] = await Promise.all([
        model.embedTexts(labels),
        imageBlob ? model.embedImageBlob(imageBlob) : model.embedImageUrl(imageUrl),
      ]);
      const cos = textEmbeds.map((t) => dot(imgEmbed, t));
      setResult({ labels, cos, probs: softmax(cos.map((v) => v / TAU)) });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const onFile = (f: File | undefined) => {
    if (!f) return;
    setImageBlob(f);
    setImageUrl(URL.createObjectURL(f));
    setResult(null);
  };

  const best = result ? argmax(result.probs) : -1;

  return (
    <div className="card wide">
      <p className="card-title">Step 2 · Zero-shot classify your own image</p>
      <p className="card-sub">
        Upload any image (it stays local) and type any labels — full sentences work
        best, exactly like the prompt templates in Chapter 7.
      </p>
      <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
        <div style={{ flex: '0 1 240px' }}>
          <img
            className="thumb"
            src={imageUrl}
            alt="classification target"
            style={{ width: '100%', maxWidth: 240, aspectRatio: '1', objectFit: 'cover' }}
          />
          <div style={{ display: 'flex', gap: 8, marginTop: 10, flexWrap: 'wrap' }}>
            <button className="btn" onClick={() => fileRef.current?.click()}>
              📁 Upload image
            </button>
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              style={{ display: 'none' }}
              onChange={(e) => onFile(e.target.files?.[0])}
            />
          </div>
          <div className="gallery-strip" style={{ marginTop: 10, gridTemplateColumns: 'repeat(8, 1fr)' }}>
            {data.items.map((it) => (
              <button
                key={it.id}
                className="gallery-pick"
                onClick={() => {
                  setImageBlob(null);
                  setImageUrl(galleryUrl(it.file));
                  setResult(null);
                }}
                title={it.caption}
              >
                <img className="thumb" src={galleryUrl(it.file)} alt={it.caption} />
              </button>
            ))}
          </div>
        </div>
        <div style={{ flex: '1 1 320px' }}>
          <p className="stage-label">Candidate labels (one per line)</p>
          <textarea
            className="caption-select"
            rows={5}
            value={labelsText}
            onChange={(e) => setLabelsText(e.target.value)}
            style={{ resize: 'vertical', fontFamily: 'inherit' }}
          />
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginTop: 10 }}>
            <button className="btn primary" onClick={classify} disabled={busy}>
              {busy ? 'Encoding…' : '▶ Classify'}
            </button>
            {busy && <span className="spinner" />}
            {error && <span className="small" style={{ color: 'var(--bad)' }}>{error}</span>}
          </div>
          {result && (
            <div style={{ marginTop: 16 }}>
              <BarChart
                rows={result.labels.map((label, i) => ({
                  label: `“${label}”`,
                  value: result.probs[i],
                  emph: i === best,
                }))}
              />
              <p className="small muted" style={{ marginTop: 8 }}>
                cosines:{' '}
                {result.cos.map((c, i) => (
                  <span key={i} className="tabular" style={{ marginRight: 10 }}>
                    {c.toFixed(3)}
                  </span>
                ))}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function LiveSearch({ model }: { model: ClipModel }) {
  const [query, setQuery] = useState('something cozy and warm');
  const [busy, setBusy] = useState(false);
  const [ranked, setRanked] = useState<{ order: number[]; scores: number[] } | null>(null);

  const search = async () => {
    if (!query.trim()) return;
    setBusy(true);
    try {
      const [q] = await model.embedTexts([query.trim()]);
      const scores = data.items.map((it) => dot(q, it.imageEmbed));
      setRanked({ order: rankDescending(scores), scores });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card wide">
      <p className="card-title">Step 3 · Search the gallery with any words</p>
      <p className="card-sub">
        Your free-text query is embedded live, then compared against the gallery's
        precomputed image embeddings — precisely how large-scale semantic image search
        works.
      </p>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        <input
          className="caption-select"
          style={{ flex: '1 1 260px' }}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && search()}
          placeholder='e.g. "something cozy and warm"'
        />
        <button className="btn primary" onClick={search} disabled={busy}>
          {busy ? 'Searching…' : '🔍 Search'}
        </button>
      </div>
      {ranked && (
        <div className="retrieval-rank" style={{ marginTop: 16 }}>
          {ranked.order.map((itemIdx, rank) => (
            <div key={itemIdx} className={`rank-item${rank === 0 ? ' top' : ''}`}>
              <span className="rank-badge">{rank + 1}</span>
              <img
                className="thumb"
                src={galleryUrl(data.items[itemIdx].file)}
                alt={data.items[itemIdx].caption}
              />
              <span className="rank-score">{ranked.scores[itemIdx].toFixed(3)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function Ch9Playground() {
  const [state, setState] = useState<LoadState>({ phase: 'idle' });

  const loadModel = async () => {
    setState({ phase: 'loading', progress: null });
    try {
      const model = await getClip((progress) =>
        setState((s) => (s.phase === 'loading' ? { phase: 'loading', progress } : s)),
      );
      setState({ phase: 'ready', model });
    } catch (e) {
      setState({
        phase: 'error',
        message: e instanceof Error ? e.message : String(e),
      });
    }
  };

  return (
    <>
      <h1>Playground: Run CLIP Live</h1>
      <p className="lede">
        Everything until now used precomputed embeddings. Time to drop the training
        wheels: load the actual model into your browser and point it at{' '}
        <em>your own</em> images and words.
      </p>

      <ModelLoader state={state} onLoad={loadModel} />

      {state.phase === 'ready' && (
        <>
          <LiveClassifier model={state.model} />
          <LiveSearch model={state.model} />
          <h2>Ideas to test its limits</h2>
          <ul>
            <li>
              <strong>Fine-grained labels:</strong> “a photo of a tabby cat” vs. “a
              photo of a siamese cat” — how confident is it now?
            </li>
            <li>
              <strong>Attributes and context:</strong> “a red car” vs. “a blue car”;
              “a dog indoors” vs. “a dog outside”.
            </li>
            <li>
              <strong>Counting (a known weakness):</strong> “one apple” vs. “three
              apples”. CLIP mostly can't count.
            </li>
            <li>
              <strong>Typographic attack:</strong> write the word “pizza” on a sticky
              note, photograph it on your laptop, and classify with labels “a photo of
              a laptop” / “a photo of a pizza”. The paper documents CLIP reading the
              text instead of seeing the object.
            </li>
          </ul>
        </>
      )}

      <div className="note">
        <span className="note-label">Why this works on GitHub Pages</span>
        There's no server: the page is static files, the model comes from a public CDN,
        and inference runs in WebAssembly inside your browser. Modern ML deployment can
        be surprisingly serverless.
      </div>
    </>
  );
}
