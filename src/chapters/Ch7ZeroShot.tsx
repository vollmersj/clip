import { useMemo, useState } from 'react';
import { data, galleryUrl, fillTemplate } from '../lib/data';
import { dot, softmax, argmax } from '../lib/math';
import { GalleryPicker } from '../components/GalleryPicker';
import { BarChart } from '../components/BarChart';

/** CLIP's learned logit scale: logits = 100 · cosine, i.e. τ = 0.01. */
const TAU = 0.01;

export function Ch7ZeroShot() {
  const [imgIdx, setImgIdx] = useState(1); // start on the dog, like the classic demo
  const [templateIdx, setTemplateIdx] = useState(1); // "a photo of a {}"
  const [enabled, setEnabled] = useState<boolean[]>(data.items.map(() => true));

  const classIdxs = data.items.map((_, c) => c).filter((c) => enabled[c]);

  const { cosines, probs } = useMemo(() => {
    const img = data.items[imgIdx].imageEmbed;
    const cosines = classIdxs.map((c) => dot(img, data.templateEmbeds[templateIdx][c]));
    return { cosines, probs: softmax(cosines.map((v) => v / TAU)) };
  }, [imgIdx, templateIdx, classIdxs.join(',')]); // eslint-disable-line react-hooks/exhaustive-deps

  const best = argmax(probs);

  // Template comparison: top-class probability for each template (fixed class set: all 8)
  const templateComparison = useMemo(() => {
    const img = data.items[imgIdx].imageEmbed;
    return data.templates.map((tpl, t) => {
      const cos = data.items.map((_, c) => dot(img, data.templateEmbeds[t][c]));
      const p = softmax(cos.map((v) => v / TAU));
      const b = argmax(p);
      return { tpl, prob: p[b], cls: data.items[b].className, correct: b === imgIdx };
    });
  }, [imgIdx]);

  const toggleClass = (c: number) => {
    const next = enabled.slice();
    next[c] = !next[c];
    if (next.filter(Boolean).length >= 2) setEnabled(next);
  };

  return (
    <>
      <h1>Zero-Shot Classification</h1>
      <p className="lede">
        Here is the payoff. We never trained a cat-vs-dog-vs-pizza classifier — yet CLIP
        can classify these images. The trick: <strong>turn the class names into
        sentences</strong>, embed them, and let cosine similarity vote.
      </p>

      <h2>The recipe</h2>
      <ol>
        <li>
          Write each candidate class into a <strong>prompt template</strong>: “cat” →
          “a photo of a cat”.
        </li>
        <li>
          Push all prompts through the <span className="tag txt">text encoder</span> —
          this builds a classifier out of thin air.
        </li>
        <li>
          Push the image through the <span className="tag img">image encoder</span>.
        </li>
        <li>
          Cosine-compare, scale by the learned 1/τ = 100, softmax → probabilities.
        </li>
      </ol>

      <div className="card wide">
        <p className="card-title">Zero-shot classifier, on real embeddings</p>
        <p className="card-sub">
          Pick an image, edit the class list, switch templates — everything updates from
          precomputed CLIP embeddings.
        </p>

        <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
          <div style={{ flex: '0 0 auto' }}>
            <img
              className="thumb"
              src={galleryUrl(data.items[imgIdx].file)}
              alt={data.items[imgIdx].caption}
              style={{ width: 148, height: 148 }}
            />
          </div>
          <div style={{ flex: '1 1 340px' }}>
            <p className="stage-label" style={{ marginBottom: 6 }}>
              Candidate classes (click to include/exclude)
            </p>
            <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap', marginBottom: 12 }}>
              {data.items.map((it, c) => (
                <button
                  key={it.id}
                  className={`class-toggle${enabled[c] ? ' on' : ''}`}
                  onClick={() => toggleClass(c)}
                >
                  {it.className}
                </button>
              ))}
            </div>
            <p className="stage-label" style={{ marginBottom: 6 }}>
              Prompt template
            </p>
            <select
              className="caption-select"
              value={templateIdx}
              onChange={(e) => setTemplateIdx(Number(e.target.value))}
            >
              {data.templates.map((tpl, t) => (
                <option key={t} value={t}>
                  “{tpl.replace('{}', '…')}”
                </option>
              ))}
            </select>
          </div>
        </div>

        <div style={{ marginTop: 18 }}>
          <BarChart
            rows={classIdxs.map((c, k) => ({
              label: `“${fillTemplate(data.templates[templateIdx], data.items[c].className)}”`,
              value: probs[k],
              emph: k === best,
            }))}
            format={(v) => `${(v * 100).toFixed(1)}%`}
          />
          <p className="small muted" style={{ marginTop: 10 }}>
            Raw cosines behind those probabilities:{' '}
            {classIdxs.map((c, k) => (
              <span key={c} className="tabular" style={{ marginRight: 10 }}>
                {data.items[c].className} {cosines[k].toFixed(3)}
              </span>
            ))}
          </p>
        </div>

        <div style={{ marginTop: 10 }}>
          <GalleryPicker selected={imgIdx} onSelect={setImgIdx} />
        </div>
      </div>

      <div className="note">
        <span className="note-label">Try this</span>
        Exclude the true class. Select the cat image, then remove <em>cat</em> from the
        class list — CLIP redistributes its belief over what's left (usually onto{' '}
        <em>dog</em>, the semantically nearest option). A classic classifier could never
        do this: its classes are baked in at training time; here the class list is just
        text you can edit.
      </div>

      <h2>Prompt engineering is real</h2>
      <p>
        The template is not cosmetic. A bare class name is ambiguous (“apple” the fruit
        or the company?) and unlike the full sentences CLIP saw during pre-training.
        For the currently selected image, each template gives:
      </p>
      <div className="card">
        <table className="readout-table">
          <tbody>
            {templateComparison.map((row, t) => (
              <tr key={t}>
                <td>
                  “{fillTemplate(row.tpl, data.items[imgIdx].className)}”
                  {t === templateIdx && (
                    <span className="muted small"> ← selected</span>
                  )}
                </td>
                <td className="tabular">
                  {row.correct ? (
                    <span style={{ color: 'var(--good-text)' }}>
                      {row.cls} {(row.prob * 100).toFixed(1)}%
                    </span>
                  ) : (
                    <span style={{ color: 'var(--bad)' }}>
                      → {row.cls}! {(row.prob * 100).toFixed(1)}%
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="small muted" style={{ margin: '10px 0 0' }}>
          Top prediction and its confidence per template, always over all 8 classes.
          The paper went further: ensembling 80 templates (“a photo of a big {'{}'}”,
          “a blurry photo of a {'{}'}”, …) buys another ~3.5% ImageNet accuracy.
        </p>
      </div>

      <details className="deep-dive">
        <summary>Deep dive: the text encoder as a “classifier factory”</summary>
        <div className="deep-dive-body">
          <p>
            A linear classifier is just a weight vector per class, scored by dot
            product. Here, those weight vectors are the normalized text embeddings —
            so the text encoder is effectively a <em>hypernetwork</em> that
            manufactures classifier weights from any class description you type. New
            task, new classes? No retraining — just new sentences. That's why CLIP is
            called <strong>zero-shot</strong>: zero task-specific training examples.
          </p>
        </div>
      </details>
    </>
  );
}
