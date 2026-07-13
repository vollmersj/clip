import { useMemo, useState } from 'react';
import { data, galleryUrl } from '../lib/data';
import { similarityMatrix, rankDescending } from '../lib/math';
import { MatrixView } from '../components/MatrixView';

export function Ch4SimilarityMatrix() {
  const [hovered, setHovered] = useState<[number, number] | null>(null);
  const [showDiag, setShowDiag] = useState(false);

  const matrix = useMemo(
    () =>
      similarityMatrix(
        data.items.map((it) => it.imageEmbed),
        data.items.map((it) => it.captionEmbed),
      ),
    [],
  );

  const n = data.items.length;
  const detail = hovered
    ? {
        img: data.items[hovered[0]],
        txt: data.items[hovered[1]],
        value: matrix[hovered[0]][hovered[1]],
        rank:
          rankDescending(matrix[hovered[0]]).indexOf(hovered[1]) + 1,
        isMatch: hovered[0] === hovered[1],
      }
    : null;

  return (
    <>
      <h1>The Similarity Matrix</h1>
      <p className="lede">
        Take a training batch of {n} (image, caption) pairs, encode everything, and
        compute the cosine similarity of <em>every image with every caption</em>. The
        result is an {n}×{n} matrix — the heart of CLIP's training step. Below: the
        real matrix for our gallery, from the real model.
      </p>

      <div className="card wide">
        <p className="card-title">
          Every image × every caption <span className="muted">(hover the cells)</span>
        </p>
        <p className="card-sub">
          Rows: <span className="tag img">image embeddings</span> I₁…I₈ · Columns:{' '}
          <span className="tag txt">caption embeddings</span> T₁…T₈ · Values: cosine
          similarity
        </p>
        <MatrixView
          values={matrix}
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
          markDiagonal={showDiag}
          onHover={setHovered}
          corner={<span>images ↓ · texts →</span>}
          legendLabel="cosine similarity"
        />
        <div className="pair-detail">
          {detail ? (
            <>
              <img
                className="thumb"
                src={galleryUrl(detail.img.file)}
                alt={detail.img.caption}
              />
              <span className="pair-detail-caption">
                I{hovered![0] + 1} × T{hovered![1] + 1}: this image vs. “
                {detail.txt.caption}”
                {detail.isMatch ? (
                  <strong style={{ color: 'var(--good-text)' }}> — the true pair</strong>
                ) : (
                  <span> — a mismatched pair</span>
                )}
              </span>
              <span className="pair-detail-score">
                {detail.value.toFixed(3)}
                <span className="sub">
                  rank {detail.rank} of {n} in its row
                </span>
              </span>
            </>
          ) : (
            <span className="pair-detail-caption muted">
              Hover a cell to inspect one (image, caption) pairing…
            </span>
          )}
        </div>
        <label
          className="small"
          style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 12, cursor: 'pointer' }}
        >
          <input
            type="checkbox"
            checked={showDiag}
            onChange={(e) => setShowDiag(e.target.checked)}
          />
          Highlight the matching pairs (the diagonal)
        </label>
      </div>

      <h2>What to notice</h2>
      <ul>
        <li>
          <strong>The diagonal wins.</strong> In every row, the true caption scores
          highest — the model was trained precisely to make that happen.
        </li>
        <li>
          <strong>Semantics leak into the off-diagonal.</strong> The cat image scores
          noticeably higher with the dog caption than with the pizza caption: related
          concepts sit closer in the shared space. The matrix is not just
          “diagonal = 1, rest = 0” — it encodes a whole geometry of meaning.
        </li>
        <li>
          <strong>One batch = free negatives.</strong> With {n} pairs we get {n} correct
          pairings and {n * n - n} incorrect ones to push away — no extra labeling
          needed. CLIP's actual batch size was 32,768, which yields about a{' '}
          <em>billion</em> incorrect pairings per batch.
        </li>
      </ul>

      <div className="note">
        <span className="note-label">Where this goes next</span>
        Training wants the diagonal to be the biggest value in each <em>row</em> and in
        each <em>column</em>. Chapter 5 turns that wish into a precise, differentiable
        loss function.
      </div>
    </>
  );
}
