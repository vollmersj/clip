import { useMemo, useState } from 'react';
import { data, galleryUrl } from '../lib/data';
import { dot, rankDescending } from '../lib/math';
import { GalleryPicker } from '../components/GalleryPicker';
import { BarChart } from '../components/BarChart';

function TextToImage() {
  const [qIdx, setQIdx] = useState(0);
  const query = data.queries[qIdx];
  const { order, scores } = useMemo(() => {
    const scores = data.items.map((it) => dot(data.queryEmbeds[qIdx], it.imageEmbed));
    return { order: rankDescending(scores), scores };
  }, [qIdx]);

  return (
    <div className="card wide">
      <p className="card-title">
        Text → image search <span className="tag txt">query</span> →{' '}
        <span className="tag img">results</span>
      </p>
      <p className="card-sub">
        None of these queries contains a gallery class name — the match is semantic, not
        string matching.
      </p>
      <select
        className="caption-select"
        value={qIdx}
        onChange={(e) => setQIdx(Number(e.target.value))}
        style={{ marginBottom: 16 }}
      >
        {data.queries.map((q, i) => (
          <option key={i} value={i}>
            “{q}”
          </option>
        ))}
      </select>
      <div className="retrieval-rank">
        {order.map((itemIdx, rank) => (
          <div key={itemIdx} className={`rank-item${rank === 0 ? ' top' : ''}`}>
            <span className="rank-badge">{rank + 1}</span>
            <img
              className="thumb"
              src={galleryUrl(data.items[itemIdx].file)}
              alt={data.items[itemIdx].caption}
            />
            <span className="rank-score">{scores[itemIdx].toFixed(3)}</span>
          </div>
        ))}
      </div>
      <p className="small muted" style={{ margin: '12px 0 0' }}>
        The whole “search engine” is: embed the query “{query}”, dot-product against
        the 8 precomputed image embeddings, sort. An image index never needs
        re-embedding for new queries.
      </p>
    </div>
  );
}

function ImageToText() {
  const [imgIdx, setImgIdx] = useState(0);
  // Candidate pool: all 8 captions + all 10 free-text queries.
  const texts = useMemo(
    () => [
      ...data.items.map((it) => ({ text: `“${it.caption}”`, embed: it.captionEmbed })),
      ...data.queries.map((q, i) => ({ text: `“${q}”`, embed: data.queryEmbeds[i] })),
    ],
    [],
  );
  const scores = useMemo(
    () => texts.map((t) => dot(data.items[imgIdx].imageEmbed, t.embed)),
    [texts, imgIdx],
  );
  const order = rankDescending(scores).slice(0, 6);

  return (
    <div className="card wide">
      <p className="card-title">
        Image → text search <span className="tag img">query</span> →{' '}
        <span className="tag txt">results</span>
      </p>
      <p className="card-sub">
        Same embeddings, opposite direction: which of {texts.length} candidate texts
        (all captions + all queries) best describes this image?
      </p>
      <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <img
          className="thumb"
          src={galleryUrl(data.items[imgIdx].file)}
          alt={data.items[imgIdx].caption}
          style={{ width: 132, height: 132 }}
        />
        <div style={{ flex: '1 1 340px' }}>
          <BarChart
            rows={order.map((tIdx, rank) => ({
              label: texts[tIdx].text,
              value: scores[tIdx],
              emph: rank === 0,
            }))}
            max={Math.max(...scores) * 1.15}
            format={(v) => v.toFixed(3)}
          />
          <p className="small muted" style={{ marginTop: 8 }}>
            Top 6 of {texts.length} candidates, by cosine similarity.
          </p>
        </div>
      </div>
      <div style={{ marginTop: 14 }}>
        <GalleryPicker selected={imgIdx} onSelect={setImgIdx} showLabels={false} />
      </div>
    </div>
  );
}

export function Ch8Retrieval() {
  return (
    <>
      <h1>Retrieval: Search in Both Directions</h1>
      <p className="lede">
        Zero-shot classification asked “which text fits this image?” over a handful of
        classes. Ask the same question over <em>thousands of images</em> instead and you
        have a search engine. Because both modalities share one space, it works in both
        directions.
      </p>

      <TextToImage />

      <ImageToText />

      <h2>Why this matters in practice</h2>
      <ul>
        <li>
          <strong>Semantic image search:</strong> photo libraries and stock-photo sites
          embed their entire collection once, then answer any natural-language query
          with one text-encoder call plus a fast nearest-neighbor lookup.
        </li>
        <li>
          <strong>The bridge runs both ways:</strong> the same index answers “find
          images for this text” and “find texts for this image” — captioning-ish
          behavior without a captioning model.
        </li>
        <li>
          <strong>Beyond search:</strong> CLIP embeddings became the backbone for
          text-to-image generators (Stable Diffusion's text conditioning, DALL·E 2's
          prior), image deduplication, content moderation, and dataset curation.
        </li>
      </ul>

      <div className="note">
        <span className="note-label">Look closer</span>
        Try the query <em>“a fluffy animal”</em> — the cat and dog should both surface
        near the top, well above the vehicles. Then flip direction: for the dog image,
        the ranked texts mix its true caption with fitting free-text queries like
        “man's best friend”. One shared space, many uses.
      </div>
    </>
  );
}
