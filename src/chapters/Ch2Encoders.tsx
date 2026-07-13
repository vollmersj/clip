import { useState } from 'react';
import { data, galleryUrl } from '../lib/data';
import { VectorStrip } from '../components/VectorStrip';
import { GalleryPicker } from '../components/GalleryPicker';

function TextEncoderDemo() {
  const [idx, setIdx] = useState(0);
  const item = data.items[idx];
  return (
    <div className="card">
      <p className="card-title">
        From sentence to vector <span className="tag txt">text</span>
      </p>
      <p className="card-sub">
        Pick a caption to see its real BPE tokens and its real CLIP embedding.
      </p>
      <select
        className="caption-select"
        value={idx}
        onChange={(e) => setIdx(Number(e.target.value))}
      >
        {data.items.map((it, i) => (
          <option key={it.id} value={i}>
            “{it.caption}”
          </option>
        ))}
      </select>

      <div className="encoder-stage">
        <p className="stage-label">1 · Tokenize (byte-pair encoding, lowercased)</p>
        <div className="token-row">
          {item.captionTokens.map((t, i) => (
            <span
              key={i}
              className={`token-chip${t === '[SOS]' || t === '[EOS]' ? ' special' : ''}`}
            >
              {t}
            </span>
          ))}
        </div>
        <p className="small muted" style={{ margin: '6px 0 0' }}>
          {item.captionTokens.length} tokens. The vocabulary has ~49,000 entries;
          sequences are capped at 77 tokens.
        </p>
      </div>

      <div className="encoder-stage">
        <p className="stage-label">
          2 · Transformer reads the sequence → the activation at{' '}
          <span className="token-chip special" style={{ fontSize: 11 }}>
            [EOS]
          </span>{' '}
          summarizes the whole sentence
        </p>
      </div>

      <div className="encoder-stage">
        <p className="stage-label">3 · Linear projection into the shared space</p>
        <VectorStrip
          values={item.captionEmbed}
          modality="txt"
          title="First 20 of 512 dimensions (real values)"
        />
        <p className="small muted" style={{ margin: '6px 0 0' }}>
          The strip shows the first 20 of {data.dim} dimensions — blue = positive, red =
          negative. These are the actual numbers the model produced for this caption.
        </p>
      </div>
    </div>
  );
}

function ImageEncoderDemo() {
  const [idx, setIdx] = useState(0);
  const [hoverPatch, setHoverPatch] = useState<number | null>(null);
  const item = data.items[idx];
  const grid = 7; // ViT-B/32 on a 224×224 input: 7 × 7 patches of 32 px

  return (
    <div className="card">
      <p className="card-title">
        From pixels to vector <span className="tag img">image</span>
      </p>
      <p className="card-sub">
        A Vision Transformer (ViT) treats an image like a sentence — made of patches
        instead of words. Hover the grid.
      </p>
      <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <div>
          <div className="patch-wrap">
            <img src={galleryUrl(item.file)} alt={item.caption} className="patch-img" />
            <div
              className="patch-grid"
              style={{ gridTemplateColumns: `repeat(${grid}, 1fr)` }}
              onMouseLeave={() => setHoverPatch(null)}
            >
              {Array.from({ length: grid * grid }, (_, p) => (
                <div
                  key={p}
                  className={`patch-cell${hoverPatch === p ? ' hl' : ''}`}
                  onMouseEnter={() => setHoverPatch(p)}
                />
              ))}
            </div>
          </div>
          <p className="small muted" style={{ margin: '8px 0 0', maxWidth: 280 }}>
            {hoverPatch === null
              ? 'The 224×224 input is cut into 7×7 = 49 patches of 32×32 pixels.'
              : `Patch ${hoverPatch + 1} of 49 → one “visual token”.`}
          </p>
        </div>
        <div style={{ flex: '1 1 300px' }}>
          <div className="encoder-stage" style={{ marginTop: 0 }}>
            <p className="stage-label">1 · Each patch → one token (a learned linear map)</p>
            <p className="small muted">
              49 patch tokens + 1 <code>[CLS]</code> token + position information.
            </p>
          </div>
          <div className="encoder-stage">
            <p className="stage-label">
              2 · Transformer layers mix information between patches
            </p>
            <p className="small muted">
              Every patch can attend to every other patch — the ears, the whiskers, the
              background all inform each other.
            </p>
          </div>
          <div className="encoder-stage">
            <p className="stage-label">3 · Linear projection into the shared space</p>
            <VectorStrip
              values={item.imageEmbed}
              modality="img"
              title="First 20 of 512 dimensions (real values)"
            />
          </div>
        </div>
      </div>
      <div style={{ marginTop: 16 }}>
        <GalleryPicker selected={idx} onSelect={setIdx} showLabels={false} />
      </div>
    </div>
  );
}

export function Ch2Encoders() {
  return (
    <>
      <h1>Two Encoders, One Goal</h1>
      <p className="lede">
        Both encoders do the same job for different media: compress their input into a
        single {data.dim}-dimensional vector, its <em>embedding</em>. What matters is
        that both vectors land in the <em>same</em> space — that's what Chapter 3 is
        about.
      </p>

      <h2>
        The text encoder <span className="tag txt">text</span>
      </h2>
      <p>
        The text encoder is a <strong>Transformer</strong> (12 layers, 512 wide, 8
        attention heads — about 63M parameters). It reads the caption as a sequence of
        tokens and summarizes it in the activation of the final <code>[EOS]</code>{' '}
        token, which is then linearly projected into the shared embedding space.
      </p>
      <TextEncoderDemo />

      <h2>
        The image encoder <span className="tag img">image</span>
      </h2>
      <p>
        For images, the CLIP authors tried two families: ResNets and{' '}
        <strong>Vision Transformers (ViT)</strong>. The best model — and the one this
        walkthrough runs — is <strong>ViT-B/32</strong>: “B” for Base size, “32” for the
        patch size in pixels.
      </p>
      <ImageEncoderDemo />

      <div className="note">
        <span className="note-label">Key point</span>
        Two different architectures, two different inputs — but the outputs are the same
        kind of object: one vector of {data.dim} numbers each. From here on, the images
        and the sentences live in the same world.
      </div>

      <details className="deep-dive">
        <summary>Deep dive: training-scale details from the paper</summary>
        <div className="deep-dive-body">
          <ul>
            <li>
              CLIP was trained on <strong>400M (image, text) pairs</strong> for 32
              epochs with a batch size of <strong>32,768</strong>.
            </li>
            <li>
              The largest ResNet (RN50x64) took <strong>18 days on 592 V100 GPUs</strong>;
              the largest ViT took 12 days on 256 GPUs.
            </li>
            <li>
              The only data augmentation was a random square crop — with this much data,
              overfitting simply wasn't the concern.
            </li>
            <li>
              Both encoders were trained <strong>from scratch</strong>, with no ImageNet
              or language-model initialization.
            </li>
          </ul>
        </div>
      </details>
    </>
  );
}
