import { data, galleryUrl } from '../lib/data';
import { dot } from '../lib/math';
import { BarChart } from '../components/BarChart';

/** Static overview diagram: two encoders feeding one shared space. */
function OverviewDiagram() {
  const cat = data.items[0];
  return (
    <svg
      viewBox="0 0 760 300"
      style={{ width: '100%', height: 'auto', display: 'block' }}
      role="img"
      aria-label="Diagram: an image goes through the image encoder, a caption goes through the text encoder, and both end up as nearby points in a shared embedding space."
    >
      <defs>
        <clipPath id="ov-img-clip">
          <rect x="24" y="40" width="88" height="88" rx="10" />
        </clipPath>
        <marker id="ov-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--muted)" />
        </marker>
      </defs>

      {/* image branch */}
      <image
        href={galleryUrl(cat.file)}
        x="24"
        y="40"
        width="88"
        height="88"
        preserveAspectRatio="xMidYMid slice"
        clipPath="url(#ov-img-clip)"
      />
      <rect x="24" y="40" width="88" height="88" rx="10" fill="none" stroke="var(--img)" strokeWidth="2" />
      <line x1="120" y1="84" x2="176" y2="84" stroke="var(--muted)" strokeWidth="1.5" markerEnd="url(#ov-arrow)" />
      <rect x="182" y="56" width="150" height="56" rx="10" fill="var(--img-soft)" stroke="var(--img)" strokeWidth="1.5" />
      <text x="257" y="80" textAnchor="middle" fontSize="14" fontWeight="650" fill="var(--ink)">
        Image Encoder
      </text>
      <text x="257" y="98" textAnchor="middle" fontSize="11.5" fill="var(--ink-2)">
        ViT or ResNet
      </text>
      <line x1="338" y1="84" x2="394" y2="84" stroke="var(--muted)" strokeWidth="1.5" markerEnd="url(#ov-arrow)" />
      <rect x="400" y="66" width="118" height="36" rx="8" fill="var(--surface)" stroke="var(--img)" strokeWidth="2" />
      <text x="459" y="89" textAnchor="middle" fontSize="13.5" fontWeight="600" fill="var(--img)">
        image vector
      </text>

      {/* text branch */}
      <rect x="24" y="188" width="150" height="56" rx="10" fill="var(--surface)" stroke="var(--txt)" strokeWidth="2" />
      <text x="99" y="212" textAnchor="middle" fontSize="11.5" fill="var(--ink)">
        “a tabby cat with blue
      </text>
      <text x="99" y="228" textAnchor="middle" fontSize="11.5" fill="var(--ink)">
        eyes looking at the camera”
      </text>
      <line x1="182" y1="216" x2="200" y2="216" stroke="var(--muted)" strokeWidth="1.5" markerEnd="url(#ov-arrow)" />
      <rect x="206" y="188" width="150" height="56" rx="10" fill="var(--txt-soft)" stroke="var(--txt)" strokeWidth="1.5" />
      <text x="281" y="212" textAnchor="middle" fontSize="14" fontWeight="650" fill="var(--ink)">
        Text Encoder
      </text>
      <text x="281" y="230" textAnchor="middle" fontSize="11.5" fill="var(--ink-2)">
        Transformer
      </text>
      <line x1="362" y1="216" x2="394" y2="216" stroke="var(--muted)" strokeWidth="1.5" markerEnd="url(#ov-arrow)" />
      <rect x="400" y="198" width="118" height="36" rx="8" fill="var(--surface)" stroke="var(--txt)" strokeWidth="2" />
      <text x="459" y="221" textAnchor="middle" fontSize="13.5" fontWeight="600" fill="var(--txt-strong)">
        text vector
      </text>

      {/* shared space */}
      <line x1="524" y1="84" x2="588" y2="128" stroke="var(--muted)" strokeWidth="1.5" markerEnd="url(#ov-arrow)" />
      <line x1="524" y1="216" x2="588" y2="176" stroke="var(--muted)" strokeWidth="1.5" markerEnd="url(#ov-arrow)" />
      <circle cx="662" cy="152" r="72" fill="var(--surface)" stroke="var(--baseline)" strokeDasharray="5 4" strokeWidth="1.5" />
      <circle cx="646" cy="138" r="7" fill="var(--img)" />
      <circle cx="668" cy="150" r="7" fill="var(--txt)" />
      <text x="662" y="196" textAnchor="middle" fontSize="12" fill="var(--ink-2)">
        same concept →
      </text>
      <text x="662" y="212" textAnchor="middle" fontSize="12" fill="var(--ink-2)">
        nearby points
      </text>
      <text x="662" y="66" textAnchor="middle" fontSize="12.5" fontWeight="650" fill="var(--ink)">
        shared embedding space
      </text>
    </svg>
  );
}

/** Teaser: real similarity scores between one image and three captions. */
function Teaser() {
  const cat = data.items[0];
  const rivals = [data.items[0], data.items[5], data.items[2]]; // cat, pizza, car
  const scores = rivals.map((r) => dot(cat.imageEmbed, r.captionEmbed));
  const best = scores.indexOf(Math.max(...scores));
  return (
    <div className="card">
      <p className="card-title">A taste of what's coming</p>
      <p className="card-sub">
        Real numbers from the actual CLIP model (ViT-B/32) — the same one you'll run in
        your browser in Chapter 9.
      </p>
      <div style={{ display: 'flex', gap: 18, alignItems: 'center', flexWrap: 'wrap' }}>
        <img
          className="thumb"
          src={galleryUrl(cat.file)}
          alt={cat.caption}
          style={{ width: 120, height: 120 }}
        />
        <div style={{ flex: '1 1 320px' }}>
          <BarChart
            rows={rivals.map((r, i) => ({
              label: `“${r.caption}”`,
              value: scores[i],
              emph: i === best,
            }))}
            max={Math.max(...scores) * 1.25}
            format={(v) => v.toFixed(3)}
          />
        </div>
      </div>
      <p className="small muted" style={{ margin: '12px 0 0' }}>
        CLIP gives the matching caption the highest score — without ever being trained
        on “cat vs. pizza vs. car” as a task. How? That's this whole walkthrough.
      </p>
    </div>
  );
}

export function Ch1BigIdea() {
  return (
    <>
      <h1>The Big Idea: Bridging Vision and Language</h1>
      <p className="lede">
        CLIP (Contrastive Language–Image Pre-training, OpenAI 2021) doesn't just look at
        images — it connects two worlds. It learns to represent{' '}
        <span className="tag img">images</span> and <span className="tag txt">text</span>{' '}
        in the <em>same</em> space, so that a photo of a cat and the sentence “a photo of
        a cat” end up close together.
      </p>

      <h2>The problem with classic image classifiers</h2>
      <p>
        A standard classifier — say, a ResNet trained on ImageNet — ends in a fixed list
        of 1,000 output classes. That design has three painful consequences:
      </p>
      <ul>
        <li>
          <strong>It only knows its list.</strong> If “golden retriever” isn't a class,
          the model literally cannot say it.
        </li>
        <li>
          <strong>Every new task needs new labels.</strong> Want to classify pizza
          styles? Collect and hand-label a new dataset, then retrain.
        </li>
        <li>
          <strong>Labels throw information away.</strong> The label “dog” says nothing
          about what the dog is doing, its color, or the scene around it. A caption does.
        </li>
      </ul>
      <p>
        CLIP's answer: <strong>learn directly from natural language</strong>. The
        internet is full of images that already come with text — alt-text, captions,
        titles. The CLIP team collected roughly{' '}
        <strong>400 million (image, text) pairs</strong> this way. No human labeling
        marathon; the supervision was already out there.
      </p>

      <h2>Two encoders, one shared space</h2>
      <p>
        CLIP trains two networks <em>at the same time</em>: an image encoder and a text
        encoder. Each turns its input into a vector — a list of numbers called an{' '}
        <em>embedding</em>. The training objective (Chapters 4–6) pulls matching pairs
        together in that space and pushes mismatched pairs apart.
      </p>
      <figure>
        <OverviewDiagram />
      </figure>

      <Teaser />

      <h2>What you'll build up to</h2>
      <p>
        By the end of the walkthrough you'll understand — and have played with — every
        stage of the pipeline:
      </p>
      <ul>
        <li>how each encoder turns its input into a vector (Chapters 2–3),</li>
        <li>
          how the <strong>similarity matrix</strong> and the{' '}
          <strong>contrastive loss</strong> shape the shared space (Chapters 4–5),
        </li>
        <li>training a tiny CLIP yourself, live in this page (Chapter 6),</li>
        <li>
          the payoff: <strong>zero-shot classification</strong> and{' '}
          <strong>retrieval</strong> with no task-specific training (Chapters 7–8),
        </li>
        <li>running the real model on your own images, in your browser (Chapter 9).</li>
      </ul>
      <div className="note">
        <span className="note-label">Navigation tip</span>
        Use the <strong>←</strong> and <strong>→</strong> arrow keys (or the buttons
        below) to move between chapters, and the <strong>Contents</strong> button above
        to jump around.
      </div>
    </>
  );
}
