import { useState } from 'react';
import { credits, galleryUrl } from '../lib/data';

interface QuizItem {
  q: string;
  options: string[];
  correct: number;
  explain: string;
}

const QUIZ: QuizItem[] = [
  {
    q: 'What does CLIP’s training objective directly optimize?',
    options: [
      'Predicting the exact words of each caption',
      'Ranking the true (image, text) pairing above all wrong pairings in the batch',
      'Reconstructing the input image from its embedding',
      'Classifying images into 1,000 fixed categories',
    ],
    correct: 1,
    explain:
      'The contrastive loss only cares that each image scores its own caption highest (and vice versa). The paper found this ~4× more compute-efficient than predicting caption words.',
  },
  {
    q: 'How does CLIP measure how well an image and a text match?',
    options: [
      'Euclidean distance between raw pixels and characters',
      'A separate neural network scores each pair',
      'Cosine similarity between their L2-normalized embeddings',
      'The number of overlapping words between the caption and image tags',
    ],
    correct: 2,
    explain:
      'Both encoders output vectors that are normalized to length 1, so a plain dot product equals the cosine of the angle between them (Chapter 3).',
  },
  {
    q: 'The temperature τ is made very LARGE (say τ = 5). What happens to the row softmax?',
    options: [
      'It sharpens — the top pair takes all the probability',
      'It flattens toward a uniform distribution',
      'It becomes exactly 0 everywhere',
      'Nothing — τ only affects training speed',
    ],
    correct: 1,
    explain:
      'Logits are cosine ÷ τ, so a large τ shrinks all differences and the softmax approaches uniform. Small τ sharpens instead. That’s why CLIP learns τ (ending near 0.01).',
  },
  {
    q: 'A demo shows “dog: 85%”. What is that number?',
    options: [
      'The cosine similarity between the image and “dog” (0.85)',
      'The fraction of the image that contains a dog',
      'A softmax probability over the candidate labels the user provided',
      'The model’s training accuracy on dog images',
    ],
    correct: 2,
    explain:
      'Raw cosines top out around 0.3. The 85% comes from softmax(cosines ÷ τ) over the label set — change the label list and the same image gets different percentages (Chapters 5 & 7).',
  },
  {
    q: 'You need to classify products into 15 brand-new categories. With CLIP zero-shot, what do you need?',
    options: [
      'A labeled training set with a few thousand examples per category',
      'Fine-tuning the image encoder for a few epochs',
      'Just the 15 category names, ideally wrapped in a prompt template',
      'Re-training the text encoder on product descriptions',
    ],
    correct: 2,
    explain:
      'The text encoder turns the category names into classifier weights on the fly — the “classifier factory” view from Chapter 7. No gradient steps needed.',
  },
];

function Quiz() {
  const [answers, setAnswers] = useState<(number | null)[]>(QUIZ.map(() => null));
  const score = answers.filter((a, i) => a === QUIZ[i].correct).length;
  const done = answers.every((a) => a !== null);

  return (
    <div>
      {QUIZ.map((item, qi) => {
        const picked = answers[qi];
        return (
          <div className="card" key={qi}>
            <p className="quiz-q">
              {qi + 1}. {item.q}
            </p>
            <div className="quiz-options">
              {item.options.map((opt, oi) => {
                let cls = 'quiz-option';
                if (picked !== null) {
                  if (oi === item.correct) cls += ' correct';
                  else if (oi === picked) cls += ' wrong';
                }
                return (
                  <button
                    key={oi}
                    className={cls}
                    disabled={picked !== null}
                    onClick={() =>
                      setAnswers((a) => a.map((v, i) => (i === qi ? oi : v)))
                    }
                  >
                    {opt}
                  </button>
                );
              })}
            </div>
            {picked !== null && <p className="quiz-explain">{item.explain}</p>}
          </div>
        );
      })}
      {done && (
        <div className="note">
          <span className="note-label">Your score</span>
          {score} / {QUIZ.length}{' '}
          {score === QUIZ.length
            ? '— perfect! You can explain CLIP to your classmates now.'
            : score >= 3
              ? '— solid. Revisit the chapters behind the ones you missed.'
              : '— worth another pass through Chapters 3–7; the interactives make it stick.'}
        </div>
      )}
    </div>
  );
}

export function Ch10Recap() {
  return (
    <>
      <h1>Recap, Pseudocode & Quiz</h1>
      <p className="lede">
        The entire system fits on one slide — the paper's own pseudocode (Figure 3).
        Every line should now read as a chapter you've already played with.
      </p>

      <pre className="codeblock">{`# image_encoder - ResNet or Vision Transformer          → Chapter 2
# text_encoder  - CBOW or Text Transformer              → Chapter 2
# I[n, h, w, c] - minibatch of aligned images
# T[n, l]       - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t             - learned temperature parameter         → Chapter 5

# extract feature representations of each modality
I_f = image_encoder(I)  # [n, d_i]
T_f = text_encoder(T)   # [n, d_t]

# joint multimodal embedding [n, d_e]                   → Chapter 3
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)

# scaled pairwise cosine similarities [n, n]            → Chapters 4–5
logits = np.dot(I_e, T_e.T) * np.exp(t)

# symmetric loss function                               → Chapters 5–6
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss   = (loss_i + loss_t) / 2`}</pre>
      <p className="small muted">
        One subtlety: the paper multiplies by <code>exp(t)</code> with t learned — same
        thing as dividing by τ, just parameterized so the scale stays positive during
        learning.
      </p>

      <h2>The pipeline in five sentences</h2>
      <ol>
        <li>Two encoders map images and texts into one shared vector space.</li>
        <li>Embeddings are L2-normalized, so similarity = cosine = dot product.</li>
        <li>
          In each batch, a symmetric cross-entropy over the similarity matrix pulls
          true pairs together and pushes the other pairings apart.
        </li>
        <li>
          After training on 400M pairs, the space is so well organized that class
          names <em>written as sentences</em> act as classifier weights — zero-shot
          classification.
        </li>
        <li>The same space powers search in both directions — retrieval.</li>
      </ol>

      <h2>What CLIP is not good at</h2>
      <ul>
        <li>
          <strong>Systematic, fine-grained tasks:</strong> counting objects,
          reading gauges, distinguishing aircraft models or lymph-node tumors —
          zero-shot CLIP lags far behind specialized models.
        </li>
        <li>
          <strong>Prompt sensitivity:</strong> as you saw in Chapter 7, wording changes
          predictions; good templates are part of the system.
        </li>
        <li>
          <strong>Typographic attacks:</strong> it can be fooled by text written on
          objects, because it learned to read.
        </li>
        <li>
          <strong>Inherited bias:</strong> 400M internet pairs import the internet's
          stereotypes and gaps; the paper dedicates a section (7) to these harms.
        </li>
      </ul>

      <h2>Check yourself</h2>
      <Quiz />

      <h2>Go further</h2>
      <ul>
        <li>
          The paper:{' '}
          <a href="https://arxiv.org/abs/2103.00020" target="_blank" rel="noreferrer">
            Radford et al., “Learning Transferable Visual Models From Natural Language
            Supervision” (2021)
          </a>
        </li>
        <li>
          Open-source reproductions at larger scale:{' '}
          <a href="https://github.com/mlfoundations/open_clip" target="_blank" rel="noreferrer">
            OpenCLIP
          </a>{' '}
          and the LAION datasets.
        </li>
        <li>
          The in-browser model used here:{' '}
          <a
            href="https://huggingface.co/Xenova/clip-vit-base-patch32"
            target="_blank"
            rel="noreferrer"
          >
            Xenova/clip-vit-base-patch32
          </a>{' '}
          via{' '}
          <a href="https://huggingface.co/docs/transformers.js" target="_blank" rel="noreferrer">
            transformers.js
          </a>
          .
        </li>
      </ul>

      <h2>Image credits</h2>
      <p className="small muted">
        Gallery photographs from Wikimedia Commons, used under their respective free
        licenses:
      </p>
      <table className="credits-table">
        <tbody>
          {credits.map((c) => (
            <tr key={c.id}>
              <td>
                <img className="thumb" src={galleryUrl(c.file)} alt={c.id} />
              </td>
              <td>
                <a href={c.descriptionUrl} target="_blank" rel="noreferrer">
                  {c.title.replace(/^File:/, '')}
                </a>
              </td>
              <td>{c.artist}</td>
              <td>
                {c.licenseUrl ? (
                  <a href={c.licenseUrl} target="_blank" rel="noreferrer">
                    {c.license}
                  </a>
                ) : (
                  c.license
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
