// Precomputes real CLIP embeddings for the tutorial's gallery images and texts
// using the same model the in-browser playground loads (Xenova/clip-vit-base-patch32),
// so precomputed numbers match live numbers. Writes src/data/embeddings.json.
import {
  AutoTokenizer,
  AutoProcessor,
  CLIPTextModelWithProjection,
  CLIPVisionModelWithProjection,
  RawImage,
  env,
} from '@huggingface/transformers';
import { readFile, writeFile } from 'node:fs/promises';

// Load from the locally downloaded copy (scripts/download_model.sh) —
// the sandbox network cuts long streaming downloads, curl -C - does not care.
env.localModelPath = new URL('../models/', import.meta.url).pathname;
env.allowRemoteModels = false;

const MODEL_ID = 'Xenova/clip-vit-base-patch32';
const GALLERY_DIR = new URL('../public/gallery/', import.meta.url).pathname;
const OUT_PATH = new URL('../src/data/embeddings.json', import.meta.url).pathname;

// The 8 concepts, in gallery order. Captions are written to match the actual
// downloaded photos (natural web-style captions, like CLIP's training pairs).
const ITEMS = [
  { id: 'cat', className: 'cat', caption: 'a tabby cat with blue eyes looking at the camera' },
  { id: 'dog', className: 'dog', caption: 'the face of a golden retriever looking up at the sky' },
  { id: 'car', className: 'car', caption: 'a red volkswagen golf parked in a showroom' },
  { id: 'airplane', className: 'airplane', caption: 'a jumbo jet flying overhead leaving white contrails' },
  { id: 'apple', className: 'apple', caption: 'a red apple on a white background' },
  { id: 'pizza', className: 'pizza', caption: 'a pizza margherita with melted mozzarella and basil' },
  { id: 'guitar', className: 'guitar', caption: 'a black and white photo of a hand playing a guitar' },
  { id: 'mountain', className: 'mountain', caption: 'two hikers walking on a trail towards a snowy mountain' },
];

// Prompt templates for the zero-shot chapter (chapter 7). '{}' is replaced by the class name.
const TEMPLATES = [
  '{}',
  'a photo of a {}',
  'an image of a {}',
  'a blurry photo of a {}',
  'a drawing of a {}',
  'a photo of a small {}',
];

// Free-text retrieval queries for chapter 8 — none mentions a class name directly,
// so ranking demonstrates semantic (not string) matching.
const QUERIES = [
  'a fluffy animal',
  "man's best friend",
  'something you can drive',
  'something that flies',
  'a musical instrument',
  'italian food',
  'a healthy snack',
  'a snowy landscape',
  'a fast vehicle',
  'a pet sitting indoors',
];

console.log('Loading model (first run downloads weights)...');
// q8 matches what the in-browser playground loads, so numbers stay consistent.
const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
const textModel = await CLIPTextModelWithProjection.from_pretrained(MODEL_ID, { dtype: 'q8' });
const processor = await AutoProcessor.from_pretrained(MODEL_ID);
const visionModel = await CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, { dtype: 'q8' });

function l2normalize(vec) {
  const norm = Math.sqrt(vec.reduce((s, x) => s + x * x, 0));
  return vec.map((x) => x / norm);
}

const round = (vec) => vec.map((x) => Math.round(x * 1e5) / 1e5);

async function embedTexts(texts) {
  const inputs = tokenizer(texts, { padding: true, truncation: true });
  const { text_embeds } = await textModel(inputs);
  const [n, d] = text_embeds.dims;
  const out = [];
  for (let i = 0; i < n; i++) {
    out.push(round(l2normalize(Array.from(text_embeds.data.slice(i * d, (i + 1) * d)))));
  }
  return out;
}

async function embedImage(file) {
  const image = await RawImage.read(GALLERY_DIR + file);
  const inputs = await processor(image);
  const { image_embeds } = await visionModel(inputs);
  return round(l2normalize(Array.from(image_embeds.data)));
}

const credits = JSON.parse(await readFile(new URL('../src/data/credits.json', import.meta.url), 'utf8'));
const fileById = Object.fromEntries(credits.map((c) => [c.id, c.file]));

console.log('Embedding gallery images...');
const imageEmbeds = [];
for (const item of ITEMS) {
  const file = fileById[item.id];
  if (!file) throw new Error(`No downloaded image for ${item.id} — run fetch_images.mjs first`);
  imageEmbeds.push(await embedImage(file));
  console.log(`  ✓ ${item.id}`);
}

console.log('Embedding captions...');
const captionEmbeds = await embedTexts(ITEMS.map((i) => i.caption));

// Per-caption BPE token strings (for the text-encoder chapter).
// Decoding each id separately yields a readable per-token string.
const captionTokens = ITEMS.map((item) => {
  const ids = tokenizer.encode(item.caption);
  return ids.map((id) => {
    const s = tokenizer.decode([id]);
    if (s === '<|startoftext|>') return '[SOS]';
    if (s === '<|endoftext|>') return '[EOS]';
    return s;
  });
});

console.log('Embedding prompt-template × class texts...');
// Fill a template and fix the article: "a photo of a airplane" → "an airplane".
// Must stay in sync with fillTemplate() in src/lib/data.ts.
const fillTemplate = (template, className) =>
  template.replace('{}', className).replace(/\ba ([aeiou])/g, 'an $1');
const templateTexts = [];
for (const template of TEMPLATES) {
  for (const item of ITEMS) {
    templateTexts.push(fillTemplate(template, item.className));
  }
}
const templateEmbedsFlat = await embedTexts(templateTexts);
// templateEmbeds[t][c] = embedding of TEMPLATES[t] filled with ITEMS[c].className
const templateEmbeds = TEMPLATES.map((_, t) =>
  ITEMS.map((_, c) => templateEmbedsFlat[t * ITEMS.length + c]),
);

console.log('Embedding retrieval queries...');
const queryEmbeds = await embedTexts(QUERIES);

const out = {
  modelId: MODEL_ID,
  dim: imageEmbeds[0].length,
  items: ITEMS.map((item, i) => ({
    ...item,
    file: fileById[item.id],
    imageEmbed: imageEmbeds[i],
    captionEmbed: captionEmbeds[i],
    captionTokens: captionTokens[i],
  })),
  templates: TEMPLATES,
  templateEmbeds,
  queries: QUERIES,
  queryEmbeds,
};

await writeFile(OUT_PATH, JSON.stringify(out));
const kb = (JSON.stringify(out).length / 1024).toFixed(0);
console.log(`\nWrote ${OUT_PATH} (${kb} KB, dim=${out.dim})`);

// Sanity check: the similarity matrix diagonal should dominate its row.
console.log('\nSanity check — cosine(image_i, caption_j) matrix:');
const dot = (a, b) => a.reduce((s, x, k) => s + x * b[k], 0);
for (let i = 0; i < ITEMS.length; i++) {
  const row = captionEmbeds.map((t) => dot(imageEmbeds[i], t));
  const best = row.indexOf(Math.max(...row));
  const marker = best === i ? '✓' : '✗ MISMATCH';
  console.log(
    `  ${ITEMS[i].id.padEnd(9)} ${row.map((x) => x.toFixed(3)).join(' ')}  ${marker}`,
  );
}
