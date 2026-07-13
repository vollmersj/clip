// Fetches openly-licensed images from Wikimedia Commons for the tutorial gallery.
// Records attribution metadata to src/data/credits.json.
import { writeFile, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function fetchWithRetry(url, tries = 6) {
  for (let i = 0; i < tries; i++) {
    try {
      const res = await fetch(url, { headers: { 'User-Agent': 'clip-tutorial-image-fetch/1.0 (educational project)' } });
      if (res.status === 429 || res.status >= 500) {
        const wait = 10000 * 2 ** i;
        console.log(`  ${res.status} for ${url}, retrying in ${wait / 1000}s...`);
        await sleep(wait);
        continue;
      }
      return res;
    } catch (err) {
      const wait = 10000 * 2 ** i;
      console.log(`  network error (${err.cause?.code ?? err.message}), retrying in ${wait / 1000}s...`);
      await sleep(wait);
    }
  }
  throw new Error(`Gave up after ${tries} tries: ${url}`);
}

const OUT_DIR = new URL('../public/gallery/', import.meta.url).pathname;
const CREDITS_PATH = new URL('../src/data/credits.json', import.meta.url).pathname;

const CONCEPTS = [
  { id: 'cat', search: 'tabby cat portrait quality image' },
  { id: 'dog', search: 'golden retriever dog quality image' },
  { id: 'car', search: 'red sports car quality image' },
  { id: 'airplane', search: 'airliner flying sky quality image' },
  { id: 'apple', search: 'red apple fruit quality image' },
  { id: 'pizza', search: 'pizza margherita quality image' },
  { id: 'guitar', search: 'acoustic guitar quality image' },
  { id: 'mountain', search: 'matterhorn mountain quality image' },
];

const OK_LICENSES = /cc0|cc by|public domain|pd/i;

async function api(params) {
  const url = new URL('https://commons.wikimedia.org/w/api.php');
  for (const [k, v] of Object.entries({ format: 'json', ...params })) url.searchParams.set(k, v);
  const res = await fetchWithRetry(url.toString());
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}

async function findImage(concept) {
  const data = await api({
    action: 'query',
    generator: 'search',
    gsrsearch: `${concept.search} filetype:bitmap`,
    gsrnamespace: '6',
    gsrlimit: '10',
    prop: 'imageinfo',
    iiprop: 'url|extmetadata|mime',
    iiurlwidth: '640',
  });
  const pages = Object.values(data.query?.pages ?? {}).sort((a, b) => a.index - b.index);
  for (const page of pages) {
    const info = page.imageinfo?.[0];
    if (!info) continue;
    if (!/jpeg|png/.test(info.mime)) continue;
    const meta = info.extmetadata ?? {};
    const license = meta.LicenseShortName?.value ?? '';
    if (!OK_LICENSES.test(license)) continue;
    return {
      id: concept.id,
      title: page.title,
      thumbUrl: info.thumburl,
      descriptionUrl: info.descriptionurl,
      artist: (meta.Artist?.value ?? 'Unknown').replace(/<[^>]*>/g, '').trim(),
      license,
      licenseUrl: meta.LicenseUrl?.value ?? '',
    };
  }
  throw new Error(`No suitable image for ${concept.id}`);
}

await mkdir(OUT_DIR, { recursive: true });
await mkdir(path.dirname(CREDITS_PATH), { recursive: true });

const credits = [];
for (const concept of CONCEPTS) {
  const img = await findImage(concept);
  const ext = img.thumbUrl.includes('.png') ? 'png' : 'jpg';
  const file = `${concept.id}.${ext}`;
  const outPath = path.join(OUT_DIR, file);
  if (!existsSync(outPath)) {
    const res = await fetchWithRetry(img.thumbUrl);
    if (!res.ok) throw new Error(`Download failed for ${concept.id}: ${res.status}`);
    const buf = Buffer.from(await res.arrayBuffer());
    await writeFile(outPath, buf);
    console.log(`✓ ${concept.id}: ${img.title} [${img.license}] -> ${file} (${(buf.length / 1024).toFixed(0)} KB)`);
    await sleep(2000);
  } else {
    console.log(`• ${concept.id}: already downloaded (${file})`);
  }
  credits.push({ ...img, file });
}

await writeFile(CREDITS_PATH, JSON.stringify(credits, null, 2));
console.log(`\nWrote ${credits.length} credits to ${CREDITS_PATH}`);
