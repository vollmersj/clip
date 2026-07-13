import raw from '../data/embeddings.json';
import creditsRaw from '../data/credits.json';

export interface GalleryItem {
  id: string;
  /** Bare class name used for zero-shot prompts, e.g. "cat". */
  className: string;
  /** Natural caption paired with the image (the "training pair" text). */
  caption: string;
  file: string;
  imageEmbed: number[];
  captionEmbed: number[];
  /** BPE token strings of the caption, including [SOS]/[EOS]. */
  captionTokens: string[];
}

export interface EmbeddingData {
  modelId: string;
  dim: number;
  items: GalleryItem[];
  templates: string[];
  /** templateEmbeds[t][c] = text embedding of templates[t] filled with items[c].className */
  templateEmbeds: number[][][];
  queries: string[];
  queryEmbeds: number[][];
}

export interface ImageCredit {
  id: string;
  title: string;
  descriptionUrl: string;
  artist: string;
  license: string;
  licenseUrl: string;
  file: string;
}

export const data = raw as EmbeddingData;
export const credits = creditsRaw as ImageCredit[];

export const galleryUrl = (file: string) =>
  `${import.meta.env.BASE_URL}gallery/${file}`;

/**
 * Fill a prompt template and fix the article ("a airplane" → "an airplane").
 * Must stay in sync with fillTemplate() in scripts/precompute.mjs, so the
 * displayed text is exactly the text that was embedded.
 */
export const fillTemplate = (template: string, className: string) =>
  template.replace('{}', className).replace(/\ba ([aeiou])/g, 'an $1');
