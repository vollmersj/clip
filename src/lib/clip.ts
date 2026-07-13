/**
 * Lazy in-browser CLIP via transformers.js (loaded on demand so the main
 * bundle stays small). Uses the same q8 ViT-B/32 weights the precomputed
 * embeddings came from, so live numbers match the rest of the walkthrough.
 */

export const MODEL_ID = 'Xenova/clip-vit-base-patch32';

export interface OverallProgress {
  loadedMB: number;
  totalMB: number;
  /** 0..1 across all files with known sizes. */
  fraction: number;
}

export interface ClipModel {
  embedTexts(texts: string[]): Promise<number[][]>;
  embedImageBlob(blob: Blob): Promise<number[]>;
  embedImageUrl(url: string): Promise<number[]>;
}

function l2normalize(vec: number[]): number[] {
  const n = Math.sqrt(vec.reduce((s, x) => s + x * x, 0)) || 1;
  return vec.map((x) => x / n);
}

let modelPromise: Promise<ClipModel> | null = null;

export function getClip(onProgress?: (p: OverallProgress) => void): Promise<ClipModel> {
  if (!modelPromise) {
    modelPromise = load(onProgress).catch((err) => {
      modelPromise = null; // allow retry after a failure
      throw err;
    });
  }
  return modelPromise;
}

async function load(onProgress?: (p: OverallProgress) => void): Promise<ClipModel> {
  const T = await import('@huggingface/transformers');
  T.env.allowLocalModels = false;

  const files = new Map<string, { loaded: number; total: number }>();
  const report = () => {
    if (!onProgress) return;
    let loaded = 0;
    let total = 0;
    for (const f of files.values()) {
      loaded += f.loaded;
      total += f.total;
    }
    onProgress({
      loadedMB: loaded / 1e6,
      totalMB: total / 1e6,
      fraction: total > 0 ? loaded / total : 0,
    });
  };
  // transformers.js emits per-file progress events during downloads
  const progress_callback = (p: {
    status: string;
    file?: string;
    loaded?: number;
    total?: number;
  }) => {
    if (p.status === 'progress' && p.file && p.total) {
      files.set(p.file, { loaded: p.loaded ?? 0, total: p.total });
      report();
    } else if (p.status === 'done' && p.file) {
      const f = files.get(p.file);
      if (f) files.set(p.file, { ...f, loaded: f.total });
      report();
    }
  };

  const [tokenizer, textModel, processor, visionModel] = await Promise.all([
    T.AutoTokenizer.from_pretrained(MODEL_ID, { progress_callback }),
    T.CLIPTextModelWithProjection.from_pretrained(MODEL_ID, {
      dtype: 'q8',
      progress_callback,
    }),
    T.AutoProcessor.from_pretrained(MODEL_ID, { progress_callback }),
    T.CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, {
      dtype: 'q8',
      progress_callback,
    }),
  ]);

  async function embedRawImage(image: InstanceType<typeof T.RawImage>): Promise<number[]> {
    const inputs = await processor(image);
    const { image_embeds } = await visionModel(inputs);
    return l2normalize(Array.from(image_embeds.data as Float32Array));
  }

  return {
    async embedTexts(texts: string[]): Promise<number[][]> {
      const inputs = tokenizer(texts, { padding: true, truncation: true });
      const { text_embeds } = await textModel(inputs);
      const [n, d] = text_embeds.dims as [number, number];
      const flat = text_embeds.data as Float32Array;
      return Array.from({ length: n }, (_, i) =>
        l2normalize(Array.from(flat.slice(i * d, (i + 1) * d))),
      );
    },
    async embedImageBlob(blob: Blob): Promise<number[]> {
      return embedRawImage(await T.RawImage.fromBlob(blob));
    },
    async embedImageUrl(url: string): Promise<number[]> {
      return embedRawImage(await T.RawImage.fromURL(url));
    },
  };
}
