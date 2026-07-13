# Inside CLIP — An Interactive Walkthrough

An interactive, self-contained teaching app that walks bachelor students through
**CLIP** (Radford et al., 2021, [arXiv:2103.00020](https://arxiv.org/abs/2103.00020))
end to end:

1. **The Big Idea** — natural-language supervision vs. fixed-class classifiers
2. **Two Encoders** — real BPE tokens, ViT patches, real 512-d embeddings
3. **A Shared Embedding Space** — draggable unit-circle cosine-similarity demo
4. **The Similarity Matrix** — the real 8×8 matrix from real CLIP embeddings
5. **The Contrastive Loss** — temperature slider, row/column softmax, symmetric cross-entropy
6. **Train It Yourself** — a tiny CLIP trained live in the page (exact InfoNCE loss + gradients)
7. **Zero-Shot Classification** — editable class lists and prompt templates
8. **Retrieval** — semantic search in both directions
9. **Playground** — loads the actual quantized ViT-B/32 into the browser (transformers.js); classify your own images, fully client-side
10. **Recap** — the paper's pseudocode annotated, limitations, self-check quiz

All similarity numbers in chapters 1–8 are **precomputed with the same model**
(`Xenova/clip-vit-base-patch32`, q8 ONNX) that the playground loads live, so
everything is consistent.

## Develop

```bash
npm install
npm run dev
```

## Test

```bash
node scripts/test_grad.ts   # numerical gradient check for the training sim
```

## Deploy to GitHub Pages

1. Create a GitHub repository and push this project to the `main` branch.
2. In the repo settings → **Pages**, set **Source: GitHub Actions**.
3. Done — `.github/workflows/deploy.yml` builds and deploys on every push.

The Vite config uses `base: './'`, so the build works from any Pages subpath.
No server is needed: the playground model is fetched from the Hugging Face CDN
at runtime and inference runs in WebAssembly in the visitor's browser.

## Regenerating the data (optional)

The gallery images and embeddings are checked in. To rebuild them:

```bash
node scripts/fetch_images.mjs     # pulls freely-licensed images from Wikimedia Commons
./scripts/download_model.sh       # downloads the q8 ONNX model into models/ (resumable)
node scripts/precompute.mjs       # writes src/data/embeddings.json
```

## Credits

- Paper: Radford et al., *Learning Transferable Visual Models From Natural
  Language Supervision*, 2021.
- In-browser model: [Xenova/clip-vit-base-patch32](https://huggingface.co/Xenova/clip-vit-base-patch32)
  via [transformers.js](https://huggingface.co/docs/transformers.js).
- Gallery photographs: Wikimedia Commons under free licenses — full attribution
  in the app's final chapter and `src/data/credits.json`.
